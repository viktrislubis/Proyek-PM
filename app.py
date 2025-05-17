from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge  
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

app = Flask(__name__)

# Load data
data = pd.read_csv('Data_Tanaman_Padi_Sumatera_version_1.csv')

# Step 1: Clean outliers with stricter threshold
Q1 = data['Produksi'].quantile(0.25)
Q3 = data['Produksi'].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data['Produksi'] < (Q1 - 2.0 * IQR)) | (data['Produksi'] > (Q3 + 2.0 * IQR)))]

# Step 2: Feature engineering
data = pd.get_dummies(data, columns=['Provinsi'], drop_first=True)
data['Hujan_Kelembapan'] = data['Curah hujan'] * data['Kelembapan']
data['Luas_Hujan_Ratio'] = data['Luas Panen'] / (data['Curah hujan'] + 1)  # Hindari divisi nol
data['Suhu_Squared'] = data['Suhu rata-rata'] ** 2
data['Luas_Suhu_Interaction'] = data['Luas Panen'] * data['Suhu rata-rata']

# Prepare features and target
provinsi_columns = [col for col in data.columns if col.startswith('Provinsi_')]
feature_columns = ['Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata', 'Hujan_Kelembapan', 
                   'Luas_Hujan_Ratio', 'Suhu_Squared', 'Luas_Suhu_Interaction'] + provinsi_columns
X = data[feature_columns]
y = np.log1p(data['Produksi'])  

# Preprocessor dan model regresi
reg_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata', 'Hujan_Kelembapan', 
                              'Luas_Hujan_Ratio', 'Suhu_Squared', 'Luas_Suhu_Interaction']),
    ('cat', 'passthrough', provinsi_columns)
])
model = Pipeline([
    ('preprocessor', reg_preprocessor),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  
    ('regressor', Ridge(alpha=1.0))  
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Evaluate model (transform back to original scale)
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)
mae = mean_absolute_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))

print(f"Linear Regression with Ridge - Mean Absolute Error: {mae}")
print(f"Linear Regression with Ridge - R² Score: {r2}")
print(f"Linear Regression with Ridge - RMSE: {rmse}")

# Clustering for categorization
cluster_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata', 'Hujan_Kelembapan', 
                              'Luas_Hujan_Ratio', 'Suhu_Squared', 'Luas_Suhu_Interaction']),
    ('cat', 'passthrough', provinsi_columns)
])
X_scaled = cluster_preprocessor.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Define categories based on cluster means
cluster_means = data.groupby('Cluster')['Produksi'].mean().sort_values()
cluster_labels = {cluster_means.index[0]: 'Rendah', cluster_means.index[1]: 'Sedang', cluster_means.index[2]: 'Tinggi'}
data['Kategori'] = data['Cluster'].map(cluster_labels)

# Route untuk halaman pengantar
@app.route('/')
def intro():
    return render_template('dashboard.html', mae=mae, r2=r2, rmse=rmse)

# Route untuk form dan prediksi
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    provinsi_list = ['Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Jambi', 'Sumatera Selatan', 'Bengkulu', 'Lampung']
    tahun_list = sorted(data['Tahun'].unique())
    show_result = False
    prediction = None
    cluster_label = None
    avg_cluster_prod = None
    error = None

    if request.method == 'POST':
        try:
            provinsi = request.form['provinsi']
            tahun = int(request.form['tahun'])
            luas = float(request.form['luas'])
            hujan = float(request.form['hujan'])
            kelembapan = float(request.form['kelembapan'])
            suhu = float(request.form['suhu'])

            # Prepare input data for prediction
            input_data = pd.DataFrame({
                'Luas Panen': [luas],
                'Curah hujan': [hujan],
                'Kelembapan': [kelembapan],
                'Suhu rata-rata': [suhu],
                'Hujan_Kelembapan': [hujan * kelembapan],
                'Luas_Hujan_Ratio': [luas / (hujan + 1)],
                'Suhu_Squared': [suhu ** 2],
                'Luas_Suhu_Interaction': [luas * suhu]
            })

            # Add dummy variables for Provinsi
            for prov in provinsi_list:
                prov_col = f'Provinsi_{prov}'
                input_data[prov_col] = 1 if prov == provinsi else 0

            # Ensure input_data has the same columns as X
            input_data = input_data[feature_columns]

            # Predict production (transform back to original scale)
            prediction_log = model.predict(input_data)[0]
            prediction = np.expm1(prediction_log)

            # Predict cluster
            cluster_input = cluster_preprocessor.transform(input_data)
            cluster = kmeans.predict(cluster_input)[0]
            cluster_label = cluster_labels[cluster]
            avg_cluster_prod = data[data['Cluster'] == cluster]['Produksi'].mean()

            show_result = True

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('index.html',
                       provinsi_list=provinsi_list,
                       tahun_list=tahun_list,
                       show_result=show_result,
                       prediction=prediction,
                       cluster_label=cluster_label,
                       avg_cluster_prod=avg_cluster_prod,
                       error=error,
                       mae=mae,
                       r2=r2,
                       rmse=rmse)


if __name__ == '__main__':
    app.run(debug=True)