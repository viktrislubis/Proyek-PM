from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)

# Load data
data = pd.read_csv('Data_Tanaman_Padi_Sumatera_version_1.csv')

# Step 1: Clean outliers (remove extreme values in Produksi)
Q1 = data['Produksi'].quantile(0.25)
Q3 = data['Produksi'].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data['Produksi'] < (Q1 - 1.5 * IQR)) | (data['Produksi'] > (Q3 + 1.5 * IQR)))]

# Step 2: Add new features (Provinsi as dummy variables and interaction term)
data = pd.get_dummies(data, columns=['Provinsi'], drop_first=True)
data['Hujan_Kelembapan'] = data['Curah hujan'] * data['Kelembapan']

# Prepare features and target
provinsi_columns = [col for col in data.columns if col.startswith('Provinsi_')]
feature_columns = ['Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata', 'Hujan_Kelembapan'] + provinsi_columns
X = data[feature_columns]
y = data['Produksi']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Clustering for categorization
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Define categories based on cluster means
cluster_means = data.groupby('Cluster')['Produksi'].mean().sort_values()
cluster_labels = {cluster_means.index[0]: 'Rendah', cluster_means.index[1]: 'Sedang', cluster_means.index[2]: 'Tinggi'}
data['Kategori'] = data['Cluster'].map(cluster_labels)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Daftar provinsi statis (semua provinsi di Sumatera)
    provinsi_list = [
        'Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Jambi',
        'Sumatera Selatan', 'Bengkulu', 'Lampung'
    ]
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
                'Hujan_Kelembapan': [hujan * kelembapan]
            })

            # Add dummy variables for Provinsi
            for prov in provinsi_list:
                prov_col = f'Provinsi_{prov}'
                input_data[prov_col] = 1 if prov == provinsi else 0

            # Ensure input_data has the same columns as X
            input_data = input_data[feature_columns]

            # Normalize input data
            input_scaled = scaler.transform(input_data)

            # Predict production
            prediction = model.predict(input_scaled)[0]

            # Predict cluster
            cluster = kmeans.predict(input_scaled)[0]
            cluster_label = cluster_labels[cluster]

            # Calculate average production for the cluster
            avg_cluster_prod = data[data['Cluster'] == cluster]['Produksi'].mean()

            show_result = True

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('index.html', provinsi_list=provinsi_list, tahun_list=tahun_list,
                           show_result=show_result, prediction=prediction, cluster_label=cluster_label,
                           avg_cluster_prod=avg_cluster_prod, error=error)

if __name__ == '__main__':
    app.run(debug=True)