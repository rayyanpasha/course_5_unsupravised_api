from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

# Initialize Flask app
app = Flask(__name__)

# Load the dataset and cluster analysis report
data_file = 'normal_points_clusters_multivariate.csv'
data = pd.read_csv(data_file)

report_file = 'cluster_analysis_report_multivariate.txt'
with open(report_file, 'r') as file:
    cluster_report = file.read()

# Define preprocessing and clustering logic
numerical_features = ['number_of_kills', 'number_of_wounded']
categorical_features = ['region', 'attack_type']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# Fit the preprocessing pipeline on the entire dataset
X = data[numerical_features + categorical_features]
X_transformed = preprocessor.fit_transform(X)

# PCA for dimensionality reduction
pca = PCA(n_components=0.95, random_state=42)
X_reduced = pca.fit_transform(X_transformed)

# Clustering with MiniBatchKMeans
optimal_k = len(data['cluster'].unique())  # Number of clusters from the dataset
kmeans = MiniBatchKMeans(n_clusters=optimal_k, init='k-means++', random_state=42, batch_size=1024, max_iter=200)
kmeans.fit(X_reduced)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    input_data = request.json

    # Extract and preprocess the input
    input_df = pd.DataFrame([input_data])
    input_transformed = preprocessor.transform(input_df)
    input_reduced = pca.transform(input_transformed)

    # Predict cluster
    cluster_label = kmeans.predict(input_reduced)[0]

    # Filter cluster details
    cluster_details = data[data['cluster'] == cluster_label]
    avg_kills = cluster_details['number_of_kills'].mean()
    avg_wounded = cluster_details['number_of_wounded'].mean()
    common_region = cluster_details['region'].mode()[0]
    common_attack = cluster_details['attack_type'].mode()[0]

    response = {
        'cluster': int(cluster_label),
        'details': {
            'average_kills': round(avg_kills, 2),
            'average_wounded': round(avg_wounded, 2),
            'most_common_region': common_region,
            'most_common_attack_type': common_attack
        }
    }

    return jsonify(response)

@app.route('/cluster-report', methods=['GET'])
def cluster_report_view():
    return jsonify({'report': cluster_report})

if __name__ == '__main__':
    app.run(debug=True)
