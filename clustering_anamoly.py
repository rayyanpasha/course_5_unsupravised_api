import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data_file = 'cleaned_file.csv'
df = pd.read_csv(data_file)

# Select variables for LOF
numerical_features = ['number_of_kills', 'number_of_wounded']
categorical_features = ['region', 'attack_type']

# Fill missing values
df[numerical_features] = df[numerical_features].fillna(0)
df[categorical_features] = df[categorical_features].fillna('Unknown')

# Encode categorical features using LabelEncoder
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Combine features for LOF
X = df[numerical_features + categorical_features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply LOF
n_neighbors = 20
lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.01)
y_pred = lof.fit_predict(X_scaled)
lof_scores = -lof.negative_outlier_factor_

# Add LOF scores and outlier flag to the original DataFrame
df['lof_score'] = lof_scores
threshold = np.mean(lof_scores) + 2 * np.std(lof_scores)
df['is_outlier'] = lof_scores > threshold

# Decode categorical columns
for col in categorical_features:
    df[col] = df[col].apply(lambda x: label_encoders[col].inverse_transform([int(x)])[0])

# Save results
outliers = df[df['is_outlier']]
normal_points = df[~df['is_outlier']]

# Save the outliers and normal points to CSV
outliers.to_csv('outliers_lof.csv', index=False)
normal_points.to_csv('normal_points_lof.csv', index=False)

# Generate PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization and analysis
X_pca = pca.fit_transform(X_scaled)

# PCA explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# PCA Loading matrix (components)
pca_loading_matrix = pd.DataFrame(pca.components_, columns=numerical_features + categorical_features)

# Analysis report generation
def generate_report():
    report = []

    # Basic dataset and outlier information
    report.append(f"Total records analyzed: {len(df)}")
    report.append(f"Number of outliers detected: {df['is_outlier'].sum()}")
    report.append(f"Percentage of outliers: {(df['is_outlier'].sum() / len(df) * 100):.2f}%")
    report.append(f"LOF Score threshold: {threshold:.2f}")

    # PCA analysis
    report.append("\nPCA Explained Variance Ratio (Top 2 Components):")
    for i, var in enumerate(explained_variance_ratio):
        report.append(f"PC{i+1}: {var:.2f}")
    total_variance = np.sum(explained_variance_ratio)
    report.append(f"Total variance explained by the first 2 components: {total_variance:.2f}")

    # PCA Loading matrix (Top components)
    report.append("\nPCA Loading Matrix (Top 2 Components):")
    report.append(pca_loading_matrix.to_string(index=False))

    # Generate deeper insights into outliers
    report.append("\nDeeper Insights into Outliers:")
    top_outliers = outliers.nlargest(10, 'lof_score')
    report.append(top_outliers[['lof_score', 'region', 'attack_type', 'number_of_kills', 'number_of_wounded']].to_string(index=False))

    return "\n".join(report)

# Generate and print the report text
report_text = generate_report()
print(report_text)

# Plot LOF Score Distribution
plt.figure(figsize=(10, 6))
sns.histplot(lof_scores, bins=50, kde=True, color='blue', alpha=0.7, label='LOF Scores')
plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
plt.title('Distribution of LOF Scores')
plt.xlabel('LOF Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Display PCA visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['is_outlier'], palette="coolwarm", marker='o')
plt.title('PCA Projection of the Data (Outliers Highlighted)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Outlier', loc='upper left')
plt.show()

# Clustering normal points
# Load normal points
normal_points = pd.read_csv('normal_points_lof.csv')

# Select features for clustering
features = normal_points[['number_of_kills', 'number_of_wounded', 'region', 'attack_type']]

# Preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['number_of_kills', 'number_of_wounded']),  # Scale numeric data
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['region', 'attack_type'])  # Encode categorical data
    ]
)

# Transform features
features_transformed = preprocessor.fit_transform(features)

# Reduce dimensions with PCA to retain 95% variance
pca = PCA(n_components=0.95, random_state=42)
features_reduced = pca.fit_transform(features_transformed)

# Elbow Method to find the optimal number of clusters
wcss = []  # Store WCSS for each k
k_range = range(1, 11)  # Test for 1 to 10 clusters
for k in k_range:
    kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', random_state=42, batch_size=1024, max_iter=200)
    kmeans.fit(features_reduced)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(k_range)
plt.grid()
plt.show()

# Apply MiniBatchKMeans with K-Means++ and the chosen number of clusters
optimal_k = 7  # Set the number of clusters based on the Elbow Method
mini_kmeans = MiniBatchKMeans(n_clusters=optimal_k, init='k-means++', random_state=42, batch_size=1024, max_iter=200)
normal_points['cluster'] = mini_kmeans.fit_predict(features_reduced)

# Visualize clusters in four quadrants with proper graph style
plt.figure(figsize=(10, 10))

# Scatter plot
plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=normal_points['cluster'], cmap='viridis', marker='o')

# Set title and labels
plt.title(f'Clusters Visualization (PCA Reduced Data, k={optimal_k}, K-Means++)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Add grid and center axis lines to create four quadrants
plt.axhline(0, color='black',linewidth=1)  # Horizontal line (y=0)
plt.axvline(0, color='black',linewidth=1)  # Vertical line (x=0)
plt.grid(True)

# Set axis limits for zoomed-in range of -25 to 25
plt.xlim(-25, 25)
plt.ylim(-25, 25)

# Display the plot
plt.show()

# Generate Cluster Analysis Report
def generate_cluster_report(data, cluster_column='cluster'):
    report = []
    for cluster in sorted(data[cluster_column].unique()):
        cluster_data = data[data[cluster_column] == cluster]
        report.append(f"\nCluster {cluster} Summary:")
        report.append(f"Number of Points: {len(cluster_data)}")
        report.append(f"Average Kills: {cluster_data['number_of_kills'].mean():.2f}")
        report.append(f"Average Wounded: {cluster_data['number_of_wounded'].mean():.2f}")
        report.append(f"Most Common Region: {cluster_data['region'].mode()[0]}")
        report.append(f"Most Common Attack Type: {cluster_data['attack_type'].mode()[0]}")
    return "\n".join(report)

# Save and display the cluster report
cluster_report = generate_cluster_report(normal_points)
with open('cluster_analysis_report_multivariate.txt', 'w') as f:
    f.write(cluster_report)
print(cluster_report)

# Save clustered data
normal_points.to_csv('normal_points_clusters_multivariate.csv', index=False)
