import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from dataset.generate_clustering_data import generate_clustering_data

def kmeans_clustering_analysis(file_path, feature_cols, n_clusters=3):
    df = pd.read_csv(file_path)

    X = df[feature_cols]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=feature_cols[0], y=feature_cols[1], hue='Cluster', data=df, palette='viridis', s=100)
    plt.title(f'K-Means Clustering ({n_clusters} Clusters)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(f'output/{os.path.basename(file_path)}_kmeans_clusters.png')
    print(f"Grafico K-Means salvato come output/{os.path.basename(file_path)}_kmeans_clusters.png")

    print("\nCaratteristiche di ogni gruppo:")
    for i in range(n_clusters):
        cluster_data = df[df['Cluster'] == i]
        print(f"\nCluster {i}:")
        print(cluster_data.describe())

if __name__ == '__main__':
    generate_clustering_data()
    kmeans_clustering_analysis('dataset/clustering_data.csv', n_clusters=3)