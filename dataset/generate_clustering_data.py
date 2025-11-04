import pandas as pd
import numpy as np

def generate_clustering_data(num_rows=30):
    np.random.seed(42) # for reproducibility
    # Generate data for 3 clusters
    data = []
    for _ in range(num_rows // 3):
        data.append([np.random.normal(1, 0.2), np.random.normal(1, 0.2)]) # Cluster 1
        data.append([np.random.normal(5, 0.2), np.random.normal(5, 0.2)]) # Cluster 2
        data.append([np.random.normal(2, 0.2), np.random.normal(2, 0.2)]) # Cluster 3

    df = pd.DataFrame(data[:num_rows], columns=['X', 'Y'])
    df.to_csv('dataset/clustering_data.csv', index=False)
    print(f"Generato clustering_data.csv con {num_rows} righe.")

if __name__ == '__main__':
    generate_clustering_data()