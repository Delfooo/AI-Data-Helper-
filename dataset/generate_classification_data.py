import pandas as pd
import numpy as np

def generate_classification_data(num_rows=20):
    np.random.seed(42) # for reproducibility
    data = {
        'Feature1': np.random.rand(num_rows) * 10,
        'Feature2': np.random.rand(num_rows) * 10,
        'Target': np.random.randint(0, 2, num_rows) # 0 or 1
    }
    df = pd.DataFrame(data)
    df.to_csv('dataset/classification_data.csv', index=False)
    print(f"Generato classification_data.csv con {num_rows} righe.")

if __name__ == '__main__':
    generate_classification_data()