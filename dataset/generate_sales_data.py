import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sales_data(num_rows=20):
    products = ['A', 'B', 'C', 'D']
    regions = ['East', 'West', 'North', 'South']
    start_date = datetime(2023, 1, 1)

    data = {
        'Date': [start_date + timedelta(days=i % 10) for i in range(num_rows)],
        'Product': np.random.choice(products, num_rows),
        'Sales': np.random.randint(50, 300, num_rows),
        'Region': np.random.choice(regions, num_rows)
    }

    df = pd.DataFrame(data)
    df.to_csv('dataset/sales_data.csv', index=False)
    print(f"Generato sales_data.csv con {num_rows} righe.")

if __name__ == '__main__':
    generate_sales_data()