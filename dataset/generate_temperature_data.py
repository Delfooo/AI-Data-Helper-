import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_temperature_data(num_rows=20):
    locations = ['Rome', 'Milan', 'Naples', 'Turin']
    start_date = datetime(2023, 1, 1)

    data = {
        'Date': [start_date + timedelta(days=i % 10) for i in range(num_rows)],
        'Location': np.random.choice(locations, num_rows),
        'Temperature': np.random.randint(0, 30, num_rows)
    }

    df = pd.DataFrame(data)
    df.to_csv('dataset/temperature_data.csv', index=False)
    print(f"Generato temperature_data.csv con {num_rows} righe.")

if __name__ == '__main__':
    generate_temperature_data()