import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from dataset.generate_sales_data import generate_sales_data
from dataset.generate_temperature_data import generate_temperature_data

def find_outliers(file_name, column):
    file_path = os.path.join("dataset", file_name)
    df = pd.read_csv(file_path)
    print(f"Prime 5 righe del dataset {file_path}:\n{df.head()}\n")

    if column not in df.columns:
        print(f"Errore: La colonna '{column}' non esiste nel dataset.")
        return

    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Errore: La colonna '{column}' non contiene dati numerici per l'analisi degli outliers.")
        return

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    file_name = os.path.basename(file_path)
    with open(f'txt/{file_name}_outliers.txt', 'w') as f:
        f.write(f"Analisi Outliers per la colonna '{column}' nel dataset {file_path}:\n")
        if not outliers.empty:
            f.write(f"Valori sospetti (outliers):\n{outliers}\n\n")
            f.write("Spiegazione: I valori anomali sono stati identificati utilizzando il metodo dell'intervallo interquartile (IQR). Un valore è considerato anomalo se si trova al di sotto di Q1 - 1.5 * IQR o al di sopra di Q3 + 1.5 * IQR, dove Q1 è il primo quartile, Q3 è il terzo quartile e IQR è la differenza tra Q3 e Q1.\n")
        else:
            f.write("Nessun valore anomalo significativo trovato.\n")

    print(f"Analisi outliers salvata in {file_path}_outliers.txt")

if __name__ == '__main__':
    generate_sales_data()
    find_outliers('dataset/sales_data.csv', 'Sales')
    generate_temperature_data()
    find_outliers('dataset/temperature_data.csv', 'Temperature')