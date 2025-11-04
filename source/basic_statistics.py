import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from ai_explainer import get_gemini_explanation
from dataset.generate_sales_data import generate_sales_data
from dataset.generate_temperature_data import generate_temperature_data

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

def basic_statistics(file_name):
    file_path = os.path.join("dataset", file_name)
    df = pd.read_csv(file_path)
    print(f"Prime 5 righe del dataset {file_path}:\n{df.head()}\n")

    numeric_cols = df.select_dtypes(include=['number'])
    stats = numeric_cols.agg(['mean', 'max', 'min'])
    missing_data = df.isnull().sum()

    file_name = os.path.basename(file_path)
    with open(f'txt/{file_name}_stats.txt', 'w') as f:
        f.write(f"Statistiche Base per {file_path}:\n")
        f.write(f"Media, Massimo, Minimo:\n{stats}\n\n")
        f.write(f"Dati Mancanti:\n{missing_data}\n")

    print(f"Statistiche salvate in {file_path}_stats.txt")

    # Genera spiegazioni con l'AI
    for col in stats.columns:
        mean_val = stats.loc['mean', col]
        max_val = stats.loc['max', col]
        min_val = stats.loc['min', col]
        prompt = f"Spiega in 2-3 frasi semplici in italiano le seguenti statistiche per la colonna {col}: media={mean_val}, massimo={max_val}, minimo={min_val}."
        explanation = get_gemini_explanation(GEMINI_API_KEY, prompt)
        print(f"\nSpiegazione AI per la colonna {col}:\n{explanation}\n")
        with open(f'txt/{file_name}_stats.txt', 'a') as f:
            f.write(f"\nSpiegazione AI per la colonna {col}:\n{explanation}\n")

if __name__ == '__main__':
    generate_sales_data()
    basic_statistics('dataset/sales_data.csv')
    generate_temperature_data()
    basic_statistics('dataset/temperature_data.csv')