import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from source.ai_explainer import get_gemini_explanation
import matplotlib.pyplot as plt
import seaborn as sns

def compare_datasets(file_path1, file_path2, comparison_name):
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    # Basic comparison of descriptive statistics
    stats1 = df1.describe().to_string()
    stats2 = df2.describe().to_string()

    prompt = f"Confronta i seguenti due dataset. Il primo dataset proviene da {file_path1} e il secondo da {file_path2}. " \
             f"Analizza le differenze principali e scrivi un confronto di circa 200 parole, " \
             f"evidenziando cosa è migliorato o peggiorato. " \
             f"Statistiche del primo dataset:\n{stats1}\n\nStatistiche del secondo dataset:\n{stats2}\n"

    print("Generazione del testo di confronto tramite Gemini...")
    comparison_text = get_gemini_explanation(prompt)
    print("Testo di confronto generato.")

    output_text_file_path = f'txt/{comparison_name}_comparison.txt'
    with open(output_text_file_path, 'w') as f:
        f.write(f"Confronto Dataset: {file_path1} vs {file_path2}\n\n")
        f.write(comparison_text)

    print(f"Report di confronto salvato come {output_text_file_path}")

    # Generate a comparative graph for a common numerical column (e.g., 'Sales' or 'Value')
    common_cols = list(set(df1.columns) & set(df2.columns))
    numerical_common_cols = [col for col in common_cols if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col])]

    if numerical_common_cols:
        col_to_plot = numerical_common_cols[0] # Take the first common numerical column
        
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'Dataset': [os.path.basename(file_path1)] * len(df1) + [os.path.basename(file_path2)] * len(df2),
            col_to_plot: pd.concat([df1[col_to_plot], df2[col_to_plot]])
        })

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Dataset', y=col_to_plot, data=plot_df)
        plt.title(f'Confronto di {col_to_plot}: {os.path.basename(file_path1)} vs {os.path.basename(file_path2)}')
        plt.ylabel(col_to_plot)
        plt.tight_layout()
        output_plot_file_path = f'output/{comparison_name}_{col_to_plot}_comparison.png'
        plt.savefig(output_plot_file_path)
        print(f"Grafico comparativo salvato come {output_plot_file_path}")
        return output_plot_file_path
    else:
        print("Nessuna colonna numerica comune trovata per il grafico comparativo.")
        return None

if __name__ == '__main__':
    # Example usage
    from dataset.generate_sales_data import generate_sales_data
    from dataset.generate_temperature_data import generate_temperature_data

    # Generate two slightly different sales datasets for comparison
    # For demonstration, we'll just use the same data twice, but in a real scenario,
    # you'd have different data files.
    generate_sales_data()
    # Assuming you have a sales_data_2025.csv for a real comparison
    # For now, let's just copy sales_data.csv to simulate a second dataset
    df_sales = pd.read_csv('dataset/sales_data.csv')
    df_sales['Sales'] = df_sales['Sales'] * 1.1 # Simulate improvement
    df_sales.to_csv('dataset/sales_data_2025.csv', index=False)

    compare_datasets('dataset/sales_data.csv', 'dataset/sales_data_2025.csv', 'sales_comparison')

    # Generate two slightly different temperature datasets for comparison
    generate_temperature_data()
    df_temp = pd.read_csv('dataset/temperature_data.csv')
    df_temp['Temperature'] = df_temp['Temperature'] + 2 # Simulate increase
    df_temp.to_csv('dataset/temperature_data_summer.csv', index=False)

    compare_datasets('dataset/temperature_data.csv', 'dataset/temperature_data_summer.csv', 'temperature_comparison')