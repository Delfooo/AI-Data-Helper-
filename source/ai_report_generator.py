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

def generate_automatic_report(file_path, report_name):
    df = pd.read_csv(file_path)

    # Basic statistics for the report
    numeric_cols = df.select_dtypes(include=['number'])
    stats = numeric_cols.describe().to_string()
    missing_data = df.isnull().sum().to_string()

    prompt = f"Genera un report di circa 200 parole per il dataset {file_path}. " \
             f"Includi i dati principali, osservazioni e consigli basati sulle seguenti statistiche: " \
             f"Statistiche descrittive:\n{stats}\n\nDati mancanti:\n{missing_data}\n"

    report_text = get_gemini_explanation(prompt)

    output_file_path = f'txt/{report_name}.txt'
    with open(output_file_path, 'w') as f:
        f.write(f"Report Automatico per {file_path}\n\n")
        f.write(report_text)

    print(f"Report salvato come {output_file_path}")

if __name__ == '__main__':
    # Example usage (assuming generate_sales_data and generate_temperature_data are available)
    from dataset.generate_sales_data import generate_sales_data
    from dataset.generate_temperature_data import generate_temperature_data

    generate_sales_data()
    generate_automatic_report('dataset/sales_data.csv', 'sales_report')

    generate_temperature_data()
    generate_automatic_report('dataset/temperature_data.csv', 'temperature_report')