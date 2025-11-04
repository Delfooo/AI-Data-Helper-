import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataset.generate_sales_data import generate_sales_data
from dataset.generate_temperature_data import generate_temperature_data

def automatic_plot(file_path, x_col, y_col, plot_type='bar', title='Grafico Automatico', color='skyblue'):
    df = pd.read_csv(file_path)

    plt.figure(figsize=(10, 6))
    if plot_type == 'bar':
        sns.barplot(x=x_col, y=y_col, data=df, color=color)
    elif plot_type == 'line':
        sns.lineplot(x=x_col, y=y_col, data=df, color=color)
    else:
        print("Tipo di grafico non supportato. Usa 'bar' o 'line'.")
        return

    plt.title(title, fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'output/{os.path.basename(file_path)}_{plot_type}_plot.png')
    print(f"Grafico salvato come output/{os.path.basename(file_path)}_{plot_type}_plot.png")

if __name__ == '__main__':
    generate_sales_data()
    automatic_plot('dataset/sales_data.csv', 'Product', 'Sales', plot_type='bar', title='Vendite per Prodotto', color='lightcoral')
    generate_temperature_data()
    automatic_plot('dataset/temperature_data.csv', 'Date', 'Temperature', plot_type='line', title='Temperature Medie per Data', color='lightseagreen')