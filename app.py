import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import atexit
import shutil
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'source')))

from source.ai_explainer import get_gemini_explanation
from source.automatic_plot import automatic_plot
from source.basic_statistics import basic_statistics
from source.find_outliers import find_outliers
from source.kmeans_model import kmeans_clustering_analysis
from source.logistic_regression_model import logistic_regression_analysis
from source.ai_report_generator import generate_automatic_report
from source.ai_dataset_comparer import compare_datasets

# Set your Gemini API key
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

st.set_page_config(layout="wide")
st.title("AIDataHelper: Analisi Dati e Funzionalità AI")

# Sidebar for navigation
st.sidebar.title("Navigazione")
analysis_type = st.sidebar.radio(
    "Scegli il tipo di analisi:",
    ("Carica Dataset", "Statistiche di Base", "Trova Outlier", "Plot Automatico",
     "Modello K-Means", "Modello Regressione Logistica", "Report Automatico AI", "Confronta Dataset AI")
)

# --- Carica Dataset --- #
if analysis_type == "Carica Dataset":
    st.header("Carica il tuo file CSV")
    uploaded_file = st.file_uploader("Scegli un file CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.session_state['file_name'] = uploaded_file.name
        if 'uploaded_files' not in st.session_state:
            st.session_state['uploaded_files'] = []
        st.session_state['uploaded_files'].append(uploaded_file.name)
        st.success(f"File {uploaded_file.name} caricato con successo!")
        st.subheader("Anteprima del Dataset:")
        st.write(df.head())
    else:
        st.info("Carica un file CSV per iniziare l'analisi.")

# Ensure df and file_name are in session_state for other analyses
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'file_name' not in st.session_state:
    st.session_state['file_name'] = None

df = st.session_state['df']
file_name = st.session_state['file_name']

if analysis_type != "Carica Dataset" and df is None:
    st.warning("Per favore, carica un dataset prima di procedere con l'analisi.")

# --- Statistiche di Base --- #
if analysis_type == "Statistiche di Base" and df is not None:
    st.header("Statistiche Descrittive")
    st.write(df.describe())
    st.subheader("Informazioni sul Dataset:")
    buffer = pd.io.common.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    if st.button("Genera Statistiche Avanzate"):
        # Save the uploaded file temporarily to be read by basic_statistics.py
        temp_file_path = os.path.join("dataset", file_name)
        df.to_csv(temp_file_path, index=False)

        st.write("Generazione statistiche avanzate...")
        # Redirect stdout to capture print statements from the script
        old_stdout = sys.stdout
        sys.stdout = captured_output = pd.io.common.StringIO()
        basic_statistics(file_name)
        sys.stdout = old_stdout
        st.text(captured_output.getvalue())

        # Display the generated stats file
        stats_file_name = f"txt/{file_name}_stats.txt"
        if os.path.exists(stats_file_name):
            with open(stats_file_name, 'r') as f:
                st.subheader("Report Statistiche Avanzate:")
                st.text(f.read())
        else:
            st.error(f"File di statistiche non trovato: {stats_file_name}")

# --- Trova Outlier --- #
if analysis_type == "Trova Outlier" and df is not None:
    st.header("Rilevamento Outlier")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        st.warning("Nessuna colonna numerica trovata per l'analisi degli outlier.")
    else:
        col_to_analyze = st.selectbox("Seleziona la colonna per l'analisi degli outlier:", numeric_cols)
        if st.button("Trova Outlier"):
            temp_file_path = os.path.join("dataset", file_name)
            df.to_csv(temp_file_path, index=False)

            st.write(f"Analisi outlier per la colonna {col_to_analyze}...")
            old_stdout = sys.stdout
            sys.stdout = captured_output = pd.io.common.StringIO()
            find_outliers(file_name, col_to_analyze)
            sys.stdout = old_stdout
            st.text(captured_output.getvalue())

            outliers_file_name = f"txt/{os.path.splitext(file_name)[0]}_outliers.txt"
            if os.path.exists(outliers_file_name):
                with open(outliers_file_name, 'r') as f:
                    st.subheader("Report Outlier:")
                    st.text(f.read())
            else:
                st.error(f"File outlier non trovato: {outliers_file_name}")

# --- Plot Automatico --- #
if analysis_type == "Plot Automatico" and df is not None:
    st.header("Generazione Plot Automatico")
    all_columns = df.columns.tolist()
    if len(all_columns) < 2:
        st.warning("Sono necessarie almeno due colonne per generare un plot.")
    else:
        x_col = st.selectbox("Seleziona la colonna per l'asse X:", all_columns)
        y_col = st.selectbox("Seleziona la colonna per l'asse Y:", all_columns)
        plot_type = st.selectbox("Seleziona il tipo di plot:", ["bar", "line", "scatter"])
        plot_title = st.text_input("Titolo del Plot:", f"Plot di {y_col} vs {x_col}")

        if st.button("Genera Plot"):
            temp_file_path = os.path.join("dataset", file_name)
            df.to_csv(temp_file_path, index=False)

            st.write(f"Generazione plot {plot_type}...")
            old_stdout = sys.stdout
            sys.stdout = captured_output = pd.io.common.StringIO()
            automatic_plot(temp_file_path, x_col, y_col, plot_type, plot_title)
            sys.stdout = old_stdout
            st.text(captured_output.getvalue())

            plot_output_path = f"output/{file_name}_{plot_type}_plot.png"
            if os.path.exists(plot_output_path):
                st.subheader("Plot Generato:")
                st.image(plot_output_path)
            else:
                st.error(f"Plot non trovato: {plot_output_path}")

# --- Modello K-Means --- #
if analysis_type == "Modello K-Means" and df is not None:
    st.header("Clustering K-Means")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Sono necessarie almeno due colonne numeriche per il clustering K-Means.")
    else:
        feature_cols = st.multiselect("Seleziona le colonne per il clustering:", numeric_cols)
        num_clusters = st.slider("Numero di Cluster (K):", 2, 10, 3)

        if st.button("Esegui K-Means"):
            if feature_cols:
                temp_file_path = os.path.join("dataset", file_name)
                df.to_csv(temp_file_path, index=False)

                st.write(f"Esecuzione K-Means con {num_clusters} cluster...")
                old_stdout = sys.stdout
                sys.stdout = captured_output = pd.io.common.StringIO()
                kmeans_clustering_analysis(temp_file_path, feature_cols, num_clusters)
                sys.stdout = old_stdout
                st.text(captured_output.getvalue())

                plot_output_path = f"output/{file_name}_kmeans_clusters.png"
                if os.path.exists(plot_output_path):
                    st.subheader("Grafico K-Means:")
                    st.image(plot_output_path)
                else:
                    st.error(f"Grafico K-Means non trovato: {plot_output_path}")
            else:
                st.warning("Seleziona almeno due colonne per il clustering.")

# --- Modello Regressione Logistica --- #
if analysis_type == "Modello Regressione Logistica" and df is not None:
    st.header("Modello di Regressione Logistica")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Sono necessarie almeno due colonne numeriche per la regressione logistica.")
    else:
        target_col = st.selectbox("Seleziona la colonna target (binaria):", df.columns.tolist())
        feature_cols = st.multiselect("Seleziona le colonne feature:", [col for col in df.columns if col != target_col])

        if st.button("Esegui Regressione Logistica"):
            if feature_cols and target_col:
                # Check if the target column is suitable for classification
                # Check if the target column is suitable for classification
                if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 10:
                    st.warning("La colonna target selezionata sembra essere continua. La regressione logistica richiede una variabile target discreta (categorica).")
                else:
                    st.write("Esecuzione Regressione Logistica in corso...")
                    temp_file_path = f"output/{file_name}_temp.csv"
                    df.to_csv(temp_file_path, index=False)
                    old_stdout = sys.stdout
                    sys.stdout = captured_output = pd.io.common.StringIO()
                    logistic_regression_analysis(temp_file_path, feature_cols, target_col)
                    sys.stdout = old_stdout
                    st.text(captured_output.getvalue())
        
                    # Display the confusion matrix plot
                    plot_output_path_lr = f"output/{file_name}_temp.csv_confusion_matrix.png"
                    if os.path.exists(plot_output_path_lr):
                        st.image(plot_output_path_lr, caption="Matrice di Confusione")
                    else:
                        st.warning(f"Impossibile trovare il grafico della matrice di confusione in {plot_output_path_lr}")
            else:
                st.warning("Seleziona le colonne feature e la colonna target.")

# --- Confronta Dataset AI --- #
if analysis_type == "Confronta Dataset AI":
    st.header("Confronto Dataset con AI")
    st.write("Carica due file CSV per confrontarli.")

    uploaded_file1 = st.file_uploader("Scegli il primo file CSV", type="csv", key="file1")
    uploaded_file2 = st.file_uploader("Scegli il secondo file CSV", type="csv", key="file2")
    comparison_name_input = st.text_input("Nome del Confronto (es. sales_comparison):")

    if uploaded_file1 is not None and uploaded_file2 is not None and comparison_name_input:
        df1 = pd.read_csv(uploaded_file1)
        df2 = pd.read_csv(uploaded_file2)

        temp_file_path1 = os.path.join("dataset", uploaded_file1.name)
        temp_file_path2 = os.path.join("dataset", uploaded_file2.name)
        df1.to_csv(temp_file_path1, index=False)
        df2.to_csv(temp_file_path2, index=False)

        if st.button("Confronta Dataset con AI"):
            st.write("Generazione confronto AI in corso...")
            old_stdout = sys.stdout
            sys.stdout = captured_output = pd.io.common.StringIO()
            plot_output_path_comparison = compare_datasets(temp_file_path1, temp_file_path2, comparison_name_input)
            sys.stdout = old_stdout
            st.text(captured_output.getvalue())

            comparison_text_file_path = f"txt/{comparison_name_input}_comparison.txt"
            if os.path.exists(comparison_text_file_path):
                with open(comparison_text_file_path, 'r') as f:
                    st.subheader("Report di Confronto AI:")
                    st.text(f.read())
                st.download_button(
                    label="Scarica Report di Confronto AI",
                    data=open(comparison_text_file_path, 'rb').read(),
                    file_name=f"{comparison_name_input}_comparison.txt",
                    mime="text/plain"
                )
            else:
                st.error(f"Report di confronto AI non trovato: {comparison_text_file_path}")

            # Display comparative plot if generated
            if plot_output_path_comparison and os.path.exists(plot_output_path_comparison):
                st.subheader("Grafico Comparativo:")
                st.image(plot_output_path_comparison)
            else:
                st.warning("Grafico comparativo non trovato. Assicurati che i dataset abbiano colonne numeriche comuni.")

    else:
        st.info("Carica due file CSV e inserisci un nome per il confronto per procedere.")


def cleanup_generated_files():
    print("Cleaning up generated files...")
    for folder in ["output", "txt"]:
        if os.path.exists(folder):
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Error deleting {item_path}: {e}")

    # Clean up uploaded dataset files
    if 'uploaded_files' in st.session_state:
        for uploaded_file_name in st.session_state['uploaded_files']:
            file_path = os.path.join("dataset", uploaded_file_name)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting uploaded file {file_path}: {e}")

    print("Cleanup complete.")

atexit.register(cleanup_generated_files)