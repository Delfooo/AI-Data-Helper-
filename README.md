# AIDataHelper

Questo progetto fornisce strumenti per l'analisi dei dati e funzionalità basate sull'intelligenza artificiale, accessibili tramite un'interfaccia utente Streamlit.

## Installazione

1.  Clona il repository:
    ```bash
    git clone URL # Sostituisci con l'URL del tuo repository
    cd AIDataHelper
    ```

2.  Crea e attiva un ambiente virtuale (raccomandato):
    ```bash
    python -m venv venv
    # Su Windows
    .\venv\Scripts\activate
    # Su macOS/Linux
    source venv/bin/activate
    ```

3.  Installa le dipendenze necessarie:
    ```bash
    pip install -r requirements.txt
    ```

## Utilizzo

Per avviare l'applicazione Streamlit, esegui il seguente comando dalla directory principale del progetto:

```bash
streamlit run app.py
```
in alternativa, puoi utilizzare il seguente comando:
```bash
python -m streamlit run app.py
```

Questo aprirà l'applicazione nel tuo browser web predefinito.

### Funzionalità dell'Applicazione Streamlit:

*   **Carica Dataset**: Carica un file CSV per iniziare l'analisi.
*   **Statistiche di Base**: Visualizza le statistiche descrittive e le informazioni sul dataset.
*   **Trova Outlier**: Identifica gli outlier in una colonna numerica selezionata.
*   **Plot Automatico**: Genera grafici a barre, a linee o a dispersione per le colonne selezionate.
*   **Modello K-Means**: Esegue il clustering K-Means su colonne numeriche selezionate.
*   **Modello Regressione Logistica**: Esegue la regressione logistica su colonne feature e target selezionate.
*   **Report Automatico AI**: Genera un report testuale di 200 parole basato sul dataset caricato, utilizzando l'AI.
*   **Confronta Dataset AI**: Confronta due dataset caricati e genera un report testuale e un grafico comparativo utilizzando l'AI.

## Struttura del Progetto

```
AIDataHelper/
├── README.md
├── requirements.txt
├── app.py
├── dataset/
│   ├── ... (file CSV generati e di esempio)
├── output/
│   ├── ... (grafici generati)
├── source/
│   ├── ai_explainer.py
│   ├── ai_report_generator.py
│   ├── ai_dataset_comparer.py
│   ├── automatic_plot.py
│   ├── basic_statistics.py
│   ├── find_outliers.py
│   ├── kmeans_model.py
│   ├── logistic_regression_model.py
│   └── ...
└── txt/
    ├── ... (file di testo generati)
```