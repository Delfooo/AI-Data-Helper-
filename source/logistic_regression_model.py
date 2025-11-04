import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset.generate_classification_data import generate_classification_data

def logistic_regression_analysis(file_path, feature_cols, target_col):
    df = pd.read_csv(file_path)

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuratezza del modello: {accuracy:.2f}")
    print(f"Matrice di Confusione:\n{conf_matrix}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice di Confusione')
    plt.xlabel('Previsto')
    plt.ylabel('Reale')
    plt.savefig(f'output/{os.path.basename(file_path)}_confusion_matrix.png')
    print(f"Matrice di confusione salvata come output/{os.path.basename(file_path)}_confusion_matrix.png")

if __name__ == '__main__':
    generate_classification_data()
    logistic_regression_analysis('dataset/classification_data.csv', ['Feature1', 'Feature2'], 'Target')