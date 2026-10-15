import os
import json
import joblib
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Pas de fenêtre graphique

import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay
)


# Charger les données de test
test = pd.read_csv('data/prepare/test.csv')

# Séparer les variables explicatives et la cible
X = test.drop(columns='fraude')
y = test['fraude']

# Charger le modèle entraîné
modele = joblib.load('models/model.pkl')

# Faire les prédictions
pred = modele.predict(X)

# Calculer les métriques
mets = {
    'f1': round(f1_score(y, pred), 4),
    'precision': round(precision_score(y, pred), 4),
    'recall': round(recall_score(y, pred), 4)
}

# Créer le dossier reports
os.makedirs('reports', exist_ok=True)

# Sauvegarder les métriques
with open('reports/metrics.json', 'w', encoding='utf-8') as f:
    json.dump(mets, f, indent=2)

# Générer la matrice de confusion
ConfusionMatrixDisplay.from_predictions(y, pred)
plt.savefig(
    'reports/confusion.png',
    dpi=110,
    bbox_inches='tight'
)

print('Évaluation :', mets)