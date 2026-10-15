import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Lire les paramètres du modèle
p = yaml.safe_load(
    open('params.yaml', encoding='utf-8')
)['entrainement']

# Charger les données d'entraînement
train = pd.read_csv('data/prepare/train.csv')

# Séparer les variables explicatives et la cible
X = train.drop(columns='fraude')
y = train['fraude']

# Entraîner le modèle
modele = RandomForestClassifier(
    n_estimators=p['n_estimators'],
    max_depth=p['max_depth'],
    random_state=p['random_state']
).fit(X, y)

# Créer le dossier models
os.makedirs('models', exist_ok=True)

# Sauvegarder le modèle
joblib.dump(modele, 'models/model.pkl')

print('Entraînement : modèle écrit dans models/model.pkl')