import csv
import numpy as np

def load_data(filename):
    """Charge les données et calcule l'IMC."""
    X, y = [], []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            weight, height = float(row['weight']), float(row['height']) / 100
            imc = weight / (height ** 2)

            features = [
                float(row['age']), float(row['height']), float(row['weight']),
                float(row['ap_hi']), float(row['ap_lo']), float(row['cholesterol']),
                float(row['gluc']), float(row['smoke']), float(row['alco']),
                float(row['active']), imc
            ]
            X.append(features)
            y.append(int(row['cardio']))
    return np.array(X), np.array(y)

def normalize(X):
    """Normalisation des données."""
    return (X - X.mean(axis=0)) / X.std(axis=0)

def describe_dataset(y):
    total = len(y)
    count_malades = np.sum(y == 1)
    count_sains = np.sum(y == 0)
    proportion_malades = count_malades / total
    proportion_sains = count_sains / total

    description = (
        f"=== Statistiques du jeu de données ===\n"
        f"Total échantillons : {total}\n"
        f"Malades : {count_malades} ({proportion_malades:.2%})\n"
        f"Sains   : {count_sains} ({proportion_sains:.2%})\n"
    )
    print(description)
    return description
