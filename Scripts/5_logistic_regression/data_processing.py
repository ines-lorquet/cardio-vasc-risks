import csv
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(filename):
    """Charge les données et applique les corrections nécessaires, y compris la gestion des valeurs aberrantes."""
    X, y = [], []
    lignes_supprimees = 0  # Compteur des lignes supprimées

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Conversion de l'âge en années
                age_years = float(row['age']) / 365.0
                
                # Vérification des seuils médicaux
                if age_years < 18 or age_years > 100:
                    lignes_supprimees += 1
                    continue

                # Encodage binaire du genre
                gender_binary = int(row['gender']) - 1

                # Vérification des seuils de taille et poids
                height = float(row['height'])
                weight = float(row['weight'])
                if height < 140 or height > 220 or weight < 30 or weight > 200:
                    lignes_supprimees += 1
                    continue

                # Vérification des seuils de pression artérielle
                ap_hi = float(row['ap_hi'])
                ap_lo = float(row['ap_lo'])
                if ap_hi < 80 or ap_hi > 200 or ap_lo < 40 or ap_lo > 130:
                    lignes_supprimees += 1
                    continue

                # Transformation continue de Cholestérol et Glucose
                cholesterol = max(0, min(int(row['cholesterol']) - 1, 2))
                glucose = max(0, min(int(row['gluc']) - 1, 2))

                # Encodage des variables binaires
                smoke = int(row['smoke'])
                alco = int(row['alco'])
                active = int(row['active'])

                # Liste des features
                features = [
                    age_years, gender_binary, height, weight, ap_hi, ap_lo,
                    cholesterol, glucose, smoke, alco, active
                ]
                # Ajout d'interactions
                features.append(cholesterol * glucose)  # Interaction cholestérol × glucose
                features.append(ap_hi * ap_lo)          # Interaction pression artérielle
                imc = weight / ((height / 100) ** 2)
                features.append(imc)
                # Nouvelle interaction fumeurs x alcool
                features.append(smoke * alco)
                X.append(features)
                y.append(int(row['cardio']))

            except ValueError:
                lignes_supprimees += 1  # Compter les lignes mal formatées
                continue

    X = np.array(X)
    y = np.array(y)

    # Normalisation des variables continues
    scaler_general = StandardScaler()
    X[:, :6] = scaler_general.fit_transform(X[:, :6])

    # Normalisation spécifique de Cholestérol et Glucose
    scaler_chol_gluc = MinMaxScaler(feature_range=(0, 2))
    X[:, 6:8] = scaler_chol_gluc.fit_transform(X[:, 6:8])

    # Normalisation des features d'interaction et IMC
    scaler_inter = StandardScaler()
    X[:, 11:] = scaler_inter.fit_transform(X[:, 11:])

    print(f"Lignes supprimées pour valeurs aberrantes : {lignes_supprimees}")

    return X, y, lignes_supprimees



def describe_dataset(X, y):
    """Affichage des statistiques du jeu de données."""
    total = len(y)
    
    # Comptage des malades et sains
    count_malades = np.sum(y == 1)
    count_sains = np.sum(y == 0)
    proportion_malades = count_malades / total
    proportion_sains = count_sains / total

    # Vérification des valeurs de Cholestérol et Glucose
    chol_counts = np.unique(X[:, 6], return_counts=True)
    proportion_chol = {f"Cholestérol {int(val)}": count / total for val, count in zip(*chol_counts)}

    gluc_counts = np.unique(X[:, 7], return_counts=True)
    proportion_gluc = {f"Glucose {int(val)}": count / total for val, count in zip(*gluc_counts)}

    # Proportions des fumeurs, alcool et activité physique
    count_fumeurs = np.sum(X[:, 8] == 1)
    count_alcool = np.sum(X[:, 9] == 1)
    count_actifs = np.sum(X[:, 10] == 1)

    proportion_fumeurs = count_fumeurs / total
    proportion_alcool = count_alcool / total
    proportion_actifs = count_actifs / total

    # Proportions croisées fumeur/alcool
    non_fumeur = X[:, 8] == 0
    non_alco = X[:, 9] == 0
    fumeur = X[:, 8] == 1
    alco = X[:, 9] == 1

    ni_fumeur_ni_alco = np.sum(non_fumeur & non_alco) / total
    non_fumeur_seul = np.sum(non_fumeur & alco) / total
    non_alco_seul = np.sum(fumeur & non_alco) / total
    fumeur_et_alco = np.sum(fumeur & alco) / total

    ni_fumeur_ni_alco_n = np.sum(non_fumeur & non_alco)
    non_fumeur_seul_n = np.sum(non_fumeur & alco)
    non_alco_seul_n = np.sum(fumeur & non_alco)
    fumeur_et_alco_n = np.sum(fumeur & alco)

    description = (
        f"=== Statistiques du jeu de données ===\n"
        f"Total échantillons : {total}\n"
        f"Malades : {count_malades} ({proportion_malades:.2%})\n"
        f"Sains   : {count_sains} ({proportion_sains:.2%})\n"
        + "\n".join([f"{key} : {val:.2%}" for key, val in proportion_chol.items()]) + "\n"
        + "\n".join([f"{key} : {val:.2%}" for key, val in proportion_gluc.items()]) + "\n"
        f"Actifs : {count_actifs} ({proportion_actifs:.2%})\n"
        f"Fumeurs : {count_fumeurs} ({proportion_fumeurs:.2%})\n"
        f"Consommation d'alcool : {count_alcool} ({proportion_alcool:.2%})\n"
        f"Ni fumeur ni alcool : {ni_fumeur_ni_alco_n} ({ni_fumeur_ni_alco:.2%})\n"
        f"Non fumeur seul (alcool oui) : {non_fumeur_seul_n} ({non_fumeur_seul:.2%})\n"
        f"Non alcool seul (fumeur oui) : {non_alco_seul_n} ({non_alco_seul:.2%})\n"
        f"Fumeur ET alcool : {fumeur_et_alco_n} ({fumeur_et_alco:.2%})\n"
    )
    
    print(description)
    return description

def split_by_glucose(X, y, feature_names):
    """Crée des sous-groupes en fonction du niveau de glucose."""
    glucose_idx = feature_names.index("Glucose (0-2)")

    low_glucose = X[X[:, glucose_idx] == 0]
    medium_glucose = X[X[:, glucose_idx] == 1]
    high_glucose = X[X[:, glucose_idx] == 2]

    y_low = y[X[:, glucose_idx] == 0]
    y_medium = y[X[:, glucose_idx] == 1]
    y_high = y[X[:, glucose_idx] == 2]

    return (low_glucose, y_low), (medium_glucose, y_medium), (high_glucose, y_high)

def filter_inactive(X, y, feature_names):
    """Retourne X, y pour les personnes inactives (activité physique = 0)."""
    active_idx = feature_names.index("Activité physique")
    mask = X[:, active_idx] == 0
    return X[mask], y[mask]
