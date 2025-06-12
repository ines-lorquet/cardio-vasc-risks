import numpy as np
from data_processing import load_data
from train_model import feature_names

def check_normalization(X, feature_names):
    print("=== Vérification de la normalisation ===")
    # Variables continues (StandardScaler): colonnes 0 à 5
    continuous_idx = list(range(6))
    continuous = X[:, continuous_idx]
    means = continuous.mean(axis=0)
    stds = continuous.std(axis=0)
    for i, name in zip(continuous_idx, feature_names[:6]):
        print(f"{name}: moyenne={means[i]:.4f}, écart-type={stds[i]:.4f}")

    # Cholestérol et Glucose (MinMaxScaler sur [0,2]): colonnes 6 et 7
    chol_gluc_idx = [6, 7]
    for i, name in zip(chol_gluc_idx, [feature_names[6], feature_names[7]]):
        col = X[:, i]
        print(f"{name}: min={col.min():.2f}, max={col.max():.2f}, valeurs uniques={np.unique(col)}")

    # Interactions (StandardScaler): colonnes 11 à la fin
    inter_idx = list(range(11, X.shape[1]))
    if inter_idx:
        inter = X[:, inter_idx]
        means = inter.mean(axis=0)
        stds = inter.std(axis=0)
        for j, idx in enumerate(inter_idx):
            print(f"{feature_names[idx]}: moyenne={means[j]:.4f}, écart-type={stds[j]:.4f}")

if __name__ == "__main__":
    X, y, _ = load_data('Data/Cleaned/cardio_train_clean.csv')
    check_normalization(X, feature_names)