import numpy as np
import pandas as pd
from data_processing import load_data
from train_model import feature_names
from logistic_regression import LogisticRegression
from metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score

# 1. Charger les données
X, y, _ = load_data('Data/Cleaned/cardio_train_clean.csv')

# 2. Créer un DataFrame pour analyse
df = pd.DataFrame(X, columns=feature_names)
df['Malade'] = y

# 3. Analyse croisée tabac/alcool
print("\n=== Proportion de malades selon tabac/alcool ===")
for var in ['Tabagisme', 'Consommation d’alcool']:
    for val in [0, 1]:
        prop = df[df[var] == val]['Malade'].mean()
        n = (df[var] == val).sum()
        print(f"{var}={val} : {prop:.2%} de malades ({n} cas)")

# 4. Corrélation avec la cible
print("\n=== Corrélation avec la cible ===")
corrs = df.corr()['Malade'].sort_values(ascending=False)
print(corrs)

# 5. Test d’ablation : retirer chaque variable et mesurer l’impact
print("\n=== Test d’ablation (retrait d’une variable à la fois) ===")
base_auc = None
for i, var in enumerate(feature_names):
    X_reduced = np.delete(X, i, axis=1)
    # Split train/test
    split = int(0.8 * len(X_reduced))
    X_train, X_test = X_reduced[:split], X_reduced[split:]
    y_train, y_test = y[:split], y[split:]
    model = LogisticRegression(learning_rate=0.1, epochs=2000, l2=0.01)
    model.fit(X_train, y_train, pos_weight=2.0)
    y_pred_proba = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    if base_auc is None:
        # Pour la première variable, entraîner sur toutes les variables
        model_full = LogisticRegression(learning_rate=0.1, epochs=2000, l2=0.01)
        model_full.fit(X_train, y_train, pos_weight=2.0)
        base_auc = auc
        print(f"Modèle complet AUC : {base_auc:.4f}")
    print(f"Sans '{var}' : AUC = {auc:.4f} (diff = {auc - base_auc:+.4f})")

print("\nAnalyse terminée.")