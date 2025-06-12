from data_processing import load_data, describe_dataset, split_by_glucose, filter_inactive
from logistic_regression import LogisticRegression
from metrics import confusion_matrix, classification_report, subgroup_performance
from visualization import plot_confusion_matrix, plot_roc_curve, plot_feature_importance, plot_correlation_matrix
from report import generate_report
import numpy as np

# Chargement des données avec comptage des valeurs aberrantes
X, y, lignes_supprimees = load_data('Data/Cleaned/cardio_train_clean.csv')
dataset_stats = describe_dataset(X, y)  # Donne des statistiques détaillées

# Séparation des données
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# Liste des noms de caractéristiques mises à jour
feature_names = [
    'Âge', 'Genre', 'Taille', 'Poids', 'Pression artérielle haute', 'Pression artérielle basse',
    'Cholestérol (0-2)', 'Glucose (0-2)', 'Tabagisme', 'Consommation d’alcool', 'Activité physique',
    'Cholestérol x Glucose', 'PA haute x PA basse', 'IMC',
    'Fumeur x Âge', 'Alcool x Âge',
    'Tabac x Sexe', 'Alcool x Sexe', 'Tabac x Activité physique', 'Alcool x Activité physique',
    'Glucose x Âge', 'Glucose x Activité physique', 'Glucose x Sexe', 'Glucose x IMC',
    'Cholestérol x Âge', 'Cholestérol x Activité physique', 'Cholestérol x Sexe', 'Cholestérol x IMC'
]


plot_correlation_matrix(X_train, y_train, feature_names)


# Entraînement du modèle
model = LogisticRegression(learning_rate=0.1, epochs=5000, l2=0.01)
model.fit(X_train, y_train, pos_weight=2.0)

# Prédictions
y_pred = model.predict(X_test, threshold=0.45)
y_pred_proba = model.predict_proba(X_test)

# Calcul des métriques
TP, TN, FP, FN = confusion_matrix(y_test, y_pred)
print(classification_report(TP, TN, FP, FN))

# Visualisations
plot_confusion_matrix(TP, TN, FP, FN)
plot_roc_curve(y_test, y_pred_proba)
plot_feature_importance(model.weights, feature_names)

# Séparation des sous-groupes par niveau de glucose
(_, y_low), (_, y_medium), (_, y_high) = split_by_glucose(X_test, y_test, feature_names)

# Calcul des performances par sous-groupe (à placer AVANT generate_report)
# Glucose
(glucose0_X, glucose0_y), (glucose1_X, glucose1_y), (glucose2_X, glucose2_y) = split_by_glucose(X_test, y_test, feature_names)
y_pred_g0 = model.predict(glucose0_X, threshold=0.45)
y_pred_g1 = model.predict(glucose1_X, threshold=0.45)
y_pred_g2 = model.predict(glucose2_X, threshold=0.45)
perf_glucose = ""
perf_glucose += subgroup_performance(glucose0_y, y_pred_g0, "Glucose 0")
perf_glucose += subgroup_performance(glucose1_y, y_pred_g1, "Glucose 1")
perf_glucose += subgroup_performance(glucose2_y, y_pred_g2, "Glucose 2")

# Cholestérol
chol_idx = feature_names.index("Cholestérol (0-2)")
perf_chol = ""
for val in [0, 1, 2]:
    mask = X_test[:, chol_idx] == val
    y_pred_chol = model.predict(X_test[mask], threshold=0.45)
    perf_chol += subgroup_performance(y_test[mask], y_pred_chol, f"Cholestérol {val}")

# Fumeurs
fumeur_idx = feature_names.index("Tabagisme")
mask_fumeur = X_test[:, fumeur_idx] == 1
y_pred_fumeur = model.predict(X_test[mask_fumeur], threshold=0.45)
perf_fumeur = subgroup_performance(y_test[mask_fumeur], y_pred_fumeur, "Fumeurs")

# Alcool
alcool_idx = feature_names.index("Consommation d’alcool")
mask_alcool = X_test[:, alcool_idx] == 1
y_pred_alcool = model.predict(X_test[mask_alcool], threshold=0.45)
perf_alcool = subgroup_performance(y_test[mask_alcool], y_pred_alcool, "Alcool")

# Inactifs
from data_processing import filter_inactive
X_inactif, y_inactif = filter_inactive(X_test, y_test, feature_names)
y_pred_inactif = model.predict(X_inactif, threshold=0.45)
perf_inactif = subgroup_performance(y_inactif, y_pred_inactif, "Inactifs")

# Rapport complet (à placer APRÈS le calcul des sous-groupes)
generate_report(
    y_test, y_pred, y_pred_proba, dataset_stats, model.weights, feature_names, lignes_supprimees,
    y_low, y_medium, y_high,
    perf_glucose, perf_chol, perf_fumeur, perf_alcool, perf_inactif
)

# Messages utilisateurs
print("Modèle entraîné avec succès.")
print("Rapport généré dans 'rapport_model.txt'.")
print("Visualisations enregistrées :")
print("- Matrice de confusion : matrice_confusion.png")
print("- Courbe ROC : roc_curve.png")
print("- Importance des caractéristiques : feature_importance.png")

from metrics import classification_report

best_f1 = 0
best_recall = 0
best_threshold = 0.5
for threshold in np.arange(0.2, 0.61, 0.01):
    y_pred = model.predict(X_test, threshold=threshold)
    TP, TN, FP, FN = confusion_matrix(y_test, y_pred)
    report = classification_report(TP, TN, FP, FN)
    lines = report.split('\n')
    recall = float(lines[1].split(':')[1])
    f1 = float(lines[2].split(':')[1])
    if recall > best_recall and f1 > best_f1 * 0.95:  # compromis
        best_recall = recall
        best_f1 = f1
        best_threshold = threshold
print(f"Meilleur seuil : {best_threshold:.2f} (Recall = {best_recall:.4f}, F1 = {best_f1:.4f})")
