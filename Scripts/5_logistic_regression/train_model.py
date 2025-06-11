from data_processing import load_data, describe_dataset, split_by_glucose
from logistic_regression import LogisticRegression
from metrics import confusion_matrix, classification_report
from visualization import plot_confusion_matrix, plot_roc_curve, plot_feature_importance, plot_correlation_matrix, plot_glucose_distribution, plot_glucose_correlation, plot_glucose_vs_cardio
from report import generate_report

# Chargement des données avec comptage des valeurs aberrantes
X, y, lignes_supprimees = load_data('Data/Cleaned/cardio_train_clean.csv')
dataset_stats = describe_dataset(X, y)  # Donne des statistiques détaillées

# Séparation des données
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# Liste des noms de caractéristiques mises à jour
feature_names = ['Âge', 'Genre', 'Taille', 'Poids', 'Pression artérielle haute', 'Pression artérielle basse',
                 'Cholestérol (0-2)', 'Glucose (0-2)',
                 'Tabagisme (0-2)', 'Consommation d’alcool (0-2)', 'Activité physique']


plot_correlation_matrix(X_train, y_train, feature_names)


# Entraînement du modèle
model = LogisticRegression(learning_rate=0.1, epochs=5000)
model.fit(X_train, y_train)

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
plot_glucose_distribution(X_train, y_train, feature_names)
plot_glucose_correlation(X_train, feature_names)
(low_glucose, y_low), (medium_glucose, y_medium), (high_glucose, y_high) = split_by_glucose(X_train, y_train, feature_names)
plot_glucose_vs_cardio(y_low, y_medium, y_high)



# Rapport complet
generate_report(y_test, y_pred, y_pred_proba, dataset_stats, model.weights, feature_names, lignes_supprimees, y_low, y_medium, y_high)

# Messages utilisateurs
print("Modèle entraîné avec succès.")
print("Rapport généré dans 'rapport_model.txt'.")
print("Visualisations enregistrées :")
print("- Matrice de confusion : matrice_confusion.png")
print("- Courbe ROC : roc_curve.png")
print("- Importance des caractéristiques : feature_importance.png")
