from data_processing import load_data, normalize, describe_dataset
from logistic_regression import LogisticRegression
from metrics import accuracy, confusion_matrix, classification_report
from visualization import plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from report import generate_report

# Chargement des données
X, y = load_data('Data/Cleaned/cardio_train_clean.csv')
dataset_stats = describe_dataset(y)  # Donne des statistiques détaillées
X = normalize(X)

# Séparation des données
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Liste des noms de caractéristiques
feature_names = ['Age', 'Height', 'Weight', 'Pressure High', 'Pressure Low', 
                 'Cholesterol', 'Gluc', 'Smoke', 'Alcohol', 'Active', 'IMC']

# Entraînement du modèle
model = LogisticRegression(learning_rate=0.1, epochs=5000)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calcul des métriques
TP, TN, FP, FN = confusion_matrix(y_test, y_pred)
print(classification_report(TP, TN, FP, FN))

# Visualisations
plot_confusion_matrix(TP, TN, FP, FN)
plot_roc_curve(y_test, y_pred_proba)
plot_feature_importance(model.weights, feature_names)

# Rapport complet
generate_report(y_test, y_pred, y_pred_proba, dataset_stats, model.weights, feature_names)

# Messages utilisateurs
print("Modèle entraîné avec succès.")
print("Rapport généré dans 'rapport_model.txt'.")
print("Visualisations enregistrées :")
print("- Matrice de confusion : matrice_confusion.png")
print("- Courbe ROC : roc_curve.png")
print("- Importance des caractéristiques : feature_importance.png")

