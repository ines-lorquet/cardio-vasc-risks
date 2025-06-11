import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

def plot_confusion_matrix(TP, TN, FP, FN):
    matrix = np.array([[TN, FP], [FN, TP]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sain', 'Malade'], yticklabels=['Sain', 'Malade'])
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.savefig("Results/5_logistic_regression/matrice_confusion.png")
    plt.close()

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Ajout des annotations des seuils sur quelques points clés
    for i in range(0, len(thresholds), len(thresholds)//10):  # Ajoute 10 points
        plt.annotate(f'{thresholds[i]:.2f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(-10,5), fontsize=8, color='blue')

    plt.xlabel('Faux Positifs')
    plt.ylabel('Vrais Positifs')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("Results/5_logistic_regression/roc_curve.png")
    plt.close()

def plot_feature_importance(weights, feature_names):
    importance = weights[1:]  # Conserver les poids originaux (positifs et négatifs)
    sorted_idx = np.argsort(np.abs(importance))[::-1]  # Trier selon l'importance absolue

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[sorted_idx], y=np.array(feature_names)[sorted_idx], hue=np.array(feature_names)[sorted_idx], palette="coolwarm", legend=False)

    plt.axvline(0, color='black', linestyle='--')  # Séparer valeurs positives/négatives
    plt.title("Importance des caractéristiques")
    plt.xlabel("Poids des caractéristiques (positifs et négatifs)")
    plt.ylabel("Caractéristiques")
    plt.tight_layout()
    plt.savefig("Results/5_logistic_regression/feature_importance.png")
    plt.close()


def plot_correlation_matrix(X, y, feature_names):
    """Affiche la matrice de corrélation des caractéristiques, y compris le risque cardiovasculaire."""
    # Convertir les données en DataFrame
    df = pd.DataFrame(X, columns=feature_names)

    # Ajouter la variable cible "Risque Cardio"
    df["Risque Cardio"] = y

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

    plt.title("Matrice de corrélation des variables (après prétraitement)")
    plt.tight_layout()
    plt.savefig("Results/5_logistic_regression/correlation_matrix.png")
    plt.close()

def plot_glucose_distribution(X, y, feature_names):
    """Affiche la distribution des niveaux de glucose pour les malades et les sains."""
    glucose_idx = feature_names.index("Glucose (0-2)")  # Trouver l'index du glucose

    df = pd.DataFrame({"Glucose": X[:, glucose_idx], "Risque Cardio": y})
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="Glucose", hue="Risque Cardio", bins=20, kde=True, palette=["blue", "red"])
    plt.xlabel("Niveau de glucose (0 = normal, 1 = moyen, 2 = élevé)")
    plt.ylabel("Nombre de cas")
    plt.title("Distribution du glucose selon le risque cardiovasculaire")
    plt.legend(["Sains", "Malades"])
    plt.tight_layout()
    plt.savefig("Results/5_logistic_regression/glucose_distribution.png")
    plt.close()

def plot_glucose_correlation(X, feature_names):
    """Affiche la corrélation entre glucose et les autres variables."""
    glucose_idx = feature_names.index("Glucose (0-2)")  # Index du glucose
    
    df = pd.DataFrame(X, columns=feature_names)

    glucose_col = feature_names[glucose_idx]

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr()[[glucose_col]], annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Corrélation du glucose avec les autres variables")
    plt.tight_layout()
    plt.savefig("Results/5_logistic_regression/glucose_correlation.png")
    plt.close()

def plot_glucose_vs_cardio(y_low, y_medium, y_high):
    """Affiche la proportion de maladies cardiovasculaires en fonction du glucose."""
    labels = ["Glucose faible", "Glucose moyen", "Glucose élevé"]
    cardio_rates = [np.mean(y_low), np.mean(y_medium), np.mean(y_high)]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=cardio_rates, hue=labels, palette=["blue", "orange", "red"], legend=False)
    plt.xlabel("Niveau de glucose")
    plt.ylabel("Proportion de cas de maladies cardiovasculaires")
    plt.title("Impact du glucose sur le risque cardiovasculaire")
    plt.tight_layout()
    plt.savefig("Results/5_logistic_regression/glucose_vs_cardio.png")
    plt.close()
