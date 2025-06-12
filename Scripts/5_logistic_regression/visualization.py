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

def plot_summary_barplot():
    """Affiche un barplot des proportions principales du jeu de données."""
    categories = [
        "Malades", "Sains", "Cholestérol 2", "Glucose 2", "Fumeurs", "Alcool",
        "Ni fumeur ni alcool", "Non fumeur seul (alcool oui)", "Non alcool seul (fumeur oui)", "Fumeur ET alcool"
    ]
    proportions = [
        49.45, 50.55, 11.46, 7.60, 8.80, 5.35, 88.49, 2.71, 6.16, 2.64
    ]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=proportions, y=categories, palette="viridis")
    plt.xlabel("Pourcentage (%)")
    plt.title("Résumé des proportions principales du jeu de données")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig("Results/5_logistic_regression/summary_barplot.png")
    plt.close()

def plot_proportion_malades_by_variable(X, y, feature_names):
    """
    Affiche un barplot de la proportion de malades (y=1) pour chaque modalité de variables catégorielles principales,
    y compris actifs, ni alcool ni tabac, alcool ET tabac.
    """
    import pandas as pd

    df = pd.DataFrame(X, columns=feature_names)
    df['Malade'] = y

    variables = [
        ("Cholestérol (0-2)", [0, 1, 2]),
        ("Glucose (0-2)", [0, 1, 2]),
        ("Tabagisme", [0, 1]),
        ("Consommation d’alcool", [0, 1]),
        ("Activité physique", [0, 1])
    ]

    # Groupes personnalisés
    groupes = {
        "Actifs": (df["Activité physique"] == 1),
        "Ni alcool ni tabac": (df["Tabagisme"] == 0) & (df["Consommation d’alcool"] == 0),
        "Alcool ET tabac": (df["Tabagisme"] == 1) & (df["Consommation d’alcool"] == 1)
    }

    plt.figure(figsize=(12, 7))
    # Variables classiques
    for i, (var, vals) in enumerate(variables):
        for j, v in enumerate(vals):
            mask = df[var] == v
            if mask.sum() > 0:
                prop = df.loc[mask, 'Malade'].mean() * 100
                plt.bar(f"{var}={v}", prop, color=sns.color_palette("tab10")[i])
                plt.text(f"{var}={v}", prop + 0.5, f"{prop:.1f}%", ha='center', fontsize=8)
    # Groupes personnalisés
    for k, (label, mask) in enumerate(groupes.items()):
        if mask.sum() > 0:
            prop = df.loc[mask, 'Malade'].mean() * 100
            plt.bar(label, prop, color="gray")
            plt.text(label, prop + 0.5, f"{prop:.1f}%", ha='center', fontsize=9, fontweight='bold')

    plt.ylabel("Proportion de malades (%)")
    plt.title("Proportion de malades par variable, modalité et sous-groupe")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig("Results/5_logistic_regression/proportion_malades_by_variable.png")
    plt.close()

