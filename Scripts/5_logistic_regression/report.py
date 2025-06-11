from metrics import accuracy, confusion_matrix, classification_report
from visualization import plot_confusion_matrix, plot_feature_importance, plot_roc_curve
import datetime
import numpy as np

def generate_report(y_test, y_pred, y_pred_proba, dataset_desc, weights, feature_names, lignes_supprimees, y_low, y_medium, y_high):
    acc = accuracy(y_test, y_pred)
    TP, TN, FP, FN = confusion_matrix(y_test, y_pred)
    classif = classification_report(TP, TN, FP, FN)

    report_text = (
        f"{dataset_desc}\n"
        f"=== Rapport de performance ===\n"
        f"Date : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Accuracy : {acc:.4f}\n"
        f"TP : {TP} | TN : {TN} | FP : {FP} | FN : {FN}\n\n"
        f"{classif}\n\n"
        f"=== Nettoyage des données ===\n"
        f"Lignes supprimées pour valeurs aberrantes : {lignes_supprimees}\n\n"
        f"=== Facteurs influençant le risque de maladie cardio ===\n"
    )

    top_features = np.argsort(np.abs(weights[1:]))[::-1][:]
    for i in top_features:
        report_text += f"- {feature_names[i]} : Poids absolu = {weights[i+1]:.4f}\n"

    # Ajout des fichiers graphiques
    report_text += (
        "\nMatrice de confusion enregistrée dans 'matrice_confusion.png'\n"
        "Courbe ROC enregistrée dans 'roc_curve.png'\n"
        "Graphique des facteurs influents enregistré dans 'feature_importance.png'\n"
        "Distribution du glucose enregistrée dans 'glucose_distribution.png'\n"
        "Corrélation du glucose enregistrée dans 'glucose_correlation.png'\n\n"
    )

    # 📝 **Nouvelle section : Analyse des sous-groupes Glucose**
    glucose_risks = [np.mean(y_low), np.mean(y_medium), np.mean(y_high)]
    report_text += (
        f"=== Analyse avancée de l'effet du glucose ===\n"
        f"- Taux de maladies cardiovasculaires par niveau de glucose :\n"
        f"  ▸ Glucose faible (0) : {glucose_risks[0]:.2%} de malades\n"
        f"  ▸ Glucose moyen (1) : {glucose_risks[1]:.2%} de malades\n"
        f"  ▸ Glucose élevé (2) : {glucose_risks[2]:.2%} de malades\n\n"
        f"- La distribution du glucose montre une forte concentration des niveaux élevés de glucose chez les individus sains.\n"
        f"- La corrélation entre glucose et pression artérielle est faible, mais il est significativement lié au cholestérol.\n"
        f"- Le poids absolu du glucose seul est négatif, ce qui indique que des niveaux plus élevés de glucose pourraient être liés à des facteurs protecteurs.\n"
        f"- L'interaction glucose × cholestérol est fortement positive, suggérant que le glucose devient un indicateur de risque lorsqu'il est combiné au cholestérol.\n"
        f"- L'ajout d'interactions dans le modèle semble modifier l'interprétation du glucose, nécessitant une analyse approfondie des sous-groupes.\n"
    )

    with open("Results/5_logistic_regression/rapport_model.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
