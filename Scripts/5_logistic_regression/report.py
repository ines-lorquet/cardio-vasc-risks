from metrics import accuracy, confusion_matrix, classification_report
from visualization import plot_confusion_matrix, plot_feature_importance, plot_roc_curve
import datetime
import numpy as np

def generate_report(y_test, y_pred, y_pred_proba, dataset_desc, weights, feature_names):
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
        f"=== Facteurs influençant le risque de maladie cardio ===\n"
    )

    top_features = np.argsort(np.abs(weights[1:]))[::-1][:5]
    for i in top_features:
        report_text += f"- {feature_names[i]} : Poids absolu = {weights[i+1]:.4f}\n"

    report_text += (
        "\nMatrice de confusion enregistrée dans 'matrice_confusion.png'\n"
        "Courbe ROC enregistrée dans 'roc_curve.png'\n"
        "Graphique des facteurs influents enregistré dans 'feature_importance.png'\n"
    )

    with open("Results/5_logistic_regression/rapport_model.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    plot_confusion_matrix(TP, TN, FP, FN)
    plot_feature_importance(weights, feature_names)
    plot_roc_curve(y_test, y_pred_proba)
