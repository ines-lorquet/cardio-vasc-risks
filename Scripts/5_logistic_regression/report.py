from metrics import accuracy, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
import datetime
import numpy as np

def generate_report(
    y_test, y_pred, y_pred_proba, dataset_desc, weights, feature_names, lignes_supprimees,
    y_low, y_medium, y_high,
    perf_glucose, perf_chol, perf_fumeur, perf_alcool, perf_inactif
):
    acc = accuracy(y_test, y_pred)
    TP, TN, FP, FN = confusion_matrix(y_test, y_pred)
    classif = classification_report(TP, TN, FP, FN)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Section 1 : Statistiques du jeu de données
    report_text = "=== 1. Statistiques du jeu de données ===\n"
    report_text += dataset_desc + "\n"

    # Section 2 : Nettoyage et préparation des données
    report_text += "=== 2. Nettoyage et préparation des données ===\n"
    report_text += f"- Lignes supprimées pour valeurs aberrantes : {lignes_supprimees}\n\n"

    # Section 3 : Entraînement et performance du modèle
    report_text += "=== 3. Entraînement et performance du modèle ===\n"
    report_text += f"- Date : {now}\n"
    report_text += f"- Accuracy : {acc:.4f} ({acc*100:.2f}%)\n"
    report_text += f"- Matrice de confusion :\n"
    report_text += f"    - TP : {TP} | TN : {TN} | FP : {FP} | FN : {FN}\n"
    report_text += "- Scores :\n"
    lines = classif.split('\n')
    for line in lines:
        if line.strip():
            # Ajoute aussi le pourcentage pour chaque score
            if ':' in line:
                metric, value = line.split(':')
                value = float(value)
                report_text += f"    - {metric.strip()} : {value:.4f} ({value*100:.2f}%)\n"
            else:
                report_text += f"    - {line.strip()}\n"
    report_text += f"- AUC (Area Under Curve) : {auc_score:.4f} ({auc_score*100:.2f}%)\n\n"

    # Section 3bis : Performances par sous-groupe
    report_text += "=== 3bis. Performances par sous-groupe ===\n"
    report_text += "---- Glucose ----\n" + perf_glucose
    report_text += "---- Cholestérol ----\n" + perf_chol
    report_text += "---- Fumeurs ----\n" + perf_fumeur
    report_text += "---- Alcool ----\n" + perf_alcool
    report_text += "---- Inactifs ----\n" + perf_inactif + "\n"
    report_text += "Conclusion :\n"
    report_text += "Le modèle est robuste pour détecter les malades dans tous les sous-groupes, surtout ceux à risque élevé (glucose ou cholestérol 2). Il reste prudent (beaucoup de faux positifs), ce qui est adapté à un contexte de prévention où il vaut mieux alerter trop que pas assez.\n"
    report_text += "Pour améliorer la précision, il faudra travailler sur la réduction des faux positifs (par exemple, ajuster le seuil, ajouter des variables, ou utiliser un modèle plus complexe).\n\n"

    # Section 4 : Explication des métriques
    report_text += "=== 4. Explication des métriques ===\n"
    report_text += "- **Accuracy (Exactitude)** : Proportion de prédictions correctes sur l’ensemble des cas.\n"
    report_text += "- **Précision** : Proportion de vrais positifs parmi les cas prédits positifs (évite les faux positifs).\n"
    report_text += "- **Rappel (Recall)** : Proportion de vrais positifs détectés parmi tous les cas réellement positifs (évite les faux négatifs).\n"
    report_text += "- **F1-score** : Moyenne harmonique entre précision et rappel, équilibre entre les deux.\n"
    report_text += "- **AUC (Area Under Curve)** : Aire sous la courbe ROC, mesure la capacité du modèle à distinguer les classes (1 = parfait, 0.5 = aléatoire).\n"

    # Section 5 : Choix du seuil et pondération des classes
    report_text += "=== 5. Choix du seuil et pondération des classes ===\n"
    report_text += "- Seuil de classification optimisé pour maximiser le rappel et réduire les faux négatifs.\n"
    report_text += "- Pondération de classe (pos_weight=2.0) pour favoriser la détection des malades.\n"
    report_text += "- Résultat : rappel et F1-score augmentés, légère baisse de la précision.\n\n"

    # Section 6 : Importance des caractéristiques
    report_text += "=== 6. Importance des caractéristiques ===\n"
    report_text += "- Variables les plus influentes (poids absolu) :\n"
    top_features = np.argsort(np.abs(weights[1:]))[::-1]
    for i in top_features:
        report_text += f"    - {feature_names[i]} : {weights[i+1]:.4f}\n"
    report_text += "\n"

    # Section 7 : Visualisations et interprétations (sans les 3 graphiques glucose)
    report_text += "=== 7. Visualisations et interprétations ===\n"
    report_text += "- Matrice de confusion (`matrice_confusion.png`) : Visualise la répartition des vrais/faux positifs et négatifs.\n"
    report_text += "- Courbe ROC (`roc_curve.png`) : Capacité du modèle à distinguer malades/sains (AUC proche de 1 = meilleur).\n"
    report_text += "- Importance des caractéristiques (`feature_importance.png`) : Poids de chaque variable dans la prédiction.\n"
    report_text += "- Matrice de corrélation (`correlation_matrix.png`) : Corrélations entre toutes les variables et la cible.\n\n"

    # Section 8 : Résumé des résultats
    report_text += "=== 8. Résumé des résultats ===\n"
    report_text += "- Le modèle détecte très bien les malades (rappel élevé), ce qui est crucial en santé publique.\n"
    report_text += "- Il fait cependant beaucoup de faux positifs (précision modérée), donc certains sains sont à tort considérés à risque.\n"
    report_text += "- L’équilibre global (F1-score) est correct.\n"
    report_text += "- L’AUC montre que le modèle distingue bien les deux classes.\n\n"

    # Section 9 : Conclusion
    report_text += "=== 9. Conclusion ===\n"
    report_text += "Ce modèle est adapté au dépistage : il privilégie la détection des malades (peu de faux négatifs), quitte à avoir plus de faux positifs. C’est souvent le choix en médecine préventive.\n"


    with open("Results/5_logistic_regression/rapport_model.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
