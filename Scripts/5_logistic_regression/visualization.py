import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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
    plt.xlabel('Faux Positifs')
    plt.ylabel('Vrais Positifs')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("Results/5_logistic_regression/roc_curve.png")
    plt.close()

def plot_feature_importance(weights, feature_names):
    importance = np.abs(weights[1:])
    sorted_idx = np.argsort(importance)[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[sorted_idx], y=np.array(feature_names)[sorted_idx])
    plt.title("Importance des caractéristiques")
    plt.xlabel("Poids absolus")
    plt.ylabel("Caractéristiques")
    plt.tight_layout()
    plt.savefig("Results/5_logistic_regression/feature_importance.png")
    plt.close()
