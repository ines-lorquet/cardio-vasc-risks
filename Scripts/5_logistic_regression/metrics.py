import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

def classification_report(TP, TN, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f"Précision : {precision:.4f}\nRappel : {recall:.4f}\nF1-score : {f1_score:.4f}"

def subgroup_performance(y_true, y_pred, label):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    report = classification_report(TP, TN, FP, FN)
    return f"--- {label} ({len(y_true)} échantillons) ---\n{report}\n"
