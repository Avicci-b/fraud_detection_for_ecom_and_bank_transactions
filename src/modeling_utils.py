import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    roc_auc_score, confusion_matrix, precision_recall_curve,
)

def evaluate_model(name, model, X_test, y_test, threshold=0.5):
    """Score one fitted model on the untouched test set. Returns (metrics dict, probabilities)."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "model": name,
        "pr_auc": average_precision_score(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    return metrics, proba

def plot_pr_curves(y_test, probas, title):
    """probas: dict of {model name: predicted probabilities}."""
    plt.figure(figsize=(7, 5))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        plt.plot(rec, prec, label=name)
    plt.axhline(np.mean(y_test), ls="--", c="gray", label="No-skill baseline")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.show()