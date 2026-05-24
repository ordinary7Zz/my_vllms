from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score


MetricsDict = Dict[str, Union[float, int]]


def compute_metrics(y_true: List[int], y_prob: List[float], y_pred: List[int]) -> MetricsDict:
    auroc = roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else float("nan")
    auprc = average_precision_score(y_true, y_prob) if len(set(y_true)) == 2 else float("nan")
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "accuracy": float(acc),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_metric_cis(
    y_true: List[int],
    y_prob: List[float],
    threshold: float,
    n_bootstrap: int = 2000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    n = len(y_true_arr)
    rng = np.random.default_rng(seed)
    metric_samples = {
        "auroc": [],
        "auprc": [],
        "accuracy": [],
        "f1": [],
        "sensitivity": [],
        "specificity": [],
    }
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample_y_true = y_true_arr[indices]
        sample_y_prob = y_prob_arr[indices]
        sample_y_pred = (sample_y_prob >= threshold).astype(int)
        metrics = compute_metrics(sample_y_true.tolist(), sample_y_prob.tolist(), sample_y_pred.tolist())
        for key in metric_samples:
            value = metrics[key]
            if not np.isnan(value):
                metric_samples[key].append(value)
    lower_q = (1.0 - alpha) / 2.0
    upper_q = 1.0 - lower_q
    cis: Dict[str, Tuple[float, float]] = {}
    for key, values in metric_samples.items():
        if not values:
            cis[key] = (float("nan"), float("nan"))
        else:
            cis[key] = (float(np.quantile(values, lower_q)), float(np.quantile(values, upper_q)))
    return cis
