"""
evaluation.py — Model evaluation metrics and comparison utilities.

All metrics come from sklearn — no values are hardcoded.
X_test / y_test are never passed to any training function.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    model_id:    str
    accuracy:    float
    precision:   float
    recall:      float
    macro_f1:    float
    weighted_f1: float
    report_dict: dict
    conf_matrix: np.ndarray
    y_pred:      np.ndarray
    y_true:      np.ndarray
    class_names: list


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    class_names: list,
    model_id:    str = "",
) -> EvalResult:
    """
    Runs model.predict on X_test (shape N×28×28×1, float32) and computes all metrics.
    No training data is touched.
    """
    raw = model.predict(X_test, verbose=0)          # (N, 10)
    y_pred = np.argmax(raw, axis=1).astype(np.int64)
    y_true = y_test.astype(np.int64)

    accuracy    = float(accuracy_score(y_true, y_pred))
    macro_f1    = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    precision   = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall      = float(recall_score(y_true, y_pred, average="weighted",    zero_division=0))

    # classification_report returns dict with class names when we pass them as target_names
    n_classes = len(class_names)
    labels = list(range(n_classes))
    report_dict = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

    return EvalResult(
        model_id=model_id,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        report_dict=report_dict,
        conf_matrix=conf_mat,
        y_pred=y_pred,
        y_true=y_true,
        class_names=class_names,
    )


# ─── Comparison ───────────────────────────────────────────────────────────────

def compare_models(baseline: EvalResult, candidate: EvalResult) -> dict:
    """
    Returns a comparison dict:
    {
        "accuracy_delta":    float,   # candidate - baseline
        "macro_f1_delta":    float,
        "weighted_f1_delta": float,
        "recommendation":    str,     # human-readable text
        "winner":            "baseline" | "candidate" | "tie",
    }
    Primary criterion: macro F1 (treats all classes equally).
    """
    acc_delta    = candidate.accuracy    - baseline.accuracy
    mf1_delta    = candidate.macro_f1   - baseline.macro_f1
    wf1_delta    = candidate.weighted_f1 - baseline.weighted_f1

    threshold = 0.005   # treat differences < 0.5 pp as a tie

    if abs(mf1_delta) < threshold:
        winner = "tie"
    elif mf1_delta > 0:
        winner = "candidate"
    else:
        winner = "baseline"

    # Per-class breakdown for recommendation text
    cand_report = candidate.report_dict
    base_report = baseline.report_dict
    better_classes, worse_classes = [], []

    for cls in candidate.class_names:
        cand_f1 = cand_report.get(cls, {}).get("f1-score", 0.0)
        base_f1 = base_report.get(cls, {}).get("f1-score", 0.0)
        delta = cand_f1 - base_f1
        if delta > 0.02:
            better_classes.append(f"{cls} (+{delta:.2f})")
        elif delta < -0.02:
            worse_classes.append(f"{cls} ({delta:.2f})")

    if winner == "tie":
        recommendation = (
            f"The models perform similarly (macro F1 delta = {mf1_delta:+.4f}). "
            "Keep the baseline unless storage or inference speed favours the candidate."
        )
    elif winner == "candidate":
        recommendation = (
            f"✅ Candidate is better (macro F1 {mf1_delta:+.4f}, "
            f"accuracy {acc_delta:+.4f}). "
        )
        if better_classes:
            recommendation += f"Improved on: {', '.join(better_classes)}. "
        if worse_classes:
            recommendation += f"Regressed on: {', '.join(worse_classes)}. "
        recommendation += "Consider promoting the candidate as the active model."
    else:
        recommendation = (
            f"⚠️ Baseline is better (macro F1 delta = {mf1_delta:+.4f}, "
            f"accuracy delta = {acc_delta:+.4f}). "
        )
        if worse_classes:
            recommendation += f"Candidate is weaker on: {', '.join(worse_classes)}. "
        recommendation += "Keep the baseline as the active model."

    return {
        "accuracy_delta":    round(acc_delta,  4),
        "macro_f1_delta":    round(mf1_delta,  4),
        "weighted_f1_delta": round(wf1_delta,  4),
        "recommendation":    recommendation,
        "winner":            winner,
    }


# ─── Table formatting ─────────────────────────────────────────────────────────

def format_comparison_table(baseline: EvalResult, candidate: EvalResult) -> list:
    """
    Returns a list of dicts suitable for pd.DataFrame / st.dataframe.
    Rows: Accuracy, Macro F1, Weighted F1, Precision, Recall.
    Columns: Metric, Baseline, Candidate, Delta.
    """
    metrics = [
        ("Accuracy",     baseline.accuracy,    candidate.accuracy),
        ("Macro F1",     baseline.macro_f1,    candidate.macro_f1),
        ("Weighted F1",  baseline.weighted_f1, candidate.weighted_f1),
        ("Precision",    baseline.precision,   candidate.precision),
        ("Recall",       baseline.recall,      candidate.recall),
    ]
    rows = []
    for name, base_val, cand_val in metrics:
        delta = cand_val - base_val
        rows.append({
            "Metric":    name,
            "Baseline":  f"{base_val:.4f}",
            "Candidate": f"{cand_val:.4f}",
            "Delta":     f"{delta:+.4f}",
        })
    return rows


# ─── Dual evaluation with non-clothing rejection ─────────────────────────────

def evaluate_dual_with_rejection(
    baseline_model,
    candidate_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    model_id: str = "candidate",
) -> dict:
    """
    Evaluates both models with strict fairness:
    - If 10 classes: standard comparison on all test samples.
    - If 11 classes:
      1. Clothing subset (labels 0..9): Baseline (10-class) and Candidate (11-class)
         are compared on identical unseen clothing test samples.
      2. Other subset (label 10): Non-clothing test samples are evaluated:
         - Candidate: Non-clothing rejection accuracy (% predicted as Other).
         - Baseline: False clothing assignment rate (% forced into clothing classes)
           and which clothing classes were falsely assigned.
      3. Full 11-class evaluation for Candidate across all test items.
    """
    is_11_class = (len(class_names) == 11 and "Other / Not Clothing" in class_names)

    if not is_11_class:
        base_eval = evaluate_model(baseline_model, X_test, y_test, class_names, "baseline_v0")
        cand_eval = evaluate_model(candidate_model, X_test, y_test, class_names, model_id)
        comp = compare_models(base_eval, cand_eval)
        return {
            "is_11_class": False,
            "baseline_clothing_eval": base_eval,
            "candidate_clothing_eval": cand_eval,
            "candidate_full_eval": cand_eval,
            "comparison": comp,
            "rejection_stats": None,
        }

    # 11-class scenario:
    clothing_classes = class_names[:10]
    clothing_mask = (y_test < 10)
    other_mask = (y_test == 10)

    X_cloth, y_cloth = X_test[clothing_mask], y_test[clothing_mask]
    X_other, y_other = X_test[other_mask], y_test[other_mask]

    # Baseline on clothing test items
    base_eval = evaluate_model(baseline_model, X_cloth, y_cloth, clothing_classes, "baseline_v0")

    # Candidate on clothing test items
    cand_cloth_raw = candidate_model.predict(X_cloth, verbose=0)
    cand_cloth_pred = np.argmax(cand_cloth_raw, axis=1)

    cand_cloth_acc  = float(accuracy_score(y_cloth, cand_cloth_pred))
    cand_cloth_mf1  = float(f1_score(y_cloth, cand_cloth_pred, average="macro",    zero_division=0))
    cand_cloth_wf1  = float(f1_score(y_cloth, cand_cloth_pred, average="weighted", zero_division=0))
    cand_cloth_prec = float(precision_score(y_cloth, cand_cloth_pred, average="weighted", zero_division=0))
    cand_cloth_rec  = float(recall_score(y_cloth, cand_cloth_pred, average="weighted",    zero_division=0))
    cand_cloth_conf = confusion_matrix(y_cloth, cand_cloth_pred, labels=list(range(10)))

    cand_clothing_eval = EvalResult(
        model_id=model_id,
        accuracy=cand_cloth_acc,
        precision=cand_cloth_prec,
        recall=cand_cloth_rec,
        macro_f1=cand_cloth_mf1,
        weighted_f1=cand_cloth_wf1,
        report_dict=classification_report(
            y_cloth, cand_cloth_pred,
            labels=list(range(10)),
            target_names=clothing_classes,
            output_dict=True,
            zero_division=0,
        ),
        conf_matrix=cand_cloth_conf,
        y_pred=cand_cloth_pred,
        y_true=y_cloth,
        class_names=clothing_classes,
    )

    # Full 11-class candidate evaluation
    cand_full_eval = evaluate_model(candidate_model, X_test, y_test, class_names, model_id)
    comp = compare_models(base_eval, cand_clothing_eval)

    # Rejection statistics on unseen non-clothing images
    rejection_stats = None
    if len(X_other) > 0:
        cand_other_raw = candidate_model.predict(X_other, verbose=0)
        cand_other_pred = np.argmax(cand_other_raw, axis=1)
        rejected_count = int((cand_other_pred == 10).sum())
        total_other = len(X_other)
        rejection_rate = float(rejected_count / total_other)

        base_other_raw = baseline_model.predict(X_other, verbose=0)
        base_other_pred = np.argmax(base_other_raw, axis=1)

        false_counts = {}
        for c_idx in range(10):
            c_count = int((base_other_pred == c_idx).sum())
            if c_count > 0:
                false_counts[clothing_classes[c_idx]] = c_count

        rejection_stats = {
            "total_other_test": total_other,
            "candidate_rejected_count": rejected_count,
            "candidate_rejection_rate": rejection_rate,
            "baseline_false_clothing_rate": 1.0,
            "baseline_false_classes": false_counts,
        }

    return {
        "is_11_class": True,
        "baseline_clothing_eval": base_eval,
        "candidate_clothing_eval": cand_clothing_eval,
        "candidate_full_eval": cand_full_eval,
        "comparison": comp,
        "rejection_stats": rejection_stats,
    }
