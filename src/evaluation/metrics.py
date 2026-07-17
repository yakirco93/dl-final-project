"""Evaluation metrics + ablation study runner.

Metrics (see proposal's Evaluation section for why these, not accuracy):
  - ROC-AUC, PR-AUC, F1, Precision@K (K per configs/base_config.yaml
    -> evaluation.precision_at_k_values)

Ablation:
  run_ablation() trains/evaluates one model per entry in
  configs/base_config.yaml -> evaluation.ablation (title_only, title_tags,
  image_only, title_image, full_model) and returns a comparison table —
  this is what answers "does the image actually add value beyond the title?"
"""
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score


def compute_metrics(y_true, y_pred_proba, k_values=(50, 100, 200)) -> dict:
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_true, [1 if p >= 0.5 else 0 for p in y_pred_proba])

    results = {"roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1}
    # TODO: precision@k — rank by y_pred_proba desc, take top k, compute
    # fraction of true positives among them, for each k in k_values.
    for k in k_values:
        results[f"precision_at_{k}"] = None  # Step 7
    return results


def run_ablation(config: dict):
    raise NotImplementedError("Step 7: run once the main model (Steps 4-5) is trained.")
