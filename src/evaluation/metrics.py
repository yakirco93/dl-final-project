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
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score


def _precision_at_k(y_true, y_pred_proba, k: int) -> float:
    order = np.argsort(y_pred_proba)[::-1][:k]
    return float(np.asarray(y_true)[order].mean())


def compute_metrics(y_true, y_pred_proba, k_values=(50, 100, 200)) -> dict:
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_true, [1 if p >= 0.5 else 0 for p in y_pred_proba])

    results = {"roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1}
    for k in k_values:
        results[f"precision_at_{k}"] = _precision_at_k(y_true, y_pred_proba, k)
    return results


# Maps each ablation-table entry (config["evaluation"]["ablation"]) onto the
# baseline in experiments/baselines_results.csv that already isolates exactly
# that input combination. No new controlled models are trained here -- the
# baselines (src/models/baselines.py) were built for precisely this purpose.
_ABLATION_TO_BASELINE = {
    "title_only": ("tfidf_logreg", "title"),
    "title_tags": ("text_only_bert", "title (+ no tags baseline exists -- see note)"),
    "image_only": ("image_only", "image"),
    "title_image": ("title_image_frozen", "title+image"),
}

# The best (highest val PR-AUC) run whose result stands in for "full_model" --
# exp07, the tuned Hybrid (AlephBERT+ResNet50), the only model in the project
# that beat text_only_bert. Path is relative to config["logging"]["output_dir"].
_FULL_MODEL_RUN_DIR = "hybrid_model2_tuned"
_FULL_MODEL_INPUTS = "title+image+site (no tags -- see note)"
_FULL_MODEL_SOURCE = "Model 2 tuned (exp07), best val checkpoint"

_NO_TAGS_NOTE = (
    "No baseline/model in this project was ever given article tags as an "
    "input -- 'title_tags' here is really title-only text (text_only_bert), "
    "and full_model is title+image+site. Tags were considered early on and "
    "explicitly dropped (see project_explanation_HE.md); this is noted here "
    "rather than silently relabeling the table."
)


def run_ablation(config: dict):
    """Assembles the ablation comparison table (does the image add value
    beyond the title?) from results that already exist on disk -- each
    baseline in experiments/baselines_results.csv isolates exactly one input
    combination, and exp07 (the winning Model 2 run) stands in for
    full_model. Reads both files fresh each call so the table always
    reflects whatever is currently on disk, rather than hardcoding numbers
    that could drift out of sync with a rerun.

    Returns a pandas DataFrame indexed by ablation entry name, with columns
    roc_auc, pr_auc, inputs, source.
    """
    import os
    import pandas as pd

    output_dir = config["logging"]["output_dir"]
    baselines = pd.read_csv(os.path.join(output_dir, "baselines_results.csv"), index_col=0)

    full_model_log_path = os.path.join(output_dir, _FULL_MODEL_RUN_DIR, "val_log.csv")
    full_model_log = pd.read_csv(full_model_log_path)
    full_model_best = full_model_log.loc[full_model_log["pr_auc"].idxmax()]

    rows = []
    for entry in config["evaluation"]["ablation"]:
        name = entry["name"]
        if name == "full_model":
            rows.append({
                "name": name,
                "inputs": _FULL_MODEL_INPUTS,
                "roc_auc": float(full_model_best["roc_auc"]),
                "pr_auc": float(full_model_best["pr_auc"]),
                "source": _FULL_MODEL_SOURCE,
            })
        elif name in _ABLATION_TO_BASELINE:
            baseline_name, inputs_label = _ABLATION_TO_BASELINE[name]
            row = baselines.loc[baseline_name]
            rows.append({
                "name": name,
                "inputs": inputs_label,
                "roc_auc": float(row["roc_auc"]),
                "pr_auc": float(row["pr_auc"]),
                "source": f"{baseline_name} baseline",
            })
        else:
            raise ValueError(f"No result mapped for ablation entry {name!r}")

    table = pd.DataFrame(rows).set_index("name")
    print(_NO_TAGS_NOTE)
    return table
