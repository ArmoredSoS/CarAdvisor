"""
eval_dm.py — Intrinsic evaluation of the DM module.

Metrics
-------
  Action selection : Accuracy, Precision, Recall, F1 (per-class + macro),
                     Confusion Matrix

Outputs (written to eval_outputs/)
-------
  figures/dm_action_confusion_matrix.pdf
  figures/dm_action_metrics_bar.pdf
  tables/dm_action_metrics.tex

Usage
-----
  # Run model inference and evaluate
  python eval_dm.py [--model-name qwen3] [--save-predictions preds_dm.json]

  # Evaluate from pre-saved predictions (no GPU needed)
  python eval_dm.py --predictions preds_dm.json

Test data format  (test_data_dm.json)
--------------------------------------
{
  "dm": [
    {
      "id": "dm_sf_01",
      "description": "optional human-readable note",
      "dialogue_state": {
        "turn_count": 1,
        "intent": "provide_information",
        "slots": { "fuel_type": "Gasoline" },
        "recommended_cars": []
      },
      "expected": { "action": "slot_filling" }
    },
    ...
  ]
}

Each entry must have:
  - id               : unique string identifier
  - dialogue_state   : the full state dict passed to DM.decide()
  - expected.action  : one of the five action strings
"""

from __future__ import annotations
import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = "eval_outputs"
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR  = os.path.join(OUTPUT_DIR, "tables")
for d in (FIG_DIR, TABLE_DIR):
    os.makedirs(d, exist_ok=True)

ACTIONS = [
    "slot_filling",
    "correct_info",
    "help_user",
    "recommend",
    "end_conversation",
]

ACTION_SHORT = {
    "slot_filling":    "Slot filling",
    "correct_info":    "Correct info",
    "help_user":       "Help user",
    "recommend":       "Recommend",
    "end_conversation":"End conv.",
}

PLOT_STYLE = {
    "font.family":     "serif",
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":      150,
}

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig]   {path}")


def plot_confusion_matrix(
    y_true: list, y_pred: list, labels: list[str],
    title: str, filename: str,
    short_labels: list[str] | None = None,
    figsize: tuple = (8, 6),
) -> np.ndarray:
    cm       = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Row-normalised proportion", fraction=0.046, pad=0.04)

    ticks       = range(len(labels))
    disp_labels = short_labels or [l.replace("_", "\n") for l in labels]
    ax.set_xticks(ticks); ax.set_xticklabels(disp_labels, ha="center")
    ax.set_yticks(ticks); ax.set_yticklabels(disp_labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i,
                    f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                    ha="center", va="center", fontsize=7.5, color=color)

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(title,        fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    _save(fig, filename)
    return cm


def plot_action_bar(y_true: list, y_pred: list) -> None:
    labels = [a for a in ACTIONS if a in y_true or a in y_pred]
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    x = np.arange(len(labels))
    w = 0.25

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5))

    ax.bar(x - w, prec, w, label="Precision", color="#1f77b4", alpha=0.85)
    ax.bar(x,     rec,  w, label="Recall",    color="#ff7f0e", alpha=0.85)
    ax.bar(x + w, f1,   w, label="F1",        color="#2ca02c", alpha=0.85)

    macro_f1 = float(np.mean(f1))
    ax.axhline(
        macro_f1, color="red", linestyle="--", linewidth=1.4,
        label=f"Macro F1 = {macro_f1:.3f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([ACTION_SHORT.get(l, l) for l in labels], fontsize=8.5)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Action Selection — Per-action Precision, Recall, F1",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, "dm_action_metrics_bar.pdf")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def _save_tex(lines: list[str], name: str) -> None:
    path = os.path.join(TABLE_DIR, name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [tex]  {path}")


def save_action_latex(y_true: list, y_pred: list) -> None:
    labels = [a for a in ACTIONS if a in y_true or a in y_pred]
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{DM Action Selection -- Per-class and Overall Metrics}",
        r"\label{tab:dm_action}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Action} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Support} \\",
        r"\hline",
    ]
    for i, lbl in enumerate(labels):
        tex.append(
            f"{ACTION_SHORT.get(lbl, lbl)} & {prec[i]:.3f} & {rec[i]:.3f} & {f1[i]:.3f} & {int(sup[i])} \\\\"
        )
    tex += [
        r"\hline",
        f"Macro avg & {np.mean(prec):.3f} & {np.mean(rec):.3f} & {np.mean(f1):.3f} & {len(y_true)} \\\\",
        f"Accuracy  & \\multicolumn{{3}}{{c}}{{{acc:.3f}}} & {len(y_true)} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    _save_tex(tex, "dm_action_metrics.tex")


# ---------------------------------------------------------------------------
# Core evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(test_cases: list[dict], predictions: list[dict]) -> dict:
    print("\n" + "=" * 60)
    print("DM EVALUATION")
    print("=" * 60)

    y_true = [tc["expected"]["action"] for tc in test_cases]
    y_pred = [p.get("action", "")      for p  in predictions]

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAction Selection Accuracy : {acc:.4f}")
    print()
    print(classification_report(y_true, y_pred, labels=ACTIONS, zero_division=0))

    plot_confusion_matrix(
        y_true, y_pred, ACTIONS,
        title="Action Selection — Confusion Matrix",
        filename="dm_action_confusion_matrix.pdf",
        short_labels=[ACTION_SHORT[a] for a in ACTIONS],
        figsize=(9, 7),
    )
    plot_action_bar(y_true, y_pred)
    save_action_latex(y_true, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ACTIONS, average="macro", zero_division=0
    )

    return {
        "action_accuracy": acc,
        "macro_precision": float(prec),
        "macro_recall":    float(rec),
        "macro_f1":        float(f1),
    }


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def run_model_inference(test_cases: list[dict], args) -> list[dict]:
    """Load the model from the project and run DM inference on all test cases."""
    import sys
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, ".")
    from utils import DM, MODELS
    import main_CarAdvisor as main_mod

    print("\nLoading model …")
    model_name, InitModel, prepare_text = MODELS[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = InitModel(model_name, dtype="auto")
    model.eval()

    dm_engine = DM(model, tokenizer, prepare_text, main_mod.dm)

    predictions = []
    n = len(test_cases)
    for i, tc in enumerate(test_cases, 1):
        print(f"  [{i:02d}/{n}] {tc['id']}", end="\r", flush=True)
        dialogue_state = tc["dialogue_state"]
        try:
            action, value = dm_engine.decide(dialogue_state)
            result = {"action": action, "value": value}
        except Exception as e:
            print(f"\n  [error] {tc['id']}: {e}")
            result = {"action": "", "value": None}
        predictions.append(result)

    print(f"\n  Done — {n} cases processed.")
    return predictions


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the DM module (action selection)."
    )
    parser.add_argument("--test-data",        default="test_data_dm.json",
                        help="Path to ground-truth DM test file (default: test_data_dm.json).")
    parser.add_argument("--predictions",      default=None,
                        help="Path to pre-saved predictions JSON. "
                             "If provided, skips model inference.")
    parser.add_argument("--save-predictions", default=None,
                        help="Save model predictions to this JSON path.")
    parser.add_argument("--model-name",       default="qwen3",
                        help="Model key defined in utils.MODELS.")
    args = parser.parse_args()

    with open(args.test_data) as f:
        all_data = json.load(f)
    test_cases = all_data["dm"]
    print(f"Loaded {len(test_cases)} test cases from {args.test_data}")

    if args.predictions:
        print(f"Loading predictions from {args.predictions} …")
        with open(args.predictions) as f:
            predictions = json.load(f)["dm"]
    else:
        predictions = run_model_inference(test_cases, args)
        if args.save_predictions:
            with open(args.save_predictions, "w") as f:
                json.dump({"dm": predictions}, f, indent=2)
            print(f"Predictions saved → {args.save_predictions}")

    results = run_evaluation(test_cases, predictions)

    print("\nSummary")
    print("-" * 40)
    for k, v in results.items():
        print(f"  {k:<25} {v:.4f}")
    print(f"\nAll outputs written to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
