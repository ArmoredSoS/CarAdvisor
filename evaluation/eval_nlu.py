"""
eval_nlu.py — Intrinsic evaluation of the NLU module.

Metrics
-------
  Intent detection : Accuracy, Precision, Recall, F1 (per-class + macro),
                     Confusion Matrix
  Slot filling     : Precision, Recall, F1 (per-slot + micro-avg),
                     Confusion Matrix aggregated over all slots

Outputs (written to eval_outputs/)
-------
  figures/nlu_intent_confusion_matrix.pdf
  figures/nlu_slot_metrics_bar.pdf
  figures/nlu_slot_confusion_matrix.pdf
  tables/nlu_intent_metrics.tex
  tables/nlu_slot_metrics.tex

Usage
-----
  # Run model inference and evaluate
  python eval_nlu.py [--model-name qwen3] [--save-predictions preds.json]

  # Evaluate from pre-saved predictions (no GPU needed)
  python eval_nlu.py --predictions preds.json
"""

from __future__ import annotations
import argparse
import json
import os
import re
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

OUTPUT_DIR   = "eval_outputs"
FIG_DIR      = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR    = os.path.join(OUTPUT_DIR, "tables")
for d in (FIG_DIR, TABLE_DIR):
    os.makedirs(d, exist_ok=True)

INTENTS = [
    "provide_information",
    "require_information",
    "correct_information",
    "request_recommendation",
    "end_conversation",
]

# Slots the NLU is instructed to extract (from system prompt)
EVAL_SLOTS = [
    "manufacturer", "vehicle_class", "fuel_type",
    "engine_displacement_l", "cylinders", "transmission",
    "drivetrain", "cargo_volume_cu_ft", "combined_mpg",
    "driving_range_miles", "electric_range_miles", "phev_blended_mpge",
    "annual_fuel_cost_usd", "co2_emissions_g_per_mile", "fe_rating_1_10",
]
# model_year excluded (always 2026, trivial), model excluded (proper names, ambiguous)

PLOT_STYLE = {
    "font.family": "serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
}

# ---------------------------------------------------------------------------
# Slot evaluation helpers
# ---------------------------------------------------------------------------

def _slot_match(predicted: str, expected: str) -> str:
    """Return 'tp', 'fp', 'fn', or 'tn' for a single slot."""
    pred = str(predicted).lower().strip()
    exp  = str(expected).lower().strip()
    if exp == "" and pred == "":
        return "tn"
    if exp != "" and pred == "":
        return "fn"
    if exp == "" and pred != "":
        return "fp"
    # Both non-empty — use same substring match as CarRecommender
    return "tp" if (pred in exp or exp in pred) else "fp"


def evaluate_slots(
    test_cases: list[dict], predictions: list[dict]
) -> tuple[dict, list[str], list[str]]:
    """
    Compute per-slot TP/FP/FN/TN counts.
    Also returns flat y_true / y_pred lists for the overall slot confusion matrix.
    Skips cases where intent does not involve slot extraction.
    """
    per_slot: dict[str, dict[str, int]] = {
        s: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for s in EVAL_SLOTS
    }
    y_true_flat, y_pred_flat = [], []

    skip_intents = {"require_information", "end_conversation"}

    for tc, pred in zip(test_cases, predictions):
        if tc["expected"]["intent"] in skip_intents:
            continue

        exp_slots  = tc["expected"].get("slots", {})
        pred_slots = pred.get("slots", {})

        for slot in EVAL_SLOTS:
            exp_val  = str(exp_slots.get(slot,  "")).strip()
            pred_val = str(pred_slots.get(slot, "")).strip()
            outcome  = _slot_match(pred_val, exp_val)
            per_slot[slot][outcome] += 1

            # Flatten: binary label per slot instance
            y_true_flat.append("filled"   if exp_val  != "" else "empty")
            y_pred_flat.append("filled"   if pred_val != "" else "empty")

    return per_slot, y_true_flat, y_pred_flat


def compute_slot_metrics(per_slot: dict) -> dict:
    """Compute precision, recall, F1 per slot and micro-averaged overall."""
    metrics: dict = {}
    total = {"tp": 0, "fp": 0, "fn": 0}

    for slot, c in per_slot.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[slot] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": tp + fn,
        }
        total["tp"] += tp; total["fp"] += fp; total["fn"] += fn

    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    metrics["__overall__"] = {
        "precision": prec, "recall": rec, "f1": f1,
        "support": tp + fn,
    }
    return metrics


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
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Row-normalised proportion", fraction=0.046, pad=0.04)

    ticks = range(len(labels))
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


def plot_slot_bar(slot_metrics: dict, overall_f1: float) -> None:
    active = {
        s: m for s, m in slot_metrics.items()
        if s != "__overall__" and m["support"] > 0
    }
    if not active:
        print("  [warn] No active slots to plot.")
        return

    slots = list(active.keys())
    precs = [active[s]["precision"] for s in slots]
    recs  = [active[s]["recall"]    for s in slots]
    f1s   = [active[s]["f1"]        for s in slots]

    x = np.arange(len(slots))
    w = 0.25

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(max(10, len(slots) * 0.9), 5))

    ax.bar(x - w, precs, w, label="Precision", color="#1f77b4", alpha=0.85)
    ax.bar(x,     recs,  w, label="Recall",    color="#ff7f0e", alpha=0.85)
    ax.bar(x + w, f1s,   w, label="F1",        color="#2ca02c", alpha=0.85)

    ax.axhline(
        overall_f1, color="red", linestyle="--", linewidth=1.4,
        label=f"Micro-avg F1 = {overall_f1:.3f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in slots], fontsize=7.5)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Slot Filling — Per-slot Precision, Recall, F1",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, "nlu_slot_metrics_bar.pdf")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def _save_tex(lines: list[str], name: str) -> None:
    path = os.path.join(TABLE_DIR, name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [tex]  {path}")


def save_intent_latex(y_true, y_pred) -> None:
    labels = [l for l in INTENTS if l in y_true or l in y_pred]
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    short = {
        "provide_information":   "Provide info",
        "require_information":   "Require info",
        "correct_information":   "Correct info",
        "request_recommendation":"Request rec.",
        "end_conversation":      "End conv.",
    }

    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{NLU Intent Detection -- Per-class and Overall Metrics}",
        r"\label{tab:nlu_intent}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Intent} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Support} \\",
        r"\hline",
    ]
    for i, lbl in enumerate(labels):
        tex.append(
            f"{short.get(lbl, lbl)} & {prec[i]:.3f} & {rec[i]:.3f} & {f1[i]:.3f} & {int(sup[i])} \\\\"
        )
    tex += [
        r"\hline",
        f"Macro avg & {np.mean(prec):.3f} & {np.mean(rec):.3f} & {np.mean(f1):.3f} & {len(y_true)} \\\\",
        f"Accuracy  & \\multicolumn{{3}}{{c}}{{{acc:.3f}}} & {len(y_true)} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    _save_tex(tex, "nlu_intent_metrics.tex")


def save_slot_latex(slot_metrics: dict) -> None:
    active = {
        s: m for s, m in slot_metrics.items()
        if s != "__overall__" and m["support"] > 0
    }

    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{NLU Slot Filling -- Per-slot Metrics}",
        r"\label{tab:nlu_slots}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Slot} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Support} \\",
        r"\hline",
    ]
    for slot, m in active.items():
        clean = slot.replace("_", r"\_")
        tex.append(
            f"{clean} & {m['precision']:.3f} & {m['recall']:.3f} & {m['f1']:.3f} & {m['support']} \\\\"
        )
    ov = slot_metrics["__overall__"]
    tex += [
        r"\hline",
        f"\\textbf{{Micro avg}} & {ov['precision']:.3f} & {ov['recall']:.3f} & {ov['f1']:.3f} & {ov['support']} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    _save_tex(tex, "nlu_slot_metrics.tex")


# ---------------------------------------------------------------------------
# Core evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(test_cases: list[dict], predictions: list[dict]) -> dict:
    print("\n" + "=" * 60)
    print("NLU EVALUATION")
    print("=" * 60)

    # ── Intent detection ──────────────────────────────────────────
    y_true = [tc["expected"]["intent"] for tc in test_cases]
    y_pred = [p.get("intent", "")      for p  in predictions]

    acc = accuracy_score(y_true, y_pred)
    print(f"\nIntent Detection Accuracy : {acc:.4f}")
    print()
    print(classification_report(y_true, y_pred, labels=INTENTS, zero_division=0))

    plot_confusion_matrix(
        y_true, y_pred, INTENTS,
        title="Intent Detection — Confusion Matrix",
        filename="nlu_intent_confusion_matrix.pdf",
        figsize=(9, 7),
    )
    save_intent_latex(y_true, y_pred)

    # ── Slot filling ──────────────────────────────────────────────
    per_slot, y_true_flat, y_pred_flat = evaluate_slots(test_cases, predictions)
    slot_metrics = compute_slot_metrics(per_slot)
    overall = slot_metrics["__overall__"]

    print(f"Slot Filling (micro-avg)")
    print(f"  Precision : {overall['precision']:.4f}")
    print(f"  Recall    : {overall['recall']:.4f}")
    print(f"  F1        : {overall['f1']:.4f}")
    print(f"  Support   : {overall['support']}")

    plot_slot_bar(slot_metrics, overall["f1"])

    plot_confusion_matrix(
        y_true_flat, y_pred_flat, ["empty", "filled"],
        title="Slot Filling — Binary Confusion Matrix\n(empty vs filled, aggregated over all slots and cases)",
        filename="nlu_slot_confusion_matrix.pdf",
        short_labels=["Empty\n(not expected)", "Filled\n(expected)"],
        figsize=(6, 5),
    )
    save_slot_latex(slot_metrics)

    return {
        "intent_accuracy":  acc,
        "slot_precision":   overall["precision"],
        "slot_recall":      overall["recall"],
        "slot_f1":          overall["f1"],
    }


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def run_model_inference(test_cases: list[dict], args) -> list[dict]:
    """Load the model from the project and run NLU inference on all test cases."""
    import sys
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, ".")
    from utils import NLU, MODELS
    import main_CarAdvisor as main_mod

    print("\nLoading model …")
    model_name, InitModel, prepare_text = MODELS[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = InitModel(model_name, dtype="auto")
    model.eval()

    nlu_engine = NLU(model, tokenizer, prepare_text, main_mod.nlu)

    predictions = []
    n = len(test_cases)
    for i, tc in enumerate(test_cases, 1):
        print(f"  [{i:02d}/{n}] {tc['id']}", end="\r", flush=True)
        history = tc.get("history", [])
        try:
            result = nlu_engine.parse(tc["user_input"], history, n_exchanges=10)
        except Exception as e:
            print(f"\n  [error] {tc['id']}: {e}")
            result = {"intent": "", "slots": {}}
        predictions.append(result)

    print(f"\n  Done — {n} cases processed.")
    return predictions


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the NLU module (intent detection + slot filling)."
    )
    parser.add_argument("--test-data",        default="test_data.json",
                        help="Path to ground-truth test file.")
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
    test_cases = all_data["nlu"]

    if args.predictions:
        print(f"Loading predictions from {args.predictions} …")
        with open(args.predictions) as f:
            predictions = json.load(f)["nlu"]
    else:
        predictions = run_model_inference(test_cases, args)
        if args.save_predictions:
            with open(args.save_predictions, "w") as f:
                json.dump({"nlu": predictions}, f, indent=2)
            print(f"Predictions saved → {args.save_predictions}")

    results = run_evaluation(test_cases, predictions)

    print("\nSummary")
    print("-" * 40)
    for k, v in results.items():
        print(f"  {k:<25} {v:.4f}")
    print(f"\nAll outputs written to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
