"""
Evaluates dialogue efficiency of the CarAdvisor system.

Metrics per dialogue:
- Task Completion:        Did the dialogue reach the expected final action?
- Turns to Completion:    How many user turns were needed?
- Turn Efficiency:        Slots extracted per user turn (higher is better)
- Redundant Questions:    Did the DM ask for a slot already filled in cumulative state?
- Premature Recommend:    Did recommend fire with fewer than 3 core slots filled,
                          without the user explicitly requesting it?
- Action Accuracy:        Fraction of turns where the DM fired the expected action

Global metrics:
- Task Completion Rate
- Mean Turns to Completion (completed dialogues only)
- Mean Turn Efficiency
- Redundant Question Rate   (redundant turns / total turns)
- Premature Recommendation Rate
- Mean Action Accuracy

The script runs NLU + DM only (no NLG needed — efficiency is a structural property
of how the pipeline routes the conversation, not of the text it generates).

Usage:
    python evaluate_efficiency.py \\
        --test-file test_dialogues.json \\
        --model-name qwen3 \\
        --device cuda:0 \\
        --verbose
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer

from main_CarAdvisor import nlu as NLU_SYSTEM_PROMPT
from utils import MODELS, NLU, DialogueManager
from Dataset import RULES

CORE_SLOTS = ["car_type", "car_price", "car_state", "car_usecase", "fuel_efficiency"]

NULL_VALUES = (None, "null", "")

# Keywords that signal the user is explicitly requesting a recommendation
EXPLICIT_RECO_KEYWORDS = [
    "recommend", "suggest", "just give me", "show me something",
    "what do you suggest", "what would you recommend", "go ahead",
    "just pick", "your choice", "surprise me"
]


def is_null(v):
    if v is None:
        return True
    if isinstance(v, str) and v.lower().strip() in ("null", ""):
        return True
    return False


def user_asked_for_recommendation(user_input: str) -> bool:
    lowered = user_input.lower()
    return any(kw in lowered for kw in EXPLICIT_RECO_KEYWORDS)


def count_filled_slots(slots: dict) -> int:
    """Count non-null slots in a state dict."""
    return sum(1 for v in slots.values() if not is_null(v))


def merge_slots(cumulative: dict, new_state: dict) -> dict:
    """Merge newly extracted non-null slots into the cumulative state."""
    updated = dict(cumulative)
    for slot, value in new_state.get("slots", {}).items():
        if not is_null(value):
            updated[slot] = value
    return updated


def load_test_cases(path: str):
    with open(path) as f:
        return json.load(f)


def run_evaluation(args):
    test_cases = load_test_cases(args.test_file)

    # ── Load model ────────────────────────────────────────────────────────────
    model_name, InitModel, prepare_text = MODELS[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = InitModel(model_name, dtype="auto")
    model.eval()

    nlu_engine = NLU(model, tokenizer, prepare_text, NLU_SYSTEM_PROMPT)
    dm_engine = DialogueManager(RULES)

    # ── Global accumulators ───────────────────────────────────────────────────
    all_completed = []
    all_turns_to_completion = []
    all_turn_efficiencies = []
    all_redundant_rates = []
    all_premature_reco = []
    all_action_accuracies = []
    sample_results = []
    parse_errors = []

    for tc in test_cases:
        dialogue_id = tc["id"]
        turns = tc["turns"]
        expected_final_action = tc["expected_final_action"]
        max_turns = tc.get("max_acceptable_turns", len(turns) + 2)

        if args.verbose:
            print(f"\n{'═' * 60}")
            print(f"[{dialogue_id}] {tc.get('description', '')}")

        # Per-dialogue state
        cumulative_slots = {}
        messages = []
        turn_results = []

        completed = False
        final_action_fired = None
        total_slots_extracted = 0
        redundant_turns = 0
        premature_reco_fired = False
        action_correct_count = 0

        for turn_idx, turn in enumerate(turns):
            user_input = turn["user"]
            expected_action = turn.get("expected_action")
            expected_slots = set(turn.get("expected_slots_extracted", []))
            check_no_redundant = turn.get("check_no_redundant_question", False)

            messages.append({"role": "user", "content": user_input})

            # ── NLU ──────────────────────────────────────────────────────────
            try:
                state = nlu_engine.parse(user_input, messages, args.n_exchanges)
            except Exception as e:
                parse_errors.append({
                    "dialogue_id": dialogue_id,
                    "turn": turn_idx,
                    "error": str(e)
                })
                messages.append({"role": "assistant", "content": "[PARSE ERROR]"})
                continue

            # ── DM ───────────────────────────────────────────────────────────
            # Build DM state using cumulative slots merged with new extraction
            merged = merge_slots(cumulative_slots, state)
            dm_state = {"intent": state["intent"], "slots": merged}
            action, value = dm_engine.decide(dm_state)
            action_str = action if value is None else f"{action}({value})"
            final_action_fired = action

            # ── Slots extracted this turn ─────────────────────────────────────
            new_slots_this_turn = {
                k: v for k, v in state.get("slots", {}).items()
                if not is_null(v) and is_null(cumulative_slots.get(k))
            }
            slots_this_turn = len(new_slots_this_turn)
            total_slots_extracted += slots_this_turn

            # ── Redundant question check ──────────────────────────────────────
            # A question is redundant if DM asks for a slot already in cumulative state
            is_redundant = False
            if action == "slot_filling" and value in cumulative_slots and not is_null(cumulative_slots[value]):
                is_redundant = True
                redundant_turns += 1

            # Extra check for dialogues that explicitly test this
            if check_no_redundant and is_redundant:
                if args.verbose:
                    print(f"  ⚠  Turn {turn_idx+1}: redundant question for slot '{value}' "
                          f"(already filled: {cumulative_slots[value]})")

            # ── Premature recommendation check ────────────────────────────────
            filled_core = [s for s in CORE_SLOTS if not is_null(merged.get(s))]
            if (action == "recommend"
                    and len(filled_core) < 3
                    and not user_asked_for_recommendation(user_input)):
                premature_reco_fired = True

            # ── Action accuracy ───────────────────────────────────────────────
            # Strip slot name from action string for comparison
            action_base = action  # e.g. "slot_filling" regardless of which slot
            expected_base = expected_action.split("(")[0] if expected_action else None
            action_correct = (action_base == expected_base) if expected_base else True
            if action_correct:
                action_correct_count += 1

            # ── Update cumulative state ───────────────────────────────────────
            cumulative_slots = merge_slots(cumulative_slots, state)

            # ── Completion check ──────────────────────────────────────────────
            if action == expected_final_action:
                completed = True

            turn_result = {
                "turn": turn_idx + 1,
                "user_input": user_input,
                "extracted_slots": {k: v for k, v in state.get("slots", {}).items() if not is_null(v)},
                "new_slots_this_turn": list(new_slots_this_turn.keys()),
                "action": action_str,
                "expected_action": expected_action,
                "action_correct": action_correct,
                "is_redundant_question": is_redundant,
                "filled_core_slots": filled_core,
                "premature_reco": premature_reco_fired and action == "recommend",
            }
            turn_results.append(turn_result)

            # Simulate assistant reply in message history (empty — no NLG)
            messages.append({"role": "assistant", "content": f"[{action_str}]"})

            if args.verbose:
                correct_str = "✓" if action_correct else "✗"
                redund_str  = " ⚠ REDUNDANT" if is_redundant else ""
                premature_str = " ⚠ PREMATURE RECO" if (premature_reco_fired and action == "recommend") else ""
                print(f"  Turn {turn_idx+1}: [{correct_str}] action={action_str} | "
                      f"new_slots={list(new_slots_this_turn.keys())}{redund_str}{premature_str}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Per-dialogue metrics ──────────────────────────────────────────────
        n_turns = len(turn_results)
        turn_efficiency = total_slots_extracted / n_turns if n_turns > 0 else 0.0
        redundant_rate = redundant_turns / n_turns if n_turns > 0 else 0.0
        action_accuracy = action_correct_count / n_turns if n_turns > 0 else 0.0
        within_budget = n_turns <= max_turns

        all_completed.append(int(completed))
        if completed:
            all_turns_to_completion.append(n_turns)
        all_turn_efficiencies.append(turn_efficiency)
        all_redundant_rates.append(redundant_rate)
        all_premature_reco.append(int(premature_reco_fired))
        all_action_accuracies.append(action_accuracy)

        sample_result = {
            "id": dialogue_id,
            "description": tc.get("description", ""),
            "completed": completed,
            "expected_final_action": expected_final_action,
            "final_action_fired": final_action_fired,
            "turns_taken": n_turns,
            "max_acceptable_turns": max_turns,
            "within_turn_budget": within_budget,
            "turn_efficiency": turn_efficiency,
            "redundant_question_rate": redundant_rate,
            "premature_recommendation": premature_reco_fired,
            "action_accuracy": action_accuracy,
            "turns": turn_results,
        }
        sample_results.append(sample_result)

        if args.verbose:
            completion_str = "COMPLETED ✓" if completed else "INCOMPLETE ✗"
            budget_str = f"within budget ({n_turns}/{max_turns})" if within_budget else f"OVER BUDGET ({n_turns}/{max_turns})"
            print(f"  → {completion_str} | {budget_str} | "
                  f"efficiency={turn_efficiency:.2f} slots/turn | "
                  f"action_acc={action_accuracy:.2f}")

    # ── Global metrics ─────────────────────────────────────────────────────────
    n = len(test_cases)
    task_completion_rate = sum(all_completed) / n
    mean_turns = (sum(all_turns_to_completion) / len(all_turns_to_completion)
                  if all_turns_to_completion else 0.0)
    mean_efficiency = sum(all_turn_efficiencies) / n
    mean_redundant_rate = sum(all_redundant_rates) / n
    premature_reco_rate = sum(all_premature_reco) / n
    mean_action_accuracy = sum(all_action_accuracies) / n

    print("\n" + "=" * 60)
    print("DIALOGUE EFFICIENCY EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total dialogues              : {n}")
    print(f"Parse errors                 : {len(parse_errors)}")
    print()
    print(f"Task Completion Rate         : {task_completion_rate:.4f}  "
          f"({sum(all_completed)}/{n})")
    print(f"Mean Turns to Completion     : {mean_turns:.2f}  "
          f"(completed dialogues only)")
    print(f"Mean Turn Efficiency         : {mean_efficiency:.4f}  "
          f"(slots extracted per turn)")
    print(f"Redundant Question Rate      : {mean_redundant_rate:.4f}  "
          f"(redundant turns / total turns)")
    print(f"Premature Recommendation Rate: {premature_reco_rate:.4f}  "
          f"({sum(all_premature_reco)}/{n})")
    print(f"Mean Action Accuracy         : {mean_action_accuracy:.4f}")

    print("\nPer-dialogue summary:")
    print(f"  {'ID':<12} {'Done':>5} {'Turns':>6} {'MaxT':>5} {'Eff':>6} "
          f"{'Redund':>7} {'PreRec':>7} {'ActAcc':>7}")
    print("  " + "-" * 65)
    for r in sample_results:
        done_str   = "✓" if r["completed"] else "✗"
        budget_str = str(r["turns_taken"]) + ("⚠" if not r["within_turn_budget"] else " ")
        pre_str    = "YES" if r["premature_recommendation"] else "no"
        print(f"  {r['id']:<12} {done_str:>5} {budget_str:>6} {r['max_acceptable_turns']:>5} "
              f"{r['turn_efficiency']:>6.2f} {r['redundant_question_rate']:>7.3f} "
              f"{pre_str:>7} {r['action_accuracy']:>7.3f}")

    if parse_errors:
        print("\nParse errors:")
        for e in parse_errors:
            print(f"  dialogue={e['dialogue_id']} turn={e['turn']}: {e['error']}")

    # ── Save ───────────────────────────────────────────────────────────────────
    results = {
        "task_completion_rate": task_completion_rate,
        "mean_turns_to_completion": mean_turns,
        "mean_turn_efficiency": mean_efficiency,
        "mean_redundant_question_rate": mean_redundant_rate,
        "premature_recommendation_rate": premature_reco_rate,
        "mean_action_accuracy": mean_action_accuracy,
        "parse_errors": parse_errors,
        "per_dialogue": sample_results,
    }

    out_path = args.output or "efficiency_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dialogue efficiency.")
    parser.add_argument("--test-file", default="test_dialogues.json")
    parser.add_argument("--model-name", choices=MODELS.keys(), default="qwen3")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--n-exchanges", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())