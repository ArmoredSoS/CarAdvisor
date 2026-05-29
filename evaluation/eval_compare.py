"""
Evaluates the car comparison pipeline end-to-end.

Two metrics:

1. Car Retrieval Accuracy (CRA)
   For each test case, measures how well the NLU extracts the right set of cars
   and whether the RecommenderService can retrieve them.

   - Retrieval Precision = |retrieved ∩ expected| / |retrieved|
   - Retrieval Recall    = |retrieved ∩ expected| / |expected|
   - Retrieval F1        = harmonic mean of the above

   A car from expected_cars that is NOT in the dataset is flagged as
   "not_in_dataset" rather than penalising the retrieval score (because no
   system can retrieve a non-existent entry).

2. Attribute Correctness Rate (ACR)
   For every (car, attribute) pair that was actually retrieved, checks whether
   the value returned by RecommenderService matches the ground-truth value in
   expected_attributes.

   ACR_per_test = # correct (car, attribute) pairs / # total (car, attribute) pairs
                  across the cars that WERE successfully retrieved.
   ACR_global   = mean(ACR_per_test)

   A special "hallucination" flag is raised when the comparison includes a car
   that is not in the dataset at all (i.e. the NLU invented a brand name).

Usage:
    python evaluate_dialogues.py \
        --test-file test_comparisons.json \
        --model-name qwen3 \
        --device cuda:0
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main_CarAdvisor import nlu as NLU_SYSTEM_PROMPT

import torch
from transformers import AutoTokenizer

from utils import MODELS, NLU, RecommenderService, DialogueManager
from Dataset import cars, RULES


COMPARISON_ATTRIBUTES = [
    "car_price", "car_type", "fuel_type", "car_usecase",
    "car_state", "car_dimensions", "fuel_efficiency", "car_design",
]


def normalise(v):
    if v is None:
        return None
    return str(v).lower().strip()


def values_match(attr, predicted, expected) -> bool:
    """True if predicted attribute value matches expected."""
    if predicted is None and expected is None:
        return True
    if predicted is None or expected is None:
        return False

    if attr == "car_price":
        try:
            return int(predicted) == int(expected)
        except (ValueError, TypeError):
            return False

    return normalise(str(predicted)) == normalise(str(expected))


def load_test_cases(path: str):
    with open(path) as f:
        return json.load(f)


def run_evaluation(args):
    test_cases = load_test_cases(args.test_file)

    reco_engine = RecommenderService(cars)

    # Optional: load NLU model to test full pipeline (NLU → retrieval)
    use_nlu = not args.skip_nlu
    use_dm = not args.skip_dm
    nlu_engine = None
    dm_engine = None
    if use_nlu:
        model_name, InitModel, prepare_text = MODELS[args.model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = InitModel(model_name, dtype="auto")
        model.eval()
        nlu_engine = NLU(model, tokenizer, prepare_text, NLU_SYSTEM_PROMPT)
        
    if use_dm:
        dm_engine = DialogueManager(RULES)

    retrieval_precisions = []
    retrieval_recalls = []
    retrieval_f1s = []
    acr_scores = []
    hallucination_flags = []
    sample_results = []
    parse_errors = []

    for tc in test_cases:
        test_id = tc["id"]
        user_input = tc["user_input"]
        expected_cars_raw = tc.get("expected_cars", [])
        expected_attrs = tc.get("expected_attributes", {})

        # Normalise expected car names
        expected_set = {normalise(c) for c in expected_cars_raw}

        # ── Step 1: NLU extraction (or use ground truth cars for retrieval-only eval) ──
        if use_nlu and nlu_engine:
            try:
                state = nlu_engine.parse(user_input, [], args.n_exchanges)
                extracted_cars = state.get("slots", {}).get("comparison_cars") or []
            except Exception as e:
                extracted_cars = []
                parse_errors.append({"id": test_id, "error": str(e)})
        else:
            # Skip NLU; use ground-truth car names to isolate retrieval quality
            extracted_cars = expected_cars_raw

        # ── Step 2: Retrieve car data (optionally routed through DM) ────────
        if use_dm and dm_engine:
            dm_state = {
                "intent": "car_comparison",
                "slots": {"comparison_cars": extracted_cars}
            }
            action, _ = dm_engine.decide(dm_state)

            if action == "slot_filling_error":
                retrieved_cars_data = []
                parse_errors.append({
                    "id": test_id,
                    "error": "DM rejected comparison_cars after NLU extraction"
                })
            elif action == "compare_cars":
                validated_slots, _ = dm_engine.validate(dm_state["slots"])
                cars_to_retrieve = validated_slots.get("comparison_cars") or extracted_cars
                retrieved_cars_data = reco_engine.get_cars_by_brand(cars_to_retrieve)
            else:
                retrieved_cars_data = []
                parse_errors.append({
                    "id": test_id,
                    "error": f"Unexpected DM action: {action}"
                })
        else:
            retrieved_cars_data = reco_engine.get_cars_by_brand(extracted_cars)

        retrieved_names_norm = {normalise(car["car_brand"]) for car in retrieved_cars_data}
        extracted_names_norm = {normalise(c) for c in extracted_cars}
        
        # Hallucination: extracted car name not in dataset at all
        hallucinated = extracted_names_norm - {normalise(c["car_brand"]) for c in cars}
        is_hallucinated = len(hallucinated) > 0
        hallucination_flags.append(is_hallucinated)

        # ── Car Retrieval Accuracy ────────────────────────────────────────────
        # Only penalise for cars that actually exist in the dataset
        valid_expected = {e for e in expected_set
                          if any(normalise(c["car_brand"]) == e for c in cars)}

        if valid_expected or retrieved_names_norm:
            tp_ret = len(retrieved_names_norm & valid_expected)
            fp_ret = len(retrieved_names_norm - valid_expected)
            fn_ret = len(valid_expected - retrieved_names_norm)

            prec_ret = tp_ret / (tp_ret + fp_ret) if (tp_ret + fp_ret) > 0 else 0.0
            rec_ret  = tp_ret / (tp_ret + fn_ret) if (tp_ret + fn_ret) > 0 else 0.0
            f1_ret   = (2 * prec_ret * rec_ret / (prec_ret + rec_ret)
                        if (prec_ret + rec_ret) > 0 else 0.0)
        else:
            prec_ret = rec_ret = f1_ret = 1.0  # Nothing to retrieve, nothing retrieved → correct

        retrieval_precisions.append(prec_ret)
        retrieval_recalls.append(rec_ret)
        retrieval_f1s.append(f1_ret)

        # ── Attribute Correctness Rate ────────────────────────────────────────
        correct_attrs = 0
        total_attrs = 0
        attr_details = {}

        for car in retrieved_cars_data:
            brand = car["car_brand"]
            brand_norm = normalise(brand)

            # Find matching expected attributes (case-insensitive brand match)
            exp_brand_key = next(
                (k for k in expected_attrs if normalise(k) == brand_norm), None
            )
            if exp_brand_key is None:
                continue  # Car was retrieved but has no ground truth → skip

            exp = expected_attrs[exp_brand_key]
            attr_details[brand] = {}

            for attr in COMPARISON_ATTRIBUTES:
                pred_val = car.get(attr)
                exp_val = exp.get(attr)

                # Convert attribute types for comparison
                if attr == "car_price":
                    try:
                        pred_val = int(pred_val)
                        exp_val = int(exp_val)
                    except (TypeError, ValueError):
                        pass
                else:
                    pred_val = normalise(pred_val)
                    exp_val = normalise(exp_val)

                match = values_match(attr, pred_val, exp_val)
                correct_attrs += int(match)
                total_attrs += 1
                attr_details[brand][attr] = {
                    "expected": exp_val,
                    "retrieved": pred_val,
                    "correct": match,
                }

        acr = correct_attrs / total_attrs if total_attrs > 0 else None
        if acr is not None:
            acr_scores.append(acr)

        sample_result = {
            "id": test_id,
            "description": tc.get("description", ""),
            "extracted_cars": extracted_cars,
            "retrieved_cars": [c["car_brand"] for c in retrieved_cars_data],
            "hallucinated_cars": list(hallucinated),
            "retrieval_precision": prec_ret,
            "retrieval_recall": rec_ret,
            "retrieval_f1": f1_ret,
            "attribute_correctness_rate": acr,
            "attribute_details": attr_details,
            "note": tc.get("note", ""),
        }
        sample_results.append(sample_result)

        if args.verbose:
            print(f"\n{'─'*60}")
            print(f"[{test_id}] {tc.get('description','')}")
            print(f"  Extracted : {extracted_cars}")
            print(f"  Retrieved : {[c['car_brand'] for c in retrieved_cars_data]}")
            print(f"  Expected  : {expected_cars_raw}")
            if hallucinated:
                print(f"  ⚠ Hallucinated : {list(hallucinated)}")
            print(f"  Retrieval  → Prec={prec_ret:.3f}  Rec={rec_ret:.3f}  F1={f1_ret:.3f}")
            print(f"  ACR        → {acr:.3f}" if acr is not None else "  ACR: N/A")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Global metrics ─────────────────────────────────────────────────────
    n = len(test_cases)
    global_ret_prec = sum(retrieval_precisions) / n
    global_ret_rec  = sum(retrieval_recalls) / n
    global_ret_f1   = sum(retrieval_f1s) / n
    global_acr      = sum(acr_scores) / len(acr_scores) if acr_scores else 0.0
    hallucination_rate = sum(hallucination_flags) / n

    print("\n" + "=" * 60)
    print("COMPARISON EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total test cases          : {n}")
    print(f"NLU used                  : {use_nlu}")
    print(f"Parse errors              : {len(parse_errors)}")
    print()
    print("Car Retrieval Accuracy:")
    print(f"  Mean Precision          : {global_ret_prec:.4f}")
    print(f"  Mean Recall             : {global_ret_rec:.4f}")
    print(f"  Mean F1                 : {global_ret_f1:.4f}")
    print()
    print(f"Attribute Correctness Rate: {global_acr:.4f}")
    print(f"Hallucination Rate        : {hallucination_rate:.4f}  "
          f"({sum(hallucination_flags)}/{n} cases)")

    print("\nPer-test summary:")
    print(f"  {'ID':<15} {'Ret-P':>7} {'Ret-R':>7} {'Ret-F1':>8} {'ACR':>7}  {'Halluc':>7}")
    print("  " + "-" * 60)
    for r in sample_results:
        acr_str = f"{r['attribute_correctness_rate']:.3f}" if r["attribute_correctness_rate"] is not None else "N/A"
        hal_str = "YES" if r["hallucinated_cars"] else "no"
        print(f"  {r['id']:<15} {r['retrieval_precision']:>7.3f} {r['retrieval_recall']:>7.3f} "
              f"{r['retrieval_f1']:>8.3f} {acr_str:>7} {hal_str:>7}")

    results = {
        "global_retrieval_precision": global_ret_prec,
        "global_retrieval_recall": global_ret_rec,
        "global_retrieval_f1": global_ret_f1,
        "global_attribute_correctness_rate": global_acr,
        "global_hallucination_rate": hallucination_rate,
        "parse_errors": parse_errors,
        "per_sample": sample_results,
    }

    out_path = args.output or "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate car comparison pipeline.")
    parser.add_argument("--test-file", default="test_compare.json")
    parser.add_argument("--model-name", choices=MODELS.keys(), default="qwen3")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--n-exchanges", type=int, default=10)
    parser.add_argument(
        "--skip-nlu", action="store_true",
        help="Skip NLU and use ground-truth car names directly (tests retrieval only)"
    )
    parser.add_argument(
        "--skip-dm", action="store_true",
        help="Skip Dialogue Manager and use ground-truth dialogue state directly (tests retrieval only)"
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())