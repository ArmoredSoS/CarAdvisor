"""
evaluate_recommendations.py

Evaluates the RecommenderService recommendation quality.

Metrics:
- Constraint Satisfaction Rate (CSR):
    For each recommended car, what fraction of the user's hard constraints does it satisfy?
    A constraint is "satisfied" if the car's attribute matches the slot value.
    Price is satisfied when car_price <= car_price slot value.

    CSR_per_test = mean(constraint_satisfaction per recommended car)
    CSR_global   = mean(CSR_per_test) across all test cases

- Hallucination Rate:
    Fraction of recommended cars that are NOT in the ground-truth valid_answers list
    for that test case.

    HR_per_test  = # recommended cars not in valid_answers / # recommended cars
    HR_global    = mean(HR_per_test) across all test cases

    NOTE: A car can be "non-hallucinated" (it exists in the dataset) but still not in
    valid_answers (it doesn't satisfy all constraints). The hallucination rate here
    measures recommendation relevance against the curated ground truth, not dataset membership.

- Hit Rate @ k:
    Fraction of test cases where AT LEAST ONE of the top-k recommendations is in valid_answers.

- Precision @ k:
    Mean fraction of the k recommendations that are in valid_answers.

This script evaluates the RecommenderService DIRECTLY (no LLM required for the recommender).
It runs the scoring/ranking engine from utils.py against the test dataset from Dataset.py.

Usage:
    python evaluate_recommendations.py \
        --test-file test_recommendations.json \
        --top-k 3
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Dataset import cars
from utils import RecommenderService

# Attributes compared as hard constraints (price uses <= comparison)
CONSTRAINT_SLOTS = [
    "car_type", "car_price", "car_state", "car_usecase",
    "fuel_type", "car_dimensions", "fuel_efficiency", "car_design",
]


def satisfies_constraint(car: dict, slot: str, value) -> bool:
    """Check whether a single car satisfies one constraint slot."""
    if value is None or value == "null" or value == "":
        return True  # Not constrained; always satisfied

    car_value = car.get(slot)
    if car_value is None:
        return False

    if slot == "car_price":
        try:
            return int(car_value) <= int(float(str(value).replace(",", "")))
        except (ValueError, TypeError):
            return False

    return str(car_value).lower().strip() == str(value).lower().strip()


def constraint_satisfaction(car: dict, constraints: dict) -> float:
    """Return fraction of active (non-null) constraints satisfied by a car."""
    active = {k: v for k, v in constraints.items()
              if v is not None and v != "null" and v != ""}
    if not active:
        return 1.0
    satisfied = sum(satisfies_constraint(car, slot, val) for slot, val in active.items())
    return satisfied / len(active)


def load_test_cases(path: str):
    with open(path) as f:
        return json.load(f)


def run_evaluation(args):
    test_cases = load_test_cases(args.test_file)
    reco_engine = RecommenderService(cars)
    k = args.top_k

    per_test_csr = []
    per_test_hr = []
    per_test_precision = []
    hit_at_k = []
    sample_results = []

    print(f"\nRunning recommendation evaluation (top-{k})...")

    for tc in test_cases:
        test_id = tc["id"]
        slots = tc["slots"]
        constraints = tc.get("constraints", slots)  # fallback to full slots
        valid_set = {v.lower().strip() for v in tc.get("valid_answers", [])}

        # Get top-k recommendations
        recs = reco_engine.recommend(slots, top_k=k)
        rec_cars = [car for car, _ in recs]
        rec_names = [car["car_brand"] for car in rec_cars]

        # ── Constraint Satisfaction Rate ─────────────────────────────────────
        if rec_cars:
            cs_scores = [constraint_satisfaction(car, constraints) for car in rec_cars]
            mean_csr = sum(cs_scores) / len(cs_scores)
        else:
            cs_scores = []
            mean_csr = 0.0
        per_test_csr.append(mean_csr)

        # ── Hallucination Rate ────────────────────────────────────────────────
        if valid_set and rec_names:
            not_in_valid = [n for n in rec_names if n.lower().strip() not in valid_set]
            hr = len(not_in_valid) / len(rec_names)
        elif not valid_set:
            hr = 0.0  # No ground truth defined; cannot penalise
            not_in_valid = []
        else:
            hr = 1.0  # Recommendations exist but valid set is empty
            not_in_valid = rec_names
        per_test_hr.append(hr)

        # ── Hit Rate @ k ──────────────────────────────────────────────────────
        hit = any(n.lower().strip() in valid_set for n in rec_names) if valid_set else None
        if hit is not None:
            hit_at_k.append(1 if hit else 0)

        # ── Precision @ k ─────────────────────────────────────────────────────
        if valid_set and rec_names:
            prec = sum(1 for n in rec_names if n.lower().strip() in valid_set) / len(rec_names)
            per_test_precision.append(prec)

        sample_results.append({
            "id": test_id,
            "description": tc.get("description", ""),
            "recommendations": rec_names,
            "csr": mean_csr,
            "hallucination_rate": hr,
            "precision_at_k": prec if valid_set and rec_names else None,
            "hit_at_k": bool(hit) if hit is not None else None,
            "not_in_valid": not_in_valid,
            "per_car_csr": {name: score for name, score in zip(rec_names, cs_scores)},
        })

        if args.verbose:
            hit_str = "✓" if hit else "✗"
            print(f"\n[{hit_str}] {test_id}: {tc.get('description','')}")
            print(f"     Recommendations: {rec_names}")
            print(f"     Valid answers  : {list(valid_set)}")
            print(f"     CSR={mean_csr:.3f}  HR={hr:.3f}  Prec@{k}={prec if valid_set and rec_names else 'N/A'}")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    global_csr  = sum(per_test_csr) / len(per_test_csr) if per_test_csr else 0.0
    global_hr   = sum(per_test_hr)  / len(per_test_hr)  if per_test_hr  else 0.0
    global_prec = sum(per_test_precision) / len(per_test_precision) if per_test_precision else 0.0
    global_hit  = sum(hit_at_k) / len(hit_at_k) if hit_at_k else 0.0

    print("\n" + "=" * 60)
    print("RECOMMENDATION EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total test cases : {len(test_cases)}")
    print(f"Top-k            : {k}")
    print()
    print(f"Constraint Satisfaction Rate (CSR) : {global_csr:.4f}")
    print(f"Hallucination Rate (HR)            : {global_hr:.4f}")
    print(f"Precision @ {k}                      : {global_prec:.4f}")
    print(f"Hit Rate @ {k}                        : {global_hit:.4f}")

    print("\nPer-test summary:")
    print(f"  {'ID':<15} {'CSR':>8} {'HR':>8} {'Prec@k':>8} {'Hit@k':>8}  Recommendations")
    print("  " + "-" * 80)
    for r in sample_results:
        hit_str = "Y" if r["hit_at_k"] else "N"
        prec_str = f"{r['precision_at_k']:.3f}" if r["precision_at_k"] is not None else "N/A"
        print(f"  {r['id']:<15} {r['csr']:>8.3f} {r['hallucination_rate']:>8.3f} "
              f"{prec_str:>8} {hit_str:>8}  {r['recommendations']}")

    # ── Save results ───────────────────────────────────────────────────────────
    results = {
        "top_k": k,
        "global_csr": global_csr,
        "global_hallucination_rate": global_hr,
        "global_precision_at_k": global_prec,
        "global_hit_rate_at_k": global_hit,
        "per_sample": sample_results,
    }

    out_path = args.output or "recommendation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate car recommendations.")
    parser.add_argument("--test-file", default="test_recom.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())