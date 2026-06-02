# Evaluation Results

## 1. Intent Classification Evaluation

| Metric | Value |
|--------|------|
| Total samples | 27 |
| Parse errors | 0 |
| Accuracy | 1.0000 |
| Macro F1 | 1.0000 |
| Weighted F1 | 1.0000 |

### Per-class report

| Class | Precision | Recall | F1 | Support |
|------|-----------|--------|----|---------|
| car_comparison | 1.00 | 1.00 | 1.00 | 6 |
| car_info | 1.00 | 1.00 | 1.00 | 6 |
| car_search | 1.00 | 1.00 | 1.00 | 8 |
| end_conversation | 1.00 | 1.00 | 1.00 | 2 |
| ignore_message | 1.00 | 1.00 | 1.00 | 5 |

============================================================

## 2. Comparison Evaluation (NLU only)

| Metric | Value |
|--------|------|
| Total cases | 10 |
| Parse errors | 0 |
| Mean Precision | 1.0000 |
| Mean Recall | 0.9500 |
| Mean F1 | 0.9667 |
| Attribute Correctness | 1.0000 |
| Hallucination Rate | 0.3000 |

### Per-test summary

| ID | Ret-P | Ret-R | Ret-F1 | ACR | Halluc |
|----|------|------|--------|-----|--------|
| comp_001 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_002 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_003 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_004 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_005 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_006 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_007 | 1.000 | 0.500 | 0.667 | 1.000 | YES |
| comp_008 | 1.000 | 1.000 | 1.000 | N/A | YES |
| comp_009 | 1.000 | 1.000 | 1.000 | 1.000 | YES |
| comp_010 | 1.000 | 1.000 | 1.000 | 1.000 | no |

============================================================

## 3. Comparison Evaluation (DM only)

| Metric | Value |
|--------|------|
| Total cases | 10 |
| Parse errors | 2 |
| Mean Precision | 0.9000 |
| Mean Recall | 0.9000 |
| Mean F1 | 0.9000 |
| Attribute Correctness | 1.0000 |
| Hallucination Rate | 0.0000 |

### Per-test summary

| ID | Ret-P | Ret-R | Ret-F1 | ACR | Halluc |
|----|------|------|--------|-----|--------|
| comp_001 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_002 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_003 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_004 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_005 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_006 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_007 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_008 | 1.000 | 1.000 | 1.000 | N/A | no |
| comp_009 | 0.000 | 0.000 | 0.000 | N/A | no |
| comp_010 | 1.000 | 1.000 | 1.000 | 1.000 | no |

============================================================

## 4. Comparison Evaluation (NLU + DM)

| Metric | Value |
|--------|------|
| Total cases | 10 |
| Parse errors | 2 |
| Mean Precision | 0.9000 |
| Mean Recall | 0.9000 |
| Mean F1 | 0.9000 |
| Attribute Correctness | 1.0000 |
| Hallucination Rate | 0.2000 |

### Per-test summary

| ID | Ret-P | Ret-R | Ret-F1 | ACR | Halluc |
|----|------|------|--------|-----|--------|
| comp_001 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_002 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_003 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_004 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_005 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_006 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_007 | 1.000 | 1.000 | 1.000 | 1.000 | no |
| comp_008 | 1.000 | 1.000 | 1.000 | N/A | YES |
| comp_009 | 0.000 | 0.000 | 0.000 | N/A | YES |
| comp_010 | 1.000 | 1.000 | 1.000 | 1.000 | no |

============================================================

## 5. Slot Filling Evaluation

| Slot | TP | FP | FN | TN | Precision | Recall | F1 |
|------|----|----|----|----|-----------|--------|----|
| car_type | 10 | 2 | 0 | 3 | 0.8333 | 1.0000 | 0.9091 |
| car_price | 6 | 1 | 1 | 7 | 0.8571 | 0.8571 | 0.8571 |
| car_state | 6 | 0 | 1 | 8 | 1.0000 | 0.8571 | 0.9231 |
| car_usecase | 9 | 0 | 2 | 4 | 1.0000 | 0.8182 | 0.9000 |
| fuel_type | 8 | 0 | 0 | 7 | 1.0000 | 1.0000 | 1.0000 |
| car_brand | 2 | 0 | 0 | 13 | 1.0000 | 1.0000 | 1.0000 |
| car_dimensions | 5 | 0 | 0 | 10 | 1.0000 | 1.0000 | 1.0000 |
| fuel_efficiency | 3 | 1 | 0 | 11 | 0.7500 | 1.0000 | 0.8571 |
| car_design | 3 | 0 | 0 | 12 | 1.0000 | 1.0000 | 1.0000 |
| MICRO AVG | 52 | 4 | 4 | - | 0.9286 | 0.9286 | 0.9286 |

============================================================

## 6. Recommendation Evaluation

| Metric | Value |
|--------|------|
| Total test cases | 10 |
| Top-k | 3 |
| CSR | 0.8587 |
| Hallucination Rate | 0.3667 |
| Precision@3 | 0.6333 |
| Hit@3 | 1.0000 |

### Per-test summary

| ID | CSR | HR | Prec@k | Hit@k | Recommendations |
|----|-----|----|--------|-------|----------------|
| reco_001 | 0.750 | 0.333 | 0.667 | Y | Fiat Panda, Renault Clio, Peugeot 208 |
| reco_002 | 0.778 | 0.333 | 0.667 | Y | Volvo XC60, Audi Q5, Alfa Romeo Tonale |
| reco_003 | 0.750 | 0.333 | 0.667 | Y | BMW 3 Series, Mercedes-Benz C-Class, Audi A4 |
| reco_004 | 1.000 | 0.333 | 0.667 | Y | Land Rover Discovery, Volkswagen Touareg, Land Rover Defender |
| reco_005 | 0.944 | 0.667 | 0.333 | Y | Peugeot 208, Volkswagen ID.3, BMW i3 |
| reco_006 | 1.000 | 0.333 | 0.667 | Y | Ferrari 488, Lamborghini Huracan, Ferrari Roma |
| reco_007 | 0.810 | 0.667 | 0.333 | Y | Audi e-tron, BMW iX, Mercedes-Benz EQC |
| reco_008 | 0.833 | 0.000 | 1.000 | Y | Renault Megane, Peugeot 308, Citroën C4 |
| reco_009 | 0.833 | 0.000 | 1.000 | Y | Fiat Panda, Renault Clio, Citroën C3 |
| reco_010 | 0.889 | 0.667 | 0.333 | Y | BMW 5 Series, Audi A4, Volvo S60 |

============================================================

## 7. Dialogue Efficiency Evaluation

| Metric | Value |
|--------|------|
| Total dialogues | 12 |
| Parse errors | 0 |
| Task completion rate | 0.75 |
| Mean turns to completion | 2.00 |
| Mean turn efficiency | 2.3889 |
| Redundant question rate | 0.0000 |
| Premature recommendation rate | 0.0000 |
| Mean action accuracy | 0.75 |

### Per-dialogue summary

| ID | Done | Turns | MaxT | Eff | Redund | PreRec | ActAcc |
|----|------|------|------|-----|--------|--------|--------|
| dial_001 | ✓ | 1 | 2 | 4.00 | 0.000 | no | 1.000 |
| dial_002 | ✓ | 3 | 5 | 1.33 | 0.000 | no | 1.000 |
| dial_003 | ✓ | 1 | 1 | 1.00 | 0.000 | no | 1.000 |
| dial_004 | ✗ | 3 | 5 | 1.33 | 0.000 | no | 0.000 |
| dial_005 | ✓ | 1 | 1 | 1.00 | 0.000 | no | 1.000 |
| dial_006 | ✗ | 2 | 5 | 2.00 | 0.000 | no | 0.500 |
| dial_007 | ✓ | 3 | 5 | 2.00 | 0.000 | no | 1.000 |
| dial_008 | ✓ | 2 | 3 | 2.00 | 0.000 | no | 0.500 |
| dial_009 | ✓ | 1 | 3 | 2.00 | 0.000 | no | 1.000 |
| dial_010 | ✓ | 1 | 2 | 8.00 | 0.000 | no | 1.000 |
| dial_011 | ✓ | 4 | 6 | 1.00 | 0.000 | no | 1.000 |
| dial_012 | ✗ | 2 | 3 | 3.00 | 0.000 | no | 0.000 |
