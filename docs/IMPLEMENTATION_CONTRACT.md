# Shared Implementation Contract

This document describes the dataset and routing interfaces used by the project.

## Dataset format
One CSV with columns `id,message,label,intent_type,case_type,group_id,split`.
- `id`: unique stable message ID (e.g., M01).
- `message`: nonempty synthetic English customer message.
- `label`: integer 0 (existing customer support) or 1 (new sales inquiry).
- `intent_type`: support or sales, consistent with label.
- `case_type`: descriptive tag including paraphrase, negation, mixed, or straightforward; tags can be extended.
- `group_id`: group near-paraphrases together to avoid leakage.
- `split`: exactly train or test, frozen prior to testing.

### Labeling precedence
1. Existing service problem, complaint, refund or fix request => **support (0)**.
2. New purchase, package or quotation without an existing-service problem => **sales (1)**.
3. Mixed complaint and purchase intent => **support (0)**.
4. Use meaning, not single keyword matches, especially for negation.
5. Review ambiguous cases together before freezing labels.

### Split invariants
- Exactly 60 rows, 30 rows per class; 40 train (20/20), 20 test (10/10).
- Near-duplicate `group_id` values must not cross train/test.
- Fit the traditional classifier and feature transformations using training messages only. DeBERTa's revised label descriptions were evaluated on the same test set and are reported as exploratory.
- The same exact 20 test IDs and gold labels are evaluated by both models.

## Function interfaces
- `predict_ml(message: str) -> int` where output is only 0 or 1.
- A decision-model adapter must validate and normalize its predicted label into integer 0 or 1.
- `route_message(message: str, predict: Callable[[str], int]) -> int`: invokes simulated agent and returns class.
- Invalid/non-conforming predictions must raise an error rather than silently assigning a class.

## Results file conventions
The `results/` directory contains `decision_results.csv`, `ml_results.csv`, and `comparison.csv`. The two model-specific exports use different column names; the comparison joins them by message `id`. Treat the DeBERTa 75% result as exploratory due to test-informed candidate-label revision.

## Boundary visualization integrity
A true 2D boundary must come from a model trained/evaluated on its two plotted numeric features. If using high-dimensional TF-IDF, clearly label a 2D projection as *illustrative* rather than the actual classifier boundary.

