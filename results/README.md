# Experiment Results

- `ml_results.csv` — Traditional Linear SVM predictions on 20 test messages (**18/20 correct**).
- `decision_results.csv` — DeBERTa zero-shot exploratory predictions (**15/20 correct**).
- `comparison.csv` — Matched predictions and error analysis by message ID.
- `boundary.png` — Actual 2D decision boundary of the fitted Linear SVM.

DeBERTa's 75% is **exploratory** because candidate-label wording was revised after viewing earlier test-set results. Do not interpret either score as real-world deployment performance.
