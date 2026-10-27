# DeBERTa Zero-Shot Experiment

## Method

- **Task:** classify English customer messages as **Support (0)** or **Sales (1)** for the KIO routing prototype.
- **Model:** [MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli), via Hugging Face Transformers zero-shot classification (MIT).
- **Training:** no project-specific training or fine-tuning of DeBERTa.
- **Data:** 60 synthetic messages (30 per class), including 20 fixed test messages (10 per class).

## Candidate-Label Experiments

| Configuration | Correct / 20 | Accuracy | Support recall | Sales recall |
| --- | ---: | ---: | ---: | ---: |
| Original labels (`Support`, `Sales`) | 14 | 70% | 90% | 50% |
| Descriptive intent labels | 15 | 75% | 100% | 50% |

The descriptive labels distinguish an existing-service problem from a new sales inquiry. They were chosen **after reviewing the original test results**. Consequently, **75% is exploratory**, not an unbiased estimate from an untouched test set.

The descriptive-label confusion matrix, with rows = actual Support/Sales and columns = predicted Support/Sales, is:

```text
             Predicted Support    Predicted Sales
Support             10                 0
Sales                5                 5
```

Five actual Sales messages were incorrectly sent to Support. See `../results/decision_results.csv` for each test message and prediction.

## Comparison

The traditional 2D TF-IDF → TruncatedSVD → Linear SVM achieved **18/20 (90%)** on the same test messages, compared with DeBERTa's exploratory **15/20 (75%)**. The SVM achieved **90% recall for both classes**.

See `../results/comparison.csv` for matched predictions and `../src/kio/decision_model.ipynb` for the decision-model notebook.

## Limitations

The test set contains only 20 synthetic messages. Results should not be generalized to real customer traffic without larger, independent testing. No claim of production readiness is made.
