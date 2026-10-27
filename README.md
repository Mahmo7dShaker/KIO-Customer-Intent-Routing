# KIO — Customer Intent Routing

**A machine-learning router for a website development agency.**

KIO reads customer messages and selects the right workflow: **Support (0)** for existing-service issues or **Sales (1)** for new project inquiries. When both intents appear, existing-service problems take priority.

**CET 351 · Machine Learning · Project 1**

## At a glance

| | Traditional ML | Decision Model |
|---|---|---|
| Approach | TF-IDF → TruncatedSVD (2D) → Linear SVM | DeBERTa-v3 zero-shot classification |
| Correct predictions | **18/20** | 15/20 |
| Accuracy | **90%** | 75%* |
| Support recall | 90% | 100% |
| Sales recall | 90% | 50% |

*The DeBERTa 75% result is exploratory: candidate-label descriptions were revised after reviewing results on the same test set. These are results on **20 synthetic messages**, not deployment performance.*

## How it works

```text
Customer message
       │
       ▼
Intent classification
       │
       ├── Support (0) ──► Support Agent
       └── Sales   (1) ──► Sales Agent
```

The agents are **print-only simulations**. This project evaluates routing, **not** a full chatbot or RAG application.

## Dataset and experiment

- **60** synthetic English messages from a website-development support/sales setting, balanced **30/30**.
- **40** training messages and **20** held-out test messages, balanced within each split.
- Four case types: straightforward (23), paraphrase (16), negation (14), mixed (7).
- Related paraphrases share a `group_id` and never cross the train/test split.
- Both approaches are compared on identical test message IDs and ground-truth labels.
- The Linear SVM was **trained on the actual 2D SVD features**; `results/boundary.png` is its genuine decision boundary.

## Project files

```text
data/dataset.csv                  # Reviewed messages, labels, and fixed split
src/kio/
  traditional_ml.ipynb            # TF-IDF, SVD, SVM, evaluation
  decision_model.ipynb            # DeBERTa zero-shot experiment
  router.py                       # Model-independent agent routing
  contracts.py                    # Prediction interface and label mapping
models/traditional_ml.joblib      # Saved traditional classifier pipeline
results/                          # Both models' predictions, comparison, boundary plot
presentation/KIO_Customer_Intent_Routing.pptx
docs/                             # Evaluation protocol, interfaces, experiment notes
tests/                            # Router tests
```

## Run locally

Use **Python 3.11–3.13** and open the notebooks in VS Code or Jupyter.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Run `src/kio/traditional_ml.ipynb` for traditional training and evaluation. The DeBERTa notebook requires downloading model weights and a working PyTorch/Transformers environment; open `src/kio/decision_model.ipynb` to inspect its setup and exploratory evaluation.

**Reproducibility note:** Some notebook cells may contain machine-specific paths or expect to run from a particular working directory. Adjust paths for your environment. The committed CSV outputs document the reported comparison.

## Presentation

[**View the complete PowerPoint deck**](presentation/KIO_Customer_Intent_Routing.pptx) · 21 editable slides with method diagrams, project code, the actual decision boundary, evaluation, limitations, and references.

## Team

- **Mahmoud Wajih Shaker**
- **Abdelrahman Ahmed**
- **Abdelrahman Tamer Ali**

## References

- CET 351 — *Project 1: Routing for Agent KIO* and course lecture materials.
- [DeBERTa-v3 MNLI/FEVER/ANLI model card](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) (MIT).
- [scikit-learn documentation](https://scikit-learn.org/stable/) — `TfidfVectorizer`, `TruncatedSVD`, `SVC`.

**Research/education prototype — not validated for production deployment.**
