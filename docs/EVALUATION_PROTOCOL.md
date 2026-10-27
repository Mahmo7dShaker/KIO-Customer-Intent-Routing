# Evaluation Protocol

1. Freeze and record the labeling rules and exact 60-row dataset.
2. Validate class balance and group-safe 40/20 split. No near duplicate in both sets.
3. Train the traditional classifier using only the training subset; fit text/feature transformations on training only.
4. Document the DeBERTa zero-shot setup and label descriptions. The revised descriptions were tested after observing earlier test results, so their 75% accuracy is exploratory.
5. Predict both models on the identical 20 test IDs; retain raw/invalid outputs and run-time errors, do not hide failures.
6. Report class-specific precision/recall/F1 (support=0, sales=1) and overall accuracy, plus both confusion matrices and real errors.
7. Explain misclassified negation and mixed intent messages and contextual limitations from the 60-message synthetic sample.
8. Check demo routing for both flags and invalid model output; save screenshots/results for offline presentation.
9. Keep exact experimental results in `results/` and cite datasets, software libraries, model license and any sourced claims in slides.
