# Diabetes Prediction — ML Model Comparison

## Overview
This project applies and compares five machine learning classification
algorithms to predict diabetes onset using the **Pima Indians Diabetes
Dataset**. It was built to satisfy the course requirements: dataset
description, preprocessing, EDA, ≥4 algorithms, one comparative study,
and full evaluation with tables/figures.

**Research question:** Which machine learning algorithm — Logistic
Regression, k-NN, Decision Tree, Naive Bayes, or SVM — most accurately
predicts diabetes onset from diagnostic measurements, and how does SVM
kernel choice affect performance?

## Dataset
- **Source:** UCI Machine Learning Repository / National Institute of
  Diabetes and Digestive and Kidney Diseases. Downloaded via the
  `jbrownlee/Datasets` GitHub mirror:
  https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv
  (Also available on Kaggle: "Pima Indians Diabetes Database".)
- **Instances:** 768 (no duplicates)
- **Features:** 8 numeric clinical measurements (Pregnancies, Glucose,
  BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
- **Target:** Outcome (0 = no diabetes, 1 = diabetes) — 65% / 35% split
- **Missing values:** No literal NaNs, but Glucose, BloodPressure,
  SkinThickness, Insulin, and BMI contain biologically impossible zeros
  (5, 35, 227, 374, and 11 respectively) that represent disguised missing
  data — handled explicitly in preprocessing.
- **Data types:** All numeric (int64/float64); no categorical encoding needed.

## Files
```
diabetes_raw.csv                     Dataset with proper column headers
add_header.py                        Adds headers to the raw CSV download

01_eda.py                            Summary stats, missingness, distributions,
                                      correlation heatmap, target balance
02_preprocessing.py                  Missing-value handling, duplicate check,
                                      stratified 80/20 split, median imputation,
                                      StandardScaler (fit on train only)
03_models.py                         Trains/evaluates 5 algorithms with
                                      5-fold stratified CV + held-out test set
04_comparative_study_svm_kernels.py  Comparative Study B: SVM kernel comparison
                                      (linear, poly, rbf, sigmoid)

figures/                             All generated plots (PNG)
summary_statistics.csv               Descriptive statistics table
zero_value_counts.csv                Disguised missing-value counts
model_comparison_results.csv         Final 5-model comparison table
svm_kernel_comparison.csv            SVM kernel comparative study table
confusion_matrices.json              Raw confusion matrix values per model
X_train_scaled.csv / X_test_scaled.csv / y_train.csv / y_test.csv
                                      Processed train/test splits (reproducible)
```

## How to reproduce
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python3 add_header.py
python3 01_eda.py
python3 02_preprocessing.py
python3 03_models.py
python3 04_comparative_study_svm_kernels.py
```
Random seed fixed at 42 throughout for reproducibility.

## Headline Results (Test Set)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Decision Tree | 0.760 | 0.639 | 0.722 | 0.678 |
| k-NN | 0.753 | 0.660 | 0.611 | 0.635 |
| SVM (RBF) | 0.740 | 0.652 | 0.556 | 0.600 |
| Naive Bayes | 0.701 | 0.567 | 0.630 | 0.597 |
| Logistic Regression | 0.708 | 0.600 | 0.500 | 0.545 |

**Comparative study (SVM kernels):** Linear kernel achieved the highest
mean 5-fold CV accuracy (0.788) but the RBF kernel generalised best to
the held-out test set (F1 = 0.600), suggesting linear SVM may be
slightly overfit to the CV folds while RBF captures non-linear decision
boundaries more robustly. Polynomial and sigmoid kernels underperformed.

Full discussion, critical analysis, and literature synthesis should be
written up by the student in the final report (see paper skeleton).

## AI Usage Statement (template — edit to reflect your actual usage)
AI assistance (Claude, Anthropic) was used for: (1) generating and
debugging the Python preprocessing/modelling pipeline, (2) producing an
initial literature-search reading list, and (3) structuring this README
and the IEEE paper skeleton. All analysis, interpretation, discussion,
and final written content in the research paper were produced by the
student. No AI-generated prose was used verbatim in the Results,
Discussion, or Conclusion sections.

## Citation
Dataset: Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., &
Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the
onset of diabetes mellitus. In Proceedings of the Symposium on Computer
Applications and Medical Care (pp. 261–265). IEEE Computer Society Press.
