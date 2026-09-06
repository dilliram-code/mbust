# A Comparative Study of Machine Learning Regression Algorithms for Predicting AI and Data-Industry Salaries

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Project-Academic%20Research-success)

> **Master of Applied Sciences in Artificial Intelligence (MAI) ---
> Machine Learning Final Project**\
> **Madan Bhandari University of Science and Technology (MBUST)**\
> **Author:** Dilli Ram Chaudhary\
> **Roll No.:** 026/MAI/03\
> **Date:** September 6, 2026

------------------------------------------------------------------------

## 📌 Project Overview

This project presents a comparative machine learning study for
predicting annual salaries of professionals working in **Artificial
Intelligence, Machine Learning, and Data-related roles**.

The study compares five supervised regression algorithms:

1.  Linear Regression
2.  Ridge Regression
3.  Decision Tree Regressor
4.  Support Vector Regression (SVR)
5.  Multi-Layer Perceptron (MLP) Regressor

The project goes beyond simply comparing model accuracy. It
investigates:

-   How different regression algorithms perform on the same
    salary-prediction problem
-   Whether feature scaling affects different model families
-   Which job-related feature groups contribute most to salary
    prediction
-   How hyperparameter tuning changes model performance
-   Whether experience level alone is sufficient for salary prediction
-   The trade-off between predictive performance, computational
    practicality, and model simplicity

The complete experimental workflow is implemented in the Jupyter
Notebook:

👉 **[Open the complete ML Salary Prediction Comparative Study
notebook](https://github.com/dilliram-code/mbust/blob/main/ML_final_project/ML_Salary_Prediction_Comparative_Study.ipynb)**

------------------------------------------------------------------------

## 🎯 Research Question

> **Which machine learning regression algorithm most accurately predicts
> the annual USD salary of an AI/data professional from job- and
> employment-related attributes, and does feature scaling improve the
> accuracy of scale-sensitive algorithms such as SVR and MLP relative to
> a scale-invariant algorithm such as Decision Tree?**

------------------------------------------------------------------------

## 🔬 Objectives

The main objectives of this study are to:

-   Build a salary prediction pipeline using real-world
    AI/ML/data-industry salary data.
-   Compare five regression algorithms under a consistent experimental
    setup.
-   Evaluate the impact of feature scaling on model performance.
-   Tune important hyperparameters using 5-fold cross-validation.
-   Perform feature-subset ablation to understand the marginal value of
    feature groups.
-   Evaluate models using MAE, MSE, RMSE, and R².
-   Identify a practical baseline model for salary prediction from
    tabular job attributes.
-   Analyze the limitations of salary prediction when important
    real-world variables are unavailable.

------------------------------------------------------------------------

## 📊 Dataset

The study uses the **AI/ML/Big Data salary survey dataset from
ai-jobs.net**, derived from a publicly available weekly-updated salary
survey and distributed under a CC0/public-domain license.

### Dataset characteristics

  Property                                  Value
  --------------------------- -------------------
  Total records                            71,913
  Candidate raw features                       16
  Target                          `salary_in_usd`
  Duplicate rows                                0
  Missing values                           40,540
  Major missing field           `isco_group_hint`
  `job_title` levels                          422
  `company_location` levels                    97
  `role_family` levels                         11

The target variable, `salary_in_usd`, is right-skewed:

-   **Mean:** approximately \$151,161
-   **Median:** approximately \$138,750
-   **Standard deviation:** approximately \$77,330
-   **Minimum:** \$15,000
-   **Maximum:** \$800,000
-   **Skewness:** 1.48

------------------------------------------------------------------------

## 🧹 Data Preparation

Several fields were removed before modeling to avoid **target leakage,
redundancy, or excessive dimensionality**.

### Removed features

  -----------------------------------------------------------------------
  Feature                             Reason
  ----------------------------------- -----------------------------------
  `salary`                            Used to derive the target salary

  `salary_currency`                   Directly related to target
                                      construction

  `salary_outlier_flag`               Derived from the target and would
                                      leak information

  `experience_level_label`            Duplicate representation

  `employment_type_label`             Duplicate representation

  `work_mode`                         Redundant with `remote_ratio`

  `job_title`                         High-cardinality/noisier
                                      representation compared with
                                      `role_family`

  `isco_group_hint`                   Approximately 56% missing
  -----------------------------------------------------------------------

### Final modeling features

The final feature set consists of:

-   `experience_level`
-   `employment_type`
-   `remote_ratio`
-   `company_size`
-   `role_family`
-   `company_location`

Rare company locations were grouped into an **`Other`** category to keep
the one-hot encoded feature space manageable.

------------------------------------------------------------------------

## ⚙️ Preprocessing Pipeline

The preprocessing workflow includes:

``` text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ├── Remove redundant/leaky features
     ├── Check duplicates
     └── Handle missing information
     │
     ▼
Feature Selection
     │
     ▼
Categorical Encoding
     │
     └── One-Hot Encoding
     │
     ▼
28 Engineered Features
     │
     ├───────────────┐
     ▼               ▼
Unscaled         Standardized
Features         Numeric Features
     │               │
     └───────┬───────┘
             ▼
        Train / Test Split
          80% / 20%
             │
             ▼
      Regression Models
             │
             ▼
       Model Evaluation
```

### Train/Test Split

-   Training set: **80%**
-   Test set: **20%**
-   Random seed: **42**

### Cross-validation

All models were additionally evaluated using **5-fold cross-validation**
on the training data.

### Computational subsampling

Because Support Vector Regression becomes computationally expensive on
large datasets, a **stratified sample of 10,000 records** was used for
the modeling experiments.

The sampling was stratified by `role_family` to preserve the
distribution of role categories.

This is an important methodological limitation: the full dataset
contains **71,913 records**, while the modeling experiments use **10,000
records**.

------------------------------------------------------------------------

## 🤖 Machine Learning Models

### 1. Linear Regression

Used as the ordinary least-squares baseline.

``` text
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

### 2. Ridge Regression

A regularized linear regression model using L2 regularization.

Ridge was particularly useful because one-hot encoding creates
correlated categorical representations and a relatively high-dimensional
feature space.

### 3. Decision Tree Regressor

A non-linear tree-based model capable of learning interactions and
non-linear relationships between features.

### 4. Support Vector Regression

An SVR model using an **RBF kernel** was evaluated to investigate a
kernel-based regression approach.

### 5. Multi-Layer Perceptron Regressor

A neural-network regression model with:

-   Hidden layer 1: 64 units
-   Hidden layer 2: 32 units
-   Early stopping

------------------------------------------------------------------------

## 📏 Evaluation Metrics

The models were evaluated using four regression metrics.

### Mean Absolute Error --- MAE

Measures the average absolute difference between predicted and actual
salaries.

``` text
MAE = average(|y - ŷ|)
```

Lower is better.

### Mean Squared Error --- MSE

Penalizes larger prediction errors more heavily.

``` text
MSE = average((y - ŷ)²)
```

Lower is better.

### Root Mean Squared Error --- RMSE

The square root of MSE, expressed in the same unit as salary.

``` text
RMSE = √MSE
```

Lower is better.

### R² --- Coefficient of Determination

Measures the proportion of target variance explained by the model.

``` text
R² = 1 - SS_res / SS_tot
```

Higher is better.

------------------------------------------------------------------------

## 🧪 Comparative Study: Feature Scaling

One of the main experiments investigates whether standardizing numeric
features changes model performance.

Each model was trained under two conditions:

1.  **Unscaled:** numeric features kept in their original units
2.  **Scaled:** numeric features standardized using `StandardScaler`

The categorical one-hot encoded variables remained unchanged because
they were already represented as binary 0/1 values.

This isolates the effect of scaling.

------------------------------------------------------------------------

## 📈 Main Results

### Test-set performance

  Model                    MAE ($) | RMSE ($)           R² 
  ---------------------- -------------------- ------------ -----------
  **Ridge Regression**             **50,993**   **72,003**   **0.220**
  Linear Regression                    51,011       72,032       0.219
  Decision Tree                        51,197       72,161       0.216
  MLP Regressor                        51,309       72,232       0.215
  SVR (RBF)                            58,644       82,543      -0.026

### Key observation

**Ridge Regression achieved the best overall test performance**, with:

-   **MAE:** \$50,993
-   **RMSE:** \$72,003
-   **R²:** 0.220

The difference between Ridge, Linear Regression, Decision Tree, and MLP
is relatively small, suggesting that the available features place a
strong limit on how much salary variation can be explained.

------------------------------------------------------------------------

## 🔍 Cross-Validation Results

  Model                      Mean R²     Std. R²   Mean MAE (\$)
  ---------------------- ----------- ----------- ---------------
  Linear Regression            0.214       0.021          50,114
  **Ridge Regression**     **0.215**   **0.020**      **50,105**
  Decision Tree                0.196       0.019          50,552
  SVR (RBF)                   -0.021       0.006          57,171
  MLP Regressor                0.208       0.018          50,443

Ridge and Linear Regression produced the strongest and most stable
cross-validation results.

------------------------------------------------------------------------

## 📐 Effect of Feature Scaling

The scaling experiment produced one of the most important findings of
the study.

  Model                 Unscaled R²   Scaled R²
  ------------------- ------------- -----------
  Linear Regression           0.219       0.219
  Ridge Regression            0.220       0.220
  Decision Tree               0.216       0.216
  SVR (RBF)                  -0.035      -0.026
  **MLP Regressor**       **0.007**   **0.215**

### Main finding

Feature scaling had **dramatically different effects depending on the
algorithm**.

The MLP improved from:

``` text
R² = 0.007
```

to:

``` text
R² = 0.215
```

after scaling.

Its MAE also decreased from approximately:

``` text
$59,818 → $51,309
```

This demonstrates that feature scaling is particularly important for
gradient-based neural-network models.

In contrast:

-   Linear Regression was essentially unchanged.
-   Ridge Regression was essentially unchanged.
-   Decision Tree was unchanged.
-   SVR improved only slightly.

------------------------------------------------------------------------

## 🎛️ Hyperparameter Tuning

Grid search with 5-fold cross-validation was used for Ridge, Decision
Tree, and SVR.

### Best parameters

  Model           Selected parameters
  --------------- ---------------------------------------------
  Ridge           `alpha = 1.0`
  Decision Tree   `max_depth = 8`, `min_samples_leaf = 5`
  SVR             `C = 100`, `gamma = scale`, `epsilon = 1.0`

Tuning improved SVR from approximately:

``` text
R² = -0.026
```

to:

``` text
R² = 0.018
```

However, SVR remained the weakest model overall.

------------------------------------------------------------------------

## 🧩 Feature-Subset Ablation

Ablation experiments were performed using the tuned Ridge model to
determine which feature groups contribute most to prediction.

  Feature subset                   MAE (\$)          R²
  ---------------------------- ------------ -----------
  **All features**               **50,993**   **0.220**
  Without `role_family`              52,650       0.176
  Without `company_location`         53,864       0.155
  Without `company_size`             51,002       0.219
  `experience_level` only            55,679       0.105

### Key finding

The two most important feature groups were:

1.  **Role family**
2.  **Company location**

Removing `company_location` reduced R² from:

``` text
0.220 → 0.155
```

Removing `role_family` reduced R² from:

``` text
0.220 → 0.176
```

By comparison, using `experience_level` alone achieved:

``` text
R² = 0.105
```

Therefore, **job specialization and geographic location provide
substantial predictive information beyond experience level alone**.

------------------------------------------------------------------------

## 💡 Key Findings

### 1. Ridge Regression was the strongest overall model

Ridge achieved the best test performance with:

``` text
R²  = 0.220
MAE = $50,993
```

It provided a strong balance between predictive performance, simplicity,
interpretability, and computational efficiency.

### 2. Scaling is model-dependent

Scaling did not materially affect Linear Regression, Ridge, or Decision
Tree.

However, scaling was critical for MLP.

``` text
MLP:
Unscaled R² = 0.007
Scaled R²   = 0.215
```

### 3. Job role matters substantially

Removing `role_family` caused a meaningful performance reduction.

### 4. Geography matters substantially

Removing `company_location` caused the largest R² decrease in the
ablation study.

### 5. Experience level alone is not enough

Although salary generally increases with experience, experience alone
explained only:

``` text
R² = 0.105
```

### 6. SVR struggled with this feature representation

The RBF-based SVR performed poorly on the sparse, one-hot encoded
feature space used in this study.

### 7. The dataset has a limited predictive ceiling

The best model explains only about **22% of salary variance**.

This indicates that important salary determinants are absent from the
available features.

------------------------------------------------------------------------

## 📉 Why Is R² Relatively Low?

Salary is influenced by many variables that are not available in the
dataset, including factors such as:

-   Specific employer
-   Individual qualifications
-   Negotiated compensation
-   Equity
-   Bonuses
-   Specialized skills
-   Industry segment
-   Exact geographic market
-   Company compensation strategy
-   Individual performance
-   Education
-   Certifications

Therefore, an R² around 0.22 should not automatically be interpreted as
a failure of machine learning. It reflects the limited explanatory
information contained in the available predictors.

------------------------------------------------------------------------

## ⚠️ Limitations

This project has several important limitations.

### 1. Modeling subsample

The original dataset contains 71,913 records, but 10,000 were sampled
for the modeling experiments because of the computational cost of SVR.

### 2. Limited feature set

Several potentially important salary determinants are unavailable.

### 3. Self-reported data

The salary survey is based on self-reported information, which can
introduce noise and reporting variability.

### 4. Geographic aggregation

Company locations were reduced to the eight common countries plus an
`Other` category to control feature dimensionality.

### 5. High-cardinality categorical data

One-hot encoding can create a sparse feature space, which is
particularly challenging for distance-based kernel methods such as RBF
SVR.

### 6. Predictive ceiling

The best model achieved R² ≈ 0.22, indicating that substantial salary
variation remains unexplained.

------------------------------------------------------------------------

## 🚀 Future Work

Several directions could improve the study:

-   Train Linear Regression, Ridge, and tree-based models on the
    complete 71,913-row dataset.
-   Evaluate **Random Forest Regression**.
-   Evaluate **Gradient Boosting** and other ensemble methods.
-   Experiment with target encoding for high-cardinality categorical
    variables.
-   Investigate frequency encoding.
-   Compare alternative representations of `job_title`.
-   Explore salary transformations such as log-salary modeling.
-   Add additional job-market variables where available.
-   Evaluate more advanced boosting algorithms.
-   Perform broader hyperparameter optimization.
-   Investigate explainability techniques such as permutation importance
    and SHAP.
-   Build a deployable salary-estimation API or dashboard.

------------------------------------------------------------------------

## 🗂️ Project Structure

The core project is organized around the final machine learning
notebook:

``` text
mbust/
└── ML_final_project/
    └── ML_Salary_Prediction_Comparative_Study.ipynb
```

The notebook contains the complete experimental workflow, including data
preparation, exploratory analysis, model training, scaling comparison,
hyperparameter tuning, evaluation, and feature-ablation experiments.

------------------------------------------------------------------------

## 🛠️ Technology Stack

-   **Python**
-   **Jupyter Notebook**
-   **Pandas**
-   **NumPy**
-   **Matplotlib**
-   **Scikit-learn**
-   **Machine Learning Regression**
-   **One-Hot Encoding**
-   **StandardScaler**
-   **GridSearchCV**
-   **5-Fold Cross-Validation**

The study reports using **scikit-learn 1.8** with a fixed random seed of
**42**.

------------------------------------------------------------------------

## ▶️ How to Run

### 1. Clone the repository

``` bash
git clone https://github.com/dilliram-code/mbust.git
cd mbust
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

Activate it on macOS/Linux:

``` bash
source .venv/bin/activate
```

On Windows:

``` bash
.venv\Scripts\activate
```

### 3. Install dependencies

Install the required packages used by the notebook:

``` bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 4. Launch Jupyter

``` bash
jupyter notebook
```

Then open:

``` text
ML_final_project/ML_Salary_Prediction_Comparative_Study.ipynb
```

> **Note:** The notebook expects the project dataset to be available at
> the path/location used in the notebook. If you clone the repository,
> verify the dataset path before running all cells.

------------------------------------------------------------------------

## 📓 Notebook

The complete implementation and experimental analysis are available
here:

**[ML Salary Prediction --- Comparative
Study](https://github.com/dilliram-code/mbust/blob/main/ML_final_project/ML_Salary_Prediction_Comparative_Study.ipynb)**

The notebook contains approximately 2,004 lines of notebook content and
is the primary executable artifact for this project.

------------------------------------------------------------------------

## 📚 Dataset Source

The salary data is derived from the public AI/ML/Big Data salary survey
published by **ai-jobs.net** and mirrored in the following repository:

**[foorilla/ai-jobs-net-salaries](https://github.com/foorilla/ai-jobs-net-salaries)**

The dataset is described in the accompanying research paper as being
available under a **CC0/public-domain license**.

------------------------------------------------------------------------

## 🎓 Academic Context

This project was completed as a **Machine Learning Final Project** for
the:

**Master of Applied Sciences in Artificial Intelligence (MAI)**\
**Department of Digital Technology**\
**Madan Bhandari University of Science and Technology**\
Chitlang, Makwanpur, Nepal

**Student:** Dilli Ram Chaudhary\
**Roll No.:** 026/MAI/03

------------------------------------------------------------------------

## 🤖 AI Usage Statement

AI assistance was used during preparation of the accompanying academic
manuscript for tasks including drafting selected sections, formatting
assistance, and locating/summarizing related-work references.

The machine learning methodology, experiments, results, and analysis are
documented in the project materials and notebook.

------------------------------------------------------------------------

## 📖 References

Selected references used in the accompanying study include:

1.  *Forecasting data science professionals' salaries using machine
    learning methods based on real data*, AIP Conference Proceedings,
    2024.
2.  *Comparative Analysis of Machine Learning Models for Employee Salary
    Prediction*, Springer, 2025.
3.  W. Jiang, *The investigation and prediction for salary trends in the
    data science industry*, Applied and Computational Engineering, 2024.
4.  Z. Feng, Z. Liu, and Y. Yin, *Comparison of deep-learning and
    conventional machine learning algorithms for salary*, 2023.
5.  H. Aminu et al., *Salary prediction model using principal component
    analysis and deep neural network algorithm*, 2023.
6.  F. Eichinger and M. Mayer, *Predicting salaries with random-forest
    regression*, Springer, 2022.
7.  J. M. H. Pinheiro et al., *The Impact of Feature Scaling in Machine
    Learning: Effects on Regression and Classification Tasks*, 2025.
8.  D. Chicco, M. J. Warrens, and G. Jurman, *The coefficient of
    determination R-squared is more informative than SMAPE, MAE, MAPE,
    MSE and RMSE in regression analysis evaluation*, PeerJ Computer
    Science, 2021.

------------------------------------------------------------------------

## ⭐ Conclusion

This comparative study shows that **simple, regularized machine learning
models can be highly competitive on structured salary data**.

The strongest overall model was **Ridge Regression**, achieving a test
R² of **0.220** and an MAE of approximately **\$50,993**.

The study also demonstrates that preprocessing choices cannot be
separated from model selection. In particular, feature scaling was
essential for the MLP Regressor but had little effect on Linear
Regression, Ridge Regression, and Decision Tree.

Most importantly, the feature-ablation experiments indicate that **role
specialization and geographic location contain substantial salary
information**, while experience level alone provides only a limited
explanation of salary variation.

The results support using **Ridge Regression as a practical baseline**
for this type of tabular salary prediction problem, while suggesting
that future work should explore richer feature representations and
ensemble methods.

------------------------------------------------------------------------
```
