# Literature Review — Starting Reading List

**Important:** These are real papers found via search to help you start your
literature review. You must actually read the full papers (not just the
abstracts below), verify they are appropriate/available through your
institution's library or open-access, and then **write the Related Work
section in your own words**, correctly cited in IEEE numbered format.
Do not copy sentences from this list into your paper — it is a pointer
list, not a source of text to reuse. Pick 6–10 of the most relevant ones
(prioritise papers actually using the Pima dataset or closely related
diabetes-prediction ML comparisons, and prioritise papers from the last
five years per your assignment brief).

---

1. **Comparison of Logistic Regression, Random Forest, SVM and KNN
   Algorithms in Diabetes Prediction** (2023–2024). Compares LR, RF, SVM,
   KNN on a 100,000-record Kaggle diabetes dataset with SMOTE for class
   imbalance.
   https://www.researchgate.net/publication/397904376

2. **Prediction of Diabetes Mellitus using Machine Learning Algorithms:
   Comparative Analysis of KNN, Random Forest and Logistic Regression**
   (2025). Uses KNN, RF, Decision Tree, LR on 15,000 patient records;
   reports RF ~92% accuracy.
   https://www.academia.edu/100626341

3. **Revolutionizing Diabetes Diagnosis: Machine Learning Techniques
   Unleashed** (2023, PMC). Reviews LR, Binary Neural Network, Decision
   Forest, Decision Tree on the Pima dataset; discusses WEKA vs.
   Python/scikit-learn tooling.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC10648466/

4. **Comparative Analysis of Logistic Regression, SVM, XGBoost, and
   Random Forest Algorithms for Diabetes Classification** (2023–2024,
   Jurnal Teknologi Sistem Informasi dan Aplikasi).
   https://doi.org/10.32493/jtsi.v7i1.38258

5. **Development and Internal Validation of a Machine Learning Algorithm
   for the Risk of Type 2 Diabetes in Children with Obesity** (2024–2025,
   PMC). Compares 8 algorithms including SVM, MLP, Decision Tree,
   ensemble methods; SVM was the best performer (AUC 0.98).
   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12370479/

6. **A Comparative Study of Explainable Machine Learning Models with
   Shapley Values for Diabetes Prediction** (2025, ScienceDirect). Adds
   an explainability (SHAP) angle to the standard model comparison.
   https://www.sciencedirect.com/science/article/pii/S2772442525000097

7. **A Comparative Study of Diabetes Prediction Based on Lifestyle
   Factors Using Machine Learning** (2025, arXiv). Decision Tree, KNN,
   Logistic Regression on BRFSS lifestyle data; useful as a contrast to
   clinical-measurement-based datasets like Pima.
   https://arxiv.org/html/2503.04137v1

8. **Optimizing Feature Selection and Machine Learning Algorithms for
   Early Detection of Prediabetes Risk: Comparative Study** (2024–2025,
   PMC). Discusses feature selection strategy alongside model
   comparison — relevant if you extend to a feature-subset comparative
   study.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12314567/

9. **Predicting Diabetes Using Supervised Machine Learning Algorithms on
   E-Health Records** (2025, ScienceDirect). KNN emerged as the top
   performer in this study — useful as a contrasting result to discuss
   against your own findings.
   https://www.sciencedirect.com/science/article/pii/S2949953425000013

10. **A Comparison of Machine Learning Algorithms for Diabetes
    Prediction** (ScienceDirect). Uses the same Pima Indian Diabetes
    dataset (768 patients, 9 attributes) directly comparable to this
    project; found LR and SVM performed best; also tried a neural
    network.
    https://www.sciencedirect.com/science/article/pii/S2405959521000205

---

## Suggested grouping for your Related Work synthesis
- **Group A — Classical ML comparisons on the Pima dataset itself**
  (papers 3, 10): directly comparable prior baselines for your results.
- **Group B — Larger-scale / alternative diabetes datasets** (papers 1,
  2, 7, 9): shows how results and "best model" claims vary with dataset
  size and feature set, motivating why single-dataset studies (like
  yours) have limited generalisability.
- **Group C — Advanced/ensemble and explainability extensions** (papers
  4, 5, 6, 8): shows where the field is heading beyond the five
  algorithms you implemented, useful for your Future Work discussion.

## A note on citation practice
When you write the Related Work section, paraphrase each paper's
approach, dataset, and headline result in 1–2 sentences of your own
words, then cite. Do not chain multiple quoted phrases together. Verify
every DOI/URL before submission — links can move.
