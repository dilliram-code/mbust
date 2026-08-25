"""
04_comparative_study_svm_kernels.py
Comparative Study B: Effect of SVM kernel choice on classification
performance for diabetes prediction.

Kernels compared: linear, polynomial (degree=3), RBF, sigmoid.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train = pd.read_csv("X_train_scaled.csv")
X_test = pd.read_csv("X_test_scaled.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

kernels = ["linear", "poly", "rbf", "sigmoid"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rows = []
for k in kernels:
    model = SVC(kernel=k, degree=3, probability=True, random_state=42)
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rows.append({
        "Kernel": k,
        "CV Accuracy (mean)": round(cv_acc.mean(), 4),
        "CV Accuracy (std)": round(cv_acc.std(), 4),
        "Test Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Test Precision": round(precision_score(y_test, y_pred), 4),
        "Test Recall": round(recall_score(y_test, y_pred), 4),
        "Test F1-score": round(f1_score(y_test, y_pred), 4),
    })

kernel_df = pd.DataFrame(rows).sort_values("Test F1-score", ascending=False)
kernel_df.to_csv("svm_kernel_comparison.csv", index=False)
print(kernel_df.to_string(index=False))

plt.figure(figsize=(8, 5))
plot_df = kernel_df.set_index("Kernel")[["Test Accuracy", "Test Precision", "Test Recall", "Test F1-score"]]
plot_df.plot(kind="bar", figsize=(9, 5))
plt.title("SVM Kernel Comparison (Diabetes Prediction)")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/svm_kernel_comparison.png", dpi=150)
plt.close()

print("\nSVM kernel comparative study complete.")
