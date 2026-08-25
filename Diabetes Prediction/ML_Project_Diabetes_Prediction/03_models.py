"""
03_models.py
Implements and compares 5 classification algorithms on the Pima Indians
Diabetes dataset:
 - Logistic Regression
 - k-Nearest Neighbours
 - Decision Tree
 - Gaussian Naive Bayes
 - Support Vector Machine (RBF kernel, used as the main SVM result)

Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix,
via both 5-fold stratified cross-validation (on the training set) and
a held-out test set.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

X_train = pd.read_csv("X_train_scaled.csv")
X_test = pd.read_csv("X_test_scaled.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Naive Bayes": GaussianNB(),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
cm_dict = {}
trained_models = {}

for name, model in models.items():
    # 5-fold CV on training set
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

    # Fit on full training set, evaluate on held-out test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    trained_models[name] = model

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm_dict[name] = cm.tolist()

    results.append({
        "Model": name,
        "CV Accuracy (mean)": round(cv_acc.mean(), 4),
        "CV Accuracy (std)": round(cv_acc.std(), 4),
        "CV F1 (mean)": round(cv_f1.mean(), 4),
        "Test Accuracy": round(acc, 4),
        "Test Precision": round(prec, 4),
        "Test Recall": round(rec, 4),
        "Test F1-score": round(f1, 4),
    })

    print(f"\n=== {name} ===")
    print(f"5-fold CV Accuracy: {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
    print("Test set classification report:")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
    print("Confusion Matrix:\n", cm)

results_df = pd.DataFrame(results).sort_values("Test F1-score", ascending=False)
results_df.to_csv("model_comparison_results.csv", index=False)
print("\n\n=== FINAL COMPARISON TABLE (sorted by Test F1-score) ===")
print(results_df.to_string(index=False))

with open("confusion_matrices.json", "w") as f:
    json.dump(cm_dict, f, indent=2)

# --- Plot: confusion matrices grid ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()
for i, (name, cm) in enumerate(cm_dict.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                xticklabels=["No Diabetes", "Diabetes"],
                yticklabels=["No Diabetes", "Diabetes"], cbar=False)
    axes[i].set_title(name)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
axes[-1].axis("off")
plt.tight_layout()
plt.savefig("figures/confusion_matrices.png", dpi=150)
plt.close()

# --- Plot: model comparison bar chart ---
plt.figure(figsize=(9, 5))
metrics_to_plot = ["Test Accuracy", "Test Precision", "Test Recall", "Test F1-score"]
plot_df = results_df.set_index("Model")[metrics_to_plot]
plot_df.plot(kind="bar", figsize=(10, 5))
plt.title("Model Comparison on Test Set")
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/model_comparison_bar.png", dpi=150)
plt.close()

print("\nModel comparison complete. Results saved to model_comparison_results.csv")
