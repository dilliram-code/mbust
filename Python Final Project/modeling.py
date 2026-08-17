"""
Feature Transformation + Machine Learning Modeling
Goal: Predict an app's Rating (regression problem) from its metadata.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
PLOT_DIR = '/home/claude/project/plots'

df = pd.read_csv('/home/claude/project/step2_features.csv')

# ---------------------------------------------------------------
# 1. Build the modeling dataframe
# ---------------------------------------------------------------
# Only rows with a known Rating (our target) are usable for supervised learning.
model_df = df.dropna(subset=['Rating']).copy()

# Drop obvious leakage / non-predictive / redundant columns
drop_cols = ['App', 'Rating', 'Genres', 'Current Ver', 'Android Ver', 'Last Updated',
             'Size', 'Installs', 'Reviews', 'Price',  # keep the engineered log/clean versions instead
             'Rating_Missing', 'Install_Bucket', 'Price_Tier', 'Reviews_per_Install']

numeric_features = ['Log_Reviews', 'Size_MB', 'Log_Installs', 'Log_Price',
                     'App_Name_Length', 'Num_Genres', 'Days_Since_Update',
                     'Update_Year', 'Update_Month']
categorical_features = ['Category', 'Type', 'Content Rating']

X = model_df[numeric_features + categorical_features]
y = model_df['Rating']

print("Modeling dataset shape:", X.shape)
print("Target (Rating) summary:\n", y.describe())

# ---------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------
# 3. Preprocessing pipeline (feature transformation)
#    - numeric: median-impute + standardize
#    - categorical: most-frequent-impute + one-hot encode
# ---------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ---------------------------------------------------------------
# 4. Define candidate models
# ---------------------------------------------------------------
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=RANDOM_STATE),
    'Lasso Regression': Lasso(alpha=0.01, random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
    'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=10,
                                            min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                     learning_rate=0.05, random_state=RANDOM_STATE),
    'SVR (RBF)': SVR(kernel='rbf', C=1.0, epsilon=0.1),
}

results = []
predictions = {}

for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    predictions[name] = preds

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

    results.append({
        'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2,
        'CV_R2_mean': cv_scores.mean(), 'CV_R2_std': cv_scores.std()
    })
    print(f"{name:20s} | MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  CV_R2={cv_scores.mean():.4f}(+/-{cv_scores.std():.4f})")

results_df = pd.DataFrame(results).sort_values('R2', ascending=False).reset_index(drop=True)
print("\n=== Model comparison (sorted by test R2) ===")
print(results_df.to_string(index=False))
results_df.to_csv('/home/claude/project/model_results.csv', index=False)

# ---------------------------------------------------------------
# 5. Plot: model comparison
# ---------------------------------------------------------------
plt.figure(figsize=(10,6))
plot_df = results_df.melt(id_vars='Model', value_vars=['MAE','RMSE'], var_name='Metric', value_name='Value')
sns.barplot(data=plot_df, x='Model', y='Value', hue='Metric', palette=['#4C72B0','#C44E52'])
plt.title('Model Comparison: MAE and RMSE (lower is better)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/21_model_comparison_error.png'); plt.close()

plt.figure(figsize=(9,6))
sns.barplot(data=results_df, x='Model', y='R2', hue='Model', palette='viridis', legend=False)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Model Comparison: R2 Score on Test Set (higher is better)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/22_model_comparison_r2.png'); plt.close()

# ---------------------------------------------------------------
# 6. Best model deep dive: actual vs predicted + residuals
# ---------------------------------------------------------------
best_name = results_df.iloc[0]['Model']
best_preds = predictions[best_name]

plt.figure(figsize=(7,7))
plt.scatter(y_test, best_preds, alpha=0.3, s=15, color='#4C72B0')
plt.plot([1,5],[1,5], color='red', linestyle='--', label='Perfect prediction')
plt.xlabel('Actual Rating'); plt.ylabel('Predicted Rating')
plt.title(f'Actual vs Predicted Rating — Best Model: {best_name}')
plt.legend()
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/23_best_model_actual_vs_predicted.png'); plt.close()

residuals = y_test.values - best_preds
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=30, kde=True, color='#DD8452')
plt.axvline(0, color='black', linestyle='--')
plt.title(f'Residual Distribution — Best Model: {best_name}')
plt.xlabel('Residual (Actual - Predicted)')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/24_best_model_residuals.png'); plt.close()

# ---------------------------------------------------------------
# 7. Feature importance (tree-based best-effort: use Random Forest regardless)
# ---------------------------------------------------------------
rf_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', RandomForestRegressor(n_estimators=300, max_depth=10,
                                                             min_samples_leaf=3,
                                                             random_state=RANDOM_STATE, n_jobs=-1))])
rf_pipe.fit(X_train, y_train)

ohe = rf_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_feature_names)

importances = rf_pipe.named_steps['model'].feature_importances_
feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(15)

plt.figure(figsize=(9,7))
sns.barplot(x=feat_imp.values, y=feat_imp.index, hue=feat_imp.index, palette='crest', legend=False)
plt.title('Top 15 Feature Importances (Random Forest)')
plt.xlabel('Importance'); plt.ylabel('Feature')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/25_feature_importance_rf.png'); plt.close()

print("\nTop 15 feature importances:\n", feat_imp)

# ---------------------------------------------------------------
# 8. Hyperparameter tuning for the best tree-based model (Random Forest)
# ---------------------------------------------------------------
param_grid = {
    'model__n_estimators': [200, 400],
    'model__max_depth': [6, 10, None],
    'model__min_samples_leaf': [1, 3, 5],
}
rf_grid_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
grid = GridSearchCV(rf_grid_pipe, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

print("\nBest RF params:", grid.best_params_)
print("Best RF CV R2:", grid.best_score_)

tuned_preds = grid.predict(X_test)
tuned_r2 = r2_score(y_test, tuned_preds)
tuned_mae = mean_absolute_error(y_test, tuned_preds)
tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_preds))
print(f"Tuned RF Test -> MAE={tuned_mae:.4f} RMSE={tuned_rmse:.4f} R2={tuned_r2:.4f}")

with open('/home/claude/project/model_summary.txt', 'w') as f:
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nBest model: {best_name}\n")
    f.write(f"\nTuned Random Forest best params: {grid.best_params_}\n")
    f.write(f"Tuned RF Test MAE={tuned_mae:.4f} RMSE={tuned_rmse:.4f} R2={tuned_r2:.4f}\n")
    f.write("\nTop 15 feature importances:\n")
    f.write(feat_imp.to_string())

print("\nSaved model_results.csv and model_summary.txt")
