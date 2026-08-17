"""
================================================================================
GOOGLE PLAY STORE APPS - FULL PIPELINE
Data Cleaning -> Feature Engineering -> EDA (25 plots) -> ML Modeling
================================================================================
Run stages in order:
  1. python clean.py                 -> step1_clean.csv
  2. python feature_engineering.py   -> step2_features.csv
  3. python eda_plots.py             -> plots/01..10
  4. python eda_plots2.py            -> plots/11..20
  5. python modeling.py              -> plots/21..25, model_results.csv
  6. python tuning.py                -> tuning_result.txt (GridSearchCV)

This file simply concatenates all of the above for reference / single-file review.
================================================================================
"""
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

df = pd.read_csv('/mnt/user-data/uploads/googleplaystore.csv')
print("Raw shape:", df.shape)

# ---- 1. Fix the known corrupted row (Category == '1.9', columns shifted) ----
bad = df[df['Category'] == '1.9']
print("\nCorrupted row(s):")
print(bad)
df = df[df['Category'] != '1.9'].copy()
print("Shape after dropping corrupted row:", df.shape)

# ---- 2. Drop exact duplicate app entries (many apps appear >1 time, scraped repeatedly) ----
print("\nDuplicate App names (rows):", df.duplicated(subset='App').sum())
df = df.drop_duplicates(subset='App', keep='first').copy()
print("Shape after de-duplication:", df.shape)

df.to_csv('/home/claude/project/step1_clean.csv', index=False)
print("\nSaved step1_clean.csv")
import pandas as pd
import numpy as np

df = pd.read_csv('/home/claude/project/step1_clean.csv')

# ---- Reviews: string -> int ----
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')

# ---- Installs: "10,000+" -> 10000 (int) ----
df['Installs'] = df['Installs'].astype(str).str.replace('[+,]', '', regex=True)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

# ---- Price: "$4.99" -> 4.99 (float) ----
df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=False)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# ---- Size: "19M"->19.0 (MB), "14k"->0.014 (MB), "Varies with device"->NaN ----
def parse_size(x):
    x = str(x).strip()
    if x == 'Varies with device' or x == 'nan':
        return np.nan
    if x.endswith('M'):
        return float(x[:-1])
    if x.endswith('k') or x.endswith('K'):
        return float(x[:-1]) / 1024.0
    try:
        return float(x)
    except ValueError:
        return np.nan

df['Size_MB'] = df['Size'].apply(parse_size)
# impute missing size with median size per Category (apps in same category tend to be similar size)
df['Size_MB'] = df.groupby('Category')['Size_MB'].transform(lambda s: s.fillna(s.median()))
df['Size_MB'] = df['Size_MB'].fillna(df['Size_MB'].median())

# ---- Last Updated -> datetime, then derive Year / Month / days since update ----
df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
df['Update_Year'] = df['Last Updated'].dt.year
df['Update_Month'] = df['Last Updated'].dt.month
snapshot_date = df['Last Updated'].max()  # dataset scrape date proxy
df['Days_Since_Update'] = (snapshot_date - df['Last Updated']).dt.days

# ---- Type: fix the single missing value using Price (Price==0 -> Free) ----
type_fill = pd.Series(np.where(df['Price'].fillna(0) == 0, 'Free', 'Paid'), index=df.index)
df['Type'] = df['Type'].fillna(type_fill)

# ---- Content Rating: fill single missing with mode ----
df['Content Rating'] = df['Content Rating'].fillna(df['Content Rating'].mode()[0])

# ---- Rating: keep NaNs for now (target-ish column), but also build a flag ----
df['Rating_Missing'] = df['Rating'].isna().astype(int)

# ---- Engineered features ----
df['App_Name_Length'] = df['App'].str.len()
df['Num_Genres'] = df['Genres'].str.split(';').apply(len)
df['Is_Free'] = (df['Type'] == 'Free').astype(int)
df['Reviews_per_Install'] = df['Reviews'] / df['Installs'].replace(0, np.nan)
df['Log_Installs'] = np.log1p(df['Installs'])
df['Log_Reviews'] = np.log1p(df['Reviews'])
df['Log_Price'] = np.log1p(df['Price'])
df['Price_Tier'] = pd.cut(df['Price'], bins=[-0.01, 0, 2, 10, 50, 500],
                           labels=['Free', 'Cheap(0-2)', 'Mid(2-10)', 'Pricey(10-50)', 'Premium(50+)'])

# Install bucket for easier grouping/plots
install_bins = [0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e10]
install_labels = ['<1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M', '10M-100M', '100M+']
df['Install_Bucket'] = pd.cut(df['Installs'], bins=install_bins, labels=install_labels)

print(df[['App','Size_MB','Installs','Price','Type','Update_Year','Days_Since_Update']].head(10))
print("\nMissing values after FE:")
print(df.isna().sum())

df.to_csv('/home/claude/project/step2_features.csv', index=False)
print("\nSaved step2_features.csv, shape:", df.shape)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 110
PLOT_DIR = '/home/claude/project/plots'

df = pd.read_csv('/home/claude/project/step2_features.csv')
df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')

# For rating-based plots, use rows where Rating isn't null
df_r = df.dropna(subset=['Rating']).copy()

# 1. Distribution of Ratings
plt.figure(figsize=(8,5))
sns.histplot(df_r['Rating'], bins=30, kde=True, color='#4C72B0')
plt.axvline(df_r['Rating'].mean(), color='red', linestyle='--', label=f"Mean={df_r['Rating'].mean():.2f}")
plt.title('Distribution of App Ratings')
plt.xlabel('Rating'); plt.ylabel('Count'); plt.legend()
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/01_rating_distribution.png'); plt.close()

# 2. Top 15 Categories by app count
plt.figure(figsize=(9,7))
cat_counts = df['Category'].value_counts().head(15)
sns.barplot(x=cat_counts.values, y=cat_counts.index, hue=cat_counts.index, palette='viridis', legend=False)
plt.title('Top 15 App Categories by Number of Apps')
plt.xlabel('Number of Apps'); plt.ylabel('Category')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/02_top_categories_count.png'); plt.close()

# 3. Average Rating by Category (top 15 by count)
plt.figure(figsize=(9,7))
top_cats = df['Category'].value_counts().head(15).index
avg_rating = df_r[df_r['Category'].isin(top_cats)].groupby('Category')['Rating'].mean().sort_values()
sns.barplot(x=avg_rating.values, y=avg_rating.index, hue=avg_rating.index, palette='mako', legend=False)
plt.title('Average Rating by Category (Top 15 Categories)')
plt.xlabel('Average Rating'); plt.ylabel('Category')
plt.xlim(3.5, 4.6)
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/03_avg_rating_by_category.png'); plt.close()

# 4. Free vs Paid count
plt.figure(figsize=(6,6))
type_counts = df['Type'].value_counts()
plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
        colors=['#55A868','#C44E52'], startangle=90, explode=[0,0.08])
plt.title('Free vs Paid Apps')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/04_free_vs_paid.png'); plt.close()

# 5. Rating: Free vs Paid boxplot
plt.figure(figsize=(6,6))
sns.boxplot(data=df_r, x='Type', y='Rating', hue='Type', palette=['#55A868','#C44E52'], legend=False)
plt.title('Rating Distribution: Free vs Paid Apps')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/05_rating_free_vs_paid.png'); plt.close()

# 6. Installs distribution by bucket
plt.figure(figsize=(9,5))
order = ['<1K','1K-10K','10K-100K','100K-1M','1M-10M','10M-100M','100M+']
inst_counts = df['Install_Bucket'].value_counts().reindex(order)
sns.barplot(x=inst_counts.index, y=inst_counts.values, hue=inst_counts.index, palette='crest', legend=False)
plt.title('Number of Apps by Install Bucket')
plt.xlabel('Installs'); plt.ylabel('Number of Apps'); plt.xticks(rotation=30)
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/06_installs_bucket.png'); plt.close()

# 7. Reviews vs Rating scatter (log reviews)
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_r, x='Log_Reviews', y='Rating', alpha=0.25, s=15, color='#4C72B0')
plt.title('Rating vs Log(Reviews)')
plt.xlabel('Log(1+Reviews)'); plt.ylabel('Rating')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/07_rating_vs_logreviews.png'); plt.close()

# 8. Size vs Rating scatter
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_r, x='Size_MB', y='Rating', alpha=0.25, s=15, color='#DD8452')
plt.title('Rating vs App Size (MB)')
plt.xlabel('Size (MB)'); plt.ylabel('Rating')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/08_rating_vs_size.png'); plt.close()

# 9. Price distribution (paid apps only)
plt.figure(figsize=(8,5))
paid = df[(df['Type']=='Paid') & (df['Price']>0) & (df['Price']<100)]
sns.histplot(paid['Price'], bins=40, color='#8172B2')
plt.title('Price Distribution of Paid Apps (< $100)')
plt.xlabel('Price ($)'); plt.ylabel('Count')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/09_price_distribution.png'); plt.close()

# 10. Content Rating counts
plt.figure(figsize=(8,5))
cr_counts = df['Content Rating'].value_counts()
sns.barplot(x=cr_counts.values, y=cr_counts.index, hue=cr_counts.index, palette='flare', legend=False)
plt.title('App Count by Content Rating')
plt.xlabel('Number of Apps'); plt.ylabel('Content Rating')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/10_content_rating_counts.png'); plt.close()

print("Batch 1 plots done")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 110
PLOT_DIR = '/home/claude/project/plots'

df = pd.read_csv('/home/claude/project/step2_features.csv')
df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
df_r = df.dropna(subset=['Rating']).copy()

# 11. Correlation heatmap of numeric features
num_cols = ['Rating','Reviews','Size_MB','Installs','Price','App_Name_Length',
            'Num_Genres','Days_Since_Update','Log_Installs','Log_Reviews']
plt.figure(figsize=(10,8))
corr = df_r[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/11_correlation_heatmap.png'); plt.close()

# 12. Total installs by category (top 15)
plt.figure(figsize=(9,7))
inst_by_cat = df.groupby('Category')['Installs'].sum().sort_values(ascending=False).head(15)
sns.barplot(x=inst_by_cat.values/1e9, y=inst_by_cat.index, hue=inst_by_cat.index, palette='rocket', legend=False)
plt.title('Total Installs by Category (Top 15, in Billions)')
plt.xlabel('Total Installs (Billions)'); plt.ylabel('Category')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/12_installs_by_category.png'); plt.close()

# 13. App updates over time (by year)
plt.figure(figsize=(9,5))
year_counts = df['Update_Year'].value_counts().sort_index()
sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o', color='#4C72B0')
plt.title('Number of App Updates by Year')
plt.xlabel('Year'); plt.ylabel('Number of Apps Updated')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/13_updates_by_year.png'); plt.close()

# 14. Rating by number of genres
plt.figure(figsize=(7,5))
sns.boxplot(data=df_r, x='Num_Genres', y='Rating', hue='Num_Genres', palette='Set2', legend=False)
plt.title('Rating Distribution by Number of Genres Tagged')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/14_rating_by_numgenres.png'); plt.close()

# 15. Top 10 most expensive apps
plt.figure(figsize=(9,6))
top_price = df.drop_duplicates('App').nlargest(10, 'Price')[['App','Price']]
sns.barplot(x='Price', y='App', data=top_price, hue='App', palette='magma', legend=False)
plt.title('Top 10 Most Expensive Apps')
plt.xlabel('Price ($)'); plt.ylabel('')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/15_top10_expensive_apps.png'); plt.close()

# 16. Rating vs Days Since Update
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_r, x='Days_Since_Update', y='Rating', alpha=0.25, s=15, color='#55A868')
sns.regplot(data=df_r, x='Days_Since_Update', y='Rating', scatter=False, color='red', line_kws={'linewidth':2})
plt.title('Rating vs Days Since Last Update')
plt.xlabel('Days Since Last Update'); plt.ylabel('Rating')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/16_rating_vs_days_since_update.png'); plt.close()

# 17. Installs vs Price for paid apps (does price hurt installs?)
plt.figure(figsize=(8,6))
paid = df[(df['Type']=='Paid') & (df['Price']>0) & (df['Price']<100)]
sns.scatterplot(data=paid, x='Price', y='Log_Installs', alpha=0.4, s=20, color='#C44E52')
plt.title('Log(Installs) vs Price for Paid Apps')
plt.xlabel('Price ($)'); plt.ylabel('Log(1+Installs)')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/17_installs_vs_price.png'); plt.close()

# 18. Violin: Rating distribution across top 8 categories
plt.figure(figsize=(11,7))
top8 = df['Category'].value_counts().head(8).index
sns.violinplot(data=df_r[df_r['Category'].isin(top8)], x='Category', y='Rating', hue='Category',
               palette='Set3', legend=False)
plt.title('Rating Distribution Across Top 8 Categories (Violin Plot)')
plt.xticks(rotation=35, ha='right')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/18_rating_violin_top8_categories.png'); plt.close()

# 19. App Name Length vs Rating
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_r, x='App_Name_Length', y='Rating', alpha=0.25, s=15, color='#8172B2')
plt.title('Rating vs App Name Length')
plt.xlabel('App Name Length (characters)'); plt.ylabel('Rating')
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/19_rating_vs_namelength.png'); plt.close()

# 20. Content Rating vs average Rating
plt.figure(figsize=(8,5))
cr_rating = df_r.groupby('Content Rating')['Rating'].mean().sort_values()
sns.barplot(x=cr_rating.values, y=cr_rating.index, hue=cr_rating.index, palette='cubehelix', legend=False)
plt.title('Average Rating by Content Rating')
plt.xlabel('Average Rating'); plt.ylabel('Content Rating')
plt.xlim(3.5,4.6)
plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/20_avg_rating_by_content_rating.png'); plt.close()

print("Batch 2 plots done")
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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
df = pd.read_csv('/home/claude/project/step2_features.csv')
model_df = df.dropna(subset=['Rating']).copy()

numeric_features = ['Log_Reviews', 'Size_MB', 'Log_Installs', 'Log_Price',
                     'App_Name_Length', 'Num_Genres', 'Days_Since_Update',
                     'Update_Year', 'Update_Month']
categorical_features = ['Category', 'Type', 'Content Rating']
X = model_df[numeric_features + categorical_features]
y = model_df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                     ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features),
                                   ('cat', categorical_transformer, categorical_features)])

param_grid = {
    'model__n_estimators': [200],
    'model__max_depth': [8, 12],
    'model__min_samples_leaf': [3, 5],
}
pipe = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1))])
grid = GridSearchCV(pipe, param_grid, cv=3, scoring='r2', n_jobs=1)
grid.fit(X_train, y_train)

preds = grid.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

with open('/home/claude/project/tuning_result.txt', 'w') as f:
    f.write(f"Best params: {grid.best_params_}\n")
    f.write(f"Best CV R2: {grid.best_score_:.4f}\n")
    f.write(f"Test MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}\n")

print("DONE")
print(grid.best_params_, grid.best_score_, mae, rmse, r2)
