# Google Play Store Apps — EDA, Feature Engineering & Predictive Modeling

**Dataset:** `googleplaystore.csv` — 10,841 rows × 13 columns (app metadata scraped from the Google Play Store)

---

## 1. Data Cleaning

Before any analysis, the raw data needed fixing:

| Issue | Fix |
|---|---|
| One row (`Life Made WI-Fi Touchscreen Photo Frame`) has all values shifted left by one column because its `Category` field was missing in the source scrape — `Category` shows `"1.9"` (a Rating value) | Row dropped |
| 1,181 duplicate `App` rows (same app scraped at different times/categories) | Kept first occurrence, dropped the rest |
| `Reviews`, `Installs`, `Price`, `Size` stored as text (`"10,000+"`, `"$4.99"`, `"19M"`) | Parsed to numeric |
| `Size` has `"Varies with device"` for many rows | Converted to `NaN`, then imputed with the **median size of that app's Category** (apps in the same category tend to be similarly sized) |
| 1,474 missing `Rating` values (~13.6% of rows) | Left as `NaN` for EDA; these rows are excluded only from rating-specific plots/modeling target, but a `Rating_Missing` flag was engineered in case "not yet rated" itself is informative |
| 1 missing `Type`, 1 missing `Content Rating` | `Type` imputed from `Price` (0 → Free, else → Paid); `Content Rating` imputed with the mode |

**Shape after cleaning:** 10,840 → 9,659 rows (after de-duplication).

```python
import pandas as pd
import numpy as np

df = pd.read_csv('googleplaystore.csv')

# Fix the corrupted row (Category == '1.9', columns shifted)
df = df[df['Category'] != '1.9'].copy()

# Drop duplicate app entries, keep first occurrence
df = df.drop_duplicates(subset='App', keep='first').copy()
```

---

## 2. Feature Engineering & Transformation

```python
# Reviews: string -> int
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')

# Installs: "10,000+" -> 10000
df['Installs'] = df['Installs'].astype(str).str.replace('[+,]', '', regex=True)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

# Price: "$4.99" -> 4.99
df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=False)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Size: "19M" -> 19.0 (MB), "14k" -> 0.014 (MB), "Varies with device" -> NaN
def parse_size(x):
    x = str(x).strip()
    if x in ('Varies with device', 'nan'):
        return np.nan
    if x.endswith('M'):
        return float(x[:-1])
    if x.endswith(('k', 'K')):
        return float(x[:-1]) / 1024.0
    try:
        return float(x)
    except ValueError:
        return np.nan

df['Size_MB'] = df['Size'].apply(parse_size)
df['Size_MB'] = df.groupby('Category')['Size_MB'].transform(lambda s: s.fillna(s.median()))
df['Size_MB'] = df['Size_MB'].fillna(df['Size_MB'].median())

# Dates
df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
df['Update_Year']  = df['Last Updated'].dt.year
df['Update_Month'] = df['Last Updated'].dt.month
df['Days_Since_Update'] = (df['Last Updated'].max() - df['Last Updated']).dt.days

# New engineered features
df['App_Name_Length']      = df['App'].str.len()
df['Num_Genres']           = df['Genres'].str.split(';').apply(len)
df['Is_Free']              = (df['Type'] == 'Free').astype(int)
df['Reviews_per_Install']  = df['Reviews'] / df['Installs'].replace(0, np.nan)

# Skew-correcting log transforms (Installs, Reviews, Price are heavily right-skewed)
df['Log_Installs'] = np.log1p(df['Installs'])
df['Log_Reviews']  = np.log1p(df['Reviews'])
df['Log_Price']    = np.log1p(df['Price'])

# Binned/categorical versions for plotting & grouping
df['Price_Tier'] = pd.cut(df['Price'], bins=[-0.01, 0, 2, 10, 50, 500],
                           labels=['Free', 'Cheap(0-2)', 'Mid(2-10)', 'Pricey(10-50)', 'Premium(50+)'])
install_bins   = [0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e10]
install_labels = ['<1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M', '10M-100M', '100M+']
df['Install_Bucket'] = pd.cut(df['Installs'], bins=install_bins, labels=install_labels)
```

**New features created:** `Size_MB`, `Update_Year`, `Update_Month`, `Days_Since_Update`, `App_Name_Length`, `Num_Genres`, `Is_Free`, `Reviews_per_Install`, `Log_Installs`, `Log_Reviews`, `Log_Price`, `Price_Tier`, `Install_Bucket`, `Rating_Missing`.

**Why the log transforms matter:** `Installs`, `Reviews`, and `Price` span many orders of magnitude (1 → 1,000,000,000+ installs). Using them raw would let a handful of mega-apps (Facebook, WhatsApp, etc.) dominate any distance-based or linear model. `log1p` compresses that scale and makes the correlation with `Rating` far more linear/usable.

---

## 3. Exploratory Data Analysis — Charts & Insights

### 3.1 Distribution of App Ratings
![Rating Distribution](plots/01_rating_distribution.png)
**Insight:** Ratings are heavily **left-skewed** — the mean is 4.17 and the median is 4.3. Most apps sit between 4.0 and 4.7; very few apps have ratings below 3.0. This means the Play Store's rating field is not a good discriminator on its own — nearly every surviving/published app is "good" (survivorship bias: badly-rated apps get removed or abandoned before they accumulate many reviews).

### 3.2 Top 15 App Categories by Count
![Top Categories](plots/02_top_categories_count.png)
**Insight:** `FAMILY` (1,832 apps) and `GAME` (959 apps) dominate the store by volume, followed by `TOOLS` and `BUSINESS`. The "Family" category is inflated because Google buckets most children's games/education apps there instead of `GAME`, so category counts should be read with that in mind.

### 3.3 Average Rating by Category
![Avg Rating by Category](plots/03_avg_rating_by_category.png)
**Insight:** Average ratings across categories are tightly clustered (mostly 4.0–4.4) — category alone barely moves the needle. Looking at categories with ≥30 rated apps, **EVENTS (4.44), EDUCATION (4.36) and ART_AND_DESIGN (4.36)** rate highest, while **DATING (3.97), MAPS_AND_NAVIGATION (4.04) and TOOLS (4.04)** rate lowest — utility apps that people are forced to use (maps, dating) tend to collect more frustrated 1-star reviews than passion-project apps.

### 3.4 Free vs Paid Apps
![Free vs Paid](plots/04_free_vs_paid.png)
**Insight:** **92.2% of apps are Free**, only 7.8% are Paid. The Play Store ecosystem is overwhelmingly ad/IAP-monetized rather than pay-to-download.

### 3.5 Rating: Free vs Paid
![Rating Free vs Paid](plots/05_rating_free_vs_paid.png)
**Insight:** Paid apps rate slightly higher on average (4.26 vs 4.17 for free). This likely reflects **selection bias** — users who pay upfront are more invested/intentional, and low-quality paid apps get filtered out by the price barrier itself.

### 3.6 Number of Apps by Install Bucket
![Install Buckets](plots/06_installs_bucket.png)
**Insight:** Installs are extremely skewed: a large share of apps sit in the low-to-mid buckets (1K–1M), while relatively few apps break past 10M+ installs. The Play Store, like most app marketplaces, follows a **power-law / long-tail distribution** — a small number of blockbuster apps capture the vast majority of installs.

### 3.7 Rating vs Log(Reviews)
![Rating vs Log Reviews](plots/07_rating_vs_logreviews.png)
**Insight:** Apps with very few reviews show huge rating variance (anywhere from 1 to 5), while apps with many reviews converge tightly around 4.0–4.5. This is the classic **"small sample noise"** effect — a handful of raters can swing an average to 5.0 or 1.0, but as review counts grow the rating stabilizes near the population mean.

### 3.8 Rating vs App Size
![Rating vs Size](plots/08_rating_vs_size.png)
**Insight:** No strong visual trend, but there's a mild positive lean — very small apps (<5MB) show more low-rating outliers, while larger, more "invested-in" apps (30–80MB) cluster more tightly around 4.0+. Correlation is weak (~0.05) so size alone isn't predictive.

### 3.9 Price Distribution (Paid Apps)
![Price Distribution](plots/09_price_distribution.png)
**Insight:** The overwhelming majority of paid apps are priced **under $5**, with a long tail of niche professional/utility apps priced up to $30-$50+. Average paid-app price is **$14.05**, pulled upward by a handful of extreme outliers (several joke apps priced at $399-$400, e.g. *"I'm Rich"* and *"most expensive app (H)"*, which exist purely as status-symbol novelty apps).

### 3.10 App Count by Content Rating
![Content Rating Counts](plots/10_content_rating_counts.png)
**Insight:** The vast majority of apps are rated **"Everyone"**, followed by **"Teen"** and **"Mature 17+"**. Very few apps target the "Adults only 18+" or "Unrated" buckets — most developers optimize for the broadest possible audience.

### 3.11 Correlation Heatmap
![Correlation Heatmap](plots/11_correlation_heatmap.png)
**Insight:** The strongest relationship in the whole numeric feature set is **Reviews ↔ Installs (r≈0.62, or 0.95 in log space)** — unsurprising, since more installs mechanically produce more opportunities to review. **Rating correlates weakly with everything** — the highest is `Log_Reviews` (r≈0.18) and the only meaningfully negative one is `Days_Since_Update` (r≈-0.13, apps that haven't been updated recently tend to rate slightly lower). This is an early warning sign that **Rating will be hard to predict accurately from this metadata**.

### 3.12 Total Installs by Category
![Installs by Category](plots/12_installs_by_category.png)
**Insight:** `GAME` (≈13.9B installs) and `COMMUNICATION` (≈11.0B installs) dwarf every other category in total install volume, even though GAME isn't the single largest category by app *count* — a small number of viral games/communication apps (WhatsApp, Messenger, etc.) carry enormous weight.

### 3.13 App Updates by Year
![Updates by Year](plots/13_updates_by_year.png)
**Insight:** Update activity ramps up sharply toward **2017-2018**, confirming this dataset was scraped in mid-2018 and that most actively-maintained apps update at least once a year. Very few apps in the dataset were last updated before 2015 — those are effectively **abandoned apps** still listed on the store.

### 3.14 Rating by Number of Genres
![Rating by Num Genres](plots/14_rating_by_numgenres.png)
**Insight:** Apps tagged with 2 genres (e.g., "Action;Adventure") show a marginally higher median rating than single-genre apps, suggesting that cross-genre appeal (or simply more thoughtful store-listing effort) correlates slightly with quality — though the effect is small.

### 3.15 Top 10 Most Expensive Apps
![Top 10 Expensive Apps](plots/15_top10_expensive_apps.png)
**Insight:** The most expensive apps ($399-$400) are almost all **novelty/joke apps** with no real functionality (e.g. status-symbol apps that do nothing but prove you spent money). True high-value utility apps (professional tools, medical references) top out much lower, generally under $100.

### 3.16 Rating vs Days Since Last Update
![Rating vs Days Since Update](plots/16_rating_vs_days_since_update.png)
**Insight:** There's a subtle but real **negative trend** (r≈-0.13) — the longer it's been since an app was updated, the lower its rating tends to be. Regularly-maintained apps (bug fixes, new features) retain user satisfaction better than stale, abandoned ones.

### 3.17 Installs vs Price (Paid Apps)
![Installs vs Price](plots/17_installs_vs_price.png)
**Insight:** There's a clear **negative relationship** — as price increases, log(installs) trends downward. Cheap ($0.99-$2.99) paid apps can still reach hundreds of thousands of installs, while apps priced above $20 rarely break past a few thousand. Price is a real barrier to adoption on Android.

### 3.18 Rating Distribution by Category (Violin, Top 8)
![Rating Violin](plots/18_rating_violin_top8_categories.png)
**Insight:** All top categories show the same basic shape — a fat concentration of mass at 4.0-4.7 with a thin lower tail — but `GAME` and `TOOLS` show visibly fatter low-rating tails than `FAMILY` or `PHOTOGRAPHY`, consistent with games/tools attracting more frustrated 1-2 star reviews (bugs, ads, crashes).

### 3.19 Rating vs App Name Length
![Rating vs Name Length](plots/19_rating_vs_namelength.png)
**Insight:** Longer app names (often keyword-stuffed for App Store Optimization, e.g. "Photo Editor & Candy Camera & Grid & ScrapBook") show a very weak positive correlation with rating (r≈0.14) — possibly because SEO-heavy names correlate with more actively marketed/maintained apps, not because the name itself matters.

### 3.20 Average Rating by Content Rating
![Avg Rating by Content Rating](plots/20_avg_rating_by_content_rating.png)
**Insight:** **"Adults only 18+"** apps show the highest average rating (4.30) but this is based on a very small sample. Among the larger buckets, **"Everyone 10+" and "Teen"** apps (4.23 each) slightly outperform the general **"Everyone"** bucket (4.17), while **"Mature 17+" and "Unrated"** trail slightly — broader-audience apps face a wider range of expectations and thus somewhat lower average satisfaction.

---
## 4. Machine Learning: Predicting App Rating

**Problem framing:** `Rating` is the only rich, continuous, quality-related label in this dataset, so we treat this as a **regression problem** — predict an app's Play Store rating (1.0–5.0) from its metadata (category, size, installs, reviews, price, content rating, recency of updates, etc.), *without* using `Rating` itself or anything that leaks it.

Rows with missing `Rating` (13.6% of the data) are dropped from the supervised learning set since they have no label to train/evaluate against — that leaves **8,196 labeled rows**.

### 4.1 Feature / target setup and preprocessing pipeline

```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

model_df = df.dropna(subset=['Rating']).copy()

numeric_features = ['Log_Reviews', 'Size_MB', 'Log_Installs', 'Log_Price',
                     'App_Name_Length', 'Num_Genres', 'Days_Since_Update',
                     'Update_Year', 'Update_Month']
categorical_features = ['Category', 'Type', 'Content Rating']

X = model_df[numeric_features + categorical_features]
y = model_df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Numeric: median-impute + standardize
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# Categorical: most-frequent-impute + one-hot encode
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
```

Raw `Reviews`, `Installs`, and `Price` are excluded in favor of their `Log_*` transformed versions (fixes the extreme right-skew shown in the EDA). `Genres`, `App`, `Current Ver`, `Android Ver`, `Last Updated`, and the bucketed helper columns are excluded to avoid redundancy/leakage.

### 4.2 Fitting multiple models

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Lasso Regression': Lasso(alpha=0.01, random_state=42),
    'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=10,
                                            min_samples_leaf=3, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                     learning_rate=0.05, random_state=42),
    'SVR (RBF)': SVR(kernel='rbf', C=1.0, epsilon=0.1),
}

results = []
for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    cv   = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2,
                     'CV_R2_mean': cv.mean(), 'CV_R2_std': cv.std()})
```

### 4.3 Model comparison results

| Model | MAE | RMSE | Test R² | CV R² (mean ± std) |
|---|---|---|---|---|
| **Random Forest** | **0.344** | **0.503** | **0.178** | 0.141 ± 0.014 |
| Gradient Boosting | 0.345 | 0.505 | 0.172 | 0.156 ± 0.019 |
| Ridge Regression | 0.346 | 0.506 | 0.168 | 0.159 ± 0.014 |
| Linear Regression | 0.346 | 0.506 | 0.168 | 0.158 ± 0.014 |
| Lasso Regression | 0.355 | 0.518 | 0.130 | 0.126 ± 0.010 |
| SVR (RBF) | 0.333 | 0.518 | 0.128 | 0.152 ± 0.020 |
| Decision Tree | 0.365 | 0.525 | 0.105 | 0.058 ± 0.032 |

![Model Comparison Error](plots/21_model_comparison_error.png)
![Model Comparison R2](plots/22_model_comparison_r2.png)

**Insight:** **Random Forest** wins on test R² (0.178) and RMSE, though every model tops out with a fairly low R² (0.10–0.18). This directly confirms what the correlation heatmap showed: **app metadata alone explains only a small fraction of what drives a Play Store rating.** Ratings are driven far more by things this dataset doesn't capture — actual app quality, bugs, UX, ad load, customer support responsiveness — than by size, price, category, or update recency. Tree ensembles (Random Forest, Gradient Boosting) slightly outperform linear models because ratings depend on **non-linear interactions** (e.g., "small AND recently updated AND free" behaves differently than any single feature alone), but the gap over plain Ridge/Linear Regression is modest.

### 4.4 Best model — actual vs predicted & residuals

![Actual vs Predicted](plots/23_best_model_actual_vs_predicted.png)
**Insight:** The model consistently **over-predicts low ratings and under-predicts high ratings** — it regresses toward the ~4.0-4.3 mean rather than confidently calling out truly bad (1-2 star) or truly excellent (4.8-5.0 star) apps. This "regression to the mean" is a textbook symptom of a **weak-signal feature set**: with so little true signal, the model's safest bet is to predict close to the population average for almost everything.

![Residuals](plots/24_best_model_residuals.png)
**Insight:** Residuals are roughly centered at 0 and bell-shaped, but with a **long negative tail** — meaning the model's biggest misses are apps that scored *much lower* than predicted (i.e., genuinely bad apps that "look fine" on paper but rate poorly for reasons the metadata can't see).

### 4.5 Feature importance (Random Forest)

![Feature Importance](plots/25_feature_importance_rf.png)
**Insight:** `Log_Reviews` is by far the most important feature (≈27% of total importance), followed by `Log_Installs`, `App_Name_Length`, `Days_Since_Update`, and `Size_MB`. Individual `Category` dummies (TOOLS, HEALTH_AND_FITNESS, FAMILY, etc.) contribute very little each — category matters far less than popularity/engagement signals and app maintenance recency.

### 4.6 Hyperparameter tuning (Random Forest)

```python
param_grid = {
    'model__n_estimators': [200, 300],
    'model__max_depth': [8, 12],
    'model__min_samples_leaf': [3, 5],
}
grid = GridSearchCV(Pipeline([('preprocessor', preprocessor),
                               ('model', RandomForestRegressor(random_state=42))]),
                     param_grid, cv=3, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)
```

**Result:** Best params: `max_depth=12, min_samples_leaf=5, n_estimators=200` → CV R² = 0.139, **Test MAE = 0.343, RMSE = 0.502, R² = 0.183** — essentially in line with the untuned Random Forest, confirming the ceiling on this dataset is a modeling-signal limitation, not an under-tuned-model limitation.

---

## 5. Overall Takeaways

1. **The dataset is very Free/Everyone/mid-rating skewed** — 92% Free apps, ratings clustered 4.0-4.7, so any model trained on it will default toward predicting "an ordinary, decently-rated free app."
2. **Popularity (reviews/installs) is the strongest available signal for rating**, but it's a weak one (r≈0.18-0.28) — highly-reviewed apps tend to have *stabilized* ratings near the population mean rather than *causally* better ratings.
3. **Update recency matters**: apps not updated in a long time rate measurably lower — this is a genuinely actionable insight for developers (keep shipping updates).
4. **Price hurts installs** but modestly *helps* average rating (self-selection of paying users).
5. **Metadata-only regression tops out around R²≈0.15-0.18** — to meaningfully predict app ratings you'd need review text/sentiment, crash/ANR rates, or user-engagement/retention data that isn't present in this store-listing dataset. This is itself a useful, honest data-science finding: know the ceiling of what your features can explain before over-engineering a model.
