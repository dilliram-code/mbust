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
