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
