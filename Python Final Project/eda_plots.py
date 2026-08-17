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
