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
