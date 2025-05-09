import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from itertools import combinations

# --- Suppress warnings ---
import warnings
warnings.filterwarnings('ignore')

# --- Visualization style ---
plt.style.use('fivethirtyeight')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# --- Display options ---
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)

# --- Load dataset ---
df = pd.read_csv('cleaned_nutrition_dataset_per100g.csv')
df.columns = df.columns.str.replace(r'\(.*\)', '', regex=True).str.strip()

print("# Nutritional Efficiency Framework: A Data-Driven Approach to Food Analysis")
print(f"\nDataset contains {df.shape[0]} food items with {df.shape[1]} nutritional variables")
print("\nThis notebook introduces a novel approach to nutritional analysis:")
print("1. Comprehensive data quality assessment and cleaning")
print("2. Development of the Nutritional Efficiency Index (NEI)")
print("3. Unsupervised learning for nutritional pattern discovery")
print("4. Nutrient-Optimized Food Combinations algorithm")

# --- DATA CLEANING ---

# Calculate expected calories based on macronutrients
df['Expected_Calories'] = df['Carbohydrates'] * 4 + df['Fat'] * 9 + df['Protein'] * 4

# Compare expected vs actual calories
df['Calorie_Ratio'] = df['Expected_Calories'] / df['Calories']
df['Calorie_Diff'] = df['Expected_Calories'] - df['Calories']

# Plot calorie discrepancy
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['Calorie_Ratio'], bins=50)
plt.title('Distribution of Calorie Ratio (Expected / Actual)')
plt.axvline(1, color='red', linestyle='--')
plt.xlim(0, 2)

plt.subplot(1, 2, 2)
sns.histplot(df['Calorie_Diff'], bins=50)
plt.title('Distribution of Calorie Difference (Expected - Actual)')
plt.axvline(0, color='red', linestyle='--')
plt.xlim(-500, 500)
plt.tight_layout()
plt.savefig('calorie_discrepancy.png')
plt.close()

# Function to detect outliers using IQR
def detect_outliers(df, column, k=1.5):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return (df[column] < lower_bound) | (df[column] > upper_bound)

# Prepare cleaned dataset
df_clean = df.copy()

# Flag entries with calorie discrepancy > 20%
df_clean['Calorie_Discrepancy_Pct'] = abs((df_clean['Expected_Calories'] - df_clean['Calories']) / df_clean['Calories'] * 100)
df_clean['Calorie_Issue'] = df_clean['Calorie_Discrepancy_Pct'] > 20

# Detect outliers in nutrients
outlier_columns = ['Calories', 'Fat', 'Carbohydrates', 'Protein',
                   'Vitamin C', 'Vitamin B11', 'Calcium', 'Iron']

for col in outlier_columns:
    df_clean[f'{col}_Outlier'] = detect_outliers(df_clean, col)

# Create combined quality flag
df_clean['Data_Quality_Issue'] = (
    df_clean['Calorie_Issue'] |
    df_clean[[f'{col}_Outlier' for col in outlier_columns]].any(axis=1)
)

quality_issues = df_clean['Data_Quality_Issue'].sum()
print(f"Entries with data quality issues: {quality_issues} ({quality_issues / len(df_clean) * 100:.1f}%)")

df_reliable = df_clean[~df_clean['Data_Quality_Issue']].copy()
print(f"\nReliable dataset size: {len(df_reliable)} entries ({len(df_reliable) / len(df_clean) * 100:.1f}% of original)")

# --- FOOD CATEGORY CREATION ---
def categorize_food(food_name):
    food_name = food_name.lower()
    if any(term in food_name for term in ['juice', 'drink', 'beverage', 'tea', 'coffee']):
        return 'Beverages'
    elif any(term in food_name for term in ['fruit', 'apple', 'orange', 'banana']):
        return 'Fruits'
    elif any(term in food_name for term in ['vegetable', 'carrot', 'broccoli', 'spinach']):
        return 'Vegetables'
    elif any(term in food_name for term in ['meat', 'beef', 'chicken']):
        return 'Meats'
    elif any(term in food_name for term in ['fish', 'salmon', 'tuna']):
        return 'Seafood'
    elif any(term in food_name for term in ['dairy', 'milk', 'cheese']):
        return 'Dairy'
    elif any(term in food_name for term in ['bread', 'pasta', 'rice']):
        return 'Grains'
    elif any(term in food_name for term in ['bean', 'lentil', 'pea']):
        return 'Legumes'
    elif any(term in food_name for term in ['nut', 'seed']):
        return 'Nuts and Seeds'
    elif any(term in food_name for term in ['dessert', 'cake', 'cookie']):
        return 'Desserts and Sweets'
    else:
        return 'Other/Mixed'

df_reliable['Food_Category'] = df_reliable['food'].apply(categorize_food)

# --- PLOT FOOD CATEGORY DISTRIBUTION ---
category_counts = df_reliable['Food_Category'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Food Categories in Reliable Dataset')
plt.ylabel('Number of Items')
plt.tight_layout()
plt.savefig('food_categories.png')
plt.close()

# --- CALCULATE MACRONUTRIENT PERCENTAGES ---
df_reliable['Carb_Cal_Pct'] = df_reliable['Carbohydrates'] * 4 / df_reliable['Calories'] * 100
df_reliable['Fat_Cal_Pct'] = df_reliable['Fat'] * 9 / df_reliable['Calories'] * 100
df_reliable['Protein_Cal_Pct'] = df_reliable['Protein'] * 4 / df_reliable['Calories'] * 100

# --- PLOT MACRONUTRIENT DISTRIBUTION BY CATEGORY ---
macro_by_category = df_reliable.groupby('Food_Category')[['Carb_Cal_Pct', 'Fat_Cal_Pct', 'Protein_Cal_Pct']].mean()
plt.figure(figsize=(14, 10))
macro_by_category.plot(kind='bar', stacked=True)
plt.title('Average Macronutrient Distribution by Food Category')
plt.xlabel('Food Category')
plt.ylabel('Percentage of Calories (%)')
plt.legend(title='Macronutrient')
plt.axhline(y=100, color='black', linestyle='--')
plt.tight_layout()
plt.savefig('macronutrient_distribution.png')
plt.close()

# --- NUTRITIONAL EFFICIENCY INDEX (NEI) ---
nutrient_densities = {
    'Protein_Density': df_reliable['Protein'] / df_reliable['Calories'] * 100,
    'Fiber_Density': df_reliable['Dietary Fiber'] / df_reliable['Calories'] * 100,
    'Vitamin_C_Density': df_reliable['Vitamin C'] / df_reliable['Calories'] * 100,
    'Calcium_Density': df_reliable['Calcium'] / df_reliable['Calories'] * 100,
    'Iron_Density': df_reliable['Iron'] / df_reliable['Calories'] * 100,
    'Vitamin_B11_Density': df_reliable['Vitamin B11'] / df_reliable['Calories'] * 100
}

for name, values in nutrient_densities.items():
    df_reliable[name] = values

# Compute NEI
df_reliable['NEI'] = (
    df_reliable['Protein_Density'] * 0.3 +
    df_reliable['Fiber_Density'] * 0.2 +
    df_reliable['Vitamin_C_Density'] * 0.1 +
    df_reliable['Calcium_Density'] * 0.15 +
    df_reliable['Iron_Density'] * 0.15 +
    df_reliable['Vitamin_B11_Density'] * 0.1
)

# Completeness Score
df_reliable['Completeness_Score'] = (
    (df_reliable['Protein'] / df_reliable['Protein'].mean()) * 0.3 +
    (df_reliable['Dietary Fiber'] / df_reliable['Dietary Fiber'].mean()) * 0.2 +
    (df_reliable['Calcium'] / df_reliable['Calcium'].mean()) * 0.15 +
    (df_reliable['Iron'] / df_reliable['Iron'].mean()) * 0.15 +
    (df_reliable['Vitamin C'] / df_reliable['Vitamin C'].mean()) * 0.1 +
    (df_reliable['Vitamin B11'] / df_reliable['Vitamin B11'].mean()) * 0.1
)

# --- CLUSTERING ---

cluster_features = [
    'Protein', 'Carbohydrates', 'Fat', 'Dietary Fiber',
    'Sugars', 'Vitamin C', 'Calcium', 'Iron', 'Vitamin B11'
]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_reliable[cluster_features])

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
df_reliable['PCA1'] = pca_result[:, 0]
df_reliable['PCA2'] = pca_result[:, 1]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_reliable['Nutrient_Cluster'] = kmeans.fit_predict(scaled_features)

# Cluster names
def get_cluster_name(row):
    descriptions = {
        0: "High Sugar, Moderate Carbs",
        1: "Protein & Calcium Rich",
        2: "High Vitamin B11, High Fat",
        3: "Low Nutrient Content",
        4: "Vitamin C Rich, Low Fat"
    }
    return descriptions[row['Nutrient_Cluster']]

df_reliable['Cluster_Name'] = df_reliable.apply(get_cluster_name, axis=1)

# --- PLOT CLUSTERS ---
plt.figure(figsize=(14, 10))
cluster_colors = sns.color_palette('viridis', n_colors=5)
for i, cluster in enumerate(sorted(df_reliable['Nutrient_Cluster'].unique())):
    cluster_data = df_reliable[df_reliable['Nutrient_Cluster'] == cluster]
    plt.scatter(
        cluster_data['PCA1'],
        cluster_data['PCA2'],
        c=[cluster_colors[i]],
        label=f"Cluster {cluster}: {cluster_data['Cluster_Name'].iloc[0]}"
    )
plt.title('Food Clusters by Nutrient Content (PCA Visualization)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.tight_layout()
plt.savefig('food_clusters.png')
plt.close()

# --- RADAR CHART OF CLUSTERS ---
def radar_chart(df, categories, title):
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for i in range(len(df)):
        values = df.iloc[i].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=f"Cluster {i}")
        ax.fill(angles, values, alpha=0.1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title(title, size=15)
    plt.legend(loc='upper right')
    return fig

cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=cluster_features
)

cluster_centers_norm = cluster_centers.copy()
for col in cluster_centers.columns:
    cluster_centers_norm[col] = cluster_centers[col] / cluster_centers[col].max()

radar_fig = radar_chart(cluster_centers_norm, cluster_features, 'Nutrient Profiles of Food Clusters')
radar_fig.savefig('cluster_radar.png')
plt.close(radar_fig)

# --- FOOD COMBINATIONS OPTIMIZER ---
def nutrient_optimized_combinations(target_nutrients, n_foods=2, max_combinations=3):
    valid_foods = df_reliable[(df_reliable['Calories'] > 10)].copy()
    food_indices = list(range(min(50, len(valid_foods))))
    all_combinations = list(combinations(food_indices, n_foods))

    results = []
    for combo in all_combinations:
        foods = valid_foods.iloc[list(combo)]
        total_nutrients = foods[list(target_nutrients.keys())].sum()
        total_calories = foods['Calories'].sum()

        match_score = sum(abs(total_nutrients[n] - t) / (t if t > 0 else 1)
                         for n, t in target_nutrients.items())

        results.append({
            'foods': foods['food'].tolist(),
            'total_calories': total_calories,
            'match_score': match_score,
            **{f'total_{k}': total_nutrients[k] for k in target_nutrients}
        })

    results_df = pd.DataFrame(results)
    return results_df.nsmallest(max_combinations, 'match_score')

# Define targets
weight_loss_target = {
    'Protein': 25,
    'Dietary Fiber': 10,
    'Fat': 10,
    'Carbohydrates': 20
}

muscle_building_target = {
    'Protein': 30,
    'Carbohydrates': 40,
    'Fat': 15,
    'Iron': 5
}

heart_health_target = {
    'Dietary Fiber': 10,
    'Fat': 15,
    'Sodium': 0.5,
    'Protein': 20,
    'Vitamin C': 30
}

# Run optimization
print("\nWeight Loss Optimized Combinations:")
weight_loss_combos = nutrient_optimized_combinations(weight_loss_target, n_foods=2, max_combinations=3)
print(weight_loss_combos[['foods', 'total_calories', 'total_Protein', 'total_Dietary Fiber', 'total_Fat', 'total_Carbohydrates']])

print("\nMuscle Building Optimized Combinations:")
muscle_combos = nutrient_optimized_combinations(muscle_building_target, n_foods=2, max_combinations=3)
print(muscle_combos[['foods', 'total_calories', 'total_Protein', 'total_Carbohydrates', 'total_Fat', 'total_Iron']])

print("\nHeart Health Optimized Combinations:")
heart_combos = nutrient_optimized_combinations(heart_health_target, n_foods=2, max_combinations=3)
print(heart_combos[['foods', 'total_calories', 'total_Dietary Fiber', 'total_Fat', 'total_Sodium', 'total_Protein', 'total_Vitamin C']])

# --- END ---
print("\nAll visualizations and outputs saved.")