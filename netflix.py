import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv("netflix_titles.csv")  # Replace with your file path

# Step 2: Keep relevant columns and drop missing values
df_clean = df[['title', 'listed_in', 'rating']].dropna()

# Step 3: Convert 'listed_in' to list of genres
df_clean['genre_list'] = df_clean['listed_in'].apply(lambda x: [g.strip() for g in x.split(',')])

# Step 4: Encode genres and ratings
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df_clean['genre_list'])
rating_encoded = pd.get_dummies(df_clean['rating'], prefix='rating')

# Step 5: Combine features
features = pd.concat([
    pd.DataFrame(genre_encoded, columns=mlb.classes_),
    rating_encoded
], axis=1)

# Step 6: Normalize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 7: Clean up any NaNs (precautionary)
features_cleaned = pd.DataFrame(scaled_features, columns=features.columns).dropna()
df_clean = df_clean.loc[features_cleaned.index].reset_index(drop=True)
features_cleaned = features_cleaned.reset_index(drop=True)

# Step 8: KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df_clean['Cluster'] = kmeans.fit_predict(features_cleaned)

# Step 9: Assign cluster names (based on common traits in each cluster)
cluster_names = {
    0: "Family & Kids",
    1: "Crime & Thrillers",
    2: "Romantic & Comedy",
    3: "Documentaries & Reality",
    4: "Action & Adventure"
}
df_clean['Cluster_Label'] = df_clean['Cluster'].map(cluster_names)

# Step 10: Reduce to 2D with PCA for plotting
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_cleaned)

plot_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
plot_df['Cluster_Label'] = df_clean['Cluster_Label']

# Step 11: Plot named clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_df, x='PCA1', y='PCA2', hue='Cluster_Label', palette='Set2')
plt.title('Netflix Show Clusters by Genre/Rating')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster Label')
plt.grid(True)
plt.tight_layout()
plt.show()
