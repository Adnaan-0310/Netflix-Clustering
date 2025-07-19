import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit page config
st.set_page_config(page_title="Netflix Show Clustering", layout="wide")
st.title("🎬 Netflix Show Clustering with K-Means")

# File uploader
uploaded_file = st.file_uploader("Upload Netflix CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head())

    # Preprocessing
    df_clean = df[['title', 'listed_in', 'rating']].dropna()
    df_clean['genre_list'] = df_clean['listed_in'].apply(lambda x: [g.strip() for g in x.split(',')])

    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df_clean['genre_list'])
    rating_encoded = pd.get_dummies(df_clean['rating'], prefix='rating')
    features = pd.concat([pd.DataFrame(genre_encoded, columns=mlb.classes_), rating_encoded], axis=1)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    features_cleaned = pd.DataFrame(scaled_features, columns=features.columns).dropna()

    df_clean = df_clean.loc[features_cleaned.index].reset_index(drop=True)
    features_cleaned = features_cleaned.reset_index(drop=True)

    # Sidebar for cluster input
    n_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=5)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_clean['Cluster'] = kmeans.fit_predict(features_cleaned)

    # Automatically assign cluster names based on most common genre per cluster
    cluster_names = {}
    for cluster_num in df_clean['Cluster'].unique():
        cluster_genres = df_clean[df_clean['Cluster'] == cluster_num]['genre_list'].explode()
        top_genre = cluster_genres.value_counts().idxmax()
        cluster_names[cluster_num] = f"{top_genre} Focus"

    df_clean['Cluster_Label'] = df_clean['Cluster'].map(cluster_names)

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_cleaned)
    plot_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    plot_df['Cluster_Label'] = df_clean['Cluster_Label']

    # Plot clusters
    st.subheader("📊 PCA Cluster Plot (with Labels)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=plot_df, x='PCA1', y='PCA2', hue='Cluster_Label', palette='Set2', ax=ax)
    ax.set_title("Netflix Show Clusters by Genre")
    ax.grid(True)
    st.pyplot(fig)

    # Show shows in selected cluster
    st.subheader("📂 Shows by Cluster")
    selected_label = st.selectbox("Choose a Cluster to View Shows", sorted(df_clean['Cluster_Label'].unique()))
    cluster_df = df_clean[df_clean['Cluster_Label'] == selected_label][['title', 'listed_in', 'rating']]
    st.dataframe(cluster_df)

else:
    st.info("📥 Please upload a Netflix CSV file to begin.")
