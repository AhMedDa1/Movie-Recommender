import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import pickle

# Load the trained parameters
with open("trained_model.pkl", "rb") as f:
    params = pickle.load(f)

user_matrix = params["user_matrix"]
movie_matrix = params["movie_matrix"]
movie_bias = params["movie_bias"]
movie_map = params["movie_map"]

# Load movie titles and genres
movies = pd.read_csv("Data/movies.csv")

# Reverse the movie_map
reverse_movie_map = {v: k for k, v in movie_map.items()}

# Extract movie titles corresponding to embeddings
movie_titles = [
    movies[movies["movieId"] == reverse_movie_map[i]]["title"].values[0]
    if reverse_movie_map[i] in movies["movieId"].values else f"Unknown {i}"
    for i in range(movie_matrix.shape[0])
]

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(movie_matrix)

# Plot the reduced embeddings
plt.figure(figsize=(12, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.6, s=10)

# Annotate some points for clarity
for i in range(0, len(movie_titles), len(movie_titles) // 100):  # Sample for clarity
    plt.annotate(
        movie_titles[i],
        (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
        fontsize=8,
        alpha=0.75,
    )

plt.title("Movie Feature Embeddings (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(alpha=0.3)

# Save the plot
output_file = "feature_embeddings_corrected.png"
plt.savefig(output_file)
print(f"Feature embeddings plot saved to {output_file}")

plt.show()
