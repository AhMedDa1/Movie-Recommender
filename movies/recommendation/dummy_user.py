import pickle
import numpy as np
import pandas as pd
from .model import Model
from .dataset import DataProcessor

class DummyUser(Model):
    def __init__(self, data, latent_d=10, lamda=0.1, gamma=0.1, tau=0.1):
        super().__init__(data, latent_d, lamda, gamma, tau)
        self.movies = pd.read_csv("Data/movies.csv")
        self.features_per_movie = self._initialize_features()

    def _initialize_features(self):
        """
        Create feature mappings for movies based on genres.
        """
        genres = sorted(list(set('|'.join(self.movies['genres'].dropna()).split('|'))))
        genre_map = {genre: idx for idx, genre in enumerate(genres)}
        features = []
        for movie_id, movie_idx in self.movie_map.items():
            # Dynamically resize features list if necessary
            while len(features) <= movie_idx:
                features.append([])

            movie_row = self.movies[self.movies['movieId'] == movie_id]
            if movie_row.empty:
                print(f"Warning: Movie ID {movie_id} not found in movies dataset.")
                continue
            movie_genres = movie_row['genres'].values[0].split('|')
            features[movie_idx] = [genre_map[g] for g in movie_genres if g in genre_map]



    def add_dummy_user(self, movie_ratings, iterations=1000):
        """
        Add a dummy user and optimize their latent vector and bias.
        """
        dummy_user_id = self.num_user
        self.num_user += 1

        self.user_matrix = np.vstack([self.user_matrix, np.zeros(self.latent_d)])
        self.user_bias = np.append(self.user_bias, 0.0)

        user_vector = self.user_matrix[dummy_user_id]
        user_bias = self.user_bias[dummy_user_id]

        rated_movie_indices = [
            self.movie_map[movie_id] for movie_id, _ in movie_ratings if movie_id in self.movie_map
        ]
        ratings = np.array([rating for movie_id, rating in movie_ratings if movie_id in self.movie_map])

        for _ in range(iterations):
            # Update bias
            residuals = ratings - np.dot(user_vector, self.movie_matrix[rated_movie_indices].T) - self.movie_bias[rated_movie_indices]
            user_bias = np.mean(residuals)

            # Update latent vector
            left_matrix = self.lamda * self.movie_matrix[rated_movie_indices].T @ self.movie_matrix[rated_movie_indices]
            right_vector = self.lamda * self.movie_matrix[rated_movie_indices].T @ (ratings - user_bias - self.movie_bias[rated_movie_indices])
            user_vector = np.linalg.solve(left_matrix + self.tau * np.eye(self.latent_d), right_vector)

        self.user_matrix[dummy_user_id] = user_vector
        self.user_bias[dummy_user_id] = user_bias

        return dummy_user_id

    def predict_with_features(self, user_id, top_n=20, fact=0.9):
        """
        Predict ratings for all movies, considering features.
        """
        user_vector = self.user_matrix[user_id]
        user_bias = self.user_bias[user_id]

        predictions = (
            np.dot(user_vector, self.movie_matrix.T) + self.movie_bias * fact + user_bias
        )

        # Include rated movies in recommendations
        recommendations = [
            (movie_id, rating) for movie_id, rating in enumerate(predictions)
        ]

        recommendations.sort(key=lambda x: x[-1], reverse=True)

        # Fetch movie metadata for the top N recommendations
        top_recommendations = []
        for movie_id, rating in recommendations[:top_n]:
            movie_row = self.movies[self.movies['movieId'] == movie_id]
            if not movie_row.empty:
                title = movie_row['title'].values[0]
                genres = movie_row['genres'].values[0]
                top_recommendations.append((title, genres, rating))

        return top_recommendations


# if __name__ == "__main__":
#     ratings_path = "Data/ratings.csv"
#     movies_path = "Data/movies.csv"

#     dataset = data(ratings_path, movies_path)

#     toy_story_id = dataset.get_movie_id("Toy Story")
#     if toy_story_id is None:
#         print("Could not find 'Toy Story' in the dataset.")
#         exit()
#     print(f"The ID for 'Toy Story' is {toy_story_id}")

#     model = DummyUser(
#         data=dataset.data_structure(dataset.data, test_size=0.2),
#         latent_d=10,
#         lamda=0.01,  
#         gamma=0.01,  
#         tau=0.1    
#     )
#     print("Training the model...")
#     model.fit(epochs=10)

#     print("Adding a dummy user...")
#     dummy_user_id = model.add_dummy_user([(toy_story_id, 5.0)])

#     print("\nGenerating recommendations with features for the dummy user...")
#     recommendations_with_features = model.predict_with_features(dummy_user_id, top_n=20, fact=0.9)

#     print("\nTop Recommendations for the Dummy User (With Features):")
#     for title, genres, predicted_rating in recommendations_with_features:
#         print(f"Title: {title}, Genres: {genres}, Predicted Rating: {predicted_rating:.2f}")
