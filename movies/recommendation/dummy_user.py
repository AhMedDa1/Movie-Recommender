# file: recommendation/dummy_user.py
import numpy as np
import pandas as pd
from .model import Model

class DummyUser(Model):
    def __init__(self, data, latent_d=10, lamda=0.1, gamma=0.1, tau=0.1):
        super().__init__(data, latent_d, lamda, gamma, tau)
        self.movies = pd.read_csv("Data/movies.csv")
        
        # We'll wait to set self.V until after we load movie_matrix from the pickle
        self.V = None

        # NO re-processing of movie_map or anything that changes # of columns
        # self.movie_idx_map = self.movie_map  # We'll do that after we get the actual loaded data.

        self.features_per_movie = []  # If you need it, do it carefully after everything is loaded.

    def calculate_dummy_user_bias(self, user_dummy, iteration, dummy_user_latent):
        bias_sum = 0
        item_counter = 0
        for (movie_id, rating) in user_dummy:
            movie_index = self.movie_map[movie_id]

            if iteration == 0:
                # λ * (r_ui - b_i)
                bias_sum += self.lamda * (rating - self.movie_bias[movie_index])
            else:
                # λ * (r_ui - (u_latent^T * V_i + b_i))
                bias_sum += self.lamda * (
                    rating
                    - (
                        np.dot(dummy_user_latent.T, self.V[:, movie_index])
                        + self.movie_bias[movie_index]
                    )
                )
            item_counter += 1

        if item_counter > 0:
            return bias_sum / ((self.lamda * item_counter) + self.tau)
        return 0

    def update_user_latent_dummy(self, user_dummy, dummy_user_bias):
        k = self.latent_d
        x = np.zeros(k)
        y = np.zeros((k, k))

        for (movie_id, actual_rating) in user_dummy:
            if movie_id not in self.movie_map:
                # If user rated a movie that wasn't in training data:
                # skip or handle differently.
                continue

            movie_index = self.movie_map[movie_id]

            # Check for out-of-bounds
            if movie_index >= self.V.shape[1]:
                # This is the actual problem scenario. You can raise an error or skip
                raise IndexError(f"Movie index {movie_index} out of range for V with shape {self.V.shape}")

            error = actual_rating - dummy_user_bias - self.movie_bias[movie_index]
            x += self.V[:, movie_index] * error
            y += np.outer(self.V[:, movie_index], self.V[:, movie_index])

        y += np.identity(k) * self.tau

        return np.linalg.solve(self.lamda * y, self.lamda * x)

    # Optional: If you want to build your features or set self.V after loading, do it in a method:
    def finalize_init(self):
        # For instance, if your movie_matrix is (numMovies, latent_d):
        # self.V becomes (latent_d, numMovies)
        self.V = self.movie_matrix.T
        # If you want a direct reference
        # self.movie_idx_map = self.movie_map
