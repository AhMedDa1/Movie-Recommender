# file: recommendation/dummy_user.py
import numpy as np
import pandas as pd
from .model import Model

class DummyUser(Model):
    def __init__(self, data, latent_d=10, lamda=0.1, gamma=0.1, tau=0.1):
        super().__init__(data, latent_d, lamda, gamma, tau)
        self.movies = pd.read_csv("Data/movies.csv")
        
        # We'll set self.V after we confirm self.movie_matrix's shape
        self.V = None

        # No re-processing of movie_map in init
        self.features_per_movie = []

    def calculate_dummy_user_bias(self, user_dummy, iteration, dummy_user_latent):
        """
        Calculate the bias for the dummy user.
        We expect self.V to be shape (latent_d, numMovies), so indexing V[:, idx] is (latent_d,).
        """
        bias_sum = 0
        item_counter = 0

        for (movie_id, rating) in user_dummy:
            movie_index = self.movie_map[movie_id]
            if iteration == 0:
                # λ * (r_ui - b_i)
                bias_sum += self.lamda * (rating - self.movie_bias[movie_index])
            else:
                # λ * (r_ui - (u_latent^T * V_i + b_i))
                # Single-movie vector: self.V[:, movie_index] => shape (latent_d,)
                pred = np.dot(dummy_user_latent, self.V[:, movie_index]) + self.movie_bias[movie_index]
                bias_sum += self.lamda * (rating - pred)

            item_counter += 1

        if item_counter > 0:
            return bias_sum / ((self.lamda * item_counter) + self.tau)
        return 0

    def update_user_latent_dummy(self, user_dummy, dummy_user_bias):
        """
        Update the dummy user's latent vector by normal equations.
        We do x += V_i * error, y += V_i outer V_i, then solve for new user vector.
        """
        k = self.latent_d
        x = np.zeros(k)
        y = np.zeros((k, k))

        for (movie_id, actual_rating) in user_dummy:
            if movie_id not in self.movie_map:
                # skip movies that weren't in training
                continue

            movie_index = self.movie_map[movie_id]
            if movie_index >= self.V.shape[1]:
                raise IndexError(f"Movie index {movie_index} out of range for V with shape {self.V.shape}")

            # Single movie vector => shape (k,)
            v_m = self.V[:, movie_index]
            error = actual_rating - dummy_user_bias - self.movie_bias[movie_index]

            x += v_m * error
            y += np.outer(v_m, v_m)

        # Add τ * I
        y += np.identity(k) * self.tau

        # Solve for user vector
        return np.linalg.solve(self.lamda * y, self.lamda * x)

    def finalize_init(self):
        """
        Transpose self.movie_matrix so that each column is a single movie vector:
        if self.movie_matrix is (9774, 10), self.V becomes (10, 9774),
        so self.V[:, idx] is (10,).
        """
        if self.movie_matrix.shape[1] == self.latent_d:
            self.V = self.movie_matrix.T
        else:
            self.V = self.movie_matrix
