import numpy as np
from tqdm import tqdm
from .util import utils


class Model(utils):
    def __init__(self, data, latent_d=7, lamda=0.1, gamma=0.04, tau=0.09):
        self.user_data, self.movie_data, self.test_user_data, self.test_movie_data, self.user_map, self.movie_map = data
        self.num_user = len(self.user_data)
        self.num_movie = len(self.movie_data)
        self.latent_d = latent_d
        self.lamda = lamda
        self.gamma = gamma
        self.tau = tau

        self.user_matrix = np.random.normal(0, 1 / np.sqrt(latent_d), (self.num_user, latent_d))
        self.movie_matrix = np.random.normal(0, 1 / np.sqrt(latent_d), (self.num_movie, latent_d))
        self.user_bias = np.zeros(self.num_user)
        self.movie_bias = np.zeros(self.num_movie)

    def predict(self, user_idx, movie_idx):
        """Predict the rating for a given user and movie index."""
        return (
            np.dot(self.user_matrix[user_idx], self.movie_matrix[movie_idx])
            + self.user_bias[user_idx]
            + self.movie_bias[movie_idx]
        )

    def test_model(self, num_samples=5):
        """Evaluate the model on test data and print sample predictions."""
        predictions = []
        for user_idx, ratings in enumerate(self.test_user_data):
            for movie_id, true_rating in ratings[:num_samples]:
                if movie_id in self.movie_map:
                    movie_idx = self.movie_map[movie_id]
                    pred_rating = self.predict(user_idx, movie_idx)
                    predictions.append((user_idx, movie_id, true_rating, pred_rating))
        return predictions

    def _update_users(self):
        """Update user latent factors and biases."""
        for user_idx in range(self.num_user):
            rated_movies = [
                self.movie_map[movie] for movie, _ in self.user_data[user_idx]
                if movie in self.movie_map and isinstance(self.movie_map[movie], int)
            ]
            if not rated_movies:
                continue

            R_u = np.array([rating for movie, rating in self.user_data[user_idx] if movie in self.movie_map])
            M_u = self.movie_matrix[rated_movies]

            A = self.lamda * M_u.T @ M_u + self.tau * np.eye(self.latent_d)
            b = self.lamda * M_u.T @ (R_u - self.movie_bias[rated_movies])
            self.user_matrix[user_idx] = np.linalg.solve(A, b)

            self.user_bias[user_idx] = (
                np.mean(R_u - (self.user_matrix[user_idx] @ M_u.T) - self.movie_bias[rated_movies])
            )

    def _update_movies(self):
        """Update movie latent factors and biases."""
        for movie_idx in range(self.num_movie):
            rated_users = [
                user for user, _ in self.movie_data[movie_idx]
                if isinstance(user, int) and 0 <= user < len(self.user_matrix)
            ]
            if not rated_users:
                continue

            R_m = np.array([rating for _, rating in self.movie_data[movie_idx]])
            U_m = self.user_matrix[rated_users]

            # Regularization with gamma
            A = self.lamda * U_m.T @ U_m + self.tau * np.eye(self.latent_d)
            b = self.lamda * U_m.T @ (R_m - self.user_bias[rated_users])
            self.movie_matrix[movie_idx] = np.linalg.solve(A, b)

            self.movie_bias[movie_idx] = (
                np.mean(R_m - (self.movie_matrix[movie_idx] @ U_m.T) - self.user_bias[rated_users])
            )

    def fit(self, epochs=15, progress_callback=None):
        train_metrics = {"loss": [], "rmse": []}
        test_metrics = {"loss": [], "rmse": []}

        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}...")
            self._update_users()
            self._update_movies()

            try:
                train_loss, train_rmse = self.compute_loss(
                    self.flatten_data(self.user_data, self.user_map, self.movie_map)
                )
                test_loss, test_rmse = self.compute_loss(
                    self.flatten_data(self.test_user_data, self.user_map, self.movie_map)
                )
                train_metrics["loss"].append(train_loss)
                train_metrics["rmse"].append(train_rmse)
                test_metrics["loss"].append(test_loss)
                test_metrics["rmse"].append(test_rmse)

                print(
                    f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train RMSE = {train_rmse:.4f} | "
                    f"Test Loss = {test_loss:.4f}, Test RMSE = {test_rmse:.4f}"
                )

                if progress_callback:
                    progress_callback(epoch + 1, epochs, train_loss, train_rmse)
            except Exception as e:
                print(f"Error in epoch {epoch + 1}: {str(e)}")
                break

        return train_metrics, test_metrics

    @staticmethod
    def flatten_data(data, user_map, movie_map):
        flattened = []
        for user_idx, ratings in enumerate(data):
            for movie_id, rating in ratings:
                if movie_id in movie_map:
                    flattened.append((user_idx, movie_map[movie_id], rating))
        return np.array(flattened, dtype=np.float32)


if __name__ == "__main__":
    from dataset import data

    dataset_path = "Data/ratings_small.csv"
    movies_path = "Data/movies.csv"
    dataset = data(dataset_path, movies_path)

    train_user_ratings, train_movie_ratings, test_user_ratings, test_movie_ratings, user_map, movie_map = dataset.data_structure(
        dataset.data, test_size=0.2
    )

    model = Model(
        data=(train_user_ratings, train_movie_ratings, test_user_ratings, test_movie_ratings, user_map, movie_map),
        latent_d=10,
        lamda=0.01,
        gamma=0.01,
        tau=0.1
    )

    model.fit(epochs=20)

    model.save_model("trained_model.pkl")
