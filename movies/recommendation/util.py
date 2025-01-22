import numpy as np
import pickle

class utils:

    def save_model(self, file_path):
        param = {
            "user_matrix": self.user_matrix,
            "movie_matrix": self.movie_matrix,
            "user_bias": self.user_bias,
            "movie_bias": self.movie_bias,
            "user_map": self.user_map,
            "movie_map": self.movie_map,
        }
        if not self.user_map or not self.movie_map:
            print("Error: Missing user_map or movie_map.")
        with open(file_path, "wb") as f:
            pickle.dump(param, f)


    def save_recommendations(self, recommendations, file_path):
        with open(file_path, "w") as f:
            for movie_id, rating in recommendations:
                f.write(f"{movie_id},{rating}\n")

    

    def compute_loss(self, data):
        user_ids, movie_ids, ratings = data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2]
        valid_idx = (user_ids < self.num_user) & (movie_ids < self.num_movie)
        user_ids, movie_ids, ratings = user_ids[valid_idx], movie_ids[valid_idx], ratings[valid_idx]

        pred = (
            np.sum(self.user_matrix[user_ids] * self.movie_matrix[movie_ids], axis=1)
            + self.user_bias[user_ids]
            + self.movie_bias[movie_ids]
        )
        errors = ratings - pred

        # Compute RMSE and Loss
        loss = (
            np.sum(errors**2) +
            self.lamda * (np.sum(self.user_bias**2) + np.sum(self.movie_bias**2)) +
            self.tau * (np.sum(self.user_matrix**2) + np.sum(self.movie_matrix**2))
        )
        rmse = np.sqrt(np.mean(errors**2))

        return loss, rmse
