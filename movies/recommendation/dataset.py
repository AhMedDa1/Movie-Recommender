import pandas as pd
import numpy as np
import os
import pickle
from django.conf import settings


class DataProcessor:
    def __init__(self, ratings_path, movies_path):
        self.ratings_path = ratings_path
        self.movies_path = movies_path

        self.ratings = pd.read_csv(self.ratings_path).drop(columns="timestamp")
        self.movies = pd.read_csv(self.movies_path)

        # Precomputed datasets
        self.train_data = None
        self.test_data = None
        self.user_map = None
        self.movie_map = None
        self.movie_genres = self._extract_genres()

    def _extract_genres(self):
        """Extract genres for movies."""
        genres_map = {}
        for _, row in self.movies.iterrows():
            movie_id = row["movieId"]
            genres = row["genres"].split("|") if pd.notnull(row["genres"]) else []
            genres_map[movie_id] = genres
        return genres_map

    def save_preprocessed_data(self):
        """Preprocess and save key datasets for faster reuse."""
        CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cached_data')
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Train/Test Data
        train_user, train_movie, test_user, test_movie, user_map, movie_map = self.data_structure(
            self.ratings.to_numpy(), test_size=0.2
        )

        # Save preprocessed data
        with open(os.path.join(CACHE_DIR, "cached_train_data.pkl"), "wb") as f:
            pickle.dump((train_user, train_movie), f)  # Save both train datasets
        with open(os.path.join(CACHE_DIR, "cached_test_data.pkl"), "wb") as f:
            pickle.dump((test_user, test_movie), f)  # Save both test datasets
        with open(os.path.join(CACHE_DIR, "cached_user_map.pkl"), "wb") as f:
            pickle.dump(user_map, f)
        with open(os.path.join(CACHE_DIR, "cached_movie_map.pkl"), "wb") as f:
            pickle.dump(movie_map, f)

        # Store data in class attributes for immediate use
        self.train_data = (train_user, train_movie)
        self.test_data = (test_user, test_movie)
        self.user_map = user_map
        self.movie_map = movie_map


    def load_cached_data(self):
        """Load preprocessed datasets."""
        CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cached_data')

        with open(os.path.join(CACHE_DIR, "cached_train_data.pkl"), "rb") as f:
            train_data = pickle.load(f)
        with open(os.path.join(CACHE_DIR, "cached_test_data.pkl"), "rb") as f:
            test_data = pickle.load(f)
        with open(os.path.join(CACHE_DIR, "cached_user_map.pkl"), "rb") as f:
            user_map = {int(k): v for k, v in pickle.load(f).items()}  # Cast keys to integers
        with open(os.path.join(CACHE_DIR, "cached_movie_map.pkl"), "rb") as f:
            movie_map = {int(k): v for k, v in pickle.load(f).items()}  # Cast keys to integers

        return train_data[0], train_data[1], test_data[0], test_data[1], user_map, movie_map



    def get_movie_name(self, movie_id):
        """Get movie name by ID."""
        row = self.movies[self.movies["movieId"] == movie_id]
        return row["title"].values[0] if not row.empty else None

    def get_movie_id(self, title):
        """Get movie ID by title."""
        row = self.movies[self.movies["title"].str.contains(title, case=False, na=False)]
        return row["movieId"].values[0] if not row.empty else None

    def data_structure(self, data, test_size=0.2):
        """Split data into train/test sets and create user and movie maps."""
        np.random.shuffle(data)
        user_map, movie_map = {}, {}
        user_rating, movie_rating = [], []
        data_by_user_train, data_by_user_test = [], []
        data_by_movie_train, data_by_movie_test = [], []

        user_ids, movie_ids, ratings = data[:, 0], data[:, 1], data[:, 2]

        for i in range(len(data)):
            user, movie, rating = user_ids[i], movie_ids[i], ratings[i]

            if user not in user_map:
                user_map[user] = len(user_rating)
                user_rating.append([])
                data_by_user_train.append([])
                data_by_user_test.append([])

            if movie not in movie_map:
                movie_map[movie] = len(movie_rating)
                movie_rating.append([])
                data_by_movie_train.append([])
                data_by_movie_test.append([])

            user_idx, movie_idx = user_map[user], movie_map[movie]
            user_rating[user_idx].append((movie, rating))
            movie_rating[movie_idx].append((user_idx, rating))

        for user_idx, ratings in enumerate(user_rating):
            if len(ratings) > 1:
                np.random.shuffle(ratings)
                split_point = int(len(ratings) * (1 - test_size))
                data_by_user_train[user_idx].extend(ratings[:split_point])
                data_by_user_test[user_idx].extend(ratings[split_point:])
            else:
                data_by_user_train[user_idx].extend(ratings)

        for movie_idx, ratings in enumerate(movie_rating):
            if len(ratings) > 1:
                np.random.shuffle(ratings)
                split_point = int(len(ratings) * (1 - test_size))
                data_by_movie_train[movie_idx].extend(ratings[:split_point])
                data_by_movie_test[movie_idx].extend(ratings[split_point:])
            else:
                data_by_movie_train[movie_idx].extend(ratings)

        return (
            data_by_user_train,
            data_by_movie_train,
            data_by_user_test,
            data_by_movie_test,
            user_map,
            movie_map,
        )

    @staticmethod
    def get_ratings_data():
        """Retrieve ratings data."""
        ratings_path = os.path.join(settings.BASE_DIR, "Data/ratings.csv")
        return pd.read_csv(ratings_path)

    @staticmethod
    def get_movies_data():
        """Retrieve movies data."""
        movies_path = os.path.join(settings.BASE_DIR, "Data/movies.csv")
        return pd.read_csv(movies_path)

    # Visualization and Plotting
    @staticmethod
    def plot_movies_by_genre():
        """Prepare data for plotting movies by genre."""
        movies_data = DataProcessor.get_movies_data()
        genres = movies_data["genres"].str.split("|").explode().value_counts()
        return genres

    @staticmethod
    def plot_ratings_per_movie():
        """Prepare data for plotting the number of ratings per movie."""
        ratings_data = DataProcessor.get_ratings_data()
        ratings_count = ratings_data.groupby("movieId").size()
        return ratings_count

    @staticmethod
    def plot_genre_distributions():
        """Prepare data for plotting distributions of genres."""
        movies_data = DataProcessor.get_movies_data()
        genres = movies_data["genres"].str.split("|").explode()
        return genres

    @staticmethod
    def plot_average_rating_by_genre():
        """Prepare data for plotting average ratings by genre."""
        ratings_data = DataProcessor.get_ratings_data()
        movies_data = DataProcessor.get_movies_data()
        merged_data = pd.merge(ratings_data, movies_data, on="movieId")
        merged_data["genres"] = merged_data["genres"].str.split("|")
        merged_data_exploded = merged_data.explode("genres")
        avg_rating_by_genre = merged_data_exploded.groupby("genres")["rating"].mean().sort_values()
        return avg_rating_by_genre
