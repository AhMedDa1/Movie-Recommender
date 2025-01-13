from django.core.management.base import BaseCommand
from recommendation.model import Model  # Import your Model class
from recommendation.dataset import data  # Import the dataset loader
import os

# Paths to save the trained model and dataset
TRAINED_MODEL_PATH = "Data/trained_model.pkl"
DATASET_PATH = "Data/ratings_small.csv"
MOVIES_PATH = "Data/movies.csv"


class Command(BaseCommand):
    help = "Train the recommendation model and save the trained model and matrices."

    def handle(self, *args, **kwargs):
        # Step 1: Load the dataset
        if not os.path.exists(DATASET_PATH) or not os.path.exists(MOVIES_PATH):
            self.stderr.write("Error: Dataset or movies file not found.")
            return

        self.stdout.write("Loading dataset...")
        dataset = data(DATASET_PATH, MOVIES_PATH)
        train_user_ratings, train_movie_ratings, test_user_ratings, test_movie_ratings, user_map, movie_map = dataset.data_structure(
            dataset.data, test_size=0.2
        )

        # Step 2: Train the model
        self.stdout.write("Training the recommendation model...")
        model = Model(
            data=(train_user_ratings, train_movie_ratings, test_user_ratings, test_movie_ratings, user_map, movie_map),
            latent_d=10,
            lamda=0.01,
            gamma=0.01,
            tau=0.1,
        )
        model.fit(epochs=20)

        # Step 3: Save the model
        self.stdout.write("Saving the trained model...")
        model.save_model(TRAINED_MODEL_PATH)

        self.stdout.write(self.style.SUCCESS("Model trained and saved successfully."))
