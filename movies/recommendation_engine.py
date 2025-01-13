import os
import pickle

# Paths for saved data
MODEL_PATH = "Data/trained_model.pkl"
U_PATH = "Data/U_matrix.pkl"
V_PATH = "Data/V_matrix.pkl"

# Global variables to hold the loaded data
trained_model = None
U_matrix = None
V_matrix = None

def load_model():
    global trained_model, U_matrix, V_matrix

    # Load trained model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            trained_model = pickle.load(f)
    else:
        print("Trained model not found. Please train it first.")

    # Load U matrix
    if os.path.exists(U_PATH):
        with open(U_PATH, "rb") as f:
            U_matrix = pickle.load(f)
    else:
        print("U matrix not found. Please train it first.")

    # Load V matrix
    if os.path.exists(V_PATH):
        with open(V_PATH, "rb") as f:
            V_matrix = pickle.load(f)
    else:
        print("V matrix not found. Please train it first.")


def add_user_rating(user_id, movie_id, rating):
    """
    Add a new rating to the U matrix.
    This is a placeholder - integrate with the real logic.
    """
    global U_matrix, V_matrix

    if U_matrix is None or V_matrix is None:
        print("Error: Model and matrices must be loaded first.")
        return

    # Example: Update U_matrix based on user ratings
    # You can use an algorithm or placeholder to adjust the embeddings
    # Here, we're just printing for demonstration purposes
    print(f"User {user_id} rated Movie {movie_id} with {rating}.")


# Call the load function during app startup
load_model()
