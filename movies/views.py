from django.contrib.auth.models import User
from django.contrib import messages
from django.shortcuts import render, redirect
import pandas as pd  
import requests
import numpy as np
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout, authenticate, login
from .recommendation_engine import U_matrix, trained_model
from .recommendation.dataset import DataProcessor
import matplotlib.pyplot as plt
import io
import base64
import pickle
import os
import seaborn as sns
from django.views.decorators.csrf import csrf_exempt
from .recommendation.model import Model
from .recommendation.dummy_user import DummyUser
from threading import Lock
import time
from contextlib import redirect_stdout, redirect_stderr
from django.http import JsonResponse, HttpResponseBadRequest





MOVIES_FILE = 'Data/movies.csv'

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            # Redirect to the page the user was trying to access
            next_url = request.GET.get('next', 'main')  # Default to 'main' if 'next' is not provided
            return redirect(next_url)
        else:
            return render(request, 'movies/login.html', {'error': 'Invalid username or password'})
    return render(request, 'movies/login.html')


def register_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('register')

        try:
            # Create user in Django
            new_user = User.objects.create_user(username=username, password=password)
            
            # Add dummy user embedding for the new user
            user_index = new_user.id  # Use the user's ID as the index
            if U_matrix is not None:
                # Append a default embedding for the new user
                U_matrix[user_index] = [0.0] * len(U_matrix[0])  # Adjust based on U's dimensions

            messages.success(request, "Account created successfully!")
            return redirect('login')
        except Exception as e:
            messages.error(request, f"Error: {str(e)}")
            return redirect('register')

    return render(request, 'movies/register.html')


def fetch_tmdb_movies():
    """Fetch popular movies from TMDB API."""
    url = "https://api.themoviedb.org/3/movie/popular"
    headers = {
        "Authorization": f"Bearer {settings.TMDB_READ_ACCESS_TOKEN}",  # Use settings for API token
        "Content-Type": "application/json;charset=utf-8",
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # Extract relevant details (title, overview, poster_path)
        movies = [
            {
                "title": movie["title"],
                "description": movie["overview"],
                "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}",
            }
            for movie in data["results"]
        ]
        return movies
    else:
        print(f"Error: Unable to fetch movies from TMDB. Status Code: {response.status_code}")
        return []

@login_required
def main_page_view(request):
    query = request.GET.get("query", "")  # Capture search query
    page = int(request.GET.get("page", 1))  # Capture current page number
    per_page = 9  # Initial number of movies to show
    load_more_per_page = 3  # Number of movies for each "Show More"

    # Fetch movies from TMDB API
    all_movies = fetch_tmdb_movies()

    # Handle autocomplete request
    if request.GET.get("autocomplete", False):
        suggestions = [{"title": movie["title"]} for movie in all_movies if query.lower() in movie["title"].lower()][:10]
        return JsonResponse(suggestions, safe=False)

    # Filter movies based on search query
    if query:
        filtered_movies = [movie for movie in all_movies if query.lower() in movie["title"].lower()]
    else:
        filtered_movies = all_movies

    # Handle "load more" request
    if request.GET.get("load_more", False):
        start_index = (page - 1) * load_more_per_page
        end_index = start_index + load_more_per_page
        return JsonResponse(filtered_movies[start_index:end_index], safe=False)

    # Limit initial movies
    start_index = 0
    end_index = per_page
    paginated_movies = filtered_movies[start_index:end_index]

    context = {
        "movies": paginated_movies,
        "query": query,
        "has_more": len(filtered_movies) > end_index,  # Show "Show More" button if more movies exist
    }
    return render(request, 'movies/main.html', context)

@login_required
@login_required
def rate_movies_view(request):
    query = request.GET.get("query", "")
    page = int(request.GET.get("page", 1))
    per_page = 3
    all_movies = fetch_tmdb_movies()

    # Autocomplete and "Load More" requests
    if request.GET.get("load_more", False):
        start_index = (page - 1) * per_page
        return JsonResponse(all_movies[start_index:start_index + per_page], safe=False)

    # Handle POST submission
    if request.method == "POST":
        movie_ratings = []
        for key, value in request.POST.items():
            if key.startswith("rating_") and value.isdigit():
                movie_ratings.append((int(key.split("_")[1]), int(value)))

        if not movie_ratings:
            messages.error(request, "No ratings were submitted. Please rate at least one movie.")
            return redirect('rate')

        # Add dummy user and save session
        try:
            with open("model_matrices.pkl", "rb") as f:
                model_data = pickle.load(f)

            model = DummyUser(
                data=([[]], [[]], [[]], [[]], model_data["user_map"], model_data["movie_map"]),
                latent_d=len(model_data["U_matrix"][0]),
                lamda=0.01, gamma=0.01, tau=0.1
            )
            model.user_matrix = model_data["U_matrix"]
            model.movie_matrix = model_data["V_matrix"]
            model.user_bias = model_data["user_bias"]
            model.movie_bias = model_data["movie_bias"]

            dummy_user_id = model.add_dummy_user(movie_ratings)
            request.session["dummy_user_id"] = dummy_user_id
        except FileNotFoundError:
            return JsonResponse({"error": "Model not trained yet."})
        except Exception as e:
            return JsonResponse({"error": f"An error occurred: {e}"})

        return redirect('recommendations')

    # Filter and paginate movies
    filtered_movies = [movie for movie in all_movies if query.lower() in movie["title"].lower()] if query else all_movies
    start_index = (page - 1) * per_page
    paginated_movies = filtered_movies[start_index:start_index + per_page]

    context = {
        "movies": paginated_movies,
        "query": query,
        "total_movies": len(all_movies),
    }
    return render(request, "movies/rate.html", context)



@login_required
def recommendations_view(request):
    movie_ratings = request.session.get("movie_ratings", {})
    dummy_user_id = request.session.get("dummy_user_id")

    if not os.path.exists("trained_model.pkl"):
        return JsonResponse({"error": "Model not trained yet."})

    # Load the model
    with open("trained_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    U_matrix = model_data["user_matrix"]
    V_matrix = model_data["movie_matrix"]
    user_bias = model_data["user_bias"]
    movie_bias = model_data["movie_bias"]
    movie_map = model_data["movie_map"]

    # Debug logs
    print(f"Loaded Model: U_matrix shape: {U_matrix.shape}, V_matrix shape: {V_matrix.shape}")
    print(f"Dummy User ID: {dummy_user_id}, User Bias: {user_bias[dummy_user_id]}")

    # Fetch movies and recommend
    all_movies = fetch_tmdb_movies()
    recommended_movies = []

    for movie in all_movies:
        movie_id = movie.get("id")
        movie_idx = movie_map.get(movie_id)
        if movie_idx and movie_id not in movie_ratings:
            try:
                pred_rating = (
                    np.dot(U_matrix[dummy_user_id], V_matrix[movie_idx]) 
                    + user_bias[dummy_user_id] 
                    + movie_bias[movie_idx]
                )
                recommended_movies.append({
                    "title": movie["title"],
                    "poster": movie["poster"],
                    "description": movie["description"],
                    "predicted_rating": round(pred_rating, 2),
                })
            except Exception as e:
                print(f"Error predicting rating for movie ID {movie_id}: {e}")
                continue

    # Sort recommendations by predicted rating
    recommended_movies.sort(key=lambda x: x["predicted_rating"], reverse=True)

    # Debug recommendations
    print(f"Top Recommendations: {recommended_movies[:10]}")

    context = {"movies": recommended_movies[:20]}
    return render(request, "movies/recommendations.html", context)



def logout_view(request):
    logout(request)
    return redirect('login')


## Initialize dataset processor
DATA_PATH = os.path.join(os.path.dirname(__file__), '../Data/')
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cached_data')

processor = DataProcessor(
    ratings_path=os.path.join(DATA_PATH, 'ratings.csv'),
    movies_path=os.path.join(DATA_PATH, 'movies.csv'),
)

os.makedirs(CACHE_DIR, exist_ok=True)
cached_files = [
    os.path.join(CACHE_DIR, "cached_train_data.pkl"),
    os.path.join(CACHE_DIR, "cached_test_data.pkl"),
    os.path.join(CACHE_DIR, "cached_user_map.pkl"),
    os.path.join(CACHE_DIR, "cached_movie_map.pkl"),
]

if all(os.path.exists(f) for f in cached_files):
    processor.load_cached_data()
else:
    processor.save_preprocessed_data()

def plot_to_base64(plot_func):
    plt.figure(figsize=(10, 6))
    plot_func()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    plt.close()
    return image_base64

def datasets_view(request):
    """View to handle datasets and visualizations."""
    action = request.GET.get("action", "")
    search_query = request.GET.get("search", "").strip()
    user_id = request.GET.get("user_id", None)
    page = int(request.GET.get("page", 1))
    per_page = 10

    data = pd.DataFrame()
    plot_image = None
    message = ""
    description = ""

    if action == "movies":
        movies = processor.movies[["movieId", "title", "genres"]]
        if search_query:
            movies = movies[movies["title"].str.contains(search_query, case=False)]
        message = f"Total Movies: {len(movies)}"
        description = "This table contains movie details including genres."
        total_pages = len(movies) // per_page + (1 if len(movies) % per_page > 0 else 0)
        data = movies.iloc[(page - 1) * per_page: page * per_page]
        data = data.to_dict(orient="records")

    elif action == "ratings":
        ratings = processor.ratings.merge(processor.movies, on="movieId", how="inner")
        ratings = ratings[["userId", "movieId", "title", "rating"]]
        if search_query:
            ratings = ratings[ratings["title"].str.contains(search_query, case=False)]
        message = f"Total Ratings: {len(ratings)}"
        description = "This table shows user ratings for movies."
        total_pages = len(ratings) // per_page + (1 if len(ratings) % per_page > 0 else 0)
        data = ratings.iloc[(page - 1) * per_page: page * per_page]
        data = data.to_dict(orient="records")

    elif action == "reprocess":
        try:
            # Reprocess the data
            processor.save_preprocessed_data()
            message = "Dataset reprocessed successfully! Cached data has been updated."
            description = "Reprocessed the training and testing data and updated the cached files."
        except Exception as e:
            message = f"Reprocessing failed: {str(e)}"
            description = "An error occurred while reprocessing the dataset."


    elif action == "train":
        train_user, _, _, _, user_map, _ = processor.load_cached_data()
        message = f"Train Data: List of lists of tuples (movieId, rating). Shape: {len(train_user)} users."
        description = "Training dataset for user ratings and movies."

        if user_id:
            try:
                user_id = int(user_id)
                if user_id in user_map:
                    user_index = user_map[user_id]
                    user_data = train_user[user_index]
                    total_pages = len(user_data) // per_page + (1 if len(user_data) % per_page > 0 else 0)
                    user_data_paginated = user_data[(page - 1) * per_page: page * per_page]
                    data = pd.DataFrame(user_data_paginated, columns=["movieId", "rating"])
                    data["title"] = data["movieId"].apply(lambda x: processor.get_movie_name(x))
                    data = data.to_dict(orient="records")
                    message = f"Train Data for User ID {user_id}. Total Movies: {len(user_data)}"
                else:
                    message = f"No Train Data Found for User ID {user_id}."
            except Exception as e:
                message = f"Error: {str(e)}"

    elif action == "test":
        _, _, test_user, _, user_map, _ = processor.load_cached_data()
        message = f"Test Data: List of lists of tuples (movieId, rating). Shape: {len(test_user)} users."
        description = "Testing dataset for user ratings and movies."

        if user_id:
            try:
                user_id = int(user_id)
                if user_id in user_map:
                    user_index = user_map[user_id]
                    user_data = test_user[user_index]
                    total_pages = len(user_data) // per_page + (1 if len(user_data) % per_page > 0 else 0)
                    user_data_paginated = user_data[(page - 1) * per_page: page * per_page]
                    data = pd.DataFrame(user_data_paginated, columns=["movieId", "rating"])
                    data["title"] = data["movieId"].apply(lambda x: processor.get_movie_name(x))
                    data = data.to_dict(orient="records")
                    message = f"Test Data for User ID {user_id}. Total Movies: {len(user_data)}"
                else:
                    message = f"No Test Data Found for User ID {user_id}."
            except Exception as e:
                message = f"Error: {str(e)}"

    elif action == "plot_power_law":
        def plot_func():
            degree_counts = processor.ratings["movieId"].value_counts()
            clusters = degree_counts.value_counts().sort_index()
            plt.loglog(clusters.index, clusters.values, marker="o", linestyle="None")
            plt.title("Power Law: Distribution of Ratings per Movie")
            plt.xlabel("Number of Ratings per Movie")
            plt.ylabel("Number of Movies")
        plot_image = plot_to_base64(plot_func)

    elif action == "plot_combination":
        plot_type = request.GET.get("plot_type", "")
        merged_data = processor.ratings.merge(processor.movies, on="movieId", how="inner")
        merged_data["genres"] = merged_data["genres"].str.split('|')
        merged_data_exploded = merged_data.explode('genres')

        if plot_type == "ratings_distribution":
            def plot_func():
                sns.histplot(merged_data["rating"], bins=10, kde=False)
                plt.title("Distribution of Ratings")
                plt.xlabel("Rating")
                plt.ylabel("Frequency")
            plot_image = plot_to_base64(plot_func)

        elif plot_type == "ratings_log":
            def plot_func():
                degree_counts = merged_data["movieId"].value_counts()
                clusters = degree_counts.value_counts().sort_index()
                plt.loglog(clusters.index, clusters.values, marker="o", linestyle="None")
                plt.title("Power Law: Distribution of Ratings per Movie")
                plt.xlabel("Number of Ratings per Movie")
                plt.ylabel("Number of Movies")
            plot_image = plot_to_base64(plot_func)

        elif plot_type == "movies_genre_distribution":
            def plot_func():
                genres_count = merged_data_exploded["genres"].value_counts()
                sns.barplot(y=genres_count.index, x=genres_count.values, palette="viridis")
                plt.title("Distribution of Movies by Genre")
                plt.xlabel("Number of Movies")
                plt.ylabel("Genre")
            plot_image = plot_to_base64(plot_func)

        elif plot_type == "average_rating_genre":
            def plot_func():
                avg_rating_by_genre = merged_data_exploded.groupby("genres")["rating"].mean().sort_values()
                sns.barplot(x=avg_rating_by_genre.values, y=avg_rating_by_genre.index, palette="coolwarm")
                plt.title("Average Rating by Genre")
                plt.xlabel("Average Rating")
                plt.ylabel("Genre")
            plot_image = plot_to_base64(plot_func)

        elif plot_type == "rating_count_genre":
            def plot_func():
                rating_count_by_genre = merged_data_exploded.groupby("genres")["rating"].count().sort_values()
                sns.barplot(x=rating_count_by_genre.values, y=rating_count_by_genre.index, palette="magma")
                plt.title("Rating Count by Genre")
                plt.xlabel("Number of Ratings")
                plt.ylabel("Genre")
            plot_image = plot_to_base64(plot_func)

        elif plot_type == "top_movies":
            def plot_func():
                top_movies = merged_data.groupby("title")["rating"].count().sort_values(ascending=False).head(10)
                sns.barplot(y=top_movies.index, x=top_movies.values, palette="Blues_d")
                plt.title("Top 10 Most Popular Movies (by Number of Ratings)")
                plt.xlabel("Number of Ratings")
                plt.ylabel("Movie Title")
            plot_image = plot_to_base64(plot_func)

    elif action == "map":
        movies = processor.movies[["movieId", "title"]].copy()
        movies["index"] = range(len(movies))
        if search_query:
            movies = movies[movies["title"].str.contains(search_query, case=False)]
        message = f"Total Movies in Map: {len(movies)}"
        description = "This table maps movies to their IDs and indices."
        total_pages = len(movies) // per_page + (1 if len(movies) % per_page > 0 else 0)
        data = movies.iloc[(page - 1) * per_page: page * per_page]
        data = data.to_dict(orient="records")

    context = {
        "data": data if isinstance(data, list) and len(data) > 0 else [],
        "plot_image": plot_image,
        "message": message,
        "description": description,
        "action": action,
        "next_page": page + 1 if 'total_pages' in locals() and page < total_pages else None,
    }

    return render(request, "movies/datasets.html", context)


@csrf_exempt
def train_model_view(request):
    if request.method == "POST":
        try:
            # Extract training parameters
            params = request.POST
            epochs = int(params.get("epochs", 15))
            latent_d = int(params.get("latent_d", 10))
            lamda = float(params.get("lamda", 0.01))
            gamma = float(params.get("gamma", 0.01))
            tau = float(params.get("tau", 0.1))

            # Load cached data
            train_user, train_movie, test_user, test_movie, user_map, movie_map = processor.load_cached_data()

            if not isinstance(train_user, list) or not isinstance(train_movie, list):
                raise ValueError("Invalid format for train_user or train_movie. Expected lists.")

            # Initialize the model
            model = Model(
                data=(train_user, train_movie, test_user, test_movie, user_map, movie_map),
                latent_d=latent_d,
                lamda=lamda,
                gamma=gamma,
                tau=tau,
            )

            progress = []

            def progress_callback(epoch, total_epochs, loss, rmse):
                progress.append({"epoch": epoch, "total_epochs": total_epochs, "loss": loss, "rmse": rmse})

            # Train the model
            train_metrics, test_metrics = model.fit(epochs=epochs, progress_callback=progress_callback)

            # Generate plots
            train_plot = plot_to_base64(lambda: (
                plt.plot(train_metrics["loss"], label="Train Loss"),
                plt.plot(test_metrics["loss"], label="Test Loss"),
                plt.xlabel("Epochs"),
                plt.ylabel("Loss"),
                plt.title("Training and Testing Loss"),
                plt.legend(),
                plt.grid()
            ))
            rmse_plot = plot_to_base64(lambda: (
                plt.plot(train_metrics["rmse"], label="Train RMSE"),
                plt.plot(test_metrics["rmse"], label="Test RMSE"),
                plt.xlabel("Epochs"),
                plt.ylabel("RMSE"),
                plt.title("Training and Testing RMSE"),
                plt.legend(),
                plt.grid()
            ))

            # Save the trained model parameters
            model.save_model("trained_model.pkl")

            # Save U, V matrices, and biases for later use
            with open("model_matrices.pkl", "wb") as f:
                pickle.dump({
                    "U_matrix": model.user_matrix,
                    "V_matrix": model.movie_matrix,
                    "user_bias": model.user_bias,
                    "movie_bias": model.movie_bias,
                    "user_map": model.user_map,
                    "movie_map": model.movie_map,
                }, f)

            # Save training and testing metrics
            with open("train_test_metrics.pkl", "wb") as f:
                pickle.dump({"train_metrics": train_metrics, "test_metrics": test_metrics}, f)

            # Save progress for visualization
            with open("progress.pkl", "wb") as f:
                pickle.dump(progress, f)

            return JsonResponse({
                "message": "Training completed.",
                "train_plot": train_plot,
                "rmse_plot": rmse_plot,
                "progress": progress,
            })

        except Exception as e:
            append_to_log(f"Error during training: {str(e)}")
            return JsonResponse({"error": f"Training failed: {str(e)}"})

    return render(request, "movies/model.html")




@csrf_exempt
def test_model_view(request):
    if request.method == "POST":
        try:
            log_file_path = LOG_FILE
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

            with open(log_file_path, "a") as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
                # Load cached data
                _, _, test_user, test_movie, user_map, movie_map = processor.load_cached_data()

                # Example of test data (sample for console output)
                test_sample = {
                    "user_data_sample": test_user[:2],  # Sample first 2 users' test data
                    "movie_data_sample": test_movie[:2],  # Sample first 2 movies' test data
                    "user_map_sample": {k: user_map[k] for k in list(user_map.keys())[:2]},  # Sample first 2 user mappings
                    "movie_map_sample": {k: movie_map[k] for k in list(movie_map.keys())[:2]},  # Sample first 2 movie mappings
                }

                # Log the test samples
                append_to_log(f"Testing Cached Data Sample: {test_sample}")
                return JsonResponse({"message": "Testing completed. Check logs for details."})
        except Exception as e:
            append_to_log(f"Error during testing: {str(e)}")
            return JsonResponse({"error": f"Testing failed: {str(e)}"})

    return JsonResponse({"error": "Only POST requests are allowed."})


def file_browser_view(request):
    """
    View to return the directory structure of the project.
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root of your Django project
    directory_structure = []

    for root, dirs, files in os.walk(base_path):
        rel_root = os.path.relpath(root, base_path)
        directory_structure.append({
            "path": rel_root,
            "directories": dirs,
            "files": files,
        })

    return JsonResponse({"structure": directory_structure})


# Global variables to store logs and provide thread-safe access
LOG_FILE = "logs/output.log"
LOG_LOCK = Lock()

def append_to_log(message):
    """Append a message to the log file thread-safely."""
    with LOG_LOCK:
        with open(LOG_FILE, "a") as f:
            f.write(message + "\n")

def terminal_output_view(request):
    """Stream the terminal output."""
    try:
        if not os.path.exists(LOG_FILE):
            return JsonResponse({"output": "No logs available."})
        
        with LOG_LOCK:
            with open(LOG_FILE, "r") as f:
                content = f.read()
        return JsonResponse({"output": content})
    except Exception as e:
        return JsonResponse({"output": f"Error reading logs: {str(e)}"})

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root of the Django project

def get_file_content(request):
    """Fetch content of a file for editing."""
    file_path = request.GET.get("file_path", "").strip()

    if not file_path:
        return HttpResponseBadRequest("No file path provided.")

    full_path = os.path.join(BASE_PATH, file_path)

    if not os.path.exists(full_path) or not full_path.endswith(".py"):
        return HttpResponseBadRequest("Invalid file path or file does not exist.")

    try:
        with open(full_path, "r") as f:
            content = f.read()
        return JsonResponse({"file_path": file_path, "content": content})
    except Exception as e:
        return JsonResponse({"error": f"Failed to read file: {str(e)}"}, status=500)


@csrf_exempt
def save_file_content(request):
    """Save the edited content of a file."""
    if request.method != "POST":
        return HttpResponseBadRequest("Invalid request method.")

    try:
        data = request.POST
        file_path = data.get("file_path", "").strip()
        content = data.get("content", "").strip()

        if not file_path:
            return HttpResponseBadRequest("No file path provided.")

        full_path = os.path.join(BASE_PATH, file_path)

        if not os.path.exists(full_path) or not full_path.endswith(".py"):
            return HttpResponseBadRequest("Invalid file path or file does not exist.")

        # Save the file content
        with open(full_path, "w") as f:
            f.write(content)

        return JsonResponse({"message": "File saved successfully."})
    except Exception as e:
        return JsonResponse({"error": f"Failed to save file: {str(e)}"}, status=500)