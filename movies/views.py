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

def fetch_poster(tmdb_id):
    """
    Fetch the poster URL for a specific movie by tmdbId.
    
    :param tmdb_id: The TMDB ID of the movie.
    :return: URL of the movie poster or None if unavailable.
    """
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={settings.TMDB_API_KEY}&language=en-US"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
    except Exception as e:
        print(f"Error fetching poster for tmdbId {tmdb_id}: {e}")
    return None


def fetch_movies_with_pagination(page, per_page, query=""):
    """
    Fetch movies from movies.csv with pagination and dynamic poster fetching.
    
    :param page: Current page number.
    :param per_page: Number of movies per page.
    :param query: Search query to filter movies by title.
    :return: Dictionary containing paginated movie details and pagination info.
    """
    movies_path = "Data/movies.csv"
    links_path = "Data/links.csv"

    # Load movies and links
    movies = pd.read_csv(movies_path)
    links = pd.read_csv(links_path)

    # Merge movies with links on movieId
    merged = movies.merge(links, on="movieId", how="inner")

    # Filter by search query if provided
    if query:
        merged = merged[merged["title"].str.contains(query, case=False, na=False)]

    # Paginate results
    total_movies = len(merged)
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paginated_movies = merged.iloc[start_index:end_index]

    # Fetch posters for the paginated results
    movies_with_posters = []
    for _, row in paginated_movies.iterrows():
        poster_url = fetch_poster(row["tmdbId"])
        movies_with_posters.append({
            "id": row["movieId"],
            "title": row["title"],
            "genres": row["genres"],
            "poster": poster_url,
        })

    return {
        "movies": movies_with_posters,
        "total_movies": total_movies,
        "has_more": end_index < total_movies,
    }

@login_required
def main_page_view(request):
    query = request.GET.get("query", "")  # Capture search query
    page = int(request.GET.get("page", 1))  # Capture current page number
    per_page = 9  # Initial number of movies to show
    load_more_per_page = 3  # Number of movies for each "Show More"

    # Determine the number of movies to fetch
    movies_to_fetch = per_page if not request.GET.get("load_more", False) else load_more_per_page

    # Fetch movies dynamically with pagination and filtering
    result = fetch_movies_with_pagination(page, movies_to_fetch, query)
    movies = result["movies"]

    # Handle autocomplete request
    if request.GET.get("autocomplete", False):
        suggestions = [{"title": movie["title"]} for movie in movies if query.lower() in movie["title"].lower()][:10]
        return JsonResponse(suggestions, safe=False)

    # Handle "Load More" request
    if request.GET.get("load_more", False):
        return JsonResponse(movies, safe=False)

    # Render initial movies
    context = {
        "movies": movies,
        "query": query,
        "has_more": result["has_more"],  # Show "Show More" button if more movies exist
    }
    return render(request, 'movies/main.html', context)

@login_required
def rate_movies_view(request):
    query = request.GET.get("query", "")  # Capture search query
    page = int(request.GET.get("page", 1))  # Current page number
    per_page = 9  # Movies per page

    # Fetch movies with pagination
    result = fetch_movies_with_pagination(page, per_page, query)
    all_movies = result["movies"]

    # Autocomplete request
    if request.GET.get("autocomplete", False):
        suggestions = [{"title": movie["title"]} for movie in all_movies if query.lower() in movie["title"].lower()][:10]
        return JsonResponse(suggestions, safe=False)

    # "Load More" request
    if request.GET.get("load_more", False):
        return JsonResponse(all_movies, safe=False)

    # Handle POST submission for saving ratings
    if request.method == "POST":
        # Initialize or retrieve existing ratings from the session
        movie_ratings = request.session.get("movie_ratings", {})
        
        for key, value in request.POST.items():
            if key.startswith("rating_") and value.isdigit():
                movie_id = int(key.split("_")[1])  # Extract movie ID from "rating_<movie_id>"
                rating = int(value)  # Convert rating to integer
                movie_ratings[movie_id] = rating  # Save or update rating in session

        if not movie_ratings:
            messages.error(request, "No ratings were submitted. Please rate at least one movie.")
            return redirect('rate')

        # Debug: Check updated ratings
        print(f"Updated Ratings: {movie_ratings}")

        # Save the ratings back into the session
        request.session["movie_ratings"] = movie_ratings
        request.session.modified = True  # Mark session as modified to ensure changes are saved

        messages.success(request, "Your ratings have been saved!")
        return redirect('recommendations')

    # Context for rendering the rate page
    context = {
        "movies": all_movies,
        "query": query,
        "total_movies": result["total_movies"],
        "has_more": result["has_more"],  # Indicates if there are more movies
    }
    return render(request, "movies/rate.html", context)



@login_required
def recommendations_view(request):
    dummy_user_ratings = request.session.get("movie_ratings", {})
    if not dummy_user_ratings:
        return JsonResponse({"error": "No ratings found. Please rate at least one movie to get recommendations."})

    # Load your .npy files from movies/recommendation/model/
    base_dir = os.path.join(os.path.dirname(__file__), "recommendation", "model")

    movie_matrix_path = os.path.join(base_dir, "movies.npy")
    movie_bias_path = os.path.join(base_dir, "m_bias.npy")
    users_path = os.path.join(base_dir, "users.npy")
    movie_mapping_path = os.path.join(base_dir, "movies_mapping.npy")
    pkl_path = os.path.join(base_dir, "movies.pkl")

    movie_matrix = np.load(movie_matrix_path)  # shape (numMovies, latent_dim) or (latent_dim, numMovies)
    movie_bias = np.load(movie_bias_path)
    user_matrix = np.load(users_path)
    movie_map_data = np.load(movie_mapping_path, allow_pickle=True)
    if movie_map_data.ndim == 0:
        movie_map_data = movie_map_data.item()  # Now it's a true Python dict or something


    with open(pkl_path, "rb") as f:
        movies_pickle_data = pickle.load(f)

    # Print shapes to debug
    print(">> Loaded movie_matrix shape:", movie_matrix.shape)
    print(">> Loaded movie_bias shape:", movie_bias.shape)
    print(">> Loaded user_matrix shape:", user_matrix.shape)
    # If movie_map_data is a dict, just confirm how many keys it has:
    if hasattr(movie_map_data, "keys"):
        print(">> Loaded movie_map_data keys:", len(movie_map_data.keys()))
    else:
        print(">> Loaded movie_map_data length:", len(movie_map_data))

    # Create DummyUser
    from .recommendation.dummy_user import DummyUser
    latent_d = movie_matrix.shape[1] if movie_matrix.shape[0] > movie_matrix.shape[1] else movie_matrix.shape[0]
    print(">> Inferred latent_d from matrix shape:", latent_d)

    model = DummyUser(
        data=([], [], [], [], {}, {}),
        latent_d=latent_d,
        lamda=0.01,
        gamma=0.01,
        tau=0.1
    )

    model.user_matrix = user_matrix
    model.movie_matrix = movie_matrix
    model.movie_bias = movie_bias

    # If movie_map_data is a dict { realMovieID: index } or something:
    # Adjust how you assign it to model.movie_map:
    if hasattr(movie_map_data, "item"):
        model.movie_map = movie_map_data.item()  # if it's a 0D object array
    else:
        model.movie_map = movie_map_data

    # finalize_init might do model.V = model.movie_matrix.T
    model.finalize_init()

    print(">> After finalize_init, shape of model.V:", model.V.shape)

    # Convert session ratings
    dummy_user_ratings_list = [(int(m), float(r)) for m, r in dummy_user_ratings.items()]

    # Build dummy user vector
    dummy_user_latent = np.zeros(model.latent_d)
    dummy_user_bias = 0
    iterations = 10
    for iteration in range(iterations):
        dummy_user_bias = model.calculate_dummy_user_bias(dummy_user_ratings_list, iteration, dummy_user_latent)
        dummy_user_latent = model.update_user_latent_dummy(dummy_user_ratings_list, dummy_user_bias)
        # Print after each iteration
        print(f">> Iter {iteration}, dummy_user_bias={dummy_user_bias}, latent shape={dummy_user_latent.shape}")

    # Score unseen movies
    scores = []
    for movie_id, movie_idx in model.movie_map.items():
        if movie_id in dummy_user_ratings:
            continue
        # Print to see if any index is too large
        if movie_idx >= model.V.shape[1]:
            print(f">> Skipping movie_id={movie_id}, index={movie_idx} out of range for model.V")
            continue

        movie_vector = model.V[:, movie_idx]
        print(f">> movie_id={movie_id}, movie_idx={movie_idx}, movie_vector shape={movie_vector.shape}")

        val = np.dot(dummy_user_latent, movie_vector) + dummy_user_bias + model.movie_bias[movie_idx]
        scores.append((movie_id, val))

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Merge with CSV for posters
    movies_df = pd.read_csv("Data/movies.csv")
    links_df = pd.read_csv("Data/links.csv")
    merged_df = movies_df.merge(links_df, on="movieId", how="inner")

    top_n = 20
    recommended_movies = []
    for movie_id, pred_score in sorted_scores[:top_n]:
        row = merged_df[merged_df["movieId"] == movie_id]
        if row.empty:
            continue
        row = row.iloc[0]
        title = row["title"]
        genres = row["genres"]
        tmdb_id = row["tmdbId"]

        poster_url = None
        if not pd.isna(tmdb_id) and str(tmdb_id).isdigit():
            poster_url = fetch_poster(int(tmdb_id))

        recommended_movies.append({
            "id": movie_id,
            "title": title,
            "genres": genres,
            "poster": poster_url,
            "predicted_rating": round(pred_score, 2),
        })

    return render(request, "movies/recommendations.html", {"movies": recommended_movies})




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