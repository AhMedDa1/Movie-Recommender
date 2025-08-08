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
FALLBACK_POSTER = 'https://via.placeholder.com/300x450?text=No+Image'

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            next_url = request.GET.get('next', 'main')  
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
    per_page = 9 
    load_more_per_page = 3  

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

    if request.method == "POST":
        movie_ratings = request.session.get("movie_ratings", {})
        new_ratings_added = False
        
        for key, value in request.POST.items():
            if key.startswith("rating_") and value.isdigit():
                movie_id = int(key.split("_")[1]) 
                rating = int(value) 
                if rating > 0:  # only persist explicit ratings
                    movie_ratings[movie_id] = rating 
                    new_ratings_added = True

        if not new_ratings_added:
            messages.error(request, "No ratings were submitted. Please rate at least one movie.")
            return redirect('rate')

        request.session["movie_ratings"] = movie_ratings
        request.session.modified = True  

        messages.success(request, "Your ratings have been saved!")
        return redirect('recommendations')

    context = {
        "movies": all_movies,
        "query": query,
        "total_movies": result["total_movies"],
        "has_more": result["has_more"],  
    }
    return render(request, "movies/rate.html", context)



@login_required
def recommendations_view(request):
    dummy_user_ratings = request.session.get("movie_ratings", {})
    if not dummy_user_ratings:
        messages.info(request, "Please rate a few movies first to get personalized recommendations.")
        return redirect('rate')

    # Paths for model artifacts
    model_dir = os.path.join(os.path.dirname(__file__), "recommendation", "model")
    pkl_item_latent = os.path.join(model_dir, "item_latent.pkl")
    pkl_item_bias = os.path.join(model_dir, "item_bias.pkl")
    pkl_movie_to_idx = os.path.join(model_dir, "movie_to_idx.pkl")
    pkl_idx_to_movie = os.path.join(model_dir, "idx_to_movie.pkl")

    # Try 1: Structured PKLs from model dir (preferred)
    used_model = None
    recommended_movies = []
    try:
        append_to_log(f"[recs] session_ratings_count={len(dummy_user_ratings)} keys_sample={list(dummy_user_ratings.keys())[:5]}")

        if all(os.path.exists(p) for p in [pkl_item_latent, pkl_item_bias, pkl_movie_to_idx, pkl_idx_to_movie]):
            append_to_log("[recs] using model-dir PKLs (item_latent/item_bias/movie_to_idx/idx_to_movie)")
            with open(pkl_item_latent, 'rb') as f:
                item_latent = pickle.load(f)
            with open(pkl_item_bias, 'rb') as f:
                item_bias = pickle.load(f)
            with open(pkl_movie_to_idx, 'rb') as f:
                movie_to_idx = pickle.load(f)
            with open(pkl_idx_to_movie, 'rb') as f:
                idx_to_movie = pickle.load(f)

            # Normalize mappings possibly stored as lists
            if isinstance(idx_to_movie, list):
                idx_to_movie = {int(i): int(mid) for i, mid in enumerate(idx_to_movie)}
            else:
                idx_to_movie = {int(k): int(v) for k, v in idx_to_movie.items()}

            if isinstance(movie_to_idx, list):
                movie_to_idx = {int(mid): int(i) for i, mid in enumerate(movie_to_idx)}
            else:
                movie_to_idx = {int(k): int(v) for k, v in movie_to_idx.items()}

            V = np.array(item_latent)
            mb = np.array(item_bias).reshape(-1)
            # Orient factors to (k, n)
            if V.shape[1] <= V.shape[0]:
                V = V.T
            k = V.shape[0]
            n = V.shape[1]
            used_model = f"model-dir-pkls k={k} n={n}"

            # Build ratings list using only movies present in mapping
            ratings_list = [(int(m), float(r)) for m, r in dummy_user_ratings.items() if int(m) in movie_to_idx]
            append_to_log(f"[recs] matched_session_ratings_in_map={len(ratings_list)} (model-dir-pkls)")

            # Cold-start dummy user solve
            lamda, tau = 0.01, 0.1
            nu = np.zeros(k)
            b = 0.0
            for _ in range(10):
                s, c = 0.0, 0
                for m_id, actual in ratings_list:
                    idx = movie_to_idx[m_id]
                    pred = float(nu @ V[:, idx] + mb[idx])
                    s += lamda * (actual - pred)
                    c += 1
                b = (s / ((lamda * c) + tau)) if c > 0 else 0.0

                x = np.zeros(k)
                y = np.zeros((k, k))
                for m_id, actual in ratings_list:
                    idx = movie_to_idx[m_id]
                    v_m = V[:, idx]
                    err = actual - b - mb[idx]
                    x += v_m * err
                    y += np.outer(v_m, v_m)
                y += np.eye(k) * tau
                nu = np.linalg.solve(lamda * y, lamda * x) if ratings_list else np.zeros(k)

            # Score vectorized and exclude rated
            preds = (nu @ V) + b + mb
            for m_id, _ in ratings_list:
                preds[movie_to_idx[m_id]] = -1e9

            # Top-N indices
            topN = 20
            top_idx = np.argsort(-preds)

            # Debug: log top-10 ids and Toy Story 2 placement if available
            try:
                top10_ids = []
                inv_map = idx_to_movie
                for idx in top_idx[:10]:
                    mid = inv_map.get(int(idx))
                    top10_ids.append(int(mid) if mid is not None else None)
                append_to_log(f"[recs][pkls] top10_ids={top10_ids}")
                # Toy Story 2 diagnostics
                TS2_ID = 3114
                if TS2_ID in movie_to_idx:
                    ts2_idx = movie_to_idx[TS2_ID]
                    ts2_score = float(nu @ V[:, ts2_idx] + b + mb[ts2_idx])
                    approx_rank = int(np.sum(preds > ts2_score)) + 1
                    append_to_log(f"[recs][pkls] TS2 score={ts2_score:.3f} approx_rank={approx_rank} of {preds.shape[0]}")
            except Exception as _e:
                append_to_log(f"[recs][pkls][log_error] {_e}")

            # Join with metadata and posters; filter to titles we can display
            movies_df = pd.read_csv("Data/movies.csv")
            links_df = pd.read_csv("Data/links.csv")
            merged_df = movies_df.merge(links_df, on="movieId", how="left")
            inv_map = idx_to_movie

            for idx in top_idx:
                mid = inv_map.get(int(idx))
                row = merged_df[merged_df["movieId"] == mid]
                if row.empty:
                    continue
                row = row.iloc[0]
                tmdb_id = row.get("tmdbId")
                poster_url = None
                if pd.notna(tmdb_id):
                    try:
                        poster_url = fetch_poster(int(tmdb_id))
                    except Exception:
                        poster_url = None
                if not poster_url:
                    poster_url = FALLBACK_POSTER

                recommended_movies.append({
                    "id": int(mid),
                    "title": row.get("title"),
                    "genres": row.get("genres"),
                    "poster": poster_url,
                    "predicted_rating": round(max(0.0, min(5.0, float(preds[idx]))), 2),
                })
                if len(recommended_movies) >= topN:
                    break

        else:
            # Try 2: Root PKL dict (U_matrix/V_matrix)
            model_dict = None
            for p in [os.path.join(settings.BASE_DIR, 'model_matrices.pkl'), os.path.join(settings.BASE_DIR, 'trained_model.pkl')]:
                if os.path.exists(p):
                    with open(p, 'rb') as f:
                        model_dict = pickle.load(f)
                    used_model = os.path.basename(p)
                    break

            if model_dict is not None:
                user_matrix = model_dict['U_matrix'] if 'U_matrix' in model_dict else model_dict.get('user_matrix')
                movie_matrix = model_dict['V_matrix'] if 'V_matrix' in model_dict else model_dict.get('movie_matrix')
                user_bias = model_dict.get('user_bias')
                movie_bias = model_dict.get('movie_bias')
                movie_map = model_dict.get('movie_map')
                append_to_log(f"[recs] using root PKL {used_model} users={getattr(user_matrix,'shape',None)} movies={getattr(movie_matrix,'shape',None)} map_size={len(movie_map) if movie_map else 0}")

                latent_d = min(movie_matrix.shape[0], movie_matrix.shape[1])
                model = DummyUser(
                    data=([], [], [], [], {}, {}),
                    latent_d=latent_d,
                    lamda=0.01,
                    gamma=0.01,
                    tau=0.1,
                )
                model.user_matrix = user_matrix
                model.movie_matrix = movie_matrix
                model.user_bias = user_bias if user_bias is not None else np.zeros(user_matrix.shape[0])
                model.movie_bias = movie_bias if movie_bias is not None else np.zeros(movie_matrix.shape[0])
                model.movie_map = {int(k): int(v) for k, v in movie_map.items()}
                model.finalize_init()

                ratings_list = [(int(m), float(r)) for m, r in dummy_user_ratings.items() if int(m) in model.movie_map]

                dummy_user_latent = np.zeros(model.latent_d)
                dummy_user_bias = 0.0
                for _ in range(10):
                    dummy_user_bias = model.calculate_dummy_user_bias(ratings_list, _, dummy_user_latent)
                    dummy_user_latent = model.update_user_latent_dummy(ratings_list, dummy_user_bias)

                scores = []
                for movie_id, movie_idx in model.movie_map.items():
                    if movie_id in dummy_user_ratings:
                        continue
                    if movie_idx >= model.V.shape[1]:
                        continue
                    movie_vector = model.V[:, movie_idx]
                    val = float(np.dot(dummy_user_latent, movie_vector) + dummy_user_bias + model.movie_bias[movie_idx])
                    val = max(0.0, min(5.0, val))
                    scores.append((movie_id, val))

                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

                movies_df = pd.read_csv("Data/movies.csv")
                links_df = pd.read_csv("Data/links.csv")
                merged_df = movies_df.merge(links_df, on="movieId", how="left")

                for movie_id, pred_score in sorted_scores[:20]:
                    row = merged_df[merged_df["movieId"] == movie_id]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    tmdb_id = row.get("tmdbId")
                    poster_url = None
                    if pd.notna(tmdb_id):
                        try:
                            poster_url = fetch_poster(int(tmdb_id))
                        except Exception:
                            poster_url = None
                    if not poster_url:
                        poster_url = FALLBACK_POSTER

                    recommended_movies.append({
                        "id": int(movie_id),
                        "title": row.get("title"),
                        "genres": row.get("genres"),
                        "poster": poster_url,
                        "predicted_rating": round(max(0.0, min(5.0, float(pred_score))), 2),
                    })

            else:
                # Try 3: NPY fallback
                append_to_log("[recs] model pickles not found, using .npy fallback")
                base_dir = os.path.join(os.path.dirname(__file__), "recommendation", "model")
                movie_matrix_path = os.path.join(base_dir, "movies.npy")
                movie_bias_path = os.path.join(base_dir, "m_bias.npy")
                users_path = os.path.join(base_dir, "users.npy")
                movie_mapping_path = os.path.join(base_dir, "movies_mapping.npy")

                movie_matrix = np.load(movie_matrix_path)
                movie_bias = np.load(movie_bias_path)
                user_matrix = np.load(users_path)
                movie_map_data = np.load(movie_mapping_path, allow_pickle=True)
                if hasattr(movie_map_data, "item"):
                    movie_map_data = movie_map_data.item()

                model = DummyUser(
                    data=([], [], [], [], {}, {}),
                    latent_d=min(movie_matrix.shape[0], movie_matrix.shape[1]),
                    lamda=0.01,
                    gamma=0.01,
                    tau=0.1,
                )
                model.user_matrix = user_matrix
                model.movie_matrix = movie_matrix
                model.movie_bias = movie_bias
                model.movie_map = movie_map_data
                model.finalize_init()

                ratings_list = [(int(m), float(r)) for m, r in dummy_user_ratings.items() if int(m) in model.movie_map]

                dummy_user_latent = np.zeros(model.latent_d)
                dummy_user_bias = 0.0
                for _ in range(10):
                    dummy_user_bias = model.calculate_dummy_user_bias(ratings_list, _, dummy_user_latent)
                    dummy_user_latent = model.update_user_latent_dummy(ratings_list, dummy_user_bias)

                scores = []
                for movie_id, movie_idx in model.movie_map.items():
                    if movie_id in dummy_user_ratings:
                        continue
                    if movie_idx >= model.V.shape[1]:
                        continue
                    movie_vector = model.V[:, movie_idx]
                    val = float(np.dot(dummy_user_latent, movie_vector) + dummy_user_bias + model.movie_bias[movie_idx])
                    val = max(0.0, min(5.0, val))
                    scores.append((movie_id, val))

                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

                movies_df = pd.read_csv("Data/movies.csv")
                links_df = pd.read_csv("Data/links.csv")
                merged_df = movies_df.merge(links_df, on="movieId", how="left")

                for movie_id, pred_score in sorted_scores[:20]:
                    row = merged_df[merged_df["movieId"] == movie_id]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    tmdb_id = row.get("tmdbId")
                    poster_url = None
                    if pd.notna(tmdb_id):
                        try:
                            poster_url = fetch_poster(int(tmdb_id))
                        except Exception:
                            poster_url = None
                    if not poster_url:
                        poster_url = FALLBACK_POSTER

                    recommended_movies.append({
                        "id": int(movie_id),
                        "title": row.get("title"),
                        "genres": row.get("genres"),
                        "poster": poster_url,
                        "predicted_rating": round(max(0.0, min(5.0, float(pred_score))), 2),
                    })

    except Exception as e:
        messages.error(request, f"Failed to build recommendations: {e}")
        append_to_log(f"[recs][error] {e}")
        recommended_movies = []

    # Popularity fallback if still empty
    if not recommended_movies:
        append_to_log("[recs] falling back to popularity due to empty recommendations")
        try:
            ratings_df = pd.read_csv("Data/ratings.csv")
            movies_df = pd.read_csv("Data/movies.csv")
            links_df = pd.read_csv("Data/links.csv")
            counts = ratings_df.groupby('movieId').size().sort_values(ascending=False)
            top_ids = counts.index.tolist()[:20]
            merged_df = movies_df.merge(links_df, on="movieId", how="left")
            for movie_id in top_ids:
                row = merged_df[merged_df["movieId"] == movie_id]
                if row.empty:
                    continue
                row = row.iloc[0]
                tmdb_id = row.get("tmdbId")
                poster_url = None
                if pd.notna(tmdb_id):
                    try:
                        poster_url = fetch_poster(int(tmdb_id))
                    except Exception:
                        poster_url = None
                if not poster_url:
                    poster_url = FALLBACK_POSTER
                recommended_movies.append({
                    "id": int(movie_id),
                    "title": row.get("title"),
                    "genres": row.get("genres"),
                    "poster": poster_url,
                    "predicted_rating": None,
                })
        except Exception as e:
            append_to_log(f"[recs][popularity_error] {e}")
            pass

    return render(request, "movies/recommendations.html", {"movies": recommended_movies})




def logout_view(request):
    logout(request)
    return redirect('login')


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
            params = request.POST
            epochs = int(params.get("epochs", 15))
            latent_d = int(params.get("latent_d", 10))
            lamda = float(params.get("lamda", 0.01))
            gamma = float(params.get("gamma", 0.01))
            tau = float(params.get("tau", 0.1))

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

            train_metrics, test_metrics = model.fit(epochs=epochs, progress_callback=progress_callback)

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

            model.save_model("trained_model.pkl")

            with open("model_matrices.pkl", "wb") as f:
                pickle.dump({
                    "U_matrix": model.user_matrix,
                    "V_matrix": model.movie_matrix,
                    "user_bias": model.user_bias,
                    "movie_bias": model.movie_bias,
                    "user_map": model.user_map,
                    "movie_map": model.movie_map,
                }, f)

            with open("train_test_metrics.pkl", "wb") as f:
                pickle.dump({"train_metrics": train_metrics, "test_metrics": test_metrics}, f)

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
                _, _, test_user, test_movie, user_map, movie_map = processor.load_cached_data()

                test_sample = {
                    "user_data_sample": test_user[:2], 
                    "movie_data_sample": test_movie[:2],
                    "user_map_sample": {k: user_map[k] for k in list(user_map.keys())[:2]}, 
                    "movie_map_sample": {k: movie_map[k] for k in list(movie_map.keys())[:2]}, 
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
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    directory_structure = []

    for root, dirs, files in os.walk(base_path):
        rel_root = os.path.relpath(root, base_path)
        directory_structure.append({
            "path": rel_root,
            "directories": dirs,
            "files": files,
        })

    return JsonResponse({"structure": directory_structure})


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

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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