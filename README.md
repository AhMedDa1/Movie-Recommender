
# Movie Recommender System

This repository contains a complete implementation of a movie recommendation system built on the MovieLens dataset. It features matrix factorization with biases and latent factors, evaluated using RMSE, and deployed as a Django-based web application. The project also includes scripts and notebooks for individual use and Docker integration for easy setup.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
  - [Using Docker](#using-docker)
  - [Manual Setup](#manual-setup)
- [Usage](#usage)
  - [Web Application](#web-application)
  - [Running Scripts](#running-scripts)
  - [Notebook Exploration](#notebook-exploration)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Results and Visualization](#results-and-visualization)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project demonstrates how advanced collaborative filtering techniques can be applied to large datasets to provide accurate and scalable recommendations. Key components include:

- Bias-only and latent factor-enhanced matrix factorization using ALS.
- A Django-based web app for users to interact with the recommendation system.
- A Python notebook for exploring results interactively.
- A Dockerized setup for hassle-free deployment.

---

## Features

- **Recommendation Models:** ALS-based models with biases, latent factors, and features.
- **Django Web App:** Register, login, rate movies, view recommendations, and explore datasets.
- **Docker Support:** Easily deploy the application using Docker.
- **Interactive Notebook:** Analyze models and evaluate recommendations step-by-step.
- **Dataset Integration:** Preprocessed MovieLens dataset included.

---

## Setup

### Using Docker
1. Ensure Docker is installed on your system.
2. Clone this repository:
   ```bash
   git clone https://github.com/your_username/Movie-Recommender.git
   cd Movie-Recommender
   ```
3. Build the Docker image:
   ```bash
   docker build -t movie-recommender .
   ```
4. Run the Docker container:
   ```bash
   docker run -p 8000:8000 movie-recommender
   ```
5. Open your browser and navigate to `http://localhost:8000`.

### Manual Setup
1. Clone the repository and navigate to the directory:
   ```bash
   git clone https://github.com/AhMedDa1/Movie-Recommender.git
   cd Movie-Recommender
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Apply migrations and run the Django development server:
   ```bash
   python manage.py migrate
   python manage.py runserver
   ```
5. Open your browser and navigate to `http://127.0.0.1:8000`.

---

## Usage

### Web Application
Once the server is running (via Docker or manual setup):
1. Register a new account or log in as an existing user.
2. Explore movie data, rate movies, and receive personalized recommendations.
3. Use visualization tools to analyze datasets and recommendations.

### Running Scripts
To run specific Python scripts without using the web interface:
1. Ensure the virtual environment is activated.
2. Example commands:
   - Train a recommendation model:
     ```bash
     python movies/recommendation/model.py
     ```
   - Test user recommendations:
     ```bash
     python movies/recommendation/dummy_user.py
     ```

### Notebook Exploration
The Jupyter notebook for this project is located at:
```
movies/recommendation/ahmeda_recommender_systemLastupdate.ipynb
```
1. Activate the virtual environment.
2. Install Jupyter if not already installed:
   ```bash
   pip install jupyter
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook
   ```
4. Open the file and explore the recommendation pipeline.

---

## Project Structure
```
Movie-Recommender/
├── Data/
│   ├── links.csv
│   ├── movies.csv
│   └── ratings.csv
├── db.sqlite3
├── Dockerfile
├── LICENSE
├── manage.py
├── movies/
│   ├── recommendation/   # Recommendation logic and data
│   │   ├── ahmeda_recommender_systemLastupdate.ipynb
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── dummy_user.py
│   ├── templates/         # HTML templates for the web app
│   └── views.py
├── requirements.txt
└── staticfiles/
```

---

## Dataset

The project uses the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/), preprocessed for analysis. Key files include:
- `ratings.csv`: User-movie ratings.
- `movies.csv`: Movie metadata.
- `links.csv`: External links to movie pages.

---

## Results and Visualization

- **Evaluation Metrics:** RMSE for both training and test datasets.
- **Model Performance:** Bias-only and latent factor models compared.
- **Visualizations:**
  - Distribution of ratings, genres, and user preferences.
  - Embedding spaces for items and genres using PCA.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- MovieLens dataset by GroupLens Research.
- Inspiration and guidance from academic studies on recommendation systems.

---
