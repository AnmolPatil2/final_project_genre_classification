import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
import logging
from dotenv import load_dotenv
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_acquisition.log"),
        logging.StreamHandler()
    ]
)

# Load API key from .env file if available
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

# If no .env file, use the provided API key
if not API_KEY:
    API_KEY = "6098500f869682492abed510ecb8aedb"

logging.info(f"Using API key: {API_KEY[:4]}...{API_KEY[-4:]}")

# Base URLs for TMDb API
BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"  # w500 is the size

# Create directories for storing data
os.makedirs("data", exist_ok=True)
os.makedirs("data/posters", exist_ok=True)
os.makedirs("data/json", exist_ok=True)

def get_movie_list_by_popularity(num_pages=50, min_votes=50):
    """
    Get a list of popular movies from TMDb
    
    Args:
        num_pages: Number of pages to fetch (20 movies per page)
        min_votes: Minimum vote count for a movie to be included
        
    Returns:
        List of movie IDs
    """
    logging.info(f"Fetching {num_pages} pages of popular movies...")
    movie_ids = []
    
    for page in tqdm(range(1, num_pages + 1), desc="Popular movies"):
        # Get popular movies
        url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&page={page}&language=en-US"
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code != 200:
            logging.error(f"Failed to get page {page}: {response.status_code} - {response.text}")
            time.sleep(1.5)  # Wait a bit longer if there's an error
            continue
            
        data = response.json()
        
        # Filter movies with enough votes
        for movie in data.get('results', []):
            if movie.get('vote_count', 0) >= min_votes:
                movie_ids.append(movie['id'])
        
        # Be nice to the API - TMDb has rate limits
        time.sleep(0.3)
    
    logging.info(f"Retrieved {len(movie_ids)} popular movie IDs")
    return movie_ids

def get_movie_list_by_genre(genre_ids, num_pages=20, min_votes=50):
    """
    Get a list of movies by genre from TMDb
    
    Args:
        genre_ids: List of genre IDs to fetch
        num_pages: Number of pages to fetch per genre (20 movies per page)
        min_votes: Minimum vote count for a movie to be included
        
    Returns:
        List of movie IDs
    """
    logging.info(f"Fetching {num_pages} pages per genre for {len(genre_ids)} genres...")
    movie_ids = []
    
    for genre_id in genre_ids:
        logging.info(f"Fetching movies for genre ID: {genre_id}")
        
        for page in tqdm(range(1, num_pages + 1), desc=f"Genre {genre_id}"):
            # Get movies by genre
            url = f"{BASE_URL}/discover/movie?api_key={API_KEY}&with_genres={genre_id}&page={page}&language=en-US&sort_by=popularity.desc"
            response = requests.get(url)
            
            # Check if the request was successful
            if response.status_code != 200:
                logging.error(f"Failed to get page {page} for genre {genre_id}: {response.status_code}")
                time.sleep(1.5)  # Wait a bit longer if there's an error
                continue
                
            data = response.json()
            
            # Filter movies with enough votes
            for movie in data.get('results', []):
                if movie.get('vote_count', 0) >= min_votes:
                    movie_ids.append(movie['id'])
            
            # Be nice to the API - TMDb has rate limits
            time.sleep(0.3)
    
    # Remove duplicates
    movie_ids = list(set(movie_ids))
    logging.info(f"Retrieved {len(movie_ids)} movie IDs by genre")
    return movie_ids

def get_movie_list_by_year(years, num_pages=10, min_votes=50):
    """
    Get a list of movies by release year from TMDb
    
    Args:
        years: List of years to fetch
        num_pages: Number of pages to fetch per year (20 movies per page)
        min_votes: Minimum vote count for a movie to be included
        
    Returns:
        List of movie IDs
    """
    logging.info(f"Fetching {num_pages} pages per year for {len(years)} years...")
    movie_ids = []
    
    for year in years:
        logging.info(f"Fetching movies for year: {year}")
        
        for page in tqdm(range(1, num_pages + 1), desc=f"Year {year}"):
            # Get movies by year
            url = f"{BASE_URL}/discover/movie?api_key={API_KEY}&primary_release_year={year}&page={page}&language=en-US&sort_by=popularity.desc"
            response = requests.get(url)
            
            # Check if the request was successful
            if response.status_code != 200:
                logging.error(f"Failed to get page {page} for year {year}: {response.status_code}")
                time.sleep(1.5)  # Wait a bit longer if there's an error
                continue
                
            data = response.json()
            
            # Filter movies with enough votes
            for movie in data.get('results', []):
                if movie.get('vote_count', 0) >= min_votes:
                    movie_ids.append(movie['id'])
            
            # Be nice to the API - TMDb has rate limits
            time.sleep(0.3)
    
    # Remove duplicates
    movie_ids = list(set(movie_ids))
    logging.info(f"Retrieved {len(movie_ids)} movie IDs by year")
    return movie_ids

def get_movie_details(movie_id):
    """
    Get details for a specific movie
    
    Args:
        movie_id: TMDb movie ID
        
    Returns:
        Movie details dictionary or None if request fails
    """
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=en-US&append_to_response=keywords"
    response = requests.get(url)
    
    if response.status_code != 200:
        logging.error(f"Failed to get details for movie {movie_id}: {response.status_code}")
        return None
        
    return response.json()

def download_poster(poster_path, movie_id):
    """
    Download movie poster
    
    Args:
        poster_path: Path to poster from TMDb
        movie_id: TMDb movie ID for filename
        
    Returns:
        Local path to saved poster or None if download fails
    """
    if not poster_path:
        return None
        
    url = f"{POSTER_BASE_URL}{poster_path}"
    response = requests.get(url)
    
    if response.status_code != 200:
        logging.error(f"Failed to download poster for movie {movie_id}: {response.status_code}")
        return None
        
    # Save poster
    local_path = f"data/posters/{movie_id}.jpg"
    with open(local_path, "wb") as f:
        f.write(response.content)
        
    return local_path

def fetch_genre_list():
    """
    Fetch list of all genres from TMDb
    
    Returns:
        Dictionary mapping genre names to IDs
    """
    url = f"{BASE_URL}/genre/movie/list?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    
    if response.status_code != 200:
        logging.error(f"Failed to get genre list: {response.status_code}")
        return {}
        
    data = response.json()
    genres = {genre['name']: genre['id'] for genre in data.get('genres', [])}
    
    logging.info(f"Retrieved {len(genres)} genres: {', '.join(genres.keys())}")
    return genres

def main():
    logging.info("Starting enhanced data acquisition...")
    
    # Test API key with a simple request
    test_url = f"{BASE_URL}/movie/550?api_key={API_KEY}"  # Fight Club (id: 550)
    test_response = requests.get(test_url)
    
    if test_response.status_code != 200:
        logging.error(f"API key test failed: {test_response.status_code} - {test_response.text}")
        logging.error("Please check your API key and try again.")
        return
    else:
        logging.info("API key test successful!")
    
    # Get genre list
    genres = fetch_genre_list()
    genre_ids = list(genres.values())
    
    # Get movie IDs from multiple sources
    # 1. Popular movies
    popular_movie_ids = get_movie_list_by_popularity(num_pages=50, min_votes=100)
    
    # 2. Movies by genre - select some genres to ensure diversity
    genre_movie_ids = get_movie_list_by_genre(genre_ids, num_pages=15, min_votes=50)
    
    # 3. Recent movies by year - last 10 years
    current_year = 2024  # Adjust as needed
    years = list(range(current_year - 10, current_year + 1))
    year_movie_ids = get_movie_list_by_year(years, num_pages=5, min_votes=50)
    
    # Combine and remove duplicates
    all_movie_ids = list(set(popular_movie_ids + genre_movie_ids + year_movie_ids))
    random.shuffle(all_movie_ids)  # Shuffle to get diverse processing order
    
    logging.info(f"Total unique movie IDs collected: {len(all_movie_ids)}")
    logging.info(f"From popularity: {len(popular_movie_ids)}")
    logging.info(f"From genres: {len(genre_movie_ids)}")
    logging.info(f"From years: {len(year_movie_ids)}")
    
    # Create dataframe to store movie data
    movies_data = []
    
    # Process each movie
    for movie_id in tqdm(all_movie_ids, desc="Processing movies"):
        movie_details = get_movie_details(movie_id)
        
        if not movie_details:
            continue
            
        # Skip movies without overview or poster
        if not movie_details.get('overview') or not movie_details.get('poster_path'):
            logging.warning(f"Movie {movie_id} missing overview or poster, skipping")
            continue
            
        # Download poster
        poster_local_path = download_poster(movie_details.get('poster_path'), movie_id)
        
        if not poster_local_path:
            continue
            
        # Extract genres
        genres = [genre['name'] for genre in movie_details.get('genres', [])]
        
        # Skip movies without genres
        if not genres:
            logging.warning(f"Movie {movie_id} has no genres, skipping")
            continue
        
        # Extract relevant data
        movie_data = {
            'id': movie_id,
            'title': movie_details.get('title'),
            'overview': movie_details.get('overview'),
            'genres': genres,
            'poster_path': poster_local_path,
            'release_date': movie_details.get('release_date'),
            'vote_average': movie_details.get('vote_average'),
            'popularity': movie_details.get('popularity'),
            'adult': movie_details.get('adult', False),
            'original_language': movie_details.get('original_language')
        }
        
        # Save individual movie JSON for reference
        with open(f"data/json/{movie_id}.json", "w") as f:
            json.dump(movie_details, f)
            
        movies_data.append(movie_data)
        
        # Save progress every 100 movies
        if len(movies_data) % 100 == 0:
            logging.info(f"Progress: {len(movies_data)} movies processed")
            
            # Save interim results
            interim_df = pd.DataFrame(movies_data)
            interim_df.to_csv(f"data/movies_data_interim_{len(movies_data)}.csv", index=False)
            interim_df.to_pickle(f"data/movies_data_interim_{len(movies_data)}.pkl")
        
        # Be nice to the API
        time.sleep(0.3)
    
    # Convert to dataframe and save
    df = pd.DataFrame(movies_data)
    
    # Save to CSV
    df.to_csv("data/movies_data.csv", index=False)
    
    # Also save as pickle for preserving list data structure (genres)
    df.to_pickle("data/movies_data.pkl")
    
    logging.info(f"Data acquisition complete. Collected {len(df)} movies.")
    
    # Print genre distribution
    if len(df) > 0:
        genre_counts = {}
        for genres_list in df['genres']:
            for genre in genres_list:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                
        logging.info("Genre distribution:")
        for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"{genre}: {count}")
    else:
        logging.warning("No movies were collected. Check the API connection and try again.")
    
    # Print summary statistics
    if len(df) > 0:
        logging.info("\nSummary Statistics:")
        logging.info(f"Total movies: {len(df)}")
        logging.info(f"Total unique genres: {len(genre_counts)}")
        logging.info(f"Average genres per movie: {sum(len(genres) for genres in df['genres']) / len(df):.2f}")
        logging.info(f"Movies with posters: {df['poster_path'].notna().sum()}")
        logging.info(f"Movies with overview: {df['overview'].notna().sum()}")

if __name__ == "__main__":
    main()