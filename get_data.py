import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
import logging
from dotenv import load_dotenv

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

def get_movie_list(num_pages=10, min_votes=100):
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
    
    for page in tqdm(range(1, num_pages + 1)):
        # Get popular movies
        url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&page={page}&language=en-US"
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code != 200:
            logging.error(f"Failed to get page {page}: {response.status_code} - {response.text}")
            time.sleep(1)  # Wait a bit longer if there's an error
            continue
            
        data = response.json()
        
        # Filter movies with enough votes
        for movie in data.get('results', []):
            if movie.get('vote_count', 0) >= min_votes:
                movie_ids.append(movie['id'])
        
        # Be nice to the API - TMDb has rate limits
        time.sleep(0.5)
    
    logging.info(f"Retrieved {len(movie_ids)} movie IDs")
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

def get_movie_overview(movie_id):
    """
    Get plot overview for a specific movie
    
    Args:
        movie_id: TMDb movie ID
        
    Returns:
        Overview text or None if request fails
    """
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    
    if response.status_code != 200:
        logging.error(f"Failed to get overview for movie {movie_id}: {response.status_code}")
        return None
        
    data = response.json()
    return data.get('overview')

def main():
    logging.info("Starting data acquisition...")
    
    # Test API key with a simple request
    test_url = f"{BASE_URL}/movie/550?api_key={API_KEY}"  # Fight Club (id: 550)
    test_response = requests.get(test_url)
    
    if test_response.status_code != 200:
        logging.error(f"API key test failed: {test_response.status_code} - {test_response.text}")
        logging.error("Please check your API key and try again.")
        return
    else:
        logging.info("API key test successful!")
    
    # Get list of popular movies
    movie_ids = get_movie_list(num_pages=20, min_votes=200)  # Reduced to 20 pages for faster processing
    
    # Create dataframe to store movie data
    movies_data = []
    
    # Process each movie
    for movie_id in tqdm(movie_ids):
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
        
        # Extract relevant data
        movie_data = {
            'id': movie_id,
            'title': movie_details.get('title'),
            'overview': movie_details.get('overview'),
            'genres': genres,
            'poster_path': poster_local_path,
            'release_date': movie_details.get('release_date'),
            'vote_average': movie_details.get('vote_average'),
            'popularity': movie_details.get('popularity')
        }
        
        # Save individual movie JSON
        with open(f"data/json/{movie_id}.json", "w") as f:
            json.dump(movie_details, f)
            
        movies_data.append(movie_data)
        
        # Be nice to the API
        time.sleep(0.5)
    
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

if __name__ == "__main__":
    main()