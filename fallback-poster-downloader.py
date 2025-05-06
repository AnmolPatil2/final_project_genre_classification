import os
import pandas as pd
import requests
import time
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import imdb  # pip install imdbpy
import json
import re
from urllib.parse import quote

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fallback_poster_download.log"),
        logging.StreamHandler()
    ]
)

# Create directory for posters if it doesn't exist
os.makedirs("data/posters", exist_ok=True)

# API key for TMDb
API_KEY = "6098500f869682492abed510ecb8aedb"

# Base URLs
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"  # w500 size

# Initialize IMDb API
ia = imdb.IMDb()

def clean_filename(filename):
    """Clean a string to make it suitable for a filename"""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def search_movie_tmdb(title, year=None):
    """Search for a movie in TMDb"""
    query = quote(title)
    url = f"{TMDB_BASE_URL}/search/movie?api_key={API_KEY}&query={query}"
    
    if year:
        url += f"&year={year}"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        logging.error(f"Failed to search for movie {title}: {response.status_code}")
        return None
    
    data = response.json()
    results = data.get('results', [])
    
    if not results:
        return None
    
    # Return the first result
    return results[0]

def download_poster_from_url(url, save_path, timeout=10):
    """Download an image from a URL and save it to a file"""
    try:
        response = requests.get(url, timeout=timeout)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            logging.error(f"Failed to download image from {url}: {response.status_code}")
            return False
    
    except Exception as e:
        logging.error(f"Error downloading image from {url}: {str(e)}")
        return False

def get_poster_from_tmdb(movie_id, title, year=None):
    """Try to get a poster from TMDb"""
    # First try direct ID
    url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path')
        
        if poster_path:
            poster_url = f"{TMDB_POSTER_BASE_URL}{poster_path}"
            save_path = f"data/posters/{movie_id}.jpg"
            
            if download_poster_from_url(poster_url, save_path):
                return save_path
    
    # If direct ID fails, try searching by title
    search_result = search_movie_tmdb(title, year)
    
    if search_result:
        poster_path = search_result.get('poster_path')
        
        if poster_path:
            poster_url = f"{TMDB_POSTER_BASE_URL}{poster_path}"
            save_path = f"data/posters/{movie_id}.jpg"
            
            if download_poster_from_url(poster_url, save_path):
                return save_path
    
    return None

def get_poster_from_imdb(title, year, movie_id):
    """Try to get a poster from IMDb"""
    try:
        # Search for the movie
        results = ia.search_movie(title)
        
        if not results:
            return None
        
        # Try to find the correct movie by matching year
        movie = None
        for result in results:
            ia.update(result)
            
            # Check if it has year information and matches
            if 'year' in result.data and year and str(result.data['year']) == str(year):
                movie = result
                break
        
        # If no match by year, use the first result
        if not movie and results:
            movie = results[0]
            ia.update(movie)
        
        if not movie:
            return None
        
        # Try to get cover url
        if 'cover url' in movie.data:
            poster_url = movie.data['cover url']
            
            # Get a larger version if available
            poster_url = poster_url.replace('._V1_SX300', '._V1_SX1000')
            
            save_path = f"data/posters/{movie_id}.jpg"
            
            if download_poster_from_url(poster_url, save_path):
                return save_path
    
    except Exception as e:
        logging.error(f"Error getting poster from IMDb for {title}: {str(e)}")
    
    return None

def process_movie(row):
    """Process a single movie to get its poster"""
    movie_id = row['id']
    title = row['title']
    
    # Extract year from release_date if available
    year = None
    if pd.notna(row.get('release_date')) and len(str(row['release_date'])) >= 4:
        year = str(row['release_date'])[:4]
    
    # Check if poster already exists
    poster_path = f"data/posters/{movie_id}.jpg"
    if os.path.exists(poster_path):
        return movie_id, poster_path, "exists"
    
    # Try to get poster from TMDb
    tmdb_poster = get_poster_from_tmdb(movie_id, title, year)
    if tmdb_poster:
        return movie_id, tmdb_poster, "tmdb"
    
    # Try to get poster from IMDb as fallback
    imdb_poster = get_poster_from_imdb(title, year, movie_id)
    if imdb_poster:
        return movie_id, imdb_poster, "imdb"
    
    return movie_id, None, "failed"

def main():
    # Load movie data
    try:
        df = pd.read_pickle("data/movies_data.pkl")
        logging.info(f"Loaded {len(df)} movies from pickle file")
    except FileNotFoundError:
        try:
            df = pd.read_csv("data/movies_data.csv")
            logging.info(f"Loaded {len(df)} movies from CSV file")
        except FileNotFoundError:
            logging.error("No movie data file found. Please run the data acquisition script first.")
            return
    
    # Check which movies need posters
    if 'poster_path' in df.columns:
        missing_posters = df[df['poster_path'].isna() | 
                             ~df['poster_path'].apply(lambda x: os.path.exists(x) if isinstance(x, str) else False)]
    else:
        # If poster_path column doesn't exist, assume all need posters
        missing_posters = df
    
    logging.info(f"Found {len(missing_posters)} movies that need posters")
    
    # Create a list to store results
    results = []
    
    # Process movies with ThreadPoolExecutor for faster downloads
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_movie, row): idx for idx, row in missing_posters.iterrows()}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading posters"):
            try:
                movie_id, poster_path, source = future.result()
                results.append((movie_id, poster_path, source))
                
                # Periodically log progress
                if len(results) % 10 == 0:
                    logging.info(f"Progress: {len(results)}/{len(missing_posters)} posters processed")
            except Exception as e:
                logging.error(f"Error processing movie: {str(e)}")
    
    # Count results by source
    sources = {"exists": 0, "tmdb": 0, "imdb": 0, "failed": 0}
    for _, _, source in results:
        sources[source] = sources.get(source, 0) + 1
    
    logging.info("Poster download summary:")
    for source, count in sources.items():
        logging.info(f"  - {source}: {count}")
    
    # Update dataframe with new poster paths
    successful_posters = {movie_id: poster_path for movie_id, poster_path, source in results 
                          if poster_path is not None}
    
    # Update poster_path for movies with new posters
    for idx, row in df.iterrows():
        movie_id = row['id']
        if movie_id in successful_posters:
            df.at[idx, 'poster_path'] = successful_posters[movie_id]
    
    # Save updated dataframe
    df.to_pickle("data/movies_data_updated.pkl")
    df.to_csv("data/movies_data_updated.csv", index=False)
    
    logging.info(f"Updated dataframe saved. Total movies with posters: {df['poster_path'].notna().sum()}")

if __name__ == "__main__":
    main()