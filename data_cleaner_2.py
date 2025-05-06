import pandas as pd
import numpy as np
import ast
import os
import re
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("movie_data_processing.log"),
        logging.StreamHandler()
    ]
)

# File paths
CSV_PATH = "/Users/anmolpatil/Desktop/Into to AI/final_project/movie_dataset_complete/movies_data.csv"
OUTPUT_DIR = "processed_data"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "splits"), exist_ok=True)

def load_data(csv_path):
    """Load the movie dataset from CSV"""
    logging.info(f"Loading data from {csv_path}")
    data = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(data)} movies")
    return data

def clean_data(df):
    """Handle missing values, duplicates and inconsistent formatting"""
    logging.info("Cleaning data...")
    df_clean = df.copy()
    
    # Convert list columns to strings for duplicate checking
    # This is a workaround for the "unhashable type: 'list'" error
    if 'genres' in df_clean.columns and df_clean['genres'].dtype == 'object':
        temp_df = df_clean.copy()
        temp_df['genres_str'] = temp_df['genres'].apply(lambda x: str(x) if isinstance(x, list) else x)
        # Drop duplicates based on id or other non-list columns
        unique_indices = temp_df.drop_duplicates(
            subset=['id', 'title', 'genres_str']
        ).index
        df_clean = df_clean.loc[unique_indices]
    else:
        # If no list columns, drop duplicates normally
        df_clean = df_clean.drop_duplicates(subset=['id', 'title'])
    
    logging.info(f"After removing duplicates: {len(df_clean)} rows")
    
    # Handle missing values
    df_clean['title'] = df_clean['title'].fillna('')
    df_clean['overview'] = df_clean['overview'].fillna('')  # Keep as empty string instead of NA
    df_clean['release_date'] = df_clean['release_date'].fillna('2000-01-01')
    df_clean['vote_average'] = df_clean['vote_average'].fillna(df_clean['vote_average'].mean())
    
    # Filter out rows with missing essential data
    df_clean = df_clean.dropna(subset=['title', 'overview', 'genres'])
    logging.info(f"After removing rows with missing essential data: {len(df_clean)} rows")
    
    print(f"Cleaned data: {df_clean.shape[0]} rows")
    return df_clean

def parse_list_field(field_value):
    """Parse a string representation of a list"""
    if isinstance(field_value, list):
        return field_value
    
    if pd.isna(field_value):
        return []
    
    try:
        # Try to parse as Python literal
        return ast.literal_eval(field_value)
    except:
        # If all else fails, try simple string split
        if isinstance(field_value, str):
            if ',' in field_value:
                return [item.strip() for item in field_value.split(',')]
            return [field_value]
        return []

def extract_features(data):
    """
    Extract specific features for genre prediction while keeping the original overview text
    """
    logging.info("Extracting features...")
    # Create a copy to avoid modifying the original
    df_features = data.copy()
    
    # Ensure overview is kept as a string
    df_features['overview'] = df_features['overview'].astype(str)
    
    # Extract other useful numeric features
    
    # 1. Title sentiment (can be useful for genre prediction)
    df_features['title_sentiment'] = df_features['title'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity if isinstance(x, str) else 0
    )
    
    # 2. Season from release date (could correlate with certain movie genres)
    def get_season(date_str):
        if not isinstance(date_str, str):
            return 'unknown'
        
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            month = date_obj.month
            
            if 3 <= month <= 5:
                return 'spring'
            elif 6 <= month <= 8:
                return 'summer'
            elif 9 <= month <= 11:
                return 'fall'
            else:
                return 'winter'
        except:
            return 'unknown'
    
    df_features['season'] = df_features['release_date'].apply(get_season)
    
    # Convert season to one-hot encoding
    season_dummies = pd.get_dummies(df_features['season'], prefix='season')
    df_features = pd.concat([df_features, season_dummies], axis=1)
    
    logging.info("Feature extraction complete.")
    return df_features

def analyze_genre_distribution(df):
    """Analyze and visualize genre distribution"""
    logging.info("Analyzing genre distribution...")
    
    # Extract all genres
    all_genres = [genre for genres_list in df['genres'] for genre in genres_list]
    genre_counts = Counter(all_genres)
    
    # Print results in the desired format
    for genre, count in genre_counts.items():
        print(f"{genre}: {count}")
    
    # Create DataFrame for plotting
    genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
    genre_df = genre_df.sort_values('Count', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(genre_df['Genre'].head(15), genre_df['Count'].head(15), color='skyblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.title('Genre Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'genre_distribution.png'))
    
    return genre_counts

def assign_groups(genre_list):
    """Assign genre groups based on the movie's genres"""
    # Mapping each genre to its corresponding group
    genre_to_group = {
        # Group 1: Drama, Music
        'Drama': 'Group 1',
        'Music': 'Group 1',
        # Group 2: Comedy, Crime, History
        'Comedy': 'Group 2',
        'Crime': 'Group 2',
        'History': 'Group 2',
        # Group 3: Thriller, Science Fiction, Fantasy, Documentary, Western
        'Thriller': 'Group 3',
        'Science Fiction': 'Group 3',
        'Fantasy': 'Group 3',
        'Documentary': 'Group 3',
        'Western': 'Group 3',
        # Group 4: Action, Horror, Mystery, War, TV Movie
        'Action': 'Group 4',
        'Horror': 'Group 4',
        'Mystery': 'Group 4',
        'War': 'Group 4',
        'TV Movie': 'Group 4',
        # Group 5: Adventure, Romance, Family, Animation
        'Adventure': 'Group 5',
        'Romance': 'Group 5',
        'Family': 'Group 5',
        'Animation': 'Group 5'
    }
    
    groups = set()  # use a set to avoid duplicates if a genre appears multiple times
    for genre in genre_list:
        if genre in genre_to_group:
            groups.add(genre_to_group[genre])
    
    # Return a sorted list to maintain consistency
    return sorted(list(groups))

def preprocess_pipeline(df):
    """Full preprocessing pipeline"""
    logging.info("Starting preprocessing pipeline...")
    
    # 1. Parse genres if they're stored as strings
    if df['genres'].dtype == 'object':
        logging.info("Converting genres strings to lists...")
        df['genres'] = df['genres'].apply(parse_list_field)
        
        # Check if parsing was successful
        list_count = sum(1 for x in df['genres'] if isinstance(x, list))
        logging.info(f"Successfully converted {list_count} out of {len(df)} genre entries to lists")
        
        # Handle any remaining non-list entries
        df['genres'] = df['genres'].apply(lambda x: [] if not isinstance(x, list) else x)
    
    # 2. Clean data (after parsing genres)
    df_clean = clean_data(df)
    
    # 3. Extract features (now keeping overview as text)
    df_features = extract_features(df_clean)
    
    # 4. Analyze genre distribution
    analyze_genre_distribution(df_features)
    
    # 5. Assign genre groups
    logging.info("Assigning genre groups...")
    df_features['groups'] = df_features['genres'].apply(assign_groups)
    
    # 6. Select final features - now including the original overview text
    df_final = df_features[['vote_average', 'title_sentiment', 'overview', 'groups']]
    
    # Log a sample to verify the format
    logging.info(f"Sample processed record: {df_final.iloc[0].to_dict()}")
    
    logging.info("Preprocessing pipeline complete.")
    return df_final

def create_train_val_test_split(df, test_size=0.2, val_size=0.15):
    """Create train/validation/test splits"""
    logging.info("Creating dataset splits...")
    
    # Convert groups to string representation for stratification (take first group)
    stratify_column = df['groups'].apply(lambda x: x[0] if len(x) > 0 else 'unknown')
    
    # First split: train and temp (val+test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=test_size + val_size,
        random_state=42,
        stratify=stratify_column
    )
    
    # Recalculate stratification column for temp dataframe
    temp_stratify = temp_df['groups'].apply(lambda x: x[0] if len(x) > 0 else 'unknown')
    
    # Second split: val and test
    val_size_adjusted = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=(1 - val_size_adjusted),
        random_state=42,
        stratify=temp_stratify
    )
    
    logging.info(f"Dataset split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
    
    return train_df, val_df, test_df

def main():
    """Main function to process movie data"""
    logging.info("Starting movie data processing")
    
    # 1. Load the data
    data = load_data(CSV_PATH)
    
    # Print initial information about the dataset
    logging.info(f"Initial data shape: {data.shape}")
    logging.info(f"Columns: {data.columns.tolist()}")
    
    # Check genres column type
    if 'genres' in data.columns:
        sample_genres = data['genres'].iloc[0] if not data.empty else None
        logging.info(f"Sample genres format: {type(sample_genres)} - {sample_genres}")
    
    # Check overview column
    if 'overview' in data.columns:
        sample_overview = data['overview'].iloc[0] if not data.empty else None
        logging.info(f"Sample overview: {type(sample_overview)} - {sample_overview[:100]}...")
    
    try:
        # 2. Preprocess data
        processed_data = preprocess_pipeline(data)
        
        # 3. Create train/val/test splits
        train_df, val_df, test_df = create_train_val_test_split(processed_data)
        
        # 4. Save to pickle files
        logging.info("Saving files to pickle format...")
        os.makedirs(os.path.join(OUTPUT_DIR, "splits"), exist_ok=True)
        train_df.to_pickle(os.path.join(OUTPUT_DIR, "splits", "train.pkl"))
        val_df.to_pickle(os.path.join(OUTPUT_DIR, "splits", "val.pkl"))
        test_df.to_pickle(os.path.join(OUTPUT_DIR, "splits", "test.pkl"))
        
        # 5. Also save as CSV for easier inspection
        logging.info("Saving files to CSV format...")
        train_df.to_csv(os.path.join(OUTPUT_DIR, "splits", "train.csv"), index=False)
        val_df.to_csv(os.path.join(OUTPUT_DIR, "splits", "val.csv"), index=False)
        test_df.to_csv(os.path.join(OUTPUT_DIR, "splits", "test.csv"), index=False)
        
        # 6. Print sample of the final data to verify format
        print("\nSample of processed data:")
        print(train_df.head(1).to_string())
        
        logging.info("Processing complete. Files saved to 'processed_data/splits/' directory")
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()