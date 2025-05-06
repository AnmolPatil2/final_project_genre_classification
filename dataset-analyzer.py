import os
import pandas as pd
import numpy as np
import json
import re
import logging
import shutil
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import ast
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_preprocessing.log"),
        logging.StreamHandler()
    ]
)

# Get the absolute path to the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set directories
data_dir = os.path.join(script_dir, "data")
raw_data_dir = os.path.join(script_dir, "movie_dataset_complete")
processed_data_dir = os.path.join(data_dir, "processed")
poster_dir = os.path.join(raw_data_dir, "posters")
processed_poster_dir = os.path.join(processed_data_dir, "posters")

# Create necessary directories
os.makedirs(data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)
os.makedirs(processed_poster_dir, exist_ok=True)
os.makedirs(os.path.join(processed_data_dir, "splits"), exist_ok=True)

class MovieDataPreprocessor:
    """Class for preprocessing movie dataset"""
    
    def __init__(self, csv_path, poster_dir, processed_dir, processed_poster_dir):
        """
        Initialize the preprocessor
        
        Args:
            csv_path: Path to the CSV file with movie data
            poster_dir: Directory containing poster images
            processed_dir: Directory to save processed data
            processed_poster_dir: Directory to save processed posters
        """
        self.csv_path = csv_path
        self.poster_dir = poster_dir
        self.processed_dir = processed_dir
        self.processed_poster_dir = processed_poster_dir
        self.df = None
        
        logging.info(f"MovieDataPreprocessor initialized with:")
        logging.info(f"  CSV path: {csv_path}")
        logging.info(f"  Poster directory: {poster_dir}")
        logging.info(f"  Processed data directory: {processed_dir}")
    
    def load_data(self):
        """Load the dataset from CSV"""
        try:
            logging.info(f"Loading data from {self.csv_path}")
            self.df = pd.read_csv(self.csv_path)
            logging.info(f"Loaded {len(self.df)} movies")
            
            # Display the first few rows and column info
            logging.info("Dataset columns:")
            for col in self.df.columns:
                logging.info(f"  - {col}: {self.df[col].dtype}")
            
            return True
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return False
    
    def clean_data(self):
        """Clean the dataset"""
        if self.df is None:
            logging.error("No data loaded. Call load_data() first.")
            return False
        
        logging.info("Cleaning data...")
        initial_count = len(self.df)
        
        # 1. Remove duplicates
        self.df = self.df.drop_duplicates(subset=['id', 'title'])
        logging.info(f"After removing duplicates: {len(self.df)} rows")
        
        # 2. Handle missing values
        missing_values = self.df.isnull().sum()
        logging.info("Missing values by column:")
        for col, count in missing_values.items():
            if count > 0:
                logging.info(f"  - {col}: {count} ({count/initial_count*100:.2f}%)")
        
        # Drop rows with missing essential data
        self.df = self.df.dropna(subset=['title', 'overview', 'genres'])
        logging.info(f"After removing rows with missing essential data: {len(self.df)} rows")
        
        # 3. Parse genres if needed
        if self.df['genres'].dtype == 'object':
            try:
                # Try to parse as JSON or list literal
                self.df['genres'] = self.df['genres'].apply(self.parse_list_field)
            except Exception as e:
                logging.error(f"Error parsing genres: {e}")
        
        # 4. Ensure genres is a list
        self.df = self.df[self.df['genres'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        logging.info(f"After ensuring genres is a non-empty list: {len(self.df)} rows")
        
        # 5. Verify poster paths and add full paths
        self.df['original_poster_path'] = self.df['poster_path'].copy()
        self.df['poster_full_path'] = self.df['poster_path'].apply(
            lambda x: os.path.join(self.poster_dir, x) if isinstance(x, str) else None
        )
        
        # Check if posters exist
        self.df['poster_exists'] = self.df['poster_full_path'].apply(
            lambda x: os.path.exists(x) if isinstance(x, str) else False
        )
        
        logging.info(f"Movies with existing posters: {self.df['poster_exists'].sum()} of {len(self.df)}")
        
        # Option: Keep only rows with existing posters
        # Uncomment the line below if you want to remove movies without posters
        # self.df = self.df[self.df['poster_exists']]
        
        # 6. Add primary genre (first genre in the list)
        self.df['primary_genre'] = self.df['genres'].apply(lambda x: x[0] if len(x) > 0 else None)
        
        # 7. Clean overview text
        self.df['overview'] = self.df['overview'].astype(str).apply(self.clean_text)
        
        # 8. Filter out very short overviews
        self.df = self.df[self.df['overview'].apply(len) >= 50]
        logging.info(f"After removing short overviews: {len(self.df)} rows")
        
        # 9. Add overview length as a feature
        self.df['overview_length'] = self.df['overview'].apply(len)
        
        # 10. Add number of genres as a feature
        self.df['genre_count'] = self.df['genres'].apply(len)
        
        logging.info(f"Data cleaning complete. Final dataset: {len(self.df)} rows")
        return True
    
    def parse_list_field(self, field_value):
        """Parse a string representation of a list"""
        if isinstance(field_value, list):
            return field_value
        
        if pd.isna(field_value):
            return []
        
        try:
            # Try to parse as JSON
            return json.loads(field_value.replace("'", '"'))
        except:
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
    
    def clean_text(self, text):
        """Clean text data"""
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub('<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub('\s+', ' ', text).strip()
        
        return text
    
    def analyze_genre_distribution(self):
        """Analyze genre distribution"""
        if self.df is None:
            logging.error("No data loaded. Call load_data() first.")
            return
        
        logging.info("Analyzing genre distribution...")
        
        # Count genre occurrences
        genre_counts = {}
        for genres in self.df['genres']:
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Create DataFrame for visualization
        genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
        genre_df = genre_df.sort_values('Count', ascending=False)
        
        # Log genre distribution
        logging.info(f"Found {len(genre_counts)} unique genres")
        logging.info("Top 10 genres:")
        for i, (genre, count) in enumerate(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]):
            logging.info(f"  {i+1}. {genre}: {count} movies ({count/len(self.df)*100:.2f}%)")
        
        # Plot genre distribution
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Count', y='Genre', data=genre_df.head(15))
        plt.title('Top 15 Genres Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(self.processed_dir, 'genre_distribution.png'))
        plt.close()
        
        return genre_df
    
    def process_posters(self, size=(224, 224), sample_limit=None):
        """
        Process poster images
        
        Args:
            size: Tuple of (width, height) to resize images
            sample_limit: Maximum number of posters to process (None for all)
        """
        if self.df is None:
            logging.error("No data loaded. Call load_data() first.")
            return
        
        # Filter to only rows with existing posters
        poster_df = self.df[self.df['poster_exists']].copy()
        
        if sample_limit and sample_limit < len(poster_df):
            poster_df = poster_df.sample(sample_limit, random_state=42)
        
        logging.info(f"Processing {len(poster_df)} poster images...")
        
        # Track processed posters
        processed_posters = []
        
        for _, row in tqdm(poster_df.iterrows(), total=len(poster_df)):
            try:
                movie_id = row['id']
                src_path = row['poster_full_path']
                dst_path = os.path.join(self.processed_poster_dir, f"{movie_id}.jpg")
                
                # Open, resize, and save the image
                img = Image.open(src_path)
                img = img.resize(size, Image.LANCZOS)
                img.save(dst_path, "JPEG", quality=90)
                
                processed_posters.append({
                    'id': movie_id,
                    'original_path': row['original_poster_path'],
                    'processed_path': f"processed/posters/{movie_id}.jpg"
                })
            except Exception as e:
                logging.error(f"Error processing poster for movie {row['id']}: {e}")
        
        logging.info(f"Successfully processed {len(processed_posters)} posters")
        
        # Update the dataframe with processed poster paths
        processed_poster_dict = {item['id']: item['processed_path'] for item in processed_posters}
        self.df['processed_poster_path'] = self.df['id'].map(processed_poster_dict)
        
        # Track which movies have processed posters
        self.df['has_processed_poster'] = self.df['processed_poster_path'].notna()
        
        return processed_posters
    
    def create_train_val_test_split(self, test_size=0.15, val_size=0.15, min_genre_count=20):
        """
        Create train/validation/test splits
        
        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set
            min_genre_count: Minimum number of movies per genre to keep the genre
        """
        if self.df is None:
            logging.error("No data loaded. Call load_data() first.")
            return
        
        logging.info("Creating dataset splits...")
        
        # Filter genres with too few examples
        genre_counts = self.df['primary_genre'].value_counts()
        valid_genres = genre_counts[genre_counts >= min_genre_count].index.tolist()
        
        df_filtered = self.df[self.df['primary_genre'].isin(valid_genres)].copy()
        
        logging.info(f"After filtering genres with fewer than {min_genre_count} examples:")
        logging.info(f"  - Original dataset: {len(self.df)} movies")
        logging.info(f"  - Filtered dataset: {len(df_filtered)} movies")
        logging.info(f"  - Kept genres ({len(valid_genres)}): {', '.join(valid_genres)}")
        
        # First split into train and temp
        train_df, temp_df = train_test_split(
            df_filtered, 
            test_size=test_size + val_size,
            random_state=42,
            stratify=df_filtered['primary_genre']
        )
        
        # Then split temp into validation and test
        val_size_adjusted = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=(1 - val_size_adjusted),
            random_state=42,
            stratify=temp_df['primary_genre']
        )
        
        logging.info(f"Dataset split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
        
        # Save splits
        train_df.to_csv(os.path.join(self.processed_dir, "splits", "train.csv"), index=False)
        val_df.to_csv(os.path.join(self.processed_dir, "splits", "val.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_dir, "splits", "test.csv"), index=False)
        
        # Also save as pickle for preserving data types
        train_df.to_pickle(os.path.join(self.processed_dir, "splits", "train.pkl"))
        val_df.to_pickle(os.path.join(self.processed_dir, "splits", "val.pkl"))
        test_df.to_pickle(os.path.join(self.processed_dir, "splits", "test.pkl"))
        
        # Save split statistics
        split_stats = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'train_genre_dist': train_df['primary_genre'].value_counts().to_dict(),
            'val_genre_dist': val_df['primary_genre'].value_counts().to_dict(),
            'test_genre_dist': test_df['primary_genre'].value_counts().to_dict(),
            'valid_genres': valid_genres
        }
        
        with open(os.path.join(self.processed_dir, "splits", "split_stats.json"), 'w') as f:
            json.dump(split_stats, f, indent=2)
        
        return train_df, val_df, test_df
    
    def save_processed_data(self):
        """Save the processed dataframe"""
        if self.df is None:
            logging.error("No data loaded. Call load_data() first.")
            return
        
        # Save as CSV and pickle
        csv_path = os.path.join(self.processed_dir, "movies_processed.csv")
        pickle_path = os.path.join(self.processed_dir, "movies_processed.pkl")
        
        self.df.to_csv(csv_path, index=False)
        self.df.to_pickle(pickle_path)
        
        logging.info(f"Saved processed data to {csv_path} and {pickle_path}")
        
        # Save dataset statistics
        stats = {
            'total_movies': len(self.df),
            'movies_with_posters': int(self.df['poster_exists'].sum()),
            'movies_with_processed_posters': int(self.df['has_processed_poster'].sum() if 'has_processed_poster' in self.df.columns else 0),
            'unique_genres': len(set(genre for genres in self.df['genres'] for genre in genres)),
            'avg_genres_per_movie': float(self.df['genre_count'].mean()),
            'avg_overview_length': float(self.df['overview_length'].mean())
        }
        
        with open(os.path.join(self.processed_dir, "dataset_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        logging.info("Dataset statistics saved")

def main():
    """Main function"""
    logging.info("Starting movie data preprocessing")
    
    # CSV path
    csv_path = os.path.join(raw_data_dir, "movies_data.csv")
    
    # Create preprocessor
    preprocessor = MovieDataPreprocessor(
        csv_path=csv_path,
        poster_dir=poster_dir,
        processed_dir=processed_data_dir,
        processed_poster_dir=processed_poster_dir
    )
    
    # Load data
    if not preprocessor.load_data():
        logging.error("Failed to load data. Exiting.")
        return
    
    # Clean data
    if not preprocessor.clean_data():
        logging.error("Failed to clean data. Exiting.")
        return
    
    # Analyze genre distribution
    genre_df = preprocessor.analyze_genre_distribution()
    
    # Process posters (resize and standardize)
    preprocessor.process_posters(size=(224, 224))
    
    # Create dataset splits
    train_df, val_df, test_df = preprocessor.create_train_val_test_split(
        test_size=0.15, 
        val_size=0.15,
        min_genre_count=20
    )
    
    # Save processed data
    preprocessor.save_processed_data()
    
    logging.info("Movie data preprocessing complete")

if __name__ == "__main__":
    main()