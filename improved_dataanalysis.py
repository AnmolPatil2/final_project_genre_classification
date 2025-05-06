import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from collections import Counter
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("genre_preprocessing.log"),
        logging.StreamHandler()
    ]
)

class MovieGenrePreprocessor:
    """Class for preprocessing movie dataset with focus on handling imbalanced genres"""
    
    def __init__(self, input_path, output_dir, top_n_genres=10, drama_downsample_ratio=0.5):
        """
        Initialize the genre preprocessor
        
        Args:
            input_path: Path to the processed movie data CSV/pickle
            output_dir: Directory to save processed outputs
            top_n_genres: Number of top genres to keep
            drama_downsample_ratio: Ratio for downsampling drama (0-1)
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.top_n_genres = top_n_genres
        self.drama_downsample_ratio = drama_downsample_ratio
        self.df = None
        self.genre_counts = None
        self.top_genres = None
        self.mlb = None
        self.genre_weights = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "splits"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        
        logging.info(f"MovieGenrePreprocessor initialized:")
        logging.info(f"  Input path: {input_path}")
        logging.info(f"  Output directory: {output_dir}")
        logging.info(f"  Top genres to keep: {top_n_genres}")
        logging.info(f"  Drama downsample ratio: {drama_downsample_ratio}")
    
    def load_data(self):
        """Load the processed movie data"""
        logging.info(f"Loading data from {self.input_path}")
        
        try:
            # Determine file type and load accordingly
            if self.input_path.endswith('.csv'):
                self.df = pd.read_csv(self.input_path)
            elif self.input_path.endswith('.pkl'):
                self.df = pd.read_pickle(self.input_path)
            else:
                raise ValueError("Data file must be CSV or pickle format")
                
            logging.info(f"Loaded {len(self.df)} movies")
            
            # Basic data validation
            if 'genres' not in self.df.columns:
                raise ValueError("Dataset must contain 'genres' column")
                
            # Check if genres are already in list format
            if not isinstance(self.df['genres'].iloc[0], list):
                logging.info("Converting genres to list format")
                self.df['genres'] = self.df['genres'].apply(self._parse_genres)
                
            # Display the first few rows
            logging.info("First few rows of the dataset:")
            pd.set_option('display.max_columns', 10)
            logging.info(f"\n{self.df.head()}")
            
            # Check for NaN values and handle them
            nan_count = self.df.isnull().sum().sum()
            if nan_count > 0:
                logging.warning(f"Found {nan_count} NaN values in the dataset")
                logging.info("NaN counts by column:")
                logging.info(f"\n{self.df.isnull().sum()}")
                
            return True
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return False
    
    def _parse_genres(self, genres_str):
        """Parse genres from string to list if needed"""
        if isinstance(genres_str, list):
            return genres_str
            
        if pd.isna(genres_str):
            return []
            
        try:
            # Try to parse as JSON
            import json
            return json.loads(genres_str.replace("'", '"'))
        except:
            try:
                # Try to parse as Python literal
                import ast
                return ast.literal_eval(genres_str)
            except:
                # Last resort: split by comma if it's a string
                if isinstance(genres_str, str):
                    return [g.strip() for g in genres_str.split(',')]
                return []
    
    def analyze_genre_distribution(self):
        """Analyze and visualize genre distribution"""
        if self.df is None:
            logging.error("No data loaded. Call load_data() first.")
            return
            
        logging.info("Analyzing genre distribution...")
        
        # Count genre occurrences
        genre_counter = Counter()
        for genres in self.df['genres'].dropna():
            if not isinstance(genres, list):
                continue
            for genre in genres:
                genre_counter[genre] += 1
                
        # Convert to DataFrame for easier handling
        self.genre_counts = pd.DataFrame.from_dict(genre_counter, orient='index', columns=['count'])
        self.genre_counts = self.genre_counts.sort_values('count', ascending=False)
        self.genre_counts['percentage'] = 100 * self.genre_counts['count'] / len(self.df)
        
        # Display genre distribution
        logging.info(f"Found {len(genre_counter)} unique genres")
        logging.info("Top 15 genres:")
        logging.info(f"\n{self.genre_counts.head(15)}")
        
        # Plot overall genre distribution
        plt.figure(figsize=(12, 8))
        sns.barplot(x='count', y=self.genre_counts.index[:15], data=self.genre_counts[:15])
        plt.title('Top 15 Genres Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualizations", "genre_distribution.png"))
        plt.close()
        
        # Plot genre percentage distribution
        plt.figure(figsize=(12, 8))
        sns.barplot(x='percentage', y=self.genre_counts.index[:15], data=self.genre_counts[:15])
        plt.title('Top 15 Genres Percentage')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualizations", "genre_percentage.png"))
        plt.close()
        
        return self.genre_counts
    
    def analyze_genre_combinations(self):
        """Analyze common genre combinations"""
        if self.df is None:
            logging.error("No data loaded. Call load_data() first.")
            return
            
        logging.info("Analyzing genre combinations...")
        
        # Count genre combination occurrences
        combination_counter = Counter()
        for genres in self.df['genres'].dropna():
            if not isinstance(genres, list) or len(genres) == 0:
                continue
                
            # Sort genres to treat [A,B] and [B,A] as the same combination
            key = ', '.join(sorted(genres))
            combination_counter[key] += 1
            
        # Convert to DataFrame
        combo_df = pd.DataFrame(list(combination_counter.items()), columns=['Combination', 'Count'])
        combo_df = combo_df.sort_values('Count', ascending=False)
        
        # Log top combinations
        logging.info(f"Found {len(combination_counter)} unique genre combinations")
        logging.info("Top 10 genre combinations:")
        logging.info(f"\n{combo_df.head(10)}")
        
        # Plot top combinations
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Count', y='Combination', data=combo_df.head(10))
        plt.title('Top 10 Genre Combinations')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualizations", "genre_combinations.png"))
        plt.close()
        
        return combo_df
    
    def filter_top_genres(self):
        """Keep only the top N genres"""
        if self.genre_counts is None:
            self.analyze_genre_distribution()
            
        # Get the top N genres
        self.top_genres = self.genre_counts.head(self.top_n_genres).index.tolist()
        logging.info(f"Keeping top {self.top_n_genres} genres: {', '.join(self.top_genres)}")
        
        # Create a new column with filtered genres
        self.df['filtered_genres'] = self.df['genres'].apply(
            lambda x: [genre for genre in x if genre in self.top_genres] if isinstance(x, list) else []
        )
        
        # Keep only movies that have at least one of the top genres
        original_count = len(self.df)
        self.df = self.df[self.df['filtered_genres'].apply(len) > 0].reset_index(drop=True)
        logging.info(f"Removed {original_count - len(self.df)} movies with no top genres")
        logging.info(f"Remaining movies: {len(self.df)}")
        
        # Visualize filtered genre distribution
        filtered_counts = Counter()
        for genres in self.df['filtered_genres']:
            for genre in genres:
                filtered_counts[genre] += 1
                
        filtered_df = pd.DataFrame.from_dict(filtered_counts, orient='index', columns=['count'])
        filtered_df = filtered_df.sort_values('count', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='count', y=filtered_df.index, data=filtered_df)
        plt.title(f'Distribution of Top {self.top_n_genres} Genres')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualizations", "filtered_genre_distribution.png"))
        plt.close()
        
        return self.df
    
    def downsample_drama(self):
        """Downsample movies with Drama as primary genre"""
        if 'Drama' not in self.top_genres:
            logging.info("Drama not in top genres, skipping downsampling")
            return self.df
            
        logging.info("Downsampling Drama genre...")
        
        # Add primary genre column if not exists
        if 'primary_genre' not in self.df.columns:
            self.df['primary_genre'] = self.df['filtered_genres'].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
            )
            
        # Identify movies with Drama as primary genre
        drama_movies = self.df[self.df['primary_genre'] == 'Drama']
        non_drama_movies = self.df[self.df['primary_genre'] != 'Drama']
        
        drama_count = len(drama_movies)
        logging.info(f"Found {drama_count} movies with Drama as primary genre")
        
        # Calculate how many drama movies to keep
        n_drama_to_keep = int(drama_count * self.drama_downsample_ratio)
        logging.info(f"Downsampling to {n_drama_to_keep} Drama movies")
        
        # Randomly sample drama movies
        drama_sample = drama_movies.sample(n_drama_to_keep, random_state=42)
        
        # Combine with non-drama movies
        self.df = pd.concat([drama_sample, non_drama_movies]).reset_index(drop=True)
        
        logging.info(f"After downsampling: {len(self.df)} total movies")
        
        # Visualize distribution after downsampling
        downsampled_counts = Counter()
        for genres in self.df['filtered_genres']:
            for genre in genres:
                downsampled_counts[genre] += 1
                
        downsampled_df = pd.DataFrame.from_dict(downsampled_counts, orient='index', columns=['count'])
        downsampled_df = downsampled_df.sort_values('count', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='count', y=downsampled_df.index, data=downsampled_df)
        plt.title('Genre Distribution After Drama Downsampling')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualizations", "downsampled_distribution.png"))
        plt.close()
        
        return self.df
    
    def encode_genres(self):
        """Encode genres as multi-hot vectors using MultiLabelBinarizer"""
        if 'filtered_genres' not in self.df.columns:
            logging.error("No filtered genres found. Call filter_top_genres() first.")
            return
            
        logging.info("Encoding genres as multi-hot vectors...")
        
        # Initialize the MultiLabelBinarizer with predefined classes
        self.mlb = MultiLabelBinarizer(classes=self.top_genres)
        
        # Transform genres to multi-hot encoding
        genre_matrix = self.mlb.fit_transform(self.df['filtered_genres'])
        
        # Convert to DataFrame
        genre_df = pd.DataFrame(genre_matrix, columns=self.mlb.classes_)
        
        # Add encoded genres to main DataFrame
        for genre in self.mlb.classes_:
            self.df[f'genre_{genre}'] = genre_df[genre]
            
        logging.info(f"Encoded {len(self.mlb.classes_)} genres as multi-hot vectors")
        
        # Show sample of encoded data
        logging.info("Sample of encoded genres:")
        logging.info(f"\n{self.df[['filtered_genres'] + [f'genre_{g}' for g in self.top_genres]].head()}")
        
        return genre_matrix
    
    def calculate_class_weights(self):
        """Calculate class weights for weighted loss function"""
        if not hasattr(self, 'mlb') or self.mlb is None:
            logging.error("No genre encoding found. Call encode_genres() first.")
            return
            
        logging.info("Calculating class weights for loss function...")
        
        # Count genre occurrences after filtering
        genre_counts = {genre: 0 for genre in self.top_genres}
        for genres in self.df['filtered_genres']:
            for genre in genres:
                genre_counts[genre] += 1
                
        # Calculate weights (inverse frequency)
        n_samples = len(self.df)
        self.genre_weights = {genre: n_samples / (count * len(genre_counts)) 
                             for genre, count in genre_counts.items()}
        
        # Normalize weights to have mean=1
        weight_sum = sum(self.genre_weights.values())
        for genre in self.genre_weights:
            self.genre_weights[genre] *= len(self.genre_weights) / weight_sum
            
        # Log weights sorted by value
        logging.info("Class weights (higher = more important):")
        for genre, weight in sorted(self.genre_weights.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  {genre}: {weight:.4f}")
            
        # Visualize weights
        weights_df = pd.DataFrame.from_dict(self.genre_weights, orient='index', columns=['weight'])
        weights_df = weights_df.sort_values('weight', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='weight', y=weights_df.index, data=weights_df)
        plt.title('Class Weights for Loss Function')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualizations", "class_weights.png"))
        plt.close()
        
        return self.genre_weights
    
    def create_stratified_splits(self, test_size=0.15, val_size=0.15):
        """Create train/val/test splits preserving multi-label distribution"""
        if not hasattr(self, 'mlb') or self.mlb is None:
            logging.error("No genre encoding found. Call encode_genres() first.")
            return
            
        logging.info("Creating stratified data splits...")
        
        # Check for NaN values in genre columns
        genre_columns = [f'genre_{genre}' for genre in self.top_genres]
        nan_count = self.df[genre_columns].isna().sum().sum()
        
        if nan_count > 0:
            logging.warning(f"Found {nan_count} NaN values in genre columns. Filling with 0.")
            self.df[genre_columns] = self.df[genre_columns].fillna(0)
        
        # Get multi-hot matrix for stratification
        X = self.df.index.values.reshape(-1, 1)
        y = self.df[genre_columns].values
        
        # First split into train and temp using MultilabelStratifiedShuffleSplit
        msss_train = MultilabelStratifiedShuffleSplit(
            n_splits=1, 
            test_size=test_size + val_size,
            random_state=42
        )
        
        train_idx, temp_idx = next(msss_train.split(X, y))
        
        # Split temp into validation and test
        val_test_ratio = val_size / (test_size + val_size)
        X_temp = X[temp_idx]
        y_temp = y[temp_idx]
        
        msss_val = MultilabelStratifiedShuffleSplit(
            n_splits=1, 
            test_size=1-val_test_ratio,
            random_state=42
        )
        
        val_idx_temp, test_idx_temp = next(msss_val.split(X_temp, y_temp))
        
        # Convert temp indices back to original indices
        val_idx = temp_idx[val_idx_temp]
        test_idx = temp_idx[test_idx_temp]
        
        # Create split dataframes
        train_df = self.df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = self.df.iloc[val_idx].copy().reset_index(drop=True)
        test_df = self.df.iloc[test_idx].copy().reset_index(drop=True)
        
        logging.info(f"Created splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        # Verify genre distribution in each split
        self._verify_split_distribution(train_df, val_df, test_df)
        
        # Save the splits
        self._save_splits(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def _verify_split_distribution(self, train_df, val_df, test_df):
        """Verify genre distribution in each split"""
        logging.info("\nGenre distribution in splits:")
        logging.info(f"{'Genre':<15} {'Train %':>10} {'Val %':>10} {'Test %':>10}")
        logging.info("-" * 50)
        
        distributions = {}
        
        for genre in self.top_genres:
            col = f'genre_{genre}'
            train_pct = 100 * train_df[col].mean()
            val_pct = 100 * val_df[col].mean()
            test_pct = 100 * test_df[col].mean()
            
            distributions[genre] = {
                'train': train_pct,
                'val': val_pct,
                'test': test_pct
            }
            
            logging.info(f"{genre:<15} {train_pct:>10.2f} {val_pct:>10.2f} {test_pct:>10.2f}")
            
        # Plot distribution comparison
        plt.figure(figsize=(14, 10))
        
        x = np.arange(len(self.top_genres))
        width = 0.25
        
        plt.bar(x - width, [distributions[g]['train'] for g in self.top_genres], width, label='Train')
        plt.bar(x, [distributions[g]['val'] for g in self.top_genres], width, label='Val')
        plt.bar(x + width, [distributions[g]['test'] for g in self.top_genres], width, label='Test')
        
        plt.xlabel('Genres')
        plt.ylabel('Percentage')
        plt.title('Genre Distribution Across Splits')
        plt.xticks(x, self.top_genres, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualizations", "split_distribution.png"))
        plt.close()
        
        return distributions
    
    def _save_splits(self, train_df, val_df, test_df):
        """Save the data splits and configuration"""
        # Save CSV formats
        train_df.to_csv(os.path.join(self.output_dir, "splits", "train.csv"), index=False)
        val_df.to_csv(os.path.join(self.output_dir, "splits", "val.csv"), index=False)
        test_df.to_csv(os.path.join(self.output_dir, "splits", "test.csv"), index=False)
        
        # Save pickle formats
        train_df.to_pickle(os.path.join(self.output_dir, "splits", "train.pkl"))
        val_df.to_pickle(os.path.join(self.output_dir, "splits", "val.pkl"))
        test_df.to_pickle(os.path.join(self.output_dir, "splits", "test.pkl"))
        
        # Save configuration information
        config = {
            'top_genres': self.top_genres,
            'genre_weights': self.genre_weights,
            'drama_downsample_ratio': self.drama_downsample_ratio,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'multi_hot_columns': [f'genre_{genre}' for genre in self.top_genres],
            'threshold_suggestions': {
                # Default starting thresholds based on class frequency
                genre: 0.5 / self.genre_weights[genre] for genre in self.top_genres
            }
        }
        
        with open(os.path.join(self.output_dir, 'genre_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
            
        logging.info(f"Saved splits and configuration to {self.output_dir}")
    
    def run_full_pipeline(self):
        """Run the complete preprocessing pipeline"""
        logging.info("Starting full preprocessing pipeline...")
        
        self.load_data()
        self.analyze_genre_distribution()
        self.analyze_genre_combinations()
        self.filter_top_genres()
        self.downsample_drama()
        self.encode_genres()
        self.calculate_class_weights()
        self.create_stratified_splits()
        
        logging.info("Preprocessing pipeline complete!")


def main():
    """Main function to run the preprocessing pipeline"""
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set paths
    data_dir = os.path.join(script_dir, "data")
    processed_data_dir = os.path.join(data_dir, "processed")
    genre_output_dir = os.path.join(processed_data_dir, "genre_balanced")
    
    # Input file from your existing preprocessing
    input_path = os.path.join(processed_data_dir, "movies_processed.pkl")
    
    # Create preprocessor with options
    preprocessor = MovieGenrePreprocessor(
        input_path=input_path,
        output_dir=genre_output_dir,
        top_n_genres=10,  # Keep top 10 genres
        drama_downsample_ratio=0.5  # Reduce drama movies by half
    )
    
    # Run entire pipeline
    preprocessor.run_full_pipeline()


if __name__ == "__main__":
    main()