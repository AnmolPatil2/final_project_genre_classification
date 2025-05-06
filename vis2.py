import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_visualization.log"),
        logging.StreamHandler()
    ]
)

class MovieGenreVisualizer:
    """Class for visualizing the preprocessed movie genre dataset"""
    
    def __init__(self, data_dir, output_dir):
        """Initialize the visualizer"""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.genre_config = None
        self.top_genres = None
        self.genre_columns = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load the preprocessed data and configuration"""
        logging.info("Loading data...")
        
        # Load splits
        self.train_df = pd.read_pickle(os.path.join(self.data_dir, "splits", "train.pkl"))
        self.val_df = pd.read_pickle(os.path.join(self.data_dir, "splits", "val.pkl"))
        self.test_df = pd.read_pickle(os.path.join(self.data_dir, "splits", "test.pkl"))
        
        # Load genre configuration
        with open(os.path.join(self.data_dir, "genre_config.json"), "r") as f:
            self.genre_config = json.load(f)
            
        self.top_genres = self.genre_config["top_genres"]
        self.genre_columns = [f"genre_{genre}" for genre in self.top_genres]
        
        logging.info(f"Loaded {len(self.train_df)} training, {len(self.val_df)} validation, and {len(self.test_df)} test samples")
        logging.info(f"Working with {len(self.top_genres)} genres: {', '.join(self.top_genres)}")
        
        return True
    
    def plot_dataset_stats(self):
        """Plot basic dataset statistics"""
        logging.info("Plotting dataset statistics...")
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle("Movie Dataset Statistics", fontsize=16)
        
        # Plot 1: Genre distribution across splits
        ax1 = axes[0, 0]
        genre_counts = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for genre in self.top_genres:
            col = f"genre_{genre}"
            genre_counts['train'].append(self.train_df[col].sum())
            genre_counts['val'].append(self.val_df[col].sum())
            genre_counts['test'].append(self.test_df[col].sum())
        
        x = np.arange(len(self.top_genres))
        width = 0.25
        
        ax1.bar(x - width, genre_counts['train'], width, label='Train')
        ax1.bar(x, genre_counts['val'], width, label='Val')
        ax1.bar(x + width, genre_counts['test'], width, label='Test')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.top_genres, rotation=45, ha='right')
        ax1.set_title('Genre Distribution Across Splits')
        ax1.set_ylabel('Count')
        ax1.legend()
        
        # Plot 2: Label cardinality (genres per movie)
        ax2 = axes[0, 1]
        
        train_cardinality = self.train_df[self.genre_columns].sum(axis=1)
        val_cardinality = self.val_df[self.genre_columns].sum(axis=1)
        test_cardinality = self.test_df[self.genre_columns].sum(axis=1)
        
        sns.histplot(train_cardinality, bins=range(1, 11), kde=False, label='Train', alpha=0.7, ax=ax2)
        sns.histplot(val_cardinality, bins=range(1, 11), kde=False, label='Val', alpha=0.7, ax=ax2)
        sns.histplot(test_cardinality, bins=range(1, 11), kde=False, label='Test', alpha=0.7, ax=ax2)
        
        ax2.set_title('Number of Genres per Movie')
        ax2.set_xlabel('Number of Genres')
        ax2.set_ylabel('Count')
        ax2.set_xticks(range(1, max(train_cardinality.max(), val_cardinality.max(), test_cardinality.max()) + 1))
        ax2.legend()
        
        # Plot 3: Overview length distribution
        ax3 = axes[1, 0]
        
        sns.histplot(self.train_df['overview_length'], bins=20, kde=True, ax=ax3)
        ax3.set_title('Overview Length Distribution')
        ax3.set_xlabel('Overview Length (characters)')
        ax3.set_ylabel('Count')
        
        # Add vertical line for mean length
        mean_length = self.train_df['overview_length'].mean()
        ax3.axvline(mean_length, color='red', linestyle='--', 
                   label=f'Mean: {mean_length:.0f} chars')
        ax3.legend()
        
        # Plot 4: Genre co-occurrence matrix (heatmap)
        ax4 = axes[1, 1]
        
        # Calculate co-occurrence matrix
        co_matrix = np.zeros((len(self.top_genres), len(self.top_genres)))
        
        for i, genre1 in enumerate(self.top_genres):
            for j, genre2 in enumerate(self.top_genres):
                if i == j:
                    # Count of movies with this genre
                    co_matrix[i, j] = self.train_df[f"genre_{genre1}"].sum()
                else:
                    # Count of movies with both genres
                    co_matrix[i, j] = ((self.train_df[f"genre_{genre1}"] == 1) & 
                                       (self.train_df[f"genre_{genre2}"] == 1)).sum()
        
        # Normalize by diagonal (convert to percentages)
        normalized_matrix = np.zeros_like(co_matrix)
        for i in range(len(self.top_genres)):
            for j in range(len(self.top_genres)):
                if co_matrix[i, i] > 0:
                    normalized_matrix[i, j] = 100 * co_matrix[i, j] / co_matrix[i, i]
        
        # Plot heatmap
        sns.heatmap(normalized_matrix, annot=True, fmt=".1f", cmap="YlGnBu", 
                   xticklabels=self.top_genres, yticklabels=self.top_genres, ax=ax4)
        ax4.set_title('Genre Co-occurrence (% of Row Genre)')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()  # Changed from plt.savefig()
        
        logging.info("Dataset statistics plotted")
    
    def plot_class_weights(self):
        """Plot the class weights from the genre configuration"""
        logging.info("Plotting class weights...")
        
        genre_weights = self.genre_config["genre_weights"]
        
        # Convert to DataFrame and sort
        weights_df = pd.DataFrame.from_dict(
            genre_weights, orient='index', columns=['weight']
        ).sort_values('weight', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=weights_df.index, y='weight', data=weights_df)
        plt.title('Class Weights for Loss Function')
        plt.xlabel('Genre')
        plt.ylabel('Weight')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()  # Changed from plt.savefig()
        
        logging.info("Class weights plotted")
    
    def plot_genre_combinations(self):
        """Plot the most common genre combinations"""
        logging.info("Plotting common genre combinations...")
        
        # Function to get combination string from binary genres
        def get_combination(row):
            genres = []
            for genre, col in zip(self.top_genres, self.genre_columns):
                if row[col] == 1:
                    genres.append(genre)
            return ", ".join(sorted(genres))
        
        # Get combinations
        self.train_df['genre_combination'] = self.train_df.apply(get_combination, axis=1)
        
        # Count combinations
        combo_counts = self.train_df['genre_combination'].value_counts()
        
        # Plot top combinations
        plt.figure(figsize=(12, 8))
        combo_df = pd.DataFrame({'Combination': combo_counts.index, 'Count': combo_counts.values})
        sns.barplot(x='Count', y='Combination', data=combo_df.head(15))
        plt.title('Top 15 Genre Combinations')
        plt.tight_layout()
        plt.show()  # Changed from plt.savefig()
        
        logging.info("Genre combinations plotted")
    
    def visualize_overview_lengths_by_genre(self):
        """Visualize overview lengths by genre"""
        logging.info("Visualizing overview lengths by genre...")
        
        # Create DataFrame for plotting
        genre_lengths = []
        
        for genre, col in zip(self.top_genres, self.genre_columns):
            # Get overview lengths for movies with this genre
            lengths = self.train_df.loc[self.train_df[col] == 1, 'overview_length']
            for length in lengths:
                genre_lengths.append({'Genre': genre, 'Overview Length': length})
        
        genre_lengths_df = pd.DataFrame(genre_lengths)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Genre', y='Overview Length', data=genre_lengths_df)
        plt.title('Overview Length by Genre')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()  # Changed from plt.savefig()
        
        logging.info("Overview lengths by genre visualized")
    
    def visualize_tsne_clusters(self):
        """Visualize data clusters using t-SNE on genre combinations"""
        logging.info("Visualizing t-SNE clusters...")
        
        # Get genre matrix
        genre_matrix = self.train_df[self.genre_columns].values
        
        # Create color mapping based on primary genre
        primary_genres = []
        for i, row in enumerate(genre_matrix):
            # Get the index of the first genre (alphabetically among those present)
            present_genres = [self.top_genres[j] for j in range(len(row)) if row[j] == 1]
            if present_genres:
                primary_genres.append(sorted(present_genres)[0])
            else:
                primary_genres.append('Unknown')
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(genre_matrix)
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Get unique primary genres and colors
        unique_genres = sorted(set(primary_genres))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_genres)))
        
        # Plot each genre
        for i, genre in enumerate(unique_genres):
            mask = [pg == genre for pg in primary_genres]
            plt.scatter(
                tsne_result[mask, 0], 
                tsne_result[mask, 1], 
                c=[colors[i]], 
                label=genre,
                alpha=0.6
            )
        
        plt.title('t-SNE Visualization of Movies by Genre')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()  # Changed from plt.savefig()
        
        logging.info("t-SNE visualization complete")
    
    def run_all_visualizations(self):
        """Run all visualization functions"""
        logging.info("Running all visualizations...")
        
        self.load_data()
        self.plot_dataset_stats()
        self.plot_class_weights()
        self.plot_genre_combinations()
        self.visualize_overview_lengths_by_genre()
        self.visualize_tsne_clusters()
        
        logging.info("All visualizations complete!")


def main():
    """Main function to run visualizations"""
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set paths
    data_dir = os.path.join(script_dir, "data", "processed", "genre_balanced")
    output_dir = os.path.join(script_dir, "visualizations", "genre_balanced")
    
    # Create visualizer
    visualizer = MovieGenreVisualizer(data_dir, output_dir)
    
    # Run all visualizations
    visualizer.run_all_visualizations()


if __name__ == "__main__":
    main()