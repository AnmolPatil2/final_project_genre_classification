import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import json
from PIL import Image
import random
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('fivethirtyeight')
sns.set_palette("Set2")

# Get the absolute path to the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Output directory for visualizations
output_dir = os.path.join(script_dir, 'visualizations')
os.makedirs(output_dir, exist_ok=True)

# Download NLTK resources if needed
try:
    nltk.download('stopwords', quiet=True)
except:
    print("Could not download NLTK stopwords. Some visualizations may not work.")

def load_data():
    """Load the processed movie data"""
    try:
        # Try to load the filtered dataset first
        filtered_path = os.path.join(script_dir, "data", "movies_filtered.csv")
        if os.path.exists(filtered_path):
            df = pd.read_csv(filtered_path)
            print(f"Loaded filtered dataset with {len(df)} movies")
        else:
            # Fall back to original dataset
            original_path = os.path.join(script_dir, "movie_dataset_complete", "movies_data.csv")
            df = pd.read_csv(original_path)
            print(f"Loaded original dataset with {len(df)} movies")
        
        # Convert genres from string to list if needed
        if isinstance(df['genres'].iloc[0], str):
            df['genres'] = df['genres'].apply(eval)
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_genre_distribution(df):
    """Plot the distribution of movie genres"""
    print("Plotting genre distribution...")
    
    # Count genres
    genre_counts = {}
    for genres_list in df['genres']:
        for genre in genres_list:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    # Convert to dataframe for plotting
    genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
    genre_df = genre_df.sort_values('Count', ascending=False)
    
    # Calculate percentages
    total_movies = len(df)
    genre_df['Percentage'] = genre_df['Count'] / total_movies * 100
    
    # Plot horizontal bar chart
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Count', y='Genre', data=genre_df)
    
    # Add percentage labels
    for i, row in enumerate(genre_df.itertuples()):
        ax.text(row.Count + 5, i, f"{row.Percentage:.1f}%", va='center')
    
    plt.title('Distribution of Movie Genres', fontsize=16)
    plt.xlabel('Number of Movies', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'genre_distribution.png'), dpi=300)
    plt.show()  # Show plot interactively
    plt.close()
    
    return genre_df

def plot_genre_combinations(df, top_n=10):
    """Plot most common genre combinations"""
    print("Plotting genre combinations...")
    
    # Get genre combinations
    genre_combos = Counter(tuple(sorted(genres)) for genres in df['genres'])
    top_combos = genre_combos.most_common(top_n)
    
    # Create dataframe
    combo_df = pd.DataFrame(top_combos, columns=['Genres', 'Count'])
    combo_df['Genres'] = combo_df['Genres'].apply(lambda x: ', '.join(x))
    
    # Plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Count', y='Genres', data=combo_df)
    
    plt.title(f'Top {top_n} Genre Combinations', fontsize=16)
    plt.xlabel('Number of Movies', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'genre_combinations.png'), dpi=300)
    plt.show()  # Show plot interactively
    plt.close()

def plot_overview_length_distribution(df):
    """Plot distribution of overview lengths"""
    print("Plotting overview length distribution...")
    
    # Calculate lengths
    df['overview_length'] = df['overview'].astype(str).apply(len)
    
    plt.figure(figsize=(12, 6))
    
    # Histogram
    sns.histplot(df['overview_length'], bins=50, kde=True)
    
    # Add vertical lines for mean and median
    plt.axvline(df['overview_length'].mean(), color='r', linestyle='--', 
                label=f'Mean: {df["overview_length"].mean():.0f} chars')
    plt.axvline(df['overview_length'].median(), color='g', linestyle='-', 
                label=f'Median: {df["overview_length"].median():.0f} chars')
    
    # Add text for min and max
    plt.text(0.05, 0.95, f"Min: {df['overview_length'].min()} chars", 
             transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.90, f"Max: {df['overview_length'].max()} chars", 
             transform=plt.gca().transAxes, fontsize=12)
    
    plt.title('Distribution of Movie Overview Lengths', fontsize=16)
    plt.xlabel('Number of Characters', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overview_length_distribution.png'), dpi=300)
    plt.show()  # Show plot interactively
    plt.close()
    
    return df['overview_length'].describe()

def generate_wordcloud_by_genre(df, top_genres=None):
    """Generate word clouds for top genres"""
    print("Generating word clouds by genre...")
    
    # Get stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()
    
    # Add more domain-specific stopwords
    custom_stopwords = {'one', 'two', 'three', 'find', 'must', 'film', 'movie', 'story'}
    stop_words.update(custom_stopwords)
    
    # Get top genres if not specified
    if top_genres is None:
        genre_counts = {}
        for genres_list in df['genres']:
            for genre in genres_list:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        top_genres = [genre for genre, _ in top_genres]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, genre in enumerate(top_genres[:8]):  # Limit to 8 genres for the subplot
        # Get all overviews for movies in this genre
        genre_overviews = ' '.join(df[df['genres'].apply(lambda x: genre in x)]['overview'].astype(str))
        
        # Generate wordcloud
        wordcloud = WordCloud(
            background_color='white',
            max_words=100,
            stopwords=stop_words,
            width=400,
            height=400,
            contour_width=1,
            contour_color='steelblue'
        ).generate(genre_overviews)
        
        # Plot
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(genre, fontsize=16)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'genre_wordclouds.png'), dpi=300)
    plt.show()  # Show plot interactively
    plt.close()

def plot_genres_over_time(df):
    """Plot how genre popularity has changed over time"""
    print("Plotting genre trends over time...")
    
    # Extract year from release_date
    try:
        if 'release_date' in df.columns:
            df['year'] = pd.to_datetime(df['release_date']).dt.year
        elif 'release_year' in df.columns:
            df['year'] = df['release_year']
        else:
            print("No release date or year column found")
            return
    except:
        print("Could not extract year from release date")
        return
    
    # Filter to reasonable year range
    df = df[(df['year'] >= 1970) & (df['year'] <= 2025)]
    
    # Get top 8 genres
    genre_counts = {}
    for genres_list in df['genres']:
        for genre in genres_list:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    top_genres = [genre for genre, count in 
                 sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:8]]
    
    # Create a dataframe for year and genre counts
    year_genre_counts = pd.DataFrame()
    
    for genre in top_genres:
        # Count movies with this genre per year
        genre_by_year = df[df['genres'].apply(lambda x: genre in x)].groupby('year').size()
        year_genre_counts[genre] = genre_by_year
    
    # Fill NaN with 0
    year_genre_counts = year_genre_counts.fillna(0)
    
    # Plot
    plt.figure(figsize=(14, 8))
    for genre in top_genres:
        plt.plot(year_genre_counts.index, year_genre_counts[genre], linewidth=2, label=genre)
    
    plt.title('Genre Popularity Over Time', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Movies', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'genre_trends_over_time.png'), dpi=300)
    plt.show()  # Show plot interactively
    plt.close()

def plot_genre_correlations(df):
    """Plot correlation matrix between genres"""
    print("Plotting genre correlations...")
    
    # Get all unique genres
    all_genres = set()
    for genres in df['genres']:
        all_genres.update(genres)
    
    # Create binary columns for each genre
    for genre in all_genres:
        df[f'is_{genre}'] = df['genres'].apply(lambda x: 1 if genre in x else 0)
    
    # Calculate correlation matrix
    genre_cols = [f'is_{genre}' for genre in all_genres]
    corr_matrix = df[genre_cols].corr()
    
    # Clean up column names for better display
    corr_matrix.columns = [col[3:] for col in corr_matrix.columns]
    corr_matrix.index = [idx[3:] for idx in corr_matrix.index]
    
    # Plot heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('Correlation Between Movie Genres', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'genre_correlations.png'), dpi=300)
    plt.show()  # Show plot interactively
    plt.close()

def visualize_poster_samples(df, save_grid=True):
    """Visualize sample posters for each genre"""
    print("Visualizing poster samples...")
    
    # Get poster directory
    poster_dir = os.path.join(script_dir, "movie_dataset_complete", "posters")
    
    # Check if we have access to the posters
    if not os.path.exists(poster_dir):
        print(f"Poster directory not found: {poster_dir}")
        return
    
    # Get top genres
    genre_counts = {}
    for genres_list in df['genres']:
        for genre in genres_list:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    top_genres = [genre for genre, count in 
                 sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:8]]
    
    # Create a figure with subplots
    if save_grid:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
    
    for i, genre in enumerate(top_genres):
        # Get movies with this genre
        genre_movies = df[df['genres'].apply(lambda x: genre in x)]
        
        # Sample 1 movie
        sampled_movie = genre_movies.sample(1).iloc[0]
        
        # Try to get poster
        poster_path = sampled_movie['poster_path']
        
        # If poster_path is just a filename, prepend the directory
        if isinstance(poster_path, str) and not os.path.isabs(poster_path):
            poster_path = os.path.join(poster_dir, poster_path)
        
        try:
            img = Image.open(poster_path)
            
            if save_grid:
                # Plot in the grid
                axes[i].imshow(img)
                axes[i].set_title(f"{genre}: {sampled_movie['title']}", fontsize=12)
                axes[i].axis('off')
            else:
                # Save individual poster
                img.save(os.path.join(output_dir, f"sample_poster_{genre}.jpg"))
        except Exception as e:
            print(f"Could not load poster for {genre}: {e}")
            if save_grid:
                axes[i].text(0.5, 0.5, f"No poster available for {genre}", 
                             ha='center', va='center')
                axes[i].axis('off')
    
    if save_grid:
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'genre_poster_samples.png'), dpi=300)
        plt.show()  # Show plot interactively
        plt.close()

def plot_multi_genre_distribution(df):
    """Plot distribution of number of genres per movie"""
    print("Plotting multi-genre distribution...")
    
    # Count number of genres per movie
    df['genre_count'] = df['genres'].apply(len)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Count movies with each number of genres
    genre_count_dist = df['genre_count'].value_counts().sort_index()
    
    # Bar plot
    ax = sns.barplot(x=genre_count_dist.index, y=genre_count_dist.values)
    
    # Add count and percentage labels
    total = len(df)
    for i, count in enumerate(genre_count_dist.values):
        percentage = count / total * 100
        ax.text(i, count + 5, f"{count}\n({percentage:.1f}%)", ha='center')
    
    plt.title('Number of Genres per Movie', fontsize=16)
    plt.xlabel('Number of Genres', fontsize=14)
    plt.ylabel('Number of Movies', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_genre_distribution.png'), dpi=300)
    plt.show()  # Show plot interactively
    plt.close()
    
    return df['genre_count'].describe()

def generate_statistics_summary(df, genre_df=None):
    """Generate a summary of key statistics about the dataset"""
    print("Generating statistics summary...")
    
    # Basic dataset stats
    total_movies = len(df)
    
    # Genre stats
    total_genres = len(set(genre for genres in df['genres'] for genre in genres))
    
    # Get genre counts if not provided
    if genre_df is None:
        genre_counts = {}
        for genres_list in df['genres']:
            for genre in genres_list:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
        genre_df = genre_df.sort_values('Count', ascending=False)
    
    top_genre = genre_df.iloc[0]['Genre']
    top_genre_count = genre_df.iloc[0]['Count']
    top_genre_percentage = top_genre_count / total_movies * 100
    
    # Overview stats
    df['overview_length'] = df['overview'].astype(str).apply(len)
    avg_overview_length = df['overview_length'].mean()
    max_overview_length = df['overview_length'].max()
    min_overview_length = df['overview_length'].min()
    
    # Multi-genre stats
    df['genre_count'] = df['genres'].apply(len)
    avg_genres_per_movie = df['genre_count'].mean()
    max_genres = df['genre_count'].max()
    single_genre_count = (df['genre_count'] == 1).sum()
    single_genre_percentage = single_genre_count / total_movies * 100
    multi_genre_percentage = 100 - single_genre_percentage
    
    # Create summary
    summary = {
        "dataset_stats": {
            "total_movies": total_movies,
            "total_genres": total_genres,
            "top_genre": top_genre,
            "top_genre_count": int(top_genre_count),
            "top_genre_percentage": float(top_genre_percentage)
        },
        "overview_stats": {
            "avg_length": float(avg_overview_length),
            "max_length": int(max_overview_length),
            "min_length": int(min_overview_length)
        },
        "genre_stats": {
            "avg_genres_per_movie": float(avg_genres_per_movie),
            "max_genres_per_movie": int(max_genres),
            "single_genre_percentage": float(single_genre_percentage),
            "multi_genre_percentage": float(multi_genre_percentage)
        }
    }
    
    # Save to file
    with open(os.path.join(output_dir, 'dataset_statistics.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print summary
    print("\n----- Dataset Statistics -----")
    print(f"Total movies: {total_movies}")
    print(f"Total genres: {total_genres}")
    print(f"Most common genre: {top_genre} ({top_genre_count} movies, {top_genre_percentage:.1f}%)")
    print(f"Average overview length: {avg_overview_length:.1f} characters")
    print(f"Average genres per movie: {avg_genres_per_movie:.2f}")
    print(f"Single-genre movies: {single_genre_percentage:.1f}%")
    print(f"Multi-genre movies: {multi_genre_percentage:.1f}%")
    
    return summary

def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Create visualizations
    genre_df = plot_genre_distribution(df)
    plot_genre_combinations(df)
    plot_overview_length_distribution(df)
    try:
        generate_wordcloud_by_genre(df)
    except Exception as e:
        print(f"Could not generate wordclouds: {e}")
    
    plot_genres_over_time(df)
    plot_genre_correlations(df)
    try:
        visualize_poster_samples(df)
    except Exception as e:
        print(f"Could not visualize poster samples: {e}")
    
    plot_multi_genre_distribution(df)
    
    # Generate summary statistics
    generate_statistics_summary(df, genre_df)
    
    print(f"\nVisualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()