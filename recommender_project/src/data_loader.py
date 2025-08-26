import pandas as pd

def load_ratings(path='data/raw/ratings.csv'):
    return pd.read_csv(path, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])

def load_movies(path='data/raw/movies.csv'):
    # The movies file has pipe-separated values and many columns
    return pd.read_csv(path, sep='|', encoding='latin-1', names=[
        'movieId', 'title', 'release_date', 'video_release_date',
        'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s',
        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
        'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
        'War', 'Western'
    ])