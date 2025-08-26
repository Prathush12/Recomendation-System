from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from surprise import accuracy
import pandas as pd

def tune_svd_model(ratings_df):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

    param_grid = {
        'n_epochs': [20, 30],
        'lr_all': [0.002, 0.005],
        'reg_all': [0.2, 0.4]
    }

    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
    gs.fit(data)

    print("Best RMSE score from tuning:", gs.best_score['rmse'])
    print("Best parameters:", gs.best_params['rmse'])

    return gs.best_estimator['rmse']


def get_top_n_recommendations(model, trainset, movies_df, user_id, n=5):
    # Convert userId to the internal Surprise ID
    inner_id = trainset.to_inner_uid(user_id)

    # All movies in the training set
    all_movie_ids = trainset._raw2inner_id_items.keys()

    # Movies the user has already rated
    rated_items = set([trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[inner_id]])

    # Filter out rated movies
    unseen_movies = [iid for iid in all_movie_ids if iid not in rated_items]

    # Predict ratings for unseen movies
    predictions = [
        (movie_id, model.predict(str(user_id), movie_id).est)
        for movie_id in unseen_movies
    ]

    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Top N movies
    top_n = predictions[:n]
    top_movie_ids = [int(movie_id) for movie_id, _ in top_n]

    return movies_df[movies_df['movieId'].isin(top_movie_ids)][['movieId', 'title']]
