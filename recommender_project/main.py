from surprise import Dataset, Reader, accuracy
from src.data_loader import load_ratings, load_movies
from src.recommender import tune_svd_model, get_top_n_recommendations

# Step 1: Load the data
ratings_df = load_ratings()
movies_df = load_movies()

# Step 2: Get best SVD model from hyperparameter tuning
model = tune_svd_model(ratings_df)

# Step 3: Build full trainset and train the tuned model
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
model.fit(trainset)

# Step 4: Evaluate using cross-validation RMSE
# This is optional now since we already got RMSE from tuning, but here’s how to test manually
testset = trainset.build_testset()
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print("Final RMSE on full trainset:", rmse)

# Step 5: Generate top-N recommendations for a user
user_id = 660  # replace with a real userId from your data
top_recs = get_top_n_recommendations(model, trainset, movies_df, user_id, n=5)

print(f"\nTop 5 movie recommendations for user {user_id}:\n")
print(top_recs)
