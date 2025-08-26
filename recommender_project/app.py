import streamlit as st
import pandas as pd
from surprise import Dataset, Reader
from src.data_loader import load_ratings, load_movies
from src.recommender import tune_svd_model, get_top_n_recommendations

@st.cache_data(show_spinner=False)
def load_data():
    ratings_df = load_ratings()
    movies_df = load_movies()
    return ratings_df, movies_df

@st.cache_resource(show_spinner=False)
def train_model(ratings_df):
    model = tune_svd_model(ratings_df)
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    model.fit(trainset)
    return model, trainset

def main():
    st.title("🎬 Movie Recommender System")

    ratings_df, movies_df = load_data()
    model, trainset = train_model(ratings_df)

    user_id = st.number_input("Enter User ID", min_value=1, max_value=ratings_df['userId'].max(), step=1)

    if st.button("Get Recommendations"):
        try:
            recommendations = get_top_n_recommendations(model, trainset, movies_df, user_id, n=5)
            st.write(f"Top 5 movie recommendations for user {user_id}:")
            st.dataframe(recommendations.reset_index(drop=True))
        except ValueError:
            st.error(f"User ID {user_id} not found in training data. Please enter a valid user ID.")

if __name__ == "__main__":
    main()
