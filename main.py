import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

users = pd.read_csv("u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
movies = pd.read_csv("Movie_Id_Titles.csv")

df = pd.merge(users, movies, on="item_id")

def recommend_movies(movie_title, k=6):
    movies_df = df.pivot_table(index="title", columns="user_id", values="rating").fillna(0)
    movies_df_metrix = csr_matrix(movies_df.values)

    model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
    model_knn.fit(movies_df_metrix)

    selected_movie_index = movies_df.index.get_loc(movie_title)
    distances, indices = model_knn.kneighbors(movies_df.iloc[selected_movie_index, :].values.reshape(1, -1), n_neighbors=k)

    recommendations = []
    for i in range(1, len(distances.flatten())):  # Skip the original movie
        recommendations.append((movies_df.index[indices.flatten()[i]], distances.flatten()[i]))

    return recommendations

st.title("Movie Recommendation System - AI ML Project")
selected_movie = st.selectbox("Select a movie:", df["title"].unique())

if st.button("Recommend similar movies"):
    recommendations = recommend_movies(selected_movie)

    st.subheader(f"Recommendations for {selected_movie}:")
    columns = st.columns(len(recommendations))
    for idx, (movie, distance) in enumerate(recommendations):
        if "The" in movie:
            modified_string = movie.rsplit('The ', 1)
            title = modified_string[0].strip()
            year = modified_string[1].strip()
            movie = f"The {title}{', ' if year else ''}{year}".strip().replace(',', '')

        title = movie.split('(')[0]

        url = f"http://www.omdbapi.com/?t={title}&apikey=3a28395a"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            posterUrl = data.get('Poster')
            Moviename = data.get('Title')
            # moviePlot = data.get('Plot')

            if posterUrl:
                with columns[idx]:
                    response = requests.get(posterUrl)
                    img = Image.open(BytesIO(response.content)).resize((200, 300))
                    st.image(img, caption=Moviename)
        else:
            st.write(f"Could not retrieve data for {movie}.")
