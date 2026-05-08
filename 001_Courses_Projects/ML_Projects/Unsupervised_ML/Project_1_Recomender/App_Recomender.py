import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("🎬 Content Similarity Engine")

@st.cache_data
def load_data():
    # Placeholder: Replace with your dataset path
    data = pd.DataFrame({
        'title': ['The Matrix', 'Inception', 'Toy Story', 'Interstellar'],
        'genres': ['Sci-Fi Action', 'Sci-Fi Thriller', 'Animation Kids', 'Sci-Fi Drama']
    })
    return data

df = load_data()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movie_list = df['title'].values
selected_movie = st.selectbox("Select a movie to find similarities:", movie_list)

if st.button("Recommend"):
    idx = df.index[df['title'] == selected_movie][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:3]
    
    st.write("Top Recommendations:")
    for i in sim_scores:
        st.write(f"- {df['title'].iloc[i[0]]}")