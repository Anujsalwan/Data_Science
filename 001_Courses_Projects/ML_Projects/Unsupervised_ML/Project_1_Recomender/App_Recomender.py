import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("🎬 Content Similarity Engine")

@st.cache_data

def load_data():
    # Updated to the raw user content URL
    url = "https://raw.githubusercontent.com/Anujsalwan/Data_Science/main/001_Courses_Projects/ML_Projects/Unsupervised_ML/Project_1_Recomender/anime.csv"
    data = pd.read_csv(url)
    return data

df = load_data()
df.columns = df.columns.str.strip()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genre'].dropna())
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movie_list = df['name'].values
selected_movie = st.selectbox("Select a movie to find similarities:", movie_list)

if st.button("Recommend"):
    idx = df.index[df['name'] == selected_movie][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:3]
    
    st.write("Top Recommendations:")
    for i in sim_scores:
        st.write(f"- {df['title'].iloc[i[0]]}")