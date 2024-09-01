import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data():
    return pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

games = load_data()


games = games.dropna(subset=['Name', 'Genre'])


games['Name'] = games['Name'].str.lower().str.replace(f'[{re.escape(string.punctuation)}]', '', regex=True)
games['Genre'] = games['Genre'].str.lower()


games['tags'] = games['Name'] + ' ' + games['Genre']


tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
tag_matrix = tfidf.fit_transform(games['tags'])


similarity = cosine_similarity(tag_matrix)


def recommend(game_name):
    clean_name = re.sub(f'[{re.escape(string.punctuation)}]', '', game_name.lower())
    
    matching_games = games[games['Name'].str.contains(clean_name, na=False)]
    
    if matching_games.empty:
        st.write(f"No game found with the name '{game_name}'. Please check the name and try again.")
        return pd.DataFrame()  
    
    index = matching_games.index[0]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_games = [games.iloc[i[0]].Name for i in similarity_scores[1:6]]
    
    
    recommendations_df = pd.DataFrame({
        'Recommended Games': recommended_games
    })
    
    return recommendations_df


st.title('Game Recommendation System')
names = games['Name']

game_name = st.selectbox('Enter a game name:', names)

if st.button('Recommend'):
    recommendations_df = recommend(game_name)
    
    if not recommendations_df.empty:
        st.write('Recommendations:')
        st.dataframe(recommendations_df, width=1000)
