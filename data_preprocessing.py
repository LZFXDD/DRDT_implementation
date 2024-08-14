import pandas as pd

import openai

# openai_api设置
openai.api_key = 'sk-fph6emDpXF4gpDqOFe827a075561421d81C4442c5e489d37'
openai.api_base = 'https://free.gpt.ge/v1'


movies = pd.read_csv(r'D:\USTC\实验室\DRDT_implementation\DRDT_implementation\ml-25m\movies.csv', encoding='utf-8')
ratings = pd.read_csv(r'D:\USTC\实验室\DRDT_implementation\DRDT_implementation\ml-25m\ratings.csv', encoding='utf-8')
ratings = ratings.iloc[:1000000]

movie_info = dict()
user_history = dict()

### 处理movie_info
def extract_title_year(text):
    title = text.split('(')[0].strip()
    if title.endswith(', The'):
        title = "The " + title[:-6]

    year = text.split('(')[-1].replace(')', '').strip()

    return title.strip(), year.strip()

for row in movies.itertuples(index=False):
    movieId, title, genres = row
    # actors = agent.actor_generation(movieId)
    movie_info[movieId] = {'title': extract_title_year(title)[0], 'year': extract_title_year(title)[1] ,'genres': genres.split('|')}

print(movie_info[1])

### 处理user_history
groups = ratings.groupby('userId')
for name, group in groups:
    group = group.sort_values(by='timestamp')

    user_movies = []
    user_ratings = []
    for _, row in group.iterrows():
        userId, movieId, rating, timestamp = row
        if rating >= 4:
            user_movies.append(int(movieId))
            user_ratings.append(rating)

    user_history[name] = [user_movies, user_ratings]

print("success")
