import pandas as pd

movies = pd.read_csv(r'D:\USTC\实验室\DRDT_implementation\DRDT_implementation\ml-25m\movies.csv', encoding='utf-8')
ratings = pd.read_csv(r'D:\USTC\实验室\DRDT_implementation\DRDT_implementation\ml-25m\ratings.csv', encoding='utf-8')
ratings = ratings.iloc[:1000000]

movie_basic_info = dict()


def extract_title_year(text):
    title = text.split('(')[0].strip()
    if title.endswith(', The'):
        title = "The " + title[:-6]

    year = text.split('(')[-1].replace(')', '').strip()

    return title.strip(), year.strip()

for row in movies.itertuples(index=False):
    movieId, title, genres = row
    # actors = agent.actor_generation(movieId)
    movie_basic_info[movieId] = {'title': extract_title_year(title)[0], 'year': extract_title_year(title)[1] ,'genres': genres.split('|')}