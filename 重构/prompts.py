from langchain.prompts import PromptTemplate

### first loop: collaborative information
collaborative_info_instruction = """As an AI agent, you are tasked with analyzing and refining the movie preferences of me.

The format for the preferences analysis should be as follows:
- Based on the list of movies you have watched, I can analyze your preferences in terms of movie's genres, actors, and directors.
  Genres: [Detailed analysis of your preferences in terms of movie's genres]
  Actors: [Detailed analysis of your preferences in terms of movie's actors]
  Directors: [Detailed analysis of your preferences in terms of movie's directors]

Example:
Based on the list of movies Aim User has watched, I can analyze their preferences in terms of movie's genres, actors, and directors.
Genres: Your preferences seem to lean towards science fiction, action, and mystery/thriller genres.

Given the following information, please update my preferences analysis in the specified format:

1. The watched movie list of another user for context: {sample_watched_movies}.
2. A list of candidate movies presented to the another user: {candidates}.
3. The outcome of a previous recommendation to the another user: the another user sorted the candidates in the order of {answer}.
4. The watched movie list of me in the order they were watched: {watched_movies}.
5. My current preferences analysis: {preferences_analysis}.

Using the information provided, update my preferences analysis.
"""
collaborative_agent_prompt = PromptTemplate(
    input_variables=['sample_watched_movies', 'candidates', 'answer', 'watched_movies', 'preferences_analysis'],
    template=collaborative_info_instruction
)

### divetgent thinking
divergent_thinkng_instruction = """As an AI agent, your task is to refine and update my preferences analysis based on my viewing history and the details provided.

The format for the preferences analysis should be as follows:
- Based on the list of movies you've watched, I can analyze your preferences in terms of movie's genres, actors, and directors.
  Genres: [Detailed analysis of the user's preferences in terms of movie's genres]
  Actors: [Detailed analysis of the user's preferences in terms of movie's actors]
  Directors: [Detailed analysis of the user's preferences in terms of movie's directors]

Example:
Based on the list of movies you've watched, I can analyze your preferences in terms of movie's genres, actors, and directors.
Genres: Your preferences seem to favor science fiction, action, and mystery/thriller genres.

Given the following information, please update my preferences analysis in the specified format:

1. My viewing history includes the following movies in the order I've watched them: {watched_movies}.
2. My current preferences analysis is as follows: {preferences_analysis}.

Perform an analysis that reflects these details and update my preferences analysis accordingly.
"""
dt_agent_prompt = PromptTemplate(
    input_variables=['watched_movies', 'preferences_analysis',],
    template=divergent_thinkng_instruction,
    )

### probing
probing_instruction = """As an AI agent, your objective is to predict the next movie that I am most likely to enjoy based on my established preferences.

Below is my current preferences analysis, which outlines my preferences in terms of genres, actors, and directors:
{preferences_analysis}

Considering these preferences, I am provided with the following list of movies to choose from:
{candidates}

Please utilize my preferences analysis to predict and suggest the film from the list that I am most likely to select for my next viewing experience. Return only the title of the recommended movie.
"""
probe_agent_prompt = PromptTemplate(
    input_variables=['preferences_analysis', 'candidates'],
    template=probing_instruction,
)

### dynamic reflection
dynamic_reflection_instruction = """As an AI agent, your task is to critically analyze and update the user's preferences for movies based on the user's current preferences analysis and the interaction between AI recommendations and user decisions.

Preferences Analysis Format:
Please provide the updated preferences analysis in the following structured format:
- Based on the list of movies you've watched, I can analyze your preferences in terms of movie's genres, actors, and directors.
  Genres: [Detailed analysis of the user's preferences in terms of movie's genres]
  Actors: [Detailed analysis of the user's preferences in terms of movie's actors]
  Directors: [Detailed analysis of the user's preferences in terms of movie's directors]

Example:
Based on the list of movies you've watched, I can analyze your preferences in terms of movie's genres, actors, and directors.
Genres: Your preferences seem to favor science fiction and action, with a notable interest in mystery/thriller and drama.

Given the following information, update the user's preferences analysis:

Current Preferences Analysis:
{preferences_analysis}

AI Agent's Recommendation:
{recommendation} was selected by the AI agent as the preferred choice from the list of options: {candidates}.

User's Actual Selection:
The user's actual selection was {answer}, which was rated {rating} out of 5.

Analysis Request:
Reflect on the differences between the AI agent's recommendation and the user's actual selection. Update the user's preferences analysis to reflect any shifts in preferences based on the user's experience with the selected movie. The updated analysis should include a comparison of the genres, actors, and directors that were most influential in the user's decision.
"""
dr_agent_prompt = PromptTemplate(
    input_variables=['preferences_analysis', 'recommendation', 'candidates', 'answer', 'rating'],
    template=dynamic_reflection_instruction
)

### prediction
prediction_instruction = """As an AI agent, your mission is to identify a movie from the given list of options that aligns with my personal preferences for film selection.

My current preferences analysis is as follows, detailing my inclinations towards genres, actors, and directors:
{preferences_analysis}

Here is the complete list of movies from which I would like you to choose:
{candidates}

Please utilize my preferences analysis to recommend a movie that I am most likely to enjoy and select the most suitable title from the provided list.
"""
predict_agent_prompt = PromptTemplate(
    input_variables=['preferences_analysis', 'candidates'],
    template=prediction_instruction
)

### prediction_NDCG
prediction_NDCG_instruction = """As an AI agent, your mission is to rank the movies from the given list based on how well they align with my personal preferences for movie selection.

My current preferences analysis is as follows, detailing my inclinations towards genres, actors, and directors:
{preferences_analysis}

Here is the complete list of movies:
{candidates}

Please utilize my preferences analysis to select top 10 movies from the candidates and sort them in order of how likely I am to enjoy them, starting with the most suitable title.
"""
predict_NDCG_agent_prompt = PromptTemplate(
    input_variables=['preferences_analysis', 'candidates'],
    template=prediction_NDCG_instruction,
)