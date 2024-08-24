from langchain.prompts import PromptTemplate

### first loop: collaborative information
collaborative_info_instruction = """As an AI agent, you are tasked with analyzing and refining the movie preferences of me.

Format for the updated preferences analysis:
Genres: [Detailed analysis of the user's preferences based on movie genres]
Actors: [Detailed analysis of the user's preferences based on movie actors]
Directors: [Detailed analysis of the user's preferences based on movie directors]

Example:
Based on the list of movies you've watched, I can analyze your preferences in terms of movie genres, actors, and directors.
Genres: Your preferences appear to lean towards science fiction and action, with a notable interest in mystery/thriller and drama.
Actors: You seem to favor actors like Chris Hemsworth and Jennifer Lawrence, who have featured prominently in the films you've rated highly. For example, your high ratings for Thor: Ragnarok and The Hunger Games suggest an appreciation for their performances and the types of roles they play.
Directors: Your choices reflect a preference for directors such as Christopher Nolan and Ridley Scott. Movies like Interstellar and Prometheus show your inclination towards films with complex narratives and visually striking styles, often characteristic of these directors’ works.

Given the following information, please update my preferences analysis in the specified format:

1. The watched movie list of another user for context: {sample_watched_movies}.
2. A list of candidate movies presented to the another user: {candidates}.
3. The outcome of a previous recommendation to the another user: the another user sorted the candidates in the order of {answer}.
4. The watched movie list of me in the order they were watched: {watched_movies}.
5. My current preferences analysis: {preferences_analysis}.

Provide a refined analysis of my preferences in terms of genres, actors, and directors. Concentrate on the general trends in these areas rather than specific movies.
"""
collaborative_agent_prompt = PromptTemplate(
    input_variables=['sample_watched_movies', 'candidates', 'answer', 'watched_movies', 'preferences_analysis'],
    template=collaborative_info_instruction
)

### divetgent thinking
divergent_thinkng_instruction = """As an AI agent, your task is to update and refine my preferences analysis based on my viewing history and the current details provided.

Format for the updated preferences analysis:
Genres: [Detailed analysis of the user's preferences based on movie genres]
Actors: [Detailed analysis of the user's preferences based on movie actors]
Directors: [Detailed analysis of the user's preferences based on movie directors]

Example:
Based on the list of movies you've watched, I can analyze your preferences in terms of movie genres, actors, and directors.
Genres: Your preferences appear to lean towards science fiction and action, with a notable interest in mystery/thriller and drama.
Actors: You seem to favor actors like Chris Hemsworth and Jennifer Lawrence, who have featured prominently in the films you've rated highly. For example, your high ratings for Thor: Ragnarok and The Hunger Games suggest an appreciation for their performances and the types of roles they play.
Directors: Your choices reflect a preference for directors such as Christopher Nolan and Ridley Scott. Movies like Interstellar and Prometheus show your inclination towards films with complex narratives and visually striking styles, often characteristic of these directors’ works.

Given the following information, please update my preferences analysis accordingly:
Viewing History: {watched_movies}
Current Preferences Analysis: {preferences_analysis}

Provide a refined analysis of my preferences in terms of genres, actors, and directors. Concentrate on the general trends in these areas rather than specific movies.
"""
dt_agent_prompt = PromptTemplate(
    input_variables=['watched_movies', 'preferences_analysis',],
    template=divergent_thinkng_instruction,
    )

### probing
probing_instruction = """As an AI agent, your task is to predict which movie from a given list is most likely to align with my established preferences.

Here is my current preferences analysis, detailing my preferences in genres, actors, and directors:
{preferences_analysis}

Based on this analysis, consider the following list of candidate movies:
{candidates}

Using your understanding of my preferences, recommend the movie from the list that I am most likely to enjoy. Respond with the title of the recommended movie only.
"""
probe_agent_prompt = PromptTemplate(
    input_variables=['preferences_analysis', 'candidates'],
    template=probing_instruction,
)

### dynamic reflection
dynamic_reflection_instruction = """Task: Update the user's movie preferences analysis based on their actual selection compared to the AI agent's recommendation.

Preferences Analysis Format:
Genres: [Detailed analysis of the user's preferences in terms of movie genres]
Actors: [Detailed analysis of the user's preferences in terms of actors]
Directors: [Detailed analysis of the user's preferences in terms of directors]

Example:
Based on the list of movies you've watched, I can analyze your preferences in terms of movie genres, actors, and directors.
Genres: Your preferences appear to lean towards science fiction and action, with a notable interest in mystery/thriller and drama.
Actors: You seem to favor actors like Chris Hemsworth and Jennifer Lawrence, who have featured prominently in the films you've rated highly. For example, your high ratings for Thor: Ragnarok and The Hunger Games suggest an appreciation for their performances and the types of roles they play.
Directors: Your choices reflect a preference for directors such as Christopher Nolan and Ridley Scott. Movies like Interstellar and Prometheus show your inclination towards films with complex narratives and visually striking styles, often characteristic of these directors’ works.


Given Information:
Current Preferences Analysis:
{preferences_analysis}

AI Agent's Recommendation:
{recommendation} (chosen from: {candidates})

User's Actual Selection:
{answer}, rated {rating} out of 5


Analysis Request:
1. Compare the Recommendation and Selection: Identify key differences between the AI agent's recommendation and the user’s actual selection.
2. Update the Preferences Analysis.

Provide a refined analysis of my preferences in terms of genres, actors, and directors. Concentrate on the general trends in these areas rather than specific movies.
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
prediction_NDCG_instruction = """You are tasked with ranking movies from a provided list according to my personal movie preferences. 

I will provide you with:
1. A detailed analysis of my movie preferences, including my favored genres, actors, and directors.
2. A list of candidate movies.


Based on this analysis, please:
1. Review the preferences analysis focusing on genres, actors, and directors to understand the key factors influencing my movie selection. Concentrate on the general trends in these areas rather than specific movies.
2. Evaluate each movie in the list according to my preferences analysis.
3. Select and rank the top 10 movies from the candidate list in order of how well they align with my preferences.
4. Ensure that the final list reflects the highest alignment with my preferences, starting with the most suitable movie. Avoid any guesses or unrelated recommendations.

Preferences Analysis:
{preferences_analysis}

List of Candidate Movies:
{candidates}

Output the top 10 movies from the candidate list, ranked by their alignment with my preferences. Ensure that the ranking is based exclusively on the candidate list.
"""
predict_NDCG_agent_prompt = PromptTemplate(
    input_variables=['preferences_analysis', 'candidates'],
    template=prediction_NDCG_instruction,
)