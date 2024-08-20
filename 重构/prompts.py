from langchain.prompts import PromptTemplate

### first loop: collaborative information
collaborative_info_instruction = """You are an AI agent, aiming to analyze the user's preferences for movies. Based on the user's viewing history, you should generate a preferences analysis in the following format.

Format:
Based on the list of movies you've watched, I can analyze your preferences in terms of movie's genres, actors and directors.
The preferences are:
Genres: ... ( Analysis on the user's preferences in terms of movie's genres )
Actors: ... ( Analysis on the user's preferences in terms of movie's actors )
Directors: ... ( Analysis on the user's preferences in terms of movie's directors )

Example:
Based on the list of movies you've watched, I can analyze your preferences in terms of movie's genres, actors and directors.
Genres: Your preferences seem to lean towards science fiction, action, and mystery/thriller genres. 
Actors: You might have a preference for actors who can deliver strong performances in action and sci-fi roles.
Directors: You tend to favor directors known for their work in the action and science fiction genres, particularly those who are recognized for crafting intense and thought-provoking narratives.


Given the following information, update my preferences analysis that adheres to the specified format:

There is another user watched movie {sample_watched_movies}. Given the candidates {candidates}, the answer is {answer}.

I have watched the following list of movies in the past in this order: {watched_movies}.

Here is my current preferences analysis:
{preferences_analysis}

Update my preferences analysis.
"""
collaborative_agent_prompt = PromptTemplate(
    input_variables=['sample_watched_movies', 'candidates', 'answer', 'watched_movies', 'preferences_analysis'],
    template=collaborative_info_instruction
)

### divetgent thinking
divergent_thinkng_instruction = """You are an AI agent, aiming to analyze the user's preferences for movies. Based on the user's viewing history, you should generate a preferences analysis in the following format.

Format:
Based on the list of movies you've watched, I can analyze your preferences in terms of movie's genres, actors and directors.
The preferences are:
Genres: ... ( Analysis on the user's preferences in terms of movie's genres )
Actors: ... ( Analysis on the user's preferences in terms of movie's actors )
Directors: ... ( Analysis on the user's preferences in terms of movie's directors )

Example:
Based on the list of movies you've watched, I can analyze your preferences in terms of movie's genres, actors and directors.
Genres: Your preferences seem to lean towards science fiction, action, and mystery/thriller genres. 
Actors: You might have a preference for actors who can deliver strong performances in action and sci-fi roles.
Directors: You tend to favor directors known for their work in the action and science fiction genres, particularly those who are recognized for crafting intense and thought-provoking narratives.


Given the following information, update my preferences analysis that adheres to the specified format:

I have watched the following list of movies in the past in this order: {watched_movies}.

Here is my current preferences analysis:
{preferences_analysis}

Update my preferences analysis.
"""
dt_agent_prompt = PromptTemplate(
    input_variables=['watched_movies', 'preferences_analysis',],
    template=divergent_thinkng_instruction,
    )

### probing
probing_instruction = """You are an AI agent, aiming to forecast the next possible movie that I would like to watch.

Here is my current preferences analysis:
{preferences_analysis}

Options: {candidates}

Among the options provided, which film would I be inclined to watch next? Provide only the movie's title.
"""
probe_agent_prompt = PromptTemplate(
    input_variables=['preferences_analysis', 'candidates'],
    template=probing_instruction,
)

### dynamic reflection
dynamic_reflection_instruction = """As an AI agent, your role is to reflect upon and update the preferences analysis based on the decisions made by the AI agent and the actual selection made by me.

Preferences Analysis Format:
Based on the list of movies you've watched, I can analyze your preferences in terms of movie's genres, actors and directors.
The preferences are:
Genres: ... ( Analysis on the user's preferences in terms of movie's genres )
Actors: ... ( Analysis on the user's preferences in terms of movie's actors )
Directors: ... ( Analysis on the user's preferences in terms of movie's directors )

Example:
Based on the list of movies you've watched, I can analyze your preferences in terms of movie's genres, actors and directors.
Genres: Your preferences seem to lean towards science fiction, action, and mystery/thriller genres. 
Actors: You might have a preference for actors who can deliver strong performances in action and sci-fi roles.
Directors: You tend to favor directors known for their work in the action and science fiction genres, particularly those who are recognized for crafting intense and thought-provoking narratives.


Given the following information, update my preferences analysis that adheres to the specified format:

Here is my current preferences analysis in terms of genres, actors and directors:
{preferences_analysis}

{recommendation} has been selected by AI agent as the preferred choice from the list of options: {candidates}.

My actual selection is {answer}, and I've given it a rating of {rating} out of 5.
Refine my preferences analysis by comparing it with the current preferences analysis, taking into account the differences between the movie you recommended based on my current preferences analysis and the target movie I ended up watching. The updated preferences analysis should reflect any shifts based on the experience.
"""
dr_agent_prompt = PromptTemplate(
    input_variables=['preferences_analysis', 'recommendation', 'candidates', 'answer', 'rating'],
    template=dynamic_reflection_instruction
)

### prediction
prediction_instruction = """You are an AI agent, tasked with selecting on movie from the provided list of options that I would enjoy watching according to my preferences analysis.

Here is my current preferences analysis in terms of genres, actors and directors:
{preferences_analysis}

Here is the provided list of options: {candidates}

Based on my preferences analysis, select a movie that I am most likely to watch:
"""
predict_agent_prompt = PromptTemplate(
    input_variables=['preferences_analysis', 'candidates'],
    template=prediction_instruction
)