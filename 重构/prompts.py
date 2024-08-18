from langchain.prompts import PromptTemplate

### first loop: collaborative information
collaborative_info_instruction = """"""
collaborative_agent_prompt = PromptTemplate()

### divetgent thinking
divergent_thinkng_instruction = """You are an AI agent, aiming to analyze the user's preferences for movies based on specified terms in the given format.

Here is the format for the user's preferences analysis, along with an example:

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

I have watched the following list of movies in the past in this order: {watched_movies}.

Here is my preferences analysis:
{preference_analysis}

Could you please update my preferences analysis according to the list provided, taking into account the movie's genres, actors and directors?
"""
dt_agent_prompt = PromptTemplate(
    input_variabls=['watched_movies', 'preference_analysis',],
    template=divergent_thinkng_instruction,
    )

### probing
probing_instruction = """"""
probe_agent_prompt = PromptTemplate()

### dynamic reflection
dynamic_reflection_instruction = """"""
dr_agent_prompt = PromptTemplate()

### prediction
prediction_instruction = """"""
predict_agent_prompt = PromptTemplate()