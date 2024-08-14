from langchain.prompts import PromptTemplate

### DT ###
divergent_thinking_instruction = """There is another user watched movie {sample_watched_movies}. Given the candidates {candidates}, the answer is {answer}.
I have watched the following movies in the past in order: {watched_movies}.
Could you please analysis the user preference according to the movie's year of release, genres and actors?
"""

dt_agent_prompt = PromptTemplate(
                            input_variables=['sample_watched_movies', 'candidates', 'answer', 'watched_movies'],
                            template=divergent_thinking_instruction,
                            )

### DR_reflection ###
dynamic_reflection_instruction = """The answer is {answer}, is that consistence with the previous preferences?
The user's rating for this movie is {rating} out of 5. From what aspect will you recommend this movie to user?
Update your preference analysis on this user.
"""

dr_agent_prompt = PromptTemplate(
                            input_variables=['answer', 'rating'],
                            template=dynamic_reflection_instruction,
                            )


### DR_probing ###
probing_instruction = """What is the next possible movie I would like to watch next? """

probe_agent_prompt = PromptTemplate(
                            template=probing_instruction,
                            )

### actor_search
actor_search_instruction = """You will act as an information gatherer. You will be provided with a movie's title and its year of release. Your task is to find and list the names of the main actors, separating each name with a comma.
The movie's title {title} and the year of release {year}."""

actor_search_prompt = PromptTemplate(
                                input_variables=['title', 'year'],
                                template=actor_search_instruction
                                )