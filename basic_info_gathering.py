import os

from langchain.prompts import PromptTemplate
from llm import AnyOpenAILLM
from prompts import actor_search_prompt
from data_preprocessing import movie_info, user_history

class Actor_search:
    def __init__(self,
                 actor_search_prompt: PromptTemplate = actor_search_prompt,
                 actor_search_llm: AnyOpenAILLM = AnyOpenAILLM(
                     temperature=0,
                     model_name='gpt-3.5-turbo',
                     model_kwargs={"stop": "\n"},
                     openai_api_key=os.environ['OPENAI_API_KEY']
                 )) -> None:
        self.agent_prompt = actor_search_prompt
        self.actor_search_llm = actor_search_llm

    def actor_generation(self, movieId: int) -> str:
        self.movieId = movieId
        return format_string(self.actor_search_llm(self._build_agent_prompt()))

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            title=movie_info[self.movieId]['title'],
            year=movie_info[self.movieId]['year'],
        )



### 字符串处理 ###
def format_string(string: str) -> str:
    return string.strip().strip('\n').replace('\n','')