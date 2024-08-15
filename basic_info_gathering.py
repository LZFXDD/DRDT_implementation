import os
import openai

# openai_api设置
openai.api_key = 'sk-fph6emDpXF4gpDqOFe827a075561421d81C4442c5e489d37'
os.environ['OPENAI_API_KEY'] = openai.api_key
openai.api_base = 'https://free.gpt.ge/v1'

from langchain.prompts import PromptTemplate
from llm import AnyOpenAILLM
from prompts import actor_search_prompt
from transfer import movie_basic_info

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
            title=movie_basic_info[self.movieId]['title'],
            year=movie_basic_info[self.movieId]['year'],
        )
### 字符串处理 ###
def format_string(string: str) -> str:
    return string.strip().strip('\n').replace('\n','')

### 程序执行
agent = Actor_search()
print(agent.actor_generation(2))


