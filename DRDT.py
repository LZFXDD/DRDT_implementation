import os

from langchain.prompts import PromptTemplate
from llm import AnyOpenAILLM
from prompts import dt_agent_prompt, dr_agent_prompt, probe_agent_prompt
from data_preprocessing import movie_info, user_history

### 设置hyperparameter T = 10
train_times = 10


class DRDT:
    def __init__(self,
                user_id: int,
                 ### prompt
                DT_agent_prompt: PromptTemplate = dt_agent_prompt,
                DR_agent_prompt: PromptTemplate = dr_agent_prompt,
                PROBE_agent_prompt: PromptTemplate = probe_agent_prompt,

                ### LLM
                DT_llm: AnyOpenAILLM = AnyOpenAILLM(
                    temperature=0,
                    model_name='gpt-3.5-turbo',
                    model_kwargs={"stop": "\n"},
                    openai_api_key = os.environ['OPENAI_API_KEY']),
                DR_llm: AnyOpenAILLM = AnyOpenAILLM(
                    temperature=0,
                    model_name='gpt-3.5-turbo',
                    model_kwargs={"stop": "\n"},
                    openai_api_key=os.environ['OPENAI_API_KEY']),
                PROBE_llm: AnyOpenAILLM = AnyOpenAILLM(
                    temperature=0,
                    model_name='gpt-3.5-turbo',
                    model_kwargs={"stop": "\n"},
                    openai_api_key=os.environ['OPENAI_API_KEY']),
                 ) -> None:
        ### prompt
        self.DT_agent_prompt = DT_agent_prompt
        self.DR_agent_prompt = DR_agent_prompt
        self.PROBE_agent_prompt = PROBE_agent_prompt

        ### llm
        self.DT_llm = DT_llm
        self.DR_llm = DR_llm
        self.PROBE_llm = PROBE_llm

        ### 运行步数
        self.step_n: int = 1

        ### 基本信息
        self.user_id = user_id
        self.history = user_history
        self.user_history = user_history[user_id]
        self.length = len(user_history[user_id][0])
        self.movie_info = movie_info
        self.analysis = ''
        ### 循环记录
        self.scratchpad = ''
    ### 循环
    def run(self):
        self.scratchpad += f"""-------------------- Loop {self.step_n} --------------------"""
        self.scratchpad += '\n'
        self.step()
        self.step_n += 1
        if self.step_n < train_times:
            self.run()
        else:
            print(self.scratchpad)

    def step(self):
        ### divigent thinking
        self.scratchpad += "DIVERGENT THINKING:\n"
        self.scratchpad += self.DT_build_agent_prompt()
        self.scratchpad += '\n\n'
        self.scratchpad += self.prompt_DT() + '\n'
        self.analysis = self.prompt_DT()
        ### probing
        self.scratchpad += "PROBING:\n"
        self.scratchpad += self.PROBE_build_agent_prompt()
        self.scratchpad += '\n'
        self.scratchpad += self.prompt_PROBE()
        self.scratchpad += '\n'
        ### dynamic reflection
        self.scratchpad += "DYNAMIC REFLECTION:\n"
        self.scratchpad += self.DR_build_agent_prompt()
        self.scratchpad += '\n'
        self.scratchpad += self.prompt_DR()
        self.scratchpad += '\n'
        self.analysis = self.prompt_DR()

    ### agent_prompt
    def DT_build_agent_prompt(self) -> str:
        watched_movies = []
        sample_watched_movies = []
        for movie_id in self.user_history[0][: self.length - train_times + self.step_n]:
            watched_movies.append(self.movie_info[movie_id]['title'])

        user_history = self.user_history[0][: self.length - train_times + self.step_n]
        last_movie = user_history[-1]
        next_movie = self.movie_info[self.user_history[0][self.length - train_times + self.step_n]]['title']

        for history in self.history.values():
            history = history[0]
            if last_movie in history and history.index(last_movie) >= self.length - train_times + self.step_n and last_movie != history[-1]:
                sample_watched_movies_id = history[history.index(last_movie) - self.length + train_times - self.step_n + 1: history.index(last_movie) + 1]
                candidate = history[history.index(last_movie) + 1]
                break

        for movie_id in sample_watched_movies_id:
            sample_watched_movies.append(self.movie_info[movie_id]['title'])

        candidates = [movie_info[candidate]['title'], next_movie]
        answer = [next_movie, movie_info[candidate]['title']]

        return self.DT_agent_prompt.format(
            sample_watched_movies=sample_watched_movies,
            candidates=candidates,
            answer=answer,
            watched_movies=watched_movies,
            preference_analysis=self.analysis
        )

    def DR_build_agent_prompt(self) -> str:
        return self.DR_agent_prompt.format(
            answer=self.movie_info[self.user_history[0][self.length - train_times + self.step_n]]['title'],
            rating=self.user_history[1][self.length - train_times + self.step_n],
            preferences_analysis=self.analysis,
            predict_movies=self.prompt_PROBE()
        )

    def PROBE_build_agent_prompt(self) -> str:
        return self.PROBE_agent_prompt.format(preferences_analysis=self.analysis)

    ### 调用llm
    def prompt_DT(self) -> str:
        return format_step(self.DT_llm(self.DT_build_agent_prompt()))

    def prompt_DR(self) -> str:
        return format_step(self.DR_llm(self.DR_build_agent_prompt()))

    def prompt_PROBE(self) -> str:
        return format_step(self.PROBE_llm(self.PROBE_build_agent_prompt()))

### 字符串处理 ###
def format_step(step: str) -> str:

    return step.strip('\n').strip().replace('\n', '')