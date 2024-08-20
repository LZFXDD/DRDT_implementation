import os
import random

from langchain.prompts import PromptTemplate
from llm import AnyOpenAILLM
from prompts import collaborative_agent_prompt, dt_agent_prompt, dr_agent_prompt, probe_agent_prompt, predict_agent_prompt
from data_preprocessing import movie_info, user_history


### 设置hyperparameter T = 10
train_times = 10

class DRDT_Agents:
    def __init__(self,
                 user_id: int,
                 ### prompt
                 COLLABOARTIVE_agent_prompt: PromptTemplate = collaborative_agent_prompt,
                 DT_agent_prompt: PromptTemplate = dt_agent_prompt,
                 DR_agent_prompt: PromptTemplate = dr_agent_prompt,
                 PROBE_agent_prompt: PromptTemplate = probe_agent_prompt,
                 PREDICT_agent_prompt: PromptTemplate = predict_agent_prompt,

                 ### LLM
                 DT_llm: AnyOpenAILLM = AnyOpenAILLM(
                    temperature=0,
                    model_name='gpt-3.5-turbo',
                    openai_api_key = os.environ['OPENAI_API_KEY']),

                 DR_llm: AnyOpenAILLM = AnyOpenAILLM(
                     temperature=0,
                     model_name='gpt-3.5-turbo',
                     openai_api_key=os.environ['OPENAI_API_KEY']),

                 PROBE_llm: AnyOpenAILLM = AnyOpenAILLM(
                     temperature=0,
                     model_name='gpt-3.5-turbo',
                     openai_api_key=os.environ['OPENAI_API_KEY']),

                 PREDICT_llm: AnyOpenAILLM = AnyOpenAILLM(
                     temperature=0,
                     model_name='gpt-3.5-turbo',
                     openai_api_key=os.environ['OPENAI_API_KEY']),
                 ):

        ### prompt
        self.collaborative_agent_prompt = COLLABOARTIVE_agent_prompt
        self.dt_agent_prompt = DT_agent_prompt
        self.dr_agent_prompt = DR_agent_prompt
        self.probe_agent_prompt = PROBE_agent_prompt
        self.predict_agent_prompt = PREDICT_agent_prompt

        ### LLM
        self.dt_llm = DT_llm
        self.dr_llm = DR_llm
        self.probe_llm = PROBE_llm
        self.predict_llm = PREDICT_llm

        ### 用户信息
        self.user_id = user_id
        self.preference_analysis = ''

        ### 基本信息
        self.step_n = 1
        self.scratchpad = ''
        self.candidates = []
        self.probing_recommendation = ''

### 单个测试
    def run(self):
        self.scratchpad += f'------------------------------ Loop {self.step_n} ------------------------------'
        self.scratchpad += '\n'
        self.candidates = self.candidates_selection(5)
        self.step()
        self.step_n += 1

        if self.step_n < train_times - 1:
            self.run()
        else:
            self.scratchpad += "PREDICTION: \n"
            self.scratchpad += self.PROBE_build_agent_prompt()
            self.scratchpad += self.prediction()
            self.scratchpad += '\n'

            print(self.scratchpad)

    def step(self) -> None:
        ### divergent thinking
        self.scratchpad += "DIVERGENT THINKING: \n"
        self.scratchpad += self.DT_build_agent_prompt()
        self.scratchpad += '\n'

        prompt_DT = self.prompt_DT()
        self.scratchpad += prompt_DT + '\n'
        self.preference_analysis = prompt_DT

        ### probing
        self.scratchpad += "PROBING:\n"
        self.scratchpad += self.PROBE_build_agent_prompt()
        self.scratchpad += '\n'

        prompt_probe = self.prompt_PROBE()
        self.scratchpad += prompt_probe + '\n'

        length = len(user_history[self.user_id][0])
        candidates = self.candidates + [movie_info[user_history[self.user_id][0][length - train_times + self.step_n]]['title']]
        for candidate in candidates:
            if candidate in prompt_probe:
                self.probing_recommendation = candidate
                break

        ### dynamic reflection
        self.scratchpad += "DYNAMIC REFLECTION:\n"
        self.scratchpad += self.DR_build_agent_prompt()
        self.scratchpad += '\n'

        prompt_DR = self.prompt_DR()
        self.scratchpad += prompt_DR + '\n'
        self.preference_analysis = prompt_DR

### 正确率计算
    # def run(self):
    #     self.candidates = self.candidates_selection(5)
    #     self.step()
    #     self.step_n += 1
    #
    #     if self.step_n < train_times - 1:
    #         self.run()
    #     else:
    #         return self.prediction()
    #
    # def step(self) -> None:
    #     ### divergent thinking
    #     self.preference_analysis = self.prompt_DT()
    #
    #     ### probing
    #     prompt_probe = self.prompt_PROBE()
    #
    #     length = len(user_history[self.user_id][0])
    #     candidates = self.candidates + [movie_info[user_history[self.user_id][0][length - train_times + self.step_n]]['title']]
    #     for candidate in candidates:
    #         if candidate in prompt_probe:
    #             self.probing_recommendation = candidate
    #             break
    #
    #     ### dynamic reflection
    #     self.preference_analysis = self.prompt_DR()



    ### 候选项集
    def candidates_selection(self, number):
        selected_candidates_indexes = random.sample(list(movie_info.keys()), number)

        selected_candidates = [movie_info[index]['title'] for index in selected_candidates_indexes]
        return selected_candidates

    ### agent_prompt
    def DT_build_agent_prompt(self) -> str:
        length = len(user_history[self.user_id][0])
        watched_movies = [movie_info[index]['title'] for index in user_history[self.user_id][0][: length - train_times + self.step_n]]

        # return self.dt_agent_prompt.format(
        #     watched_movies=watched_movies,
        #     preferences_analysis=self.preference_analysis
        # )
        if self.step_n == 1:
            # 需要变量的初始化
            sample_watched_movies = []
            candidates = self.candidates_selection(2)


            last_movieId = user_history[self.user_id][0][-1]

            for userId, history in user_history.items():
                if self.user_id != userId and last_movieId in history[0] and last_movieId != history[0][-1]:
                    sample_watched_movies_id = history[0][: history[0].index(last_movieId) + 1]
                    sample_watched_movies = [movie_info[index]['title'] for index in sample_watched_movies_id]
                    candidates.append(movie_info[history[0][history[0].index(last_movieId) + 1]]['title'])
                    break

            answer = random.choice([[candidates[2], candidates[0], candidates[1]], [candidates[2], candidates[1], candidates[0]]])
            return self.collaborative_agent_prompt.format(
                sample_watched_movies=sample_watched_movies,
                candidates=candidates,
                answer=answer,
                watched_movies=watched_movies,
                preferences_analysis=self.preference_analysis
            )

        else:
            return self.dt_agent_prompt.format(
                watched_movies=watched_movies,
                preferences_analysis=self.preference_analysis
            )

    def DR_build_agent_prompt(self) -> str:
        length = len(user_history[self.user_id][0])

        return self.dr_agent_prompt.format(
            preferences_analysis=self.preference_analysis,
            recommendation=self.probing_recommendation,
            candidates=self.candidates + [movie_info[user_history[self.user_id][0][length - train_times + self.step_n]]['title']],
            answer=movie_info[user_history[self.user_id][0][length - train_times + self.step_n]]['title'],
            rating=user_history[self.user_id][1][length - train_times + self.step_n]
        )

    def PROBE_build_agent_prompt(self) -> str:
        length = len(user_history[self.user_id][0])

        return self.probe_agent_prompt.format(
            preferences_analysis=self.preference_analysis,
            candidates=self.candidates + [movie_info[user_history[self.user_id][0][length - train_times + self.step_n]]['title']]
        )

    def PREDICT_build_agent_prompt(self) -> str:
        return self.predict_agent_prompt.format(
            preferences_analysis=self.preference_analysis,
            candidates=self.candidates + [movie_info[user_history[self.user_id][0][-1]]['title']]
        )

    ### 调用llm
    def prompt_DT(self) -> str:
        return format_step(self.dt_llm(self.DT_build_agent_prompt()))

    def prompt_PROBE(self) -> str:
        return format_step(self.probe_llm(self.PROBE_build_agent_prompt()))

    def prompt_DR(self) -> str:
        return format_step(self.dr_llm(self.DR_build_agent_prompt()))

    def prediction(self) -> str:
        return format_step(self.predict_llm(self.PREDICT_build_agent_prompt()))

### 字符串处理
def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')