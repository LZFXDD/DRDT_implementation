import os

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
                    model_kwargs={"stop": '\n'},
                    openai_api_key = os.environ['OPENAI_API_KEY']),

                 DR_llm: AnyOpenAILLM = AnyOpenAILLM(
                     temperature=0,
                     model_name='gpt-3.5-turbo',
                     model_kwargs={"stop": '\n'},
                     openai_api_key=os.environ['OPENAI_API_KEY']),

                 PROBE_llm: AnyOpenAILLM = AnyOpenAILLM(
                     temperature=0,
                     model_name='gpt-3.5-turbo',
                     model_kwargs={"stop": '\n'},
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

        ### 基本信息
        self.step_n = 1
        self.user_id = user_id
        self.user_history = user_history
        self.movie_info = movie_info
        self.preference_analysis = ''
        self.scratchpad = ''

    def run(self):
        self.scratchpad += f'------------------------------ Loop {self.step_n} ------------------------------'
        self.scratchpad += '\n'
        self.step()
        self.step_n += 1
        if self.step_n < train_times:
            self.run()
        else:
            print(self.scratchpad)

    def step(self) -> None:
        if self.step_n == 1:
            pass
        else:
            pass

    ### agent_prompt
    def DT_build_agent_prompt(self) -> str:
        pass

    def DR_build_agent_prompt(self) -> str:
        pass

    def PROBE_build_agent_prompt(self) -> str:
        pass


    ### 调用llm
    def prompt_DT(self) -> str:
        return format_step(self.dt_llm(self.DT_build_agent_prompt()))

    def prompt_PROBE(self) -> str:
        return format_step(self.probe_llm(self.PROBE_build_agent_prompt()))

    def prompt_DR(self) -> str:
        return format_step(self.dr_llm(self.DR_build_agent_prompt()))

### 字符串处理
def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')