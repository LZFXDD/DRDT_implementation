import os

from langchain.prompts import PromptTemplate
from llm import AnyOpenAILLM
from prompts import dt_agent_prompt, dr_agent_prompt, probe_agent_prompt
from data_preprocessing import movie_info, user_history

### 设置hyperparameter T = 10
train_times = 10


class DRDT:
    def __init__(self,
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

    ### agent_prompt
    def DT_build_agent_prompt(self) -> str:
        return self.DT_agent_prompt(

        )

    def DR_build_agent_prompt(self) -> str:
        return self.DR_agent_prompt()

    def PROBE_build_agent_prompt(self) -> str:
        return self.PROBE_agent_prompt()

    ### 调用llm
    def prompt_DT(self) -> str:
        return format_step(self.DT_llm(self.DT_build_agent_prompt()))

    def prompt_DR(self) -> str:
        return format_step(self.DR_llm(self.DR_build_agent_prompt()))

    def prompt_PROBE(self) -> str:
        return format_step(self.PROBE_llm(self.PROBE_build_agent_prompt()))

    ### 一个loop
    def step(self) -> None:
        pass

    ### 训练
    def run(self, times: int = train_times) -> None:
        pass
### 字符串处理 ###
def format_step(step: str) -> str:

    return step.strip('\n').strip().replace('\n', '')