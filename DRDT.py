from langchain.prompts import PromptTemplate
from llm import AnyOpenAILLM
from prompts import dt_agent_prompt, dr_agent_prompt, probe_agent_prompt, critique_agent_prompt
class DRDT:
    def __init__(self,
                DT_agent_prompt: PromptTemplate = dt_agent_prompt,
                DR_agent_prompt: PromptTemplate = dr_agent_prompt,
                PROBE_agent_prompt: PromptTemplate = probe_agent_prompt,
                CRITIQUE_agent_prompt: PromptTemplate = critique_agent_prompt,
                 ) -> None:
        self.DT_agent_prompt = DT_agent_prompt
        self.DR_agent_prompt = DR_agent_prompt
        self.PROBE_agent_prompt = PROBE_agent_prompt
        self.CRITIQUE_agent_prompt = critique_agent_prompt
    ### openai调用相关
    def DT_build_agent_prompt(self) -> str:
        return self.DT_agent_prompt()

    def DR_build_agent_prompt(self) -> str:
        return self.DR_agent_prompt()

    def probe_build_agent_prompt(self) -> str:
        return self.PROBE_agent_prompt()

    def critique_build_agent_prompt(self) -> str:
        return self.CRITIQUE_agent_prompt()
### 字符串处理 ###
def format_step(step: str) -> str:

    return step.strip('\n').strip().replace('\n', '')