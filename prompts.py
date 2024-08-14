from langchain.prompts import PromptTemplate

### DT ###
divergent_thinking_instruction = """"""

dt_agent_prompt = PromptTemplate(
                            input_variables=[],
                            template=divergent_thinking_instruction,
                            )

### DR_reflection ###
dynamic_reflection_instruction = """"""

dr_agent_prompt = PromptTemplate(
                            template=dynamic_reflection_instruction,
                            )


### DR_probing ###
probing_instruction = """"""

probe_agent_prompt = PromptTemplate(
                            template=probing_instruction,
                            )

### DR_critiqueing ###
critiquing_instruction = """"""

critique_agent_prompt = PromptTemplate(
                            template=critiquing_instruction,
                            )