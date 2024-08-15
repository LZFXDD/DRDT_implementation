import openai

# openai_api设置
openai.api_key = 'sk-fph6emDpXF4gpDqOFe827a075561421d81C4442c5e489d37'
openai.api_base = 'https://free.gpt.ge/v1'

from DRDT import DRDT

test = DRDT(user_id=1)
test.run()
