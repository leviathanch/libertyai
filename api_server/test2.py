import sys
import time

from LibertyAI.liberty_chatbot import initialize_chatbot
from LibertyAI.liberty_config import get_configuration
from LibertyAI.liberty_embedding import LibertyEmbeddings
from LibertyAI.liberty_llm import LibertyLLM

config = get_configuration()

chatbot = initialize_chatbot(
    name="Mr. Human",
    email="user@test.com",
    llm = LibertyLLM(
        echo = True,
        endpoint = config.get('API', 'GENERATION_ENDPOINT'),
    ),
    emb = None,
)

uuid = chatbot.start_generations("Hello there!\nHow are you doing?")
print(uuid)

text = ""
i = 0
while text != "[DONE]":
    text = chatbot.get_part(uuid, i)
    if text == "[BUSY]":
        time.sleep(0.1)
        continue

    i += 1
    if text != "[DONE]":
        sys.stdout.write(text)
        sys.stdout.flush()

print("\n\n")
