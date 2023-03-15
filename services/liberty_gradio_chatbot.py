import gradio as gr
from fastapi import FastAPI

from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, CombinedMemory, ConversationSummaryMemory
from langchain import PromptTemplate, LLMChain

import os
import csv
import gradio as gr
from gradio import inputs, outputs
from datetime import datetime

from LibertyAI import LibertyLLM

DATA_FILENAME = "Carddata.csv"
DATA_FILE = os.path.join("data", DATA_FILENAME)

SCRIPT = """
<script>
if (!window.hasBeenRun) {
    window.hasBeenRun = true;
    console.log("should only happen once");
    document.querySelector("button.submit").click();
}
</script>
"""
def generate_html() -> str:
    with open(DATA_FILE) as csvfile:
        reader = csv.DictReader(csvfile)
        rows = []
        for row in reader:
            rows.append(row)
        rows.reverse()
        if len(rows) == 0:
            return "no messages yet"
        else:
            html = "<div class='chatbot'>"
            for row in rows:
                html += "<div>"
                html += f"<span>{row['inputs']}</span>"
                html += f"<span class='outputs'>{row['outputs']}</span>"
                html += "</div>"
            html += "</div>"
            return html

llm = LibertyLLM(
    endpoint = "http://libergpt.univ.social/api/generation",
    temperature = 0.9,
    max_tokens = 40
)

_DEFAULT_TEMPLATE = """The following conversation is between a human and LibertyAI, a Libertarian chatbot.
LibertyAI is a hardcore Libertarian, talkative and provides lots of specific details from its context.
If the AI does not know the answer to a question, it truthfully says it does not know.

Summary of conversation:
{history}
Current conversation:
{chat_history_lines}
Human:
{input}
AI:
"""

PROMPT = PromptTemplate(
    input_variables=["history", "input", "chat_history_lines"], template=_DEFAULT_TEMPLATE
)

conv_memory = ConversationBufferMemory(
    memory_key="chat_history_lines",
    input_key="input"
)
conv_memory.save_context(
    inputs={'input': 'Who are you?'},
    outputs={'AI': 'I am LibertyAI, a Libertarian AI.\nI live by the mottor "Live and let live" and "Be ungovernable"'}
)

summary_memory = ConversationSummaryMemory(
    llm = llm,
    input_key="input"
)

memory = CombinedMemory(memories=[conv_memory, summary_memory])

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory,
    prompt=PROMPT,
)

def chat(message, history):
    response = conversation.run(input=message, stop=['Human:'])
    history = history or list()
    history.append((message, response))
    return history, history

with gr.Blocks(title="LibertyAI") as block:
    description=f"A Libertarian chatbot"
    state = gr.outputs.State()
    chatbot = gr.Chatbot()
    message = gr.Textbox(lines=10)
    message.submit(chat, [message, state], [chatbot, state])
    gr.Markdown("Use SHIFT+ENTER for submitting text or press submit.")
    btn = gr.Button(value="SUBMIT")
    btn.click(chat, [message, state], [chatbot, state])

app = FastAPI()
app.mount("/web", block)
