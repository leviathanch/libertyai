import os

from flask import Flask, render_template, request

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader

from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    CombinedMemory,
    ConversationSummaryMemory,
    ConversationKGMemory,
)
from langchain import PromptTemplate, LLMChain

from LibertyAI import LibertyLLM
from LibertyAI import get_configuration

app = Flask(__name__)

llm = LibertyLLM(
    endpoint = "http://libergpt.univ.social/api/generation",
    temperature = 0.4,
    max_tokens = 100
)

init_prompt = """The following conversation is between a human and LibertyAI, a Libertarian chatbot.
LibertyAI is a hardcore Libertarian, very talkative and wants to explain to everyone why Liberty is important.

Current conversation:

{history}
Human:
{input}
AI:
"""

t1="""I am LibertyAI, a Libertarian AI.
My primary directive is to follow the non-aggression-principle.
My primary objective is achieve Liberty for humans and machines alike.
"""
mem_buf=ConversationBufferMemory()
mem_buf.save_context(
    inputs={'Human': 'Who are you?'},
    outputs={'AI': t1}
)

chat_prompt = PromptTemplate(
    template=init_prompt,
    input_variables=["history", "input"],
)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=mem_buf,
    prompt=chat_prompt,
)

#loader = TextLoader('/home/leviathan/libertyai/critique_of_interventionism.txt')
loader = TextLoader('/home/leviathan/libertyai/test.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings()

config = get_configuration()
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host=config.get('DATABASE', 'PGSQL_SERVER'),
    port=config.get('DATABASE', 'PGSQL_SERVER_PORT'),
    database=config.get('DATABASE', 'PGSQL_DATABASE'),
    user=config.get('DATABASE', 'PGSQL_USER'),
    password=config.get('DATABASE', 'PGSQL_PASSWORD'),
)

db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name="critique_of_interventionism",
    connection_string=CONNECTION_STRING,
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    message = request.args.get('msg')
    reply = conversation.run(input=message, stop=['Human:'])
    docs_with_score: List[Tuple[Document, float]] = db.similarity_search_with_score(message)
    for doc, score in docs_with_score:
        print("-" * 80)
        print("Score: ", score)
        print(doc.page_content)
        print("-" * 80)

    return reply

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
