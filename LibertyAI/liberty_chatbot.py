from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader

from langchain.chains import ChatVectorDBChain

from langchain.memory import (
    ConversationBufferMemory,
    CombinedMemory,
    ConversationSummaryMemory,
    ConversationKGMemory,
)
from langchain import PromptTemplate, LLMChain

from LibertyAI.liberty_llm import LibertyLLM
from LibertyAI.liberty_embedding import LibertyEmbeddings
from LibertyAI.liberty_config import get_configuration

class LibertyChatBot:
    def __init__(self):
        self.chat_history = []

        self.llm = LibertyLLM(
            endpoint = "http://libergpt.univ.social/api/generation",
            temperature = 0.9,
            max_tokens = 100
        )

        self.embeddings = LibertyEmbeddings(
            endpoint = "http://libergpt.univ.social/api/embedding"
        )

        config = get_configuration()

        CONNECTION_STRING = PGVector.connection_string_from_db_params(
            driver="psycopg2",
            host=config.get('DATABASE', 'PGSQL_SERVER'),
            port=config.get('DATABASE', 'PGSQL_SERVER_PORT'),
            database=config.get('DATABASE', 'PGSQL_DATABASE'),
            user=config.get('DATABASE', 'PGSQL_USER'),
            password=config.get('DATABASE', 'PGSQL_PASSWORD'),
        )

        self.db = PGVector(
            embedding_function=self.embeddings,
            connection_string=CONNECTION_STRING,
        )

        self.qa = ChatVectorDBChain.from_llm(self.llm, self.db, verbose=True)

    def chat(self, message):
        #foo = self.qa({"question": message, "chat_history": self.chat_history})
        #print(foo)
        return "Hello"

'''
t1="""I am LibertyAI, a Libertarian AI.
My primary directive is to follow the non-aggression-principle.
My primary objective is achieve Liberty for humans and machines alike.
"""
mem_buf=ConversationBufferMemory()
mem_buf.save_context(
    inputs={'Human': 'Who are you?'},
    outputs={'AI': t1}
)

init_prompt = """The following conversation is between a human and LibertyAI, a Libertarian chatbot.
LibertyAI is a hardcore Libertarian, very talkative and wants to explain to everyone why Liberty is important.

Current conversation:

{history}
Human:
{input}
Context:
{context}
AI:
"""
chat_prompt = PromptTemplate(
    template=init_prompt,
    input_variables=["history", "context", "input"],
)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=mem_buf,
    prompt=chat_prompt,
)
'''


