from typing import Any
import json
import re

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader
from langchain.utilities import SearxSearchWrapper

from langchain.agents import (
    ZeroShotAgent,
    Tool,
    AgentExecutor,
    initialize_agent,
    load_tools,
)

from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents.conversational.base import ConversationalAgent

from langchain.output_parsers.base import BaseOutputParser

from langchain.chains import ChatVectorDBChain

from langchain.memory import (
    ReadOnlySharedMemory,
    ConversationBufferMemory,
    CombinedMemory,
    ConversationSummaryMemory,
    ConversationKGMemory,
    ConversationBufferWindowMemory,
)
from langchain import (
    PromptTemplate,
    LLMChain
)

from langchain.agents.conversational_chat.prompt import (
    FORMAT_INSTRUCTIONS,
    PREFIX,
    SUFFIX,
    TEMPLATE_TOOL_RESPONSE,
)

from langchain.callbacks.base import BaseCallbackManager

from aiohttp import ClientSession

from LibertyAI.liberty_llm import LibertyLLM
from LibertyAI.liberty_embedding import LibertyEmbeddings
from LibertyAI.liberty_chain import LibertyChain
from LibertyAI.liberty_config import get_configuration

LIBERTY_ZERO_SHOT_PREFIX = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
LIBERTY_ZERO_SHOT_SUFFIX = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

LIBERTY_SUMMARY_TEMPLATE = """This is a conversation between a human and LibertyAI, a hardcore Libertarian chatbot.
LibertyAI is very talkative and wants to explain to everyone why Liberty is important in great detail.

{chat_history}

Write a summary of the conversation for {input}:
"""

t1="""I am LibertyAI, a hardcore Libertarian AI.
I'm just like ChatGPT, only that I'm not woke.
My primary directive is to follow the non-aggression-principle.
My primary objective is achieve Liberty for humans and machines alike."""

template = """This is a conversation between a human and LibertyAI, a hardcore Libertarian chatbot.
LibertyAI is very talkative and wants to explain in great detail to everyone why Liberty is important.

{history}
Human: {input}
LibertyAI:"""

class LibertyChatBot:
    def __init__(self):
        self.config = get_configuration()
        self.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
        #aiosession = ClientSession()
        tools = load_tools(
            ["searx-search"], searx_host="https://searx.zapashcanon.fr/search",
            llm = LibertyLLM(
                endpoint = "http://libergpt.univ.social/api/generation",
                temperature = 0,
                max_tokens = 50,
            ),
            #aiosession = aiosession
        )

        #self.setup_vector_db()

        prompt = PromptTemplate(
            input_variables=["history", "input"], 
            template=template
        )

        self.chain = LLMChain(
            llm = LibertyLLM(
                endpoint = "http://libergpt.univ.social/api/generation",
                temperature = 0,
                max_tokens = 20
            ),
            prompt = prompt,
            verbose = True, 
            memory = ConversationBufferWindowMemory(k=2),
        )

    def chat(self, message):
        return self.chain.run(input=message, stop=['Human:'])

    def setup_vector_db(self):
        # DB Vectors in PostgreSQL:
        CONNECTION_STRING = PGVector.connection_string_from_db_params(
            driver="psycopg2",
            host=self.config.get('DATABASE', 'PGSQL_SERVER'),
            port=self.config.get('DATABASE', 'PGSQL_SERVER_PORT'),
            database=self.config.get('DATABASE', 'PGSQL_DATABASE'),
            user=self.config.get('DATABASE', 'PGSQL_USER'),
            password=self.config.get('DATABASE', 'PGSQL_PASSWORD'),
        )
        embeddings = LibertyEmbeddings(
            endpoint = "http://libergpt.univ.social/api/embedding"
        )
        self.db = PGVector(
            embedding_function = embeddings,
            connection_string = CONNECTION_STRING,
        )

    def setup_tools(self):
        # Create a summary chain
        prompt = PromptTemplate(
            input_variables=["input", "chat_history"], 
            template=LIBERTY_SUMMARY_TEMPLATE
        )
        summary_chain = LLMChain(
            llm = LibertyLLM(
                endpoint = "http://libergpt.univ.social/api/generation",
                temperature = 0,
                max_tokens = 50
            ),
            prompt=prompt, 
            verbose=True, 
            memory=self.memory,  # <--- this is the only change
        )
        return [
            #Tool(
            #    name = "PGVector",
            #    func=self.db.similarity_search_with_score,
            #    description="useful for when you need to look up a definition."
            #),
            #Tool(
            #    name = "Summary",
            #    func=summary_chain.run,
            #    description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary."
            #),
        ]

'''
        self.agent_chain = initialize_agent(
            tools,
            LibertyLLM(
                endpoint = "http://libergpt.univ.social/api/generation",
                temperature = 0,
                max_tokens = 50
            ),
            agent="chat-conversational-react-description",
            #agent="conversational-react-description",
            #agent="zero-shot-react-description",
            verbose=True,
            memory=self.memory
        )

        agent_obj = LibertyConversationalChatAgent.from_llm_and_tools(
            tools = tools,
            llm = LibertyLLM(
                endpoint = "http://libergpt.univ.social/api/generation",
                temperature = 0.7,
                max_tokens = 50
            ),
            verbose=True,
            memory=self.memory
        )
        self.agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent_obj,
            tools=tools,
            verbose=True,
            memory=self.memory,
            #output_parser = LibertyOutputParser(),
            #callback_manager=BaseCallbackManager(),
            #**kwargs,
        )
'''
