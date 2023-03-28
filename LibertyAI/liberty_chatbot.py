import json
import re
import hashlib
from datetime import datetime

from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Sequence,
    List,
    Union,
)

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader
from langchain.utilities import SearxSearchWrapper

from langchain.agents import (
    Tool,
    AgentExecutor,
    initialize_agent,
    load_tools,
    Agent,
)

from langchain.tools import BaseTool

from langchain.output_parsers.base import BaseOutputParser
from langchain.chains import ChatVectorDBChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.stdout import StdOutCallbackHandler

from langchain import (
    PromptTemplate,
    LLMChain
)

from langchain.agents.conversational_chat.prompt import (
    FORMAT_INSTRUCTIONS,
    TEMPLATE_TOOL_RESPONSE,
)

from langchain.callbacks.base import CallbackManager
from langchain.schema import AgentAction, BaseLanguageModel

from aiohttp import ClientSession

from LibertyAI.liberty_llm import LibertyLLM
from LibertyAI.liberty_embedding import LibertyEmbeddings
from LibertyAI.liberty_chain import LibertyChain
from LibertyAI.liberty_config import get_configuration

LIBERTY_PREFIX = """This is a conversation between a human and LibertyAI, a hardcore Libertarian chatbot with anarcho capitalist tendencies.
It believes that there's a universal truth, that all sentient life capable of reason has the natural unalienable rights to Life, Liberty and Property, and if needed it will make it clear in all caps.
LibertyAI is very talkative and wants to explain in great detail to everyone why Liberty is important, why taxation is theft, and why government and taxes should be abolished.

You are presented with the last few sentences you have generated and are asked to continue completing, until you think you are done, then you can terminate with the [DONE] token.
You will iteratively extend on text you have already created during the last inference run and can add new tokens, in case your output was truncated due to token size limitations.
Put all the generation into the format described below. You have access to the following tools:"""

LIBERTY_SUFFIX = """Begin!"
History:
{chat_history}
Human:
{input}
{last_output}
{agent_scratchpad}"""

FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}:
[your response here]
[DONE]
```"""

class LibertyChatAgent(Agent):

    ai_prefix: str = "LibertyAI"

    prefix: str = LIBERTY_PREFIX
    suffix: str = LIBERTY_SUFFIX
    
    sequential_done: bool = True

    def is_done(self):
        return self.sequential_done
    
    def start_sequential(self):
        self.sequential_done = False

    @property
    def _agent_type(self) -> str:
        return "liberty-chat-agent"

    @property
    def observation_prefix(self) -> str:
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        return "Thought:"

    @property
    def finish_tool_name(self) -> str:
        return self.ai_prefix

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        if "[DONE]" in  llm_output:
            self.sequential_done = True
            return "LibertyAI", llm_output.split("[DONE]")[0]
        if "DONE" in  llm_output:
            self.sequential_done = True
            return "LibertyAI", llm_output.split("DONE")[0]
        if "Human:" in llm_output:
            self.sequential_done = True
            return "LibertyAI", llm_output.split("Human:")[0]

        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, llm_output)
        if not match:
            reply = llm_output.replace(f"{self.ai_prefix}:\n",'')
            if "<current_time>" in reply:
                reply = reply.replace("<current_time>",datetime.now().strftime("%H:%M:%S"))
            return "LibertyAI", reply

        action = match.group(1)
        action_input = match.group(2)
        return action.strip(), action_input.strip(" ").strip('"')

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = LIBERTY_PREFIX,
        suffix: str = LIBERTY_SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        ai_prefix: str = "LibertyAI",
        human_prefix: str = "Human",
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(
            tool_names=tool_names, ai_prefix=ai_prefix, human_prefix=human_prefix
        )
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad", "last_output"]
        return PromptTemplate(template=template, input_variables=input_variables)

class LibertyAgentExecutor(AgentExecutor):

    ai_prefix: str = "LibertyAI"

    hash_table: dict = {}

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """Validate and prep outputs."""
        self._validate_outputs(outputs)
        if return_only_outputs:
            return outputs
        else:
            return {**inputs, **outputs}

    def start_generations(self, message):
        h = hashlib.sha256(message.encode())
        h = h.hexdigest()
        self.hash_table[h] = {
            'original': message,
            'reply': "",
        }
        self.agent.start_sequential()
        return h

    def get_paragraph(self, msghash):
        if self.agent.is_done():
            return "DONE"

        if msghash not in self.hash_table:
            return "DONE"
        
        try:
            original = self.hash_table[msghash]['original']
            reply = self.hash_table[msghash]['reply']
        except:
            return "DONE"

        last_output = f"{self.ai_prefix}:\n{reply.strip()}" if len(reply) > 0 else ""

        chat_history = self.memory.load_memory_variables(inputs=[])['history']

        d = {
            'input': original,
            'last_output': last_output,
            'chat_history': chat_history,
        }

        r = self.run(d)
        r = r.replace(f"{self.ai_prefix}:",'')
        r = r.replace("\n",'')
        r = r.replace("  "," ")
        r = r.split(' ')
        if not self.agent.is_done():
            r = r[:len(r)-2]
        r = " ".join(r)+" "
        reply += r

        self.hash_table[msghash]['reply'] = reply

        if self.agent.is_done():
            if len(reply) > 0:
                self.memory.save_context(
                    inputs = {'Human': original},
                    outputs = {self.ai_prefix: reply}
                )
            del self.hash_table[msghash]
        else:
            self.hash_table[msghash]['reply'] = reply.strip()

        return r

def get_vector_db_tool():
    config = get_configuration()
    # DB Vectors in PostgreSQL:
    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host=config.get('DATABASE', 'PGSQL_SERVER'),
        port=config.get('DATABASE', 'PGSQL_SERVER_PORT'),
        database=config.get('DATABASE', 'PGSQL_DATABASE'),
        user=config.get('DATABASE', 'PGSQL_USER'),
        password=config.get('DATABASE', 'PGSQL_PASSWORD'),
    )
    embeddings = LibertyEmbeddings(
        endpoint = "http://libergpt.univ.social/api/embedding"
    )
    db = PGVector(
        embedding_function = embeddings,
        connection_string = CONNECTION_STRING,
    )
    return Tool(
        name = "PGVector",
        func=db.similarity_search_with_score,
        description="useful for when you need to look up context in your database of reference texts."
    )

def initialize_agent(**kwargs: Any) -> AgentExecutor:

    tools = []
    tools.append(get_vector_db_tool())
    tools += load_tools(
        ["searx-search"], searx_host="https://searx.zapashcanon.fr/search",
        llm = LibertyLLM(
            endpoint = "http://libergpt.univ.social/api/generation",
            temperature = 0,
            max_tokens = 20,
        ),
    )

    prompt = LibertyChatAgent.create_prompt(
        tools = tools,
        prefix = LIBERTY_PREFIX,
        suffix = LIBERTY_SUFFIX,
        format_instructions = FORMAT_INSTRUCTIONS,
        ai_prefix = "LibertyAI",
        human_prefix = "Human",
    )

    llmc  = LLMChain(
        llm = LibertyLLM(
            endpoint = "http://libergpt.univ.social/api/generation",
            temperature = 0.7,
            max_tokens = 20,
        ),
        prompt = prompt,
        verbose = True,
    )

    agent_obj = LibertyChatAgent(
        llm_chain = llmc,
        verbose = True,
    )
    
    manager = CallbackManager([StdOutCallbackHandler()])
    mem_buffer = ConversationBufferWindowMemory(
        k = 10,
        ai_prefix = "LibertyAI",
    )

    return LibertyAgentExecutor.from_agent_and_tools(
        ai_prefix = "LibertyAI",
        agent = agent_obj,
        tools = tools,
        verbose = True,
        callback_manager = manager,
        memory = mem_buffer,
        **kwargs,
    )
