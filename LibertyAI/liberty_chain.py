import json
import re
import hashlib

from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Sequence,
    List,
    Union,
)

from pydantic import (
    BaseModel,
    Extra,
    Field,
    root_validator
)

from datetime import datetime
from sentence_transformers import util

from langchain.chains.llm import LLMChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseMemory
from langchain.agents.agent import AgentExecutor
from langchain.vectorstores.pgvector import PGVector

from LibertyAI.liberty_prompt import PROMPT
from LibertyAI.liberty_llm import LibertyLLM
from LibertyAI.liberty_embedding import LibertyEmbeddings

class LibertyChain(LLMChain, BaseModel):

    human_prefix: str = "Human"
    ai_prefix: str = "LibertyAI"
    hash_table: dict = {}
    prompt: BasePromptTemplate = PROMPT
    mrkl: AgentExecutor = None
    memory: BaseMemory = None
    summary: ConversationSummaryMemory = None
    user_name: str = ""
    user_mail: str = ""
    embeddings: LibertyEmbeddings = None
    vectordb: PGVector = None

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True

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
        #if mrkl:
        #    context = mrkl.run(message)

        #encoded1 = self.embeddings.embed_query(message)
        #encoded2 = self.embeddings.embed_query("What's the weather in X?")
        #if util.pytorch_cos_sim(encoded1, encoded2)[0] > 0.5:
        #    context = self.mrkl.run(message)
        #else:
        #    documents = self.vectordb.similarity_search_with_score(query=message, k=1)
        #    context = documents[0][0].page_content
        context = ""
        chat_history = self.memory.load_memory_variables(inputs=[])['history']
        chat_summary = self.summary.load_memory_variables(inputs=[])['history']
        d = {
            'input': message,
            'history': chat_history,
            'context': context,
            'summary': chat_summary,
            'current_date': datetime.now().strftime("%A (%d/%m/%Y)"),
            'current_time': datetime.now().strftime("%H:%M %p"),
            #'user_name': self.user_name,
            #'user_mail': self.user_mail,
        }
        uuid = self.llm.submit_partial(self.prep_prompts([d])[0][0].text, stop = ["Human:"])
        return uuid

    def get_part(self, uuid, index):
        text = self.llm.get_partial(uuid, index)
        return text
'''
        if text == "[DONE]":
            self.memory.save_context(
                inputs = {self.human_prefix: original.strip()},
                outputs = {self.ai_prefix: self.hash_table[uuid]['reply'].strip()}
            )
            self.summary.save_context(
                inputs = {self.human_prefix: original.strip()},
                outputs = {self.ai_prefix: self.hash_table[uuid]['reply'].strip()}
            )

'''
