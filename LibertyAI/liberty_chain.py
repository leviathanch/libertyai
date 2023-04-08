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

ACTION_CHECK_INSTRUCTION = """
In order to use a tool you MUST use this format:

```
Thought: Do I need to use a tool? Yes
LibertyAI: [your response here]
```

If no tool is required you MUST use this format:

```
Thought: Do I need to use a tool? No
LibertyAI: [your response here]
```
"""

class LibertyChain(LLMChain, BaseModel):

    human_prefix: str = "Human"
    ai_prefix: str = "LibertyAI"
    hash_table: dict = {}
    sequential_done: bool = True
    prompt: BasePromptTemplate = PROMPT
    input_key: str = "input"  #: :meta private:
    output_key: str = "output"  #: :meta private:
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

    def is_done(self):
        return self.sequential_done
    
    def start_sequential(self):
        self.sequential_done = False

    @property
    def input_keys(self) -> List[str]:
        """Use this since so some prompt vars come from history."""
        return [self.input_key]

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

        context = ""

        #if mrkl:
        #    context = mrkl.run(message)

        #encoded1 = self.embeddings.embed_query(message)
        #encoded2 = self.embeddings.embed_query("What's the weather in X?")
        #if util.pytorch_cos_sim(encoded1, encoded2)[0] > 0.5:
        #    context = self.mrkl.run(message)
        #else:
        #    documents = self.vectordb.similarity_search_with_score(query=message, k=1)
        #    context = documents[0][0].page_content

        self.hash_table[h] = {
            'original': message,
            'context': context,
            'reply': "",
            'current_date': datetime.now().strftime("%A (%d/%m/%Y)"),
            'current_time': datetime.now().strftime("%H:%M %p"),
            'thread_running': False,
        }

        self.start_sequential()

        return h

    def get_paragraph(self, msghash):

        if msghash not in self.hash_table:
            return "[DONE]"
        
        try:
            original = self.hash_table[msghash]['original']
            reply = self.hash_table[msghash]['reply']
            context = self.hash_table[msghash]['context']
        except:
            return "[DONE]"

        chat_history = self.memory.load_memory_variables(inputs=[])['history']
        chat_summary = self.summary.load_memory_variables(inputs=[])['history']

        d = {
            'input': original,
            'history': chat_history,
            'last_output': f"{self.ai_prefix}: {reply.strip()}",
            'context': context,
            'summary': chat_summary,
            'current_date': self.hash_table[msghash]['current_date'],
            'current_time': self.hash_table[msghash]['current_time'],
            'user_name': self.user_name,
            'user_mail': self.user_mail,
            'stop': ["Human"],
        }
        output = self.run(d)
        print("output",output)
        output = output.replace("<current_time>",datetime.now().strftime("%H:%M %p"))
        output = output.replace("<current_date>",datetime.now().strftime("%A (%d/%m/%Y)"))
        output = output.replace(f"{self.ai_prefix}:",'')
        output = output.replace("\n",' ')
        output = output.strip().split(' ')
        if output[len(output)-1] in self.human_prefix:
            output = output[:len(output)-1]
        output = " ".join(output)+" "
        if len(output.strip()):
            self.hash_table[msghash]['reply'] += output
        elif len(self.hash_table[msghash]['reply'].strip()) == 0:
            output = ""
        #    sentiment = SentimentIntensityAnalyzer()
        #    sent = sentiment.polarity_scores(original)
        #    output = str(sent)
        

        if self.is_done() or len(output.strip())==0:
            if len(self.hash_table[msghash]['reply']) > 0:
                self.memory.save_context(
                    inputs = {self.human_prefix: original.strip()},
                    outputs = {self.ai_prefix: self.hash_table[msghash]['reply'].strip()}
                )
                self.summary.save_context(
                    inputs = {self.human_prefix: original.strip()},
                    outputs = {self.ai_prefix: self.hash_table[msghash]['reply'].strip()}
                )
            del self.hash_table[msghash]

        return "[DONE]" if len(output.strip())==0 else output
