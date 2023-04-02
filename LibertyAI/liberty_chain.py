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

from LibertyAI.liberty_prompt import PROMPT
from langchain.chains.llm import LLMChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseMemory

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

'''


import json
import re
import hashlib
from datetime import datetime, timezone
'''

class LibertyChain(LLMChain, BaseModel):

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    hash_table: dict = {}
    sequential_done: bool = True
    memory: BaseMemory = Field(default_factory=ConversationBufferMemory)
    prompt: BasePromptTemplate = PROMPT
    input_key: str = "input"  #: :meta private:
    output_key: str = "output"  #: :meta private:

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
        self.hash_table[h] = {
            'original': message,
            'reply': "",
        }
        self.start_sequential()
        return h

    def get_paragraph(self, msghash):

        if msghash not in self.hash_table:
            return "[DONE]"
        
        try:
            original = self.hash_table[msghash]['original']
            reply = self.hash_table[msghash]['reply']
        except:
            return "[DONE]"

        chat_history = self.memory.load_memory_variables(inputs=[])['history']

        d = {
            'input': original,
            'history': chat_history,
            'last_output': reply.strip(),
            'context': "",
            'stop': ["Human"],
        }
        output = self.run(d)
        output = output.replace(f"{self.ai_prefix}:",'')
        output = output.replace("\n",' ')
        output = output.strip().split(' ')
        if output[len(output)-1] in self.human_prefix:
            output = output[:len(output)-1]
        output = " ".join(output)+" "
        if len(output.strip()):
            self.hash_table[msghash]['reply'] += output
        elif len(self.hash_table[msghash]['reply'].strip()) == 0:
            sentiment = SentimentIntensityAnalyzer()
            sent = sentiment.polarity_scores(original)
            output = str(sent)

        if self.is_done() or len(output.strip())==0:
            if len(self.hash_table[msghash]['reply']) > 0:
                self.memory.save_context(
                    inputs = {self.human_prefix: original.strip()},
                    outputs = {self.ai_prefix: self.hash_table[msghash]['reply'].strip()}
                )
            del self.hash_table[msghash]

        return "[DONE]" if len(output.strip())==0 else output
