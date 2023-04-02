from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Sequence,
    List,
    Union,
)

from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain

from LibertyAI.liberty_llm import LibertyLLM
from LibertyAI.liberty_embedding import LibertyEmbeddings
from LibertyAI.liberty_chain import LibertyChain
from LibertyAI.liberty_config import get_configuration

def initialize_chatbot(**kwargs: Any) -> LibertyChain:
    mem = ConversationSummaryMemory(
        llm = LibertyLLM(
            endpoint = "https://libergpt.univ.social/api/generation",
            temperature = 0.7,
            max_tokens = 20,
            verbose = True,
        ),
    )
    chain = LibertyChain(
        memory = mem,
        llm = LibertyLLM(
            endpoint = "https://libergpt.univ.social/api/generation",
            temperature = 0.7,
            max_tokens = 20,
            verbose = True,
        ),
        verbose = True,
    );
    return chain


'''
    mem_buffer = ConversationBufferWindowMemory(
        k = 10,
        human_prefix = kwargs['name'],
        ai_prefix = "LibertyAI",
    )

'''
