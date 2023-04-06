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
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

from LibertyAI.liberty_llm import LibertyLLM
from LibertyAI.liberty_chain import LibertyChain
from LibertyAI.liberty_embedding import LibertyEmbeddings
from LibertyAI.liberty_config import get_configuration
from LibertyAI.liberty_agent import (
    get_zero_shot_agent,
    get_vector_db
)

def initialize_chatbot(**kwargs: Any) -> LibertyChain:
    llm = LibertyLLM(
        endpoint = "https://libergpt.univ.social/api/generation",
        temperature = 0.7,
        max_tokens = 20,
        verbose = True,
    )
    
    sum_mem = ConversationSummaryMemory(
        ai_prefix = "LibertyAI",
        llm = LibertyLLM(
            endpoint = "https://libergpt.univ.social/api/generation",
            temperature = 0.7,
            max_tokens = 20,
            verbose = True,
        )
    )
    
    conv_mem = ConversationBufferWindowMemory(
        ai_prefix = "LibertyAI",
        k = 1,
    )

    emb = LibertyEmbeddings(
        endpoint = "https://libergpt.univ.social/api/embedding"
    )
    
    vecdb = get_vector_db()

    chain = LibertyChain(
        summary = sum_mem,
        memory = conv_mem,
        llm = llm,
        mrkl = get_zero_shot_agent( llm ),
        verbose = True,
        user_name = kwargs['name'],
        user_mail = kwargs['email'],
        embeddings = emb,
        vectordb = vecdb,
    );

    return chain
