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

from LibertyAI.liberty_chain import LibertyChain
from LibertyAI.liberty_config import get_configuration
from LibertyAI.liberty_agent import (
    get_zero_shot_agent,
    get_vector_db
)

def initialize_chatbot(**kwargs: Any) -> LibertyChain:

    main_llm = kwargs['llm']
    main_emb = kwargs['emb']

    #sum_mem = ConversationSummaryMemory(
    #    ai_prefix = "LibertyAI",
    #    llm = main_llm,
    #)
    
    conv_mem = ConversationBufferWindowMemory(
        ai_prefix = "LibertyAI",
        k = 1,
    )

    vecdb = get_vector_db()

    chain = LibertyChain(
        #summary = sum_mem,
        summary = None,
        memory = conv_mem,
        llm = main_llm,
        mrkl = get_zero_shot_agent( main_llm ),
        verbose = True,
        user_name = kwargs['name'],
        user_mail = kwargs['email'],
        embeddings = main_emb,
        vectordb = vecdb,
    );

    return chain
