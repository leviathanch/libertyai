from langchain.embeddings.huggingface import HuggingFaceEmbeddings

_prefix = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question."""
_suffix = """## Example:

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

_eg_template = """## Example:

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question: {answer}"""

def key_word_extractor(llm):
    rds = Redis.from_documents(docs, embeddings,redis_url="redis://localhost:6379")
    rds.similarity_search(query)

    #example_selector = SemanticSimilarityExampleSelector(vectorstore=eg_store, k=4)

    #_eg_prompt = PromptTemplate(
    #    template=_eg_template,
    #    input_variables=["chat_history", "question", "answer"],
    #)

    prompt = FewShotPromptTemplate(
        prefix=_prefix,
        suffix=_suffix,
        #example_selector=example_selector,
        example_prompt=_eg_prompt,
        input_variables=["question", "chat_history"],
    )

    return LLMChain(llm=llm, prompt=prompt)
