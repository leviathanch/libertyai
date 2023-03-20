from typing import Optional, List, Mapping, Any, Dict, Tuple

from langchain.chains import ConversationChain
from langchain.vectorstores.pgvector import PGVector

from langchain import PromptTemplate

LIBERTY_PROMPT_TEMPLATE = """This is a conversation between a human and LibertyAI, a hardcore Libertarian chatbot.
LibertyAI is very talkative and wants to explain to everyone why Liberty is important in great detail.

{history}
Human:
{input}
AI:
"""

class LibertyChain(ConversationChain):
    vstore: PGVector

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        print(inputs)
        question = inputs["input"]
        chat_history_str = self._get_chat_history(inputs["history"])
        if chat_history_str:
            new_question = self.key_word_extractor.run(
                question=question, chat_history=chat_history_str
            )
        else:
            new_question = question
        print(new_question)
        docs = self.vstore.similarity_search(new_question, k=4)
        new_inputs = inputs.copy()
        new_inputs["input"] = new_question
        new_inputs["history"] = chat_history_str
        answer, _ = self.chain.combine_docs(docs, **new_inputs)
        return {"answer": answer}
