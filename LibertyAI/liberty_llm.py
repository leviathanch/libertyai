from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests
import os

class LibertyLLM(LLM):

    endpoint: str

    json_headers = {
        'content-type': 'application/json',
    }

    @property
    def _llm_type(self) -> str:
        return "liberty"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        #if stop is not None:
        #    raise ValueError("stop kwargs are not permitted.")
        json_data = {
            'input' : prompt,
            'temperature' : 0.4,
            'API_KEY': os.environ['LIBERTYAI_API_KEY'],
            'max_new_tokens': 20,
        }
        print('json_data',json_data)
        response = requests.post(
            self.endpoint,
            json = json_data,
        )
        reply = response.json()

        return reply['generated_text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}

