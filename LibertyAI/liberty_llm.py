from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

import requests
from LibertyAI.liberty_config import get_configuration

class LibertyLLM(LLM):

    endpoint: str
    temperature: float
    max_tokens: int

    json_headers = {
        'content-type': 'application/json',
    }

    @property
    def _llm_type(self) -> str:
        return "liberty"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        config = get_configuration()
        json_data = {
            'input' : prompt,
            'temperature' : self.temperature,
            'API_KEY': config.get('DEFAULT', 'API_KEY'),
            'max_new_tokens': self.max_tokens,
        }
        if stop is not None:
            json_data['stop_tokens'] = stop

        response = requests.post(
            self.endpoint,
            json = json_data,
        )
        try:
            reply = response.json()
        except:
            print("LibertyLLM_call",response)
            return ""

        if 'choices' in reply:
            choices = reply['choices']
            if len(choices) > 0:
                choice = choices[0]
                if 'text' in choice:
                    text = choice['text']
                    return text[(len(prompt)-1):].strip()
                else:
                    return ""
            else:
                return ""
        else:
            return ""
        #text = reply['choices'][0] #['text']
        #print(text)
        #text = text[len(prompt):]
        #print(text)
        return ""

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            'max_new_tokens': self.max_tokens,
            'temperature' : self.temperature,
        }

