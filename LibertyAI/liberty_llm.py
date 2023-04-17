import time

from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

import requests
from LibertyAI.liberty_config import get_configuration

class LibertyLLM(LLM):

    endpoint: str
    #temperature: float
    #max_tokens: int

    @property
    def _llm_type(self) -> str:
        return "liberty"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        uuid = self.submit_partial(prompt, stop)
        if not uuid:
            return "[DONE]"

        ret = ""
        text = ""
        i = 0
        while text != "[DONE]":
            text = self.get_partial(uuid, i)
            if text == "[BUSY]":
                time.sleep(0.1)
                continue
            i += 1
            if text != "[DONE]":
                ret += text

        return ret

    def submit_partial(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt = prompt.replace("[DONE]", b'\xf0\x9f\x96\x95'.decode()).replace("[BUSY]", b'\xf0\x9f\x96\x95'.decode())
        config = get_configuration()
        jd = {'text' : prompt,}
        if stop:
            jd['stop'] = stop

        response = requests.post(
            self.endpoint+'/submit',
            json = jd,
        )
        reply = response.json()
        if 'uuid' in reply:
            return reply['uuid']
        else:
            return None

    def get_partial(self, uuid, index):
        config = get_configuration()
        response = requests.post(
            self.endpoint+'/fetch',
            json = {'uuid' : uuid, 'index': str(index) },
        )
        reply = response.json()
        text = ""
        if 'text' in reply:
            text = reply['text']

        return text
