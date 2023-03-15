from flask import Flask, request
from gevent.pywsgi import WSGIServer
import os
import threading

from transformers import LLaMATokenizer, LLaMAForCausalLM, pipeline

from LibertyAI import get_configuration

app = Flask(__name__)
#app.debug = True

@app.route('/api/generation', methods=['POST'])
def generation():
    data = request.get_json()
    text = data['input']
    key = data['API_KEY']
    try:
        temp = data['temperature']
    except:
        temp = 0
    try:
        max_tokens = data['max_new_tokens']
    except:
        max_tokens = 20

    if key == os.environ['LIBERTYAI_API_KEY']:
        sem.acquire()
        response = pipe(text, temperature=float(temp), max_new_tokens=int(max_tokens))
        sem.release()
        return response[0]
    else:
        return {'error': "Invalid API key"}

if __name__ == '__main__':
    config = get_configuration()
    tokenizer = LLaMATokenizer.from_pretrained(
        config.get('DEFAULT', 'TokenizerDir')
    )
    model = LLaMAForCausalLM.from_pretrained(
        config.get('DEFAULT', 'LLMDir'),
        device_map="auto",
        torch_dtype="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
    )
    sem = threading.Semaphore()
    http_server = WSGIServer(('', int(config.get('DEFAULT', 'ModelServicePort'))), app)
    http_server.serve_forever()

