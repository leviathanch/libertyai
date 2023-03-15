from flask import Flask, request
from gevent.pywsgi import WSGIServer
import os
import threading

from transformers import LLaMATokenizer, LLaMAForCausalLM, pipeline

from LibertyAI import get_configuration

app = Flask(__name__)

@app.route('/generation', methods=['POST'])
def generation():
    text = request.form['input']
    key = request.form['API_KEY']
    try:
        temp = request.form['temperature']
    except:
        temp = 0
    try:
        max_tokens = request.form['max_new_tokens']
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
        device_map="balanced",
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

