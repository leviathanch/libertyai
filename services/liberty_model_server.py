from flask import Flask, request
from gevent.pywsgi import WSGIServer
import os

from transformers import LLaMATokenizer, LLaMAForCausalLM, pipeline

from LibertyAI import get_configuration

app = Flask(__name__)

API_KEY = os.environ['LIBERTYAI_API_KEY']

BUSY_STATE = False

@app.route('/generation', methods=['POST'])
def chat():
    text = request.form['input']
    temp = request.form['temperatue']
    key = request.form['API_KEY']
    if key == API_KEY:
        if BUSY_STATE:
            return {'busy': "Please try again later"}
        else:
            BUSY_STATE = True
            response = pipe(text, temperature=temp)
            BUSY_STATE = False
            return {'response': response}
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
    http_server = WSGIServer(('', int(config.get('DEFAULT', 'ModelServicePort'))), app)
    http_server.serve_forever()

