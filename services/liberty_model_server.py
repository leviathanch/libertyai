from typing import Any

from flask import Flask, request
from gevent.pywsgi import WSGIServer

import threading

from transformers import LLaMATokenizer, LLaMAForCausalLM, pipeline

from LibertyAI import get_configuration

from langchain.llms import HuggingFacePipeline

from langchain.embeddings.huggingface import HuggingFaceEmbeddings

app = Flask(__name__)
#app.debug = True


def get_pipeline() -> Any:
    pipe = pipeline(
        "feature-extraction",
        model=model,
        tokenizer=tokenizer,
    )
    return pipe


def inference_fn(pipeline: Any, prompt: str) -> Any:
    return pipeline(prompt)

@app.route('/api/embedding', methods=['POST'])
def embedding():
    data = request.get_json()

    try:
        key = data['API_KEY']
    except:
        return {'error': "Invalid API key"}

    try:
        text = data['text']
    except:
        return {'error': "No text provided"}

    if key == config.get('DEFAULT', 'API_KEY'):
        sem.acquire()
        output = embedding.embed_query(text)
        sem.release()
        return {'embedding': output}
    else:
        return {'error': "Invalid API key"}

@app.route('/api/generation', methods=['POST'])
def generation():
    data = request.get_json()

    try:
        key = data['API_KEY']
    except:
        return {'error': "Invalid API key"}

    try:
        text = data['input']
    except:
        return {'error': "No input field provided"}

    try:
        temp = data['temperature']
    except:
        temp = 0

    try:
        max_tokens = data['max_new_tokens']
    except:
        max_tokens = 20

    try:
        stop_tokens = data['stop_tokens']
    except:
        stop_tokens = None

    if key == config.get('DEFAULT', 'API_KEY'):
        sem.acquire()
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=float(temp),
            max_new_tokens=int(max_tokens),
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        if stop_tokens:
            generated_text = llm(text, stop=stop_tokens)
        else:
            generated_text = llm(text)

        sem.release()
        return {'generated_text': generated_text}
    else:
        return {'error': "Invalid API key"}

if __name__ == '__main__':
    config = get_configuration()
    tokenizer = LLaMATokenizer.from_pretrained(
        config.get('DEFAULT', 'TokenizerDir')
    )
    print("Loading model...")
    dmap = {
        'model.embed_tokens': 1,
        'model.layers.0': 0,
        'model.layers.1': 0,
        'model.layers.2': 0,
        'model.layers.3': 0,
        'model.layers.4': 0,
        'model.layers.5': 0,
        'model.layers.6': 0,
        'model.layers.7': 0,
        'model.layers.8': 0,
        'model.layers.9': 0,
        'model.layers.10': 0,
        'model.layers.11': 0,
        'model.layers.12': 0,
        'model.layers.13': 1,
        'model.layers.14': 1,
        'model.layers.15': 1,
        'model.layers.16': 1,
        'model.layers.17': 1,
        'model.layers.18': 1,
        'model.layers.19': 1,
        'model.layers.20': 1,
        'model.layers.21': 1,
        'model.layers.22': 1,
        'model.layers.23': 1,
        'model.layers.24': 1,
        'model.layers.25': 1,
        'model.layers.26': 1,
        'model.layers.27': 1,
        'model.layers.28': 1,
        'model.layers.29': 1,
        'model.layers.30': 1,
        'model.layers.31': 1,
        'model.norm': 1,
        'lm_head': 1
    }

    model = LLaMAForCausalLM.from_pretrained(
        config.get('DEFAULT', 'LLMDir'),
        device_map=dmap,
        torch_dtype="auto",
    )
    print("Loaded model.")
    
    print("Load embedding model")
    embedding = HuggingFaceEmbeddings()
    print("Loaded model.")

    sem = threading.Semaphore()
    http_server = WSGIServer(('', int(config.get('DEFAULT', 'ModelServicePort'))), app)
    http_server.serve_forever()

