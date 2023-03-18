from typing import Any

from flask import Flask, request
from gevent.pywsgi import WSGIServer

import threading
import torch

from transformers import LLaMATokenizer, LLaMAForCausalLM, pipeline

from LibertyAI import get_configuration

from langchain.llms import HuggingFacePipeline

from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

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
        output = embedding.encode(text)
        sem.release()
        return {'embedding': output.tolist()}
    else:
        return {'error': "Invalid API key"}

if __name__ == '__main__':
    config = get_configuration()
    embedding = SentenceTransformer('microsoft/mpnet-base')
    sem = threading.Semaphore(10)
    http_server = WSGIServer(('', int(config.get('DEFAULT', 'EmbeddingServicePort'))), app)
    http_server.serve_forever()

