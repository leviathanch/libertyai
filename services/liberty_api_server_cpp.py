from typing import Any

from flask import Flask, request
from gevent.pywsgi import WSGIServer

import threading

from LibertyAI.liberty_config import get_configuration

from llama_cpp import Llama
import torch

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sentence_transformers import SentenceTransformer, util

import argparse

def load_model(config):
    llm = Llama(model_path=config.get('DEFAULT', 'LLAMA_CPP_MODEL'))
    return llm

def register_model(app):
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
            temp = float(data['temperature'])
        except:
            temp = 0

        try:
            max_new_tokens = int(data['max_new_tokens'])
        except:
            max_new_tokens = 20

        try:
            stop_tokens = data['stop']
        except:
            stop_tokens = []
        
        if key == config.get('DEFAULT', 'API_KEY'):
            sem.acquire()
            output = llm(text, stop=stop_tokens, max_tokens=max_new_tokens, temperature=temp, echo=True)
            sem.release()
            return output
        else:
            return {'error': "Invalid API key"}

def embed_text(text):
    return embedding_model.encode([text])

def register_embedding(app):
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
            output = embed_text(text)
            torch.cuda.empty_cache()
            sem.release()
            return {'embedding': output[0].tolist()}
        else:
            return {'error': "Invalid API key"}

def register_sentiment(app):
    @app.route('/api/sentiment', methods=['POST'])
    def sentiment():
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
            sent = sentiment_model.polarity_scores(text)
            sem.release()
            return sent
        else:
            return {'error': "Invalid API key"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='LibertyAI: API server',
        description='Choose what API services to run',
        epilog='Give me Liberty or give me death - Patrick Henry, 1775'
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-m', '--model', action='store_true')
    parser.add_argument('-e', '--embeddings', action='store_true')
    parser.add_argument('-s', '--sentiment', action='store_true')
    args = parser.parse_args()
    if args.model or args.embeddings:
        config = get_configuration()
        sem = threading.Semaphore(10)
        app = Flask(__name__)
        if args.model:
            llm = load_model(config)
            register_model(app)
        if args.embeddings:
            embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            register_embedding(app)
        if args.sentiment:
            sentiment_model = SentimentIntensityAnalyzer()
            register_sentiment(app)
        http_server = WSGIServer(('', int(config.get('DEFAULT', 'APIServicePort'))), app)
        http_server.serve_forever()
    else:
        parser.print_help()
