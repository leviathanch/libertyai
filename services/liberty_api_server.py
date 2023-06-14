from typing import Any
import gc
import uuid

from flask import Flask, request
from gevent.pywsgi import WSGIServer

import torch.multiprocessing as mp

import threading
import torch
from typing import Any, Dict, List, Mapping, Optional, Set

import os, copy, types, gc, sys
import numpy as np

from LibertyAI import get_configuration

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import ADA_TOKEN_COUNT

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sentence_transformers import SentenceTransformer, util

import argparse

from rwkv.model import RWKV
from rwkv.utils import PIPELINE
import tokenizers

def load_model(config):
    args = types.SimpleNamespace()
    args.RUN_DEVICE = "cuda"
    args.FLOAT_MODE = "fp16"
    os.environ["RWKV_JIT_ON"] = '1'
    os.environ["RWKV_RUN_DEVICE"] = '0'
    model_kwargs = {}
    model = RWKV(
        "/home/user/RWKV/RWKV-4-Raven-3B-v9-Eng99%-Other1%-20230411-ctx4096.pth",
        strategy="cuda:0 fp16 *10 -> cuda:1 fp16 *8",
        **model_kwargs
    )
    tokenizer = tokenizers.Tokenizer.from_file("/home/user/RWKV/20B_tokenizer.json")
    model.share_memory()
    model.eval()
    return model, tokenizer

tokens = {}
processes = {}

def run_rnn(tokens):
    AVOID_REPEAT = "，：？！"
    AVOID_REPEAT_TOKENS = []
    for i in AVOID_REPEAT:
        dd = tokenizer.encode(i)
        assert len(dd) == 1
        AVOID_REPEAT_TOKENS += dd

    model_tokens = []
    model_state = None
    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    out, model_state = model.forward(tokens, model_state)
    #out = out.tolist()
    print("out", out)
    out[0] = -999999999  # disable <|endoftext|>
    #out[187] += newline_adj # adjust \n probability
    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999
    return out, model_tokens, model_state

def generation_job(data):

    sem.acquire()
    logits, model_tokens, model_state = run_rnn(tokenizer.encode(data['prompt']).ids)
    sem.release()

    begin = len(model_tokens)
    out_last = begin
    decoded = ""
    for i in range(int(data['max_tokens_per_generation'])):
        sem.acquire()
        token = tokenizer.sample_logits(
            logits,
            model_tokens,
            1024, # args.ctx_len,
            temperature=data['temperature'],
            top_p=data['top_p'],
        )
        sem.release()

        sem.acquire()
        logits, model_tokens, model_state = run_rnn([token])
        sem.release()
        xxx = tokenizer.decode(model_tokens[out_last:])
        print(xxx)

def register_model(app):
    @app.route('/api/completion/submit', methods=['POST'])
    def completion_submit():
        data = request.get_json()
        if "text" not in data:
            return {'error': "No input field provided"}

        uid = str(uuid.uuid4())
        job_params = {}
        job_params['max_tokens_per_generation'] = int(data['max_new_tokens']) if 'max_new_tokens' in data else 256
        job_params['temperature'] = float(data['temperature']) if 'temperature' in data else 1.0
        job_params['top_p'] = float(data['top_p']) if 'top_p' in data else 0.5
        job_params['CHUNK_LEN'] = int(data['CHUNK_LEN']) if 'CHUNK_LEN' in data else 256
        job_params['prompt'] = data['text']
        job_params['uuid'] = uid
        tokens[uid] = []
        print("Starting job "+uid)
        #processes[uid] = mp.Process(target=generation_job, args=(job_params, tokenizer, model))
        #processes[uid].start()
        torch.jit.fork(generation_job, job_params)
        return {'uuid': uid}

    @app.route('/api/completion/fetch', methods=['POST'])
    def completion_fetch():
        data = request.get_json()
        if "uuid" not in data:
            return {'text': "[DONE]"}
        uid = data["uuid"]
        if "index" not in data:
            return {'text': "[DONE]"}
        sem.acquire()
        text = tokens[uid][index] if index < len(tokens[uid]) else "[BUSY]"            
        sem.release()
        return {'text': text}

def embed_text(text):
    return embedding_model.encode([text])

def register_embedding(app):
    @app.route('/api/embedding', methods=['POST'])
    def embedding():
        data = request.get_json()

        try:
            text = data['text']
        except:
            return {'error': "No text provided"}

        sem.acquire()
        gc.collect()
        output = embed_text(text)
        gc.collect()
        torch.cuda.empty_cache()
        sem.release()
        return {'embedding': output[0].tolist()}

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

        if key == config.get('API', 'KEY'):
            sem.acquire()
            gc.collect()
            sent = sentiment_model.polarity_scores(text)
            gc.collect()
            torch.cuda.empty_cache()
            sem.release()
            return sent
        else:
            return {'error': "Invalid API key"}

if __name__ == '__main__':
    mp.set_start_method('spawn')
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
        gc.freeze()
        gc.enable()
        if args.model:
            model, tokenizer = load_model(config)
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
