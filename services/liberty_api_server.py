from typing import Any
import gc

from flask import Flask, request
from gevent.pywsgi import WSGIServer

import threading
import torch

from transformers import LLaMATokenizer, LLaMAForCausalLM, pipeline
from peft import PeftModel

from LibertyAI import get_configuration

from langchain.llms import HuggingFacePipeline

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import ADA_TOKEN_COUNT

from sentence_transformers import SentenceTransformer, util

import argparse

def load_model(config):
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
        'model.layers.13': 0,
        'model.layers.14': 0,
        'model.layers.15': 0,
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
        'lm_head': 1}

    model = LLaMAForCausalLM.from_pretrained(
        "chavinlo/alpaca-native",
        torch_dtype=torch.float16,
        device_map = dmap,
    )

    return model

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
        
        try:
            output_scores = data['output_scores']
        except:
            output_scores = True

        try:
            top_p = data['top_p']
        except:
            top_p = 0.75

        try:
            num_beams = data['num_beams']
        except:
            num_beams = 4

        if key == config.get('DEFAULT', 'API_KEY'):
            sem.acquire()
            pipe = pipeline(
                "text-generation",
                model = model,
                tokenizer = tokenizer,
                temperature = float(temp),
                max_new_tokens = int(max_tokens),
                output_scores = output_scores,
                top_p = top_p,
                num_beams = num_beams,
            )
            llm = HuggingFacePipeline(pipeline=pipe)

            gc.collect()
            if stop_tokens:
                generated_text = llm(text, stop=stop_tokens)
            else:
                generated_text = llm(text)

            gc.collect()
            torch.cuda.empty_cache()
            sem.release()
            return {'generated_text': generated_text}
        else:
            return {'error': "Invalid API key"}

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_text(text):
    encoded_input = tokenizer([text], return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

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
            gc.collect()
            output = embed_text(text)
            gc.collect()
            torch.cuda.empty_cache()
            sem.release()
            return {'embedding': output[0].tolist()}
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
    args = parser.parse_args()
    if args.model or args.embeddings:
        config = get_configuration()
        sem = threading.Semaphore(10)
        app = Flask(__name__)
        tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        model = load_model(config)
        gc.freeze()
        gc.enable()
        if args.model:
            register_model(app)
        if args.embeddings:
            register_embedding(app)
        http_server = WSGIServer(('', int(config.get('DEFAULT', 'APIServicePort'))), app)
        http_server.serve_forever()
    else:
        parser.print_help()
