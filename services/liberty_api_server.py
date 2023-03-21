from typing import Any

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
    llama_model = LLaMAForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        #load_in_8bit=True,
        device_map="auto",
    )
    alpaca_model = PeftModel.from_pretrained(llama_model, "tloen/alpaca-lora-7b")
    return llama_model, alpaca_model

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
    # Tokenize sentences
    encoded_input = tokenizer([text], return_tensors='pt') #, padding=True, truncation=True
    # Compute token embeddings
    with torch.no_grad():
        model_output = llama_model(**encoded_input)
    # Perform pooling. In this case, max pooling.
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
            output = embed_text(text)
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
        llama_model, alpaca_model = load_model(config)
        if args.model:
            register_model(app)
        if args.embeddings:
            register_embedding(app)
        http_server = WSGIServer(('', int(config.get('DEFAULT', 'APIServicePort'))), app)
        http_server.serve_forever()
    else:
        parser.print_help()
