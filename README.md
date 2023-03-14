# LibertyAI: A Libertarian ChatGPT

This project is called LibertyAI, it's goal is to use FaceBook LLaMA for recreating
ChatGPT only without all the nasty woke censorship going on.

For the back end LangChain is being used.

In order to avoid the annoying loading time when restart the application, the model
is loaded separately and exposed to other applications through an API interface.

In order to avoid someone taxing your GPU power, by using your API, when you expose
the interface to the interwebs, I've introduced an API key feature.

You can just generate one and set it as environment variable by running

    export LIBERTYAI_API_KEY=`pwgen 24`

or you can use any other way for generating an API key for protecting your private
computing resources, you're free to do whatever you want, it's your GPUs.

Until the huggingface support for LLaMA is in the official transformers repository
I suggest you install my merge up the tracked HF repo with the incorporated changes
from https://github.com/zphang/transformers/tree/llama_push, by running

    pip3 install https://github.com/leviathanch/transformers/archive/refs/heads/main.tar.gz

after that you can install LibertyAI with

    pip3 install -e .

## Web interface

You can go into the folder web and execute:

    npm ci
    npm run dev # or: npm run build



