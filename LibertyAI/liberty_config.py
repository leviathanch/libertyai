import appdirs
import os
import hashlib
import configparser
from pathlib import Path

def get_configuration():
    config_dir = appdirs.user_config_dir("LibertyAI")
    filename = os.path.join(config_dir, 'config.ini')
    # Load from file
    config = configparser.ConfigParser()

    if os.path.exists(filename):
        config.read(filename)
    else:
        # General stuff for getting it to start up
        config.set('DEFAULT', 'ModelServicePort', value='5001')
        config.set('DEFAULT', 'EmbeddingServicePort', value='5002')
        config.set('DEFAULT', 'LLMDir', value='/home/user/HF_LLaMA/llama-7b')
        config.set('DEFAULT', 'TokenizerDir', value='/home/user/HF_LLaMA/tokenizer')
        config.set('DEFAULT', 'API_KEY', value=hashlib.sha256(os.urandom(32)).hexdigest())
        # Data base settings
        config.add_section('DATABASE')
        config.set('DATABASE', 'PGSQL_SERVER', value='localhost')
        config.set('DATABASE', 'PGSQL_SERVER_PORT', value='5432')
        config.set('DATABASE', 'PGSQL_DATABASE', value='libertyai')
        config.set('DATABASE', 'PGSQL_USER', value='libertyai')
        config.set('DATABASE', 'PGSQL_PASSWORD', value='libertyai')

        Path(os.path.dirname(os.path.abspath(filename))).mkdir( parents=True, exist_ok=True )
        with open(filename, 'w') as cf:
            config.write(cf)
            cf.close()

    return config


