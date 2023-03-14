import appdirs
import os
import configparser
from pathlib import Path

def get_configuration():
    config_dir = appdirs.user_config_dir("LibertyAI")
    filename = os.path.join(config_dir, 'config.ini')
    defaults = {
        
    }
    # Load from file
    config = configparser.ConfigParser()

    if os.path.exists(filename):
        config.read(filename)
    else:
        config.set('DEFAULT', 'ModelServicePort', value='5001')
        config.set('DEFAULT', 'LLMDir', value='~/HF_LLaMA/llama-7b')
        config.set('DEFAULT', 'TokenizerDir', value='~/HF_LLaMA/tokenizer')
        Path(os.path.dirname(os.path.abspath(filename))).mkdir( parents=True, exist_ok=True )
        with open(filename, 'w') as cf:
            config.write(cf)
            cf.close()

    return config


