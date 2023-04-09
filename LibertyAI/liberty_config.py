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
        config.set('DEFAULT', 'APIServicePort', value='5001')
        config.set('DEFAULT', 'LLAMA_CPP_MODEL', '')
        # API
        config.add_section('API')
        config.set('API', 'KEY', value=hashlib.sha256(os.urandom(32)).hexdigest())
        config.set('API', 'GENERATION_ENDPOINT', "https://libergpt.univ.social/api/generation")
        config.set('API', 'EMBEDDING_ENDPOINT', "https://libergpt.univ.social/api/embedding")
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


