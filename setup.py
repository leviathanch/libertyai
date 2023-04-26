from setuptools import setup

name='LibertyAI'
version='0.1'
release=version+'.0'

setup(
    name=name,
    version=version,
    author='A ChatGPT clone',
    author_email='leviathan@libresilicon.com',
    packages=['LibertyAI'],
    scripts=[
        'services/liberty_api_server.py',
    ],
    url='https://redmine.libresilicon.com/projects/danube-river',
    license='LICENSE.txt',
    description='A clone of ChatGPT, based on LLaMA, without woke censorship',
    long_description=open('README.md').read(),
    install_requires=[
        'argparse',
        'appdirs',
        'bs4',
        'peft',
        'numpy',
        'transformers',
        'sentence_transformers',
        'datasets',
        'loralib',
        'sentencepiece',
        'bitsandbytes',
        'langchain',
        'langdetect',
        'pgvector',
        'psycopg2',
        'gevent',
        'typing',
        'SQLAlchemy==1.4.18',
        'spacy',
        'vaderSentiment',
        'word2number',
    ],
)
