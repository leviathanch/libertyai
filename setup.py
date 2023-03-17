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
        'services/liberty_model_server.py'
    ],
    url='https://redmine.libresilicon.com/projects/danube-river',
    license='LICENSE.txt',
    description='A clone of ChatGPT, based on LLaMA, without woke censorship',
    long_description=open('README.md').read(),
    install_requires=[
        'transformers',
        'sentence_transformers',
        'flask',
        'flask-login',
        'flask-sqlalchemy',
        'langchain',
        'pgvector',
        'psycopg2',
        'gevent',
        'typing',
    ],
)
