# Chatbot service (DBUS)

In this folder, create a virtual environment by running

    python3 -m venv chatbotenv

Change into the environment by running

    source chatbotenv/bin/activate

Install all the dependencies by running

    pip3 install -r requirements.txt

Then go one level up and install the LibertyAI module

    cd ..
    pip3 install -e .

You can test, whether all the dependencies are satisfied by running
the uwsgi server manually:

    chatbotenv/bin/uwsgi --ini chatbot.ini
