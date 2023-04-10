import requests
import time
import sys

response = requests.post(
    "http://localhost:8080/api/submit",
    json = { 'text' : "Taxation is theft because",},
)

reply = response.json()

if 'uuid' in reply:
    uuid = reply['uuid']
    print("Got UUID", uuid)
    text = ""
    i = 0
    while text != "[DONE]":
        response = requests.post(
            "http://localhost:8080/api/fetch",
            json = {'uuid' : uuid, 'index': str(i) },
        )
        reply = response.json()
        text = reply['text']
        if text == "[BUSY]":
            time.sleep(0.1)
            continue

        i += 1
        if text != "[DONE]":
            sys.stdout.write(text)
            sys.stdout.flush()
    
    print("\n\n")
