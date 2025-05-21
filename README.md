## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Build an easy-to-use application that can identify and highlight named entities (such as names, places, organizations) from user-input text, using a pre-trained language model and a simple interactive interface.

### DESIGN STEPS:

### STEP 1:
Set up access to the pre-trained NER model API by configuring authentication and endpoint URLs.

### STEP 2:
Create a function to send text input to the model API and process the model’s output to identify named entities.

### STEP 3:
Develop a Gradio interface for users to input text, display the recognized entities highlighted, and provide example inputs for testing.
### PROGRAM:
```
import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ['HF_API_NER_BASE']
hf_api_key = os.environ['HF_API_KEY']

def get_completion(inputs, parameters=None, ENDPOINT_URL=API_URL):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters:
        data["parameters"] = parameters
    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            merged_tokens.append(token)
    return merged_tokens

def ner(input):
    output = get_completion(input)
    return {"text": input, "entities": merge_tokens(output)}
gr.close_all()
demo = gr.Interface(fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="Named Entity Recognition",
    description="Find entities in the given text",
    allow_flagging="never",
    examples=["I'm Cynthia, I live in Chennai and I study engineering", "My name is Serena, I work at the High Court"]
)
demo.launch(share=True, server_port=7860)
```

### OUTPUT:
![Screenshot 2025-05-21 200245](https://github.com/user-attachments/assets/559e30fc-73f9-4ea9-9470-eb6fa02f4b72)

### RESULT:
A prototype application for Named Entity Recognition (NER) was successfully designed and developed by leveraging a fine-tuned BART model. The application was deployed using the Gradio framework, enabling easy user interaction and effective evaluation of the model’s performance.
