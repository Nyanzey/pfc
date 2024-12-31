from openai import OpenAI
import torch
from transformers import pipeline
import json

with open('config.json', 'r') as file:
    config = json.load(file)

if config['Text-to-Text']['source'] == 'meta':

    model_id = config['Text-to-Text']['model']
    pipe_llama = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

if config['Text-to-Text']['requires_key']:
    api_text_key = input(f"An API key is required for {config['Text-to-Text']['model']} by {config['Text-to-Text']['source']}: ")

if config['Text-to-Image']['requires_key']:
    api_image_key = input(f"An API key is required for {config['Text-to-Image']['model']} by {config['Text-to-Image']['source']}: ")

def query_openai_api(model, user_prompt, system_prompt):
    client = OpenAI(api_key=api_text_key)

    completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": system_prompt
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": user_prompt
                }
            ]
        }
    ])
    return completion.choices[0].message.content

def query_image_openai_api(prompt, model):
    client = OpenAI(api_key=api_image_key)

    response = client.images.generate(
        model=model,
        prompt=prompt,
        n=1,
        size="1024x1024",
        response_format="url"
    )
    return response.data[0].url

def query_llama(user_prompt, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    outputs = pipe_llama(
        messages,
        max_new_tokens=2048,
    )

    return outputs[0]["generated_text"][-1]['content']

def query_image_sd(prompt, model):
    # to be implemented
    pass

def text_query(user_prompt, system_prompt):
    if config['source'] == 'openai':
        return query_openai_api(config['model'], user_prompt, system_prompt)
    elif config['source'] == 'meta':
        return query_llama(user_prompt, system_prompt)
    return None
