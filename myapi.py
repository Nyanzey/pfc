from openai import OpenAI

def query_openai_api(model, user_prompt, system_prompt):
    with open('apikey.txt', 'r') as f:
        key = f.read()
    client = OpenAI(api_key=key)

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
    with open('apikey.txt', 'r') as f:
        key = f.read()
    client = OpenAI(api_key=key)

    response = client.images.generate(
        model=model,
        prompt=prompt,
        n=1,
        size="1024x1024",
        response_format="url"
    )
    return response.data[0].url

def query_llama(prompt, model):
    # to be implemented
    pass

def query_image_sd(prompt, model):
    # to be implemented
    pass

def text_query(model_info, user_prompt, system_prompt):
    if model_info['source'] == 'openai':
        return query_openai_api(model_info['model'], user_prompt, system_prompt)
    elif model_info['source'] == 'meta':
        return query_llama(user_prompt, system_prompt)
    return None
