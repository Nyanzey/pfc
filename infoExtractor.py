import torch
import io
import sys
import numpy as np
import openai
import re
#client = openai.OpenAI()

"""
def query_openai_api(user_prompt, system_prompt, model):
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
    return completion.choices[0].message

def query_image_api(prompt, model):
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url
"""
    
def get_txt_prompt(type, input):
    with open(f"./staticPrompts/{type}.txt", 'r') as file:
        prompt = file.read()
    if type == "createDC":
        prompt = prompt.replace("![narrative]", input["narrative"])
    elif type == "refineDC":
        prompt = prompt.replace("![descriptions]", input["descriptions"])
    elif type == "updateDC":   
        prompt = prompt.replace("![descriptions]", input["descriptions"])
        prompt = prompt.replace("![segment]", input["segment"])
    elif type == "segment":
        prompt = prompt.replace("![narrative]", input["narrative"])
    else:
        prompt = prompt.replace("![descriptions]", input["descriptions"])
        prompt = prompt.replace("![segment]", input["segment"])
        prompt = prompt.replace("![initial]", input["initial"])
    return prompt

def parse_DC(response):
    # Initialize the dictionary to hold character and scene descriptions
    parsed_data = {
        "characters": {},
        "scene": ""
    }
    
    # Use regex to find all character descriptions
    character_pattern = r'\[(.*?)\]\((.*?)\)'
    characters = re.findall(character_pattern, response)

    # Populate the characters in the dictionary
    for name, description in characters:
        parsed_data["characters"][name] = description.strip()

    # Find the scene description
    scene_pattern = r'\[scene\]\((.*?)\)'
    scene_match = re.search(scene_pattern, response)
    
    if scene_match:
        parsed_data["scene"] = scene_match.group(1).strip()
    
    return parsed_data

def parse_segment(response):
    # Initialize the dictionary to hold fragments and prompts
    parsed_data = []

    # Use regex to match the fragments and their corresponding prompts
    pattern = r'\[([^\]]+)\]\[([^\]-]+)-([^\]]+)\]\(([^)]+)\)'
    matches = re.findall(pattern, response)

    # Populate the parsed data
    for fragment, appearance_change, scene_change, prompt in matches:
        parsed_data.append({
            "fragment": fragment.strip(),
            "appearance_change": appearance_change.strip(),
            "scene_change": scene_change.strip(),
            "prompt": prompt.strip()
        })

    return parsed_data

def parse_final_prompt(response):
    # Use regex to extract the final prompt
    pattern = r'final prompt: "(.*?)"'
    match = re.search(pattern, response)

    if match:
        return match.group(1).strip()
    else:
        return None
    
def DC_to_string(DC):
    result = []

    # Process characters
    if "characters" in DC:
        for character, description in DC["characters"].items():
            result.append(f"[{character}]({description})")

    # Process scene
    if "scene" in DC:
        result.append(f"[Scene]({DC['scene']})")

    return "\n".join(result)

def DC_to_descriptions(DC):
    result = []

    # Process characters
    if "characters" in DC:
        for character, description in DC["characters"].items():
            result.append(f'{character}:"{description}"')

    # Process scene
    if "scene" in DC:
        description = DC['scene']
        result.append(f'Scene:"{description}"')

    return "\n".join(result)

with open("./dynamicPrompts/segments.txt", 'r') as file:
    segments = file.read()

print(parse_segment(segments))