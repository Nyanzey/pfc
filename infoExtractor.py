import torch
import io
import sys
import numpy as np
import openai
import re
from pathlib import Path
    
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

    if 'scene' in parsed_data["characters"]:
        del parsed_data["characters"]['scene']

    # Find the scene description
    scene_pattern = r'\[scene\]\((.*?)\)'
    scene_match = re.search(scene_pattern, response)
    
    if scene_match:
        parsed_data["scene"] = scene_match.group(1).strip()
    
    return parsed_data

def parse_segment(response):
    # Initialize the dictionary to hold fragments and prompts
    parsed_data = []

    # Update the regex to account for possible spaces within the [YES-NO] segments
    pattern = r'\[([^\]]+)\]\s*\[\s*([^\]-]+)\s*-\s*([^\]]+)\s*\]\s*\(([^)]+)\)'
    matches = re.findall(pattern, response)

    # Populate the parsed data
    for fragment, appearance_change, scene_change, prompt in matches:
        parsed_data.append({
            "fragment": fragment.strip(),
            "appearance_change": True if appearance_change.strip().upper() == 'YES' else False,
            "scene_change": True if scene_change.strip().upper() == 'YES' else False,
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
    