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
    parsed_data = {
        "characters": {},
        "scene": ""
    }
    
    character_pattern = r'\[(.*?)\]\((.*?)\)'
    characters = re.findall(character_pattern, response)

    for name, description in characters:
        parsed_data["characters"][name] = description.strip()

    if 'scene' in parsed_data["characters"]:
        del parsed_data["characters"]['scene']

    if 'Scene' in parsed_data["characters"]:
        del parsed_data["characters"]['Scene']

    scene_pattern = r'(?i)\[scene\]\((.*?)\)'
    scene_match = re.search(scene_pattern, response)
    
    if scene_match:
        parsed_data["scene"] = scene_match.group(1).strip()
    
    return parsed_data

def parse_segment(response):
    parsed_data = []

    pattern = r'\[([^\]]+)\]\s*\[\s*([^\]-]+)\s*-\s*([^\]]+)\s*\]\s*\(([^)]+)\)'
    matches = re.findall(pattern, response)

    for fragment, appearance_change, scene_change, prompt in matches:
        parsed_data.append({
            "fragment": fragment.strip(),
            "appearance_change": True if appearance_change.strip().upper() == 'YES' else False,
            "scene_change": True if scene_change.strip().upper() == 'YES' else False,
            "prompt": prompt.strip()
        })

    return parsed_data

def parse_final_prompt(response):
    pattern = r'final prompt: "(.*?)"'
    match = re.search(pattern, response)

    if match:
        return match.group(1).strip()
    else:
        return None
    
def DC_to_string(DC):
    result = []

    if "characters" in DC:
        for character, description in DC["characters"].items():
            result.append(f"[{character}]({description})")

    if "scene" in DC:
        result.append(f"[Scene]({DC['scene']})")

    return "\n".join(result)

def DC_to_descriptions(DC):
    result = []

    if "characters" in DC:
        for character, description in DC["characters"].items():
            result.append(f'{character}:"{description}"')

    if "scene" in DC:
        description = DC['scene']
        result.append(f'scene:"{description}"')

    return "\n".join(result)
    