import re
import os
import myapi
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

def get_characteristics(input_path, model_info, regenerate_always=False):
    with open(input_path, 'r', encoding='utf-8') as file:
        story = file.read()

    input_dict = {'narrative': story}
    system_prompt = "You are a story analyzer."
    dictionary_path = './dynamicPrompts/dictionary.txt'

    if os.path.exists(dictionary_path) and not regenerate_always:
        print('Using buffered DC')
        raw_DC = Path(dictionary_path).read_text(encoding='utf-8')
    else:
        print('Creating DC .....')
        createDC_prompt = get_txt_prompt('createDC', input_dict)
        
        raw_DC = myapi.text_query(model_info, createDC_prompt, system_prompt)

        with open(dictionary_path, 'w', encoding='utf-8') as f:
            f.write(raw_DC)

    print(raw_DC)
    return parse_DC(raw_DC)

def segment_story(input_path, model_info, regenerate_always=False):
    with open(input_path, 'r', encoding='utf-8') as file:
        story = file.read()

    input_dict = {'narrative': story}
    system_prompt = "You are a story analyzer."
    segments_path = './dynamicPrompts/segments.txt'

    if os.path.exists(segments_path) and not regenerate_always:
        print('Using buffered segments')
        raw_segments = Path(segments_path).read_text(encoding='utf-8')
    else:
        print('Segmenting story .....')
        segments_prompt = get_txt_prompt('segment', input_dict)

        raw_segments = myapi.text_query(model_info, segments_prompt, system_prompt)
        
        with open(segments_path, 'w', encoding='utf-8') as f:
            f.write(raw_segments)

    print(raw_segments)
    return parse_segment(raw_segments)

def custom_segment_story(input_path, regenerate_always=False):
    # to be implemented
    pass