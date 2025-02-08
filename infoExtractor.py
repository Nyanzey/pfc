import re
import os
import myapi
from pathlib import Path
import json

class InfoExtractor:
    def __init__(self, input_path=None, config_path=None, save_path=None, model_manager:myapi.ModelManager=None):
        self.input_path = input_path
        self.save_path = save_path
        self.model_manager = model_manager
        self.DC = None
        self.segments = None

        if config_path:
            with open(config_path, 'r') as file:
                self.config = json.load(file)

    def format_prompt(self, type, input):
        with open(f"./staticPrompts/{type}.txt", 'r') as file:
            prompt = file.read()

        if type == "createDC":
            prompt = prompt.replace("![narrative]", input["narrative"])
        elif type == "updateDC":   
            prompt = prompt.replace("![descriptions]", input["descriptions"])
            prompt = prompt.replace("![segment]", input["segment"])
        elif type == "segment":
            prompt = prompt.replace("![narrative]", input["narrative"])
            prompt = prompt.replace("![context]", input["context"])
        else:
            prompt = prompt.replace("![descriptions]", input["descriptions"])
            prompt = prompt.replace("![segment]", input["segment"])
            prompt = prompt.replace("![initial]", input["initial"])

        return prompt

    def parse_info(self, response):
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

    def parse_segment(self, response):
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

    def parse_final_prompt(self, response):
        pattern = r'final prompt: "(.*?)"'
        match = re.search(pattern, response)

        if match:
            return match.group(1).strip()
        else:
            return None
    
    def info_to_string(self):
        result = []

        if "characters" in self.DC:
            for character, description in self.DC["characters"].items():
                result.append(f"[{character}]({description})")

        if "scene" in self.DC:
            result.append(f"[Scene]({self.DC['scene']})")

        return "\n".join(result)

    # DC_to_descriptions
    def format_info(self):
        result = []

        if "characters" in self.DC:
            for character, description in self.DC["characters"].items():
                result.append(f'{character}:"{description}"')

        if "scene" in self.DC:
            description = self.DC['scene']
            result.append(f'scene:"{description}"')

        return "\n".join(result)
    
    def segments_to_string(self):
        result = []

        if self.segments:
            for segment in self.segments:
                result.append(f"[{segment['fragment']}]\n[{segment['appearance_change']} - {segment['scene_change']}]\n({segment['prompt']})")

        return "\n".join(result)

    def get_characteristics(self, regenerate_always=False):
        with open(self.input_path, 'r', encoding='utf-8') as file:
            story = file.read()

        input_dict = {'narrative': story}
        system_prompt = "You are a story analyzer."

        if os.path.exists(self.save_path + '/dictionary.txt') and not regenerate_always:
            print('Using buffered DC')
            raw_DC = Path(self.save_path + '/dictionary.txt').read_text(encoding='utf-8')
        else:
            print('Creating DC .....')
            createDC_prompt = self.format_prompt('createDC', input_dict)
            
            raw_DC = self.model_manager.text_query(createDC_prompt, system_prompt)

            with open(self.save_path + '/dictionary.txt', 'w', encoding='utf-8') as f:
                f.write(raw_DC)

        self.DC = self.parse_info(raw_DC)
        return self.DC

    def append_segments(self, src, dest):
        context_idx = src.find('[Context]')
        segments = src[:context_idx]
        context = src[context_idx:]
        if dest:
            dest = dest + segments
        else:
            dest = segments

        pattern = r'\[Context\]\((.*?)\)'
        match = re.search(pattern, context)
        if match:
            context = match.group(1).strip()
        else:
            context = 'No context found in the format specified'
        return dest, context

    def llm_part_segment(self):
        with open(self.input_path, 'r', encoding='utf-8') as file:
            story = file.read()

        total_length = len(story)
        context_length = self.config["Context_Length"]

        lines = story.split('\n')
        parts = []
        current = []
        total = 0
        for line in lines:
            print('line:', line)
            current.append(line)
            total += len(line)
            print('total:', total)
            if total >= (context_length/100)*total_length:
                parts.append('\n'.join(current))
                current = []
                total = 0
        if current:
            parts.append('\n'.join(current))
        print(f'split into {len(parts)} parts')
                    
        context = "This is the beginning of the story so there is no context to provide yet."
        all_segments = None
        for part in parts:
            input_dict = {'narrative': part, 'context': context}
            system_prompt = "You are a story analyzer."
                
            segments_prompt = self.format_prompt('segment', input_dict)

            raw_segments = self.model_manager.text_query(segments_prompt, system_prompt)
            print(raw_segments)
            all_segments, context = self.append_segments(raw_segments, all_segments)

        
        return all_segments

    def custom_segment(self):
        # to be implemented
        pass

    def segment_story(self, segment_method=llm_part_segment, regenerate_always=False):

        if os.path.exists(self.save_path + '/segments.txt') and not regenerate_always:
                print('Using buffered segments')
                all_segments = Path(self.save_path + '/segments.txt').read_text(encoding='utf-8')
        else:
            print('Segmenting story .....')
            all_segments = segment_method(self)

        self.segments = self.parse_segment(all_segments)
        return self.segments

    def save_all(self):
        with open(self.save_path + '/dictionary.txt', 'w') as file:
            file.write(self.info_to_string())

        with open(self.save_path + '/segments.txt', 'w') as file:
            file.write(self.segments_to_string())
