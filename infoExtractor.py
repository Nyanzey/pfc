import re
import os
import myapi
from pathlib import Path
import json
import spacy
import copy


trigger_keywords = set([
    # Location Changes
    "arrived", "entered", "stepped into", "left", "exited", "moved", "traveled", "reached", "drove to", "flew to", "sailed to", "inside", "outside", "underground", "underwater", "upstairs", "downstairs", "beyond", "across", "through", "city", "town", "village", "forest", "cave", "castle", "school", "home", "hospital", "battlefield", "alley",
    # Character Appearances & Disappearances
    "saw", "noticed", "met", "was greeted by", "appeared", "emerged", "introduced", "joined", "encountered", "disappeared", "vanished", "left", "walked away", "ran off", "faded into the distance", "turned away",
    # Major Actions & Events
    "exploded", "shattered", "collapsed", "crumbled", "destroyed", "burned", "burst", "imploded", "attacked", "fought", "clashed", "struck", "fired", "dodged", "defended", "lunged", "stabbed", "punched", "ran", "sprinted", "chased", "pursued", "fled", "escaped", "rushed", "hurried", "changed", "transformed", "turned into", "mutated", "evolved", "shifted", "morphed", "storm began", "thunder roared", "rain poured", "snow started", "earthquake struck", "tsunami hit",
    # Time Progression
    "later", "suddenly", "moments later", "hours passed", "days later", "weeks after", "years passed", "meanwhile", "at that moment", "just then", "in the meantime", "all of a sudden", "unexpectedly", "spring arrived", "summer heat", "autumn leaves", "winter snow", "the sun set", "dawn broke",
    # Emotional or Psychological Shifts
    "stunned", "gasped", "shocked", "heart pounded", "eyes widened", "breath caught", "silence fell", "the air grew tense", "an eerie feeling", "something felt wrong", "an uneasy silence", "finally", "at last", "relief washed over", "everything settled", "quiet returned",
    # Narrative Markers & Dialogue Cues
    "later that day", "a few hours passed", "as the sun rose", "by nightfall", "the next morning",
    # Mystery & Suspense Triggers
    "uncovered", "discovered", "revealed", "found", "noticed", "realized", "figured out", "stumbled upon", "a shadow moved", "the air felt strange", "something wasn't right", "an ominous presence"
])

# Helper functions
def extract_active_characters(doc, character_list):
    active_characters = set()
    for token in doc:
        # If the token is a verb, find its subject
        if token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass") and child.text in character_list:
                    active_characters.add(child.text)
                elif child.dep_ == "pobj" and token.lemma_ in ("fight", "run", "help", "join", "defend"):
                    active_characters.add(child.text)
    return active_characters

class InfoExtractor:
    def __init__(self, input_path=None, config_path=None, save_path=None, model_manager:myapi.ModelManager=None, logger=None):
        self.input_path = input_path
        self.save_path = save_path
        self.model_manager = model_manager
        self.DC = []
        self.segments = None
        self.logger = logger

        if config_path:
            with open(config_path, 'r') as file:
                self.config = json.load(file)

        if save_path:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)

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
        elif type == "identifyCharsInP":
            prompt = prompt.replace("![paragraph]", input["paragraph"])
        else:
            prompt = prompt.replace("![descriptions]", input["descriptions"])
            prompt = prompt.replace("![segment]", input["segment"])
            prompt = prompt.replace("![initial]", input["initial"])

        return prompt

    def parse_info(self, response, whole=False):
        if not whole:
            # Original single-dictionary parsing logic
            parsed_data = {
                "characters": {},
                "scene": ""
            }

            character_pattern = r'\[(.*?)\]\((.*?)\)'
            characters = re.findall(character_pattern, response)

            for name, description in characters:
                parsed_data["characters"][name] = description.strip()

            # Remove misparsed scene
            parsed_data["characters"].pop('scene', None)
            parsed_data["characters"].pop('Scene', None)

            # Scene extraction
            scene_pattern = r'(?i)\[scene\]\((.*?)\)'
            scene_match = re.search(scene_pattern, response)
            if scene_match:
                parsed_data["scene"] = scene_match.group(1).strip()

            return parsed_data

        else:
            # Multi-version parsing logic
            timeline = []
            current_entry = None

            lines = response.splitlines()
            header_pattern = r'^DC version (\d+) from segment (\d+) to (None|\d+):$'
            character_pattern = r'\[(.*?)\]\((.*?)\)'

            for line in lines:
                line = line.strip()

                # Detect new dictionary version header
                header_match = re.match(header_pattern, line)
                if header_match:
                    if current_entry:
                        timeline.append(current_entry)

                    version = int(header_match.group(1))
                    start_segment = int(header_match.group(2))
                    end_segment_raw = header_match.group(3)
                    end_segment = int(end_segment_raw) if end_segment_raw != "None" else None

                    current_entry = {
                        "version": version,
                        "start_segment": start_segment,
                        "end_segment": end_segment,
                        "dc": {
                            "characters": {},
                            "scene": ""
                        }
                    }

                # Parse character and scene lines
                elif current_entry:
                    char_scene_match = re.match(character_pattern, line)
                    if char_scene_match:
                        name = char_scene_match.group(1).strip()
                        desc = char_scene_match.group(2).strip()
                        if name.lower() == "scene":
                            current_entry["dc"]["scene"] = desc
                        else:
                            current_entry["dc"]["characters"][name] = desc

            # Don't forget to append the last one
            if current_entry:
                timeline.append(current_entry)

            print(f"Parsed {len(timeline)} DC versions.")
            return timeline

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
    
    def get_dict_version(self, segment):
        for i in range(len(self.DC)):
            if self.DC[i]['start_segment'] <= segment and (self.DC[i]['end_segment'] is None or self.DC[i]['end_segment'] > segment):
                return i
        return -1

    def info_to_string(self, segment):
        result = []

        # Save all versions of the DC
        if segment == -1:
            for i in range(len(self.DC)):
                result.append(f"DC version {i} from segment {self.DC[i]['start_segment']} to {self.DC[i]['end_segment']}:\n")
                last_dict = self.DC[i]['dc']
                if "characters" in last_dict:
                    for character, description in last_dict["characters"].items():
                        result.append(f"[{character}]({description})")

                if "scene" in last_dict:
                    result.append(f"[Scene]({last_dict['scene']})")
                result.append("\n")
            return "\n".join(result)

        dict_version = self.get_dict_version(segment)
        last_dict = self.DC[dict_version]['dc']
        if "characters" in last_dict:
            for character, description in last_dict["characters"].items():
                result.append(f"[{character}]({description})")

        if "scene" in last_dict:
            result.append(f"[Scene]({last_dict['scene']})")

        return "\n".join(result)

    # DC_to_descriptions
    def format_info(self, segment):
        result = []

        dict_version = self.get_dict_version(segment)
        last_dict = self.DC[dict_version]['dc']
        if "characters" in last_dict:
            for character, description in last_dict["characters"].items():
                result.append(f'{character}:"{description}"')

        if "scene" in last_dict:
            description = last_dict['scene']
            result.append(f'scene:"{description}"')

        return "\n".join(result)
    
    def segments_to_string(self):
        result = []

        if self.segments:
            for segment in self.segments:
                result.append(f"[{segment['fragment']}]\n[{'YES' if segment['appearance_change'] else 'NO'}-{'YES' if segment['scene_change'] else 'NO'}]\n({segment['prompt']})")

        return "\n".join(result)

    def get_characteristics(self, regenerate_always=False):
        with open(self.input_path, 'r', encoding='utf-8') as file:
            story = file.read()

        input_dict = {'narrative': story}
        system_prompt = "You are a story analyzer."

        if os.path.exists(self.save_path + '/dictionary.txt') and not regenerate_always:
            if self.logger:
                self.logger.log('Using buffered DC')
            raw_DC = Path(self.save_path + '/dictionary.txt').read_text(encoding='utf-8')
            self.DC = self.parse_info(raw_DC, whole=True)
        else:
            if self.logger:
                self.logger.log('Creating DC .....')
            createDC_prompt = self.format_prompt('createDC', input_dict)
            
            raw_DC = self.model_manager.text_query(createDC_prompt, system_prompt)

            with open(self.save_path + '/dictionary.txt', 'w', encoding='utf-8') as f:
                f.write(raw_DC)

            self.append_dc(self.parse_info(raw_DC, whole=False), 0)

    def process_changes(self):
        for i in range(len(self.segments)):
            segment = self.segments[i]
            if segment['appearance_change'] or segment['scene_change']:
                if self.logger:
                    self.logger.log(f'Segment {i} has changes')
                input_dict = {
                    'descriptions': self.format_info(i),
                    'segment': segment['fragment']
                }

                update_prompt = self.format_prompt('updateDC', input_dict)
                updatedDC_raw = self.model_manager.text_query(update_prompt, 'You are a story analyzer.')

                updatedDC = self.parse_info(updatedDC_raw)
                temp = self.DC[-1]['dc']
                if "characters" in updatedDC:
                    for character, description in updatedDC["characters"].items():
                        if len(description) > 0:
                            temp["characters"][character] = description
                if "scene" in updatedDC and len(updatedDC["scene"]) > 0:
                    temp["scene"] = updatedDC["scene"]

                self.append_dc(temp, i)
    
    def append_dc(self, newdc, start):
        self.DC.append({
            'start_segment': start,
            'end_segment': None,
            'dc': newdc
        })
        if len(self.DC) > 1:
            self.DC[-2]['end_segment'] = start

        self.logger.log(f'Appending DC: {newdc} at segment {start}')

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
            if self.logger:
                self.logger.log('line: ' + line)
            current.append(line)
            total += len(line)
            if self.logger:
                self.logger.log('total:' + str(total))
            if total >= (context_length/100)*total_length:
                parts.append('\n'.join(current))
                current = []
                total = 0
        if current:
            parts.append('\n'.join(current))
        if self.logger:
            self.logger.log(f'split into {len(parts)} parts')
                    
        context = "This is the beginning of the story so there is no context to provide yet."
        all_segments = None
        for part in parts:
            input_dict = {'narrative': part, 'context': context}
            system_prompt = "You are a story analyzer."
                
            segments_prompt = self.format_prompt('segment', input_dict)

            raw_segments = self.model_manager.text_query(segments_prompt, system_prompt)
            if self.logger:
                self.logger.log(f'LLM input: {segments_prompt}')
                self.logger.log(raw_segments)
            all_segments, context = self.append_segments(raw_segments, all_segments)

        
        return all_segments

    def format_segments_out(self, segments):
        result = ''
        for segment in segments:
            result += f"[{segment}][YES-YES](An incredibly hard and dangerous stick made of hot meat about to bust)"
        return result

    # Algorithm idea:
    # Step 1: Separate the story in sentences based on points
    # Step 2: Initialize feature dictionary (characters, location, keywords)
    # Step 3: For each sentence:
        # Step 3.1: Look for characers by finding substrings in the sentence (use the DC), use keywords to consider only characters that interact in the current scene.
        # Step 3.2: User the NER model to find locations
        # Step 3.3: Check for special keywords that indicate a scene change. Use keywods for characters as well.
        # Step 3.4: Based on the information retrieved, decide if the sentence starts a new segment or not.
    def custom_segment(self):
        with open(self.input_path, 'r', encoding='utf-8') as file:
            story = file.read()

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(story)
        sentences = doc.sents
        current_info = {'characters': set(), 'location': set(), 'keywords': set()}
        char_list = [char for char in self.DC['characters'].keys()] # not updated
        char_list = char_list + self.config['check_subjects']
        current_segment = ''
        segments = []
        for sentence in sentences:
            self.logger.log(f'sentence: {sentence.text}')
            last_info = copy.deepcopy(current_info)
            # Step 3.1
            active_characters = extract_active_characters(sentence, char_list)
            current_info['characters'] = active_characters

            # Step 3.2
            entities = self.model_manager.recognize_entities(sentence.text)
            current_info['location'].update(entities['places'])

            # Step 3.3
            words = set(sentence.text.lower().split())
            current_info["keywords"] = words.intersection(trigger_keywords)

            # Step 3.4
            char_diff = current_info["characters"]-last_info["characters"]
            loc_diff = current_info["location"]-last_info["location"]
            if char_diff or loc_diff or current_info["keywords"]:
                if current_segment:
                    segments.append(current_segment)
                current_segment = sentence.text
            else:
                current_segment += sentence.text
        if current_segment:
            segments.append(current_segment)

        return self.format_segments_out(segments)

    def segment_story(self, segment_method=llm_part_segment, regenerate_always=False):

        if os.path.exists(self.save_path + '/segments.txt') and not regenerate_always:
            if self.logger:
                self.logger.log('Using buffered segments')
            all_segments = Path(self.save_path + '/segments.txt').read_text(encoding='utf-8')
        else:
            if self.logger:
                self.logger.log('Segmenting story .....')
            all_segments = segment_method()

        self.segments = self.parse_segment(all_segments)
        return self.segments

    def save_all(self):
        with open(self.save_path + '/dictionary.txt', 'w', encoding='utf-8') as file:
            file.write(self.info_to_string(-1))

        with open(self.save_path + '/segments.txt', 'w', encoding='utf-8') as file:
            file.write(self.segments_to_string())
