import infoExtractor as IE
from PIL import Image
import myapi
import json
import os
from torch.nn.functional import cosine_similarity
import torch
import open_clip
from threading import Thread
from queue import Queue
import base64

class SceneGenerator:
    def __init__(self, config_path=None, save_path=None, output_image_path=None, info_extractor:IE.InfoExtractor=None, model_manager:myapi.ModelManager=None, logger=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.info_extractor = info_extractor # For segment and DC

        # For text-image similarity
        model_name = 'hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' # Was good during testing so not considered in config
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(model_name, cache_dir='F:\\modelscache', device=self.device)
        self.clip_model.eval() # Changing mode to evaluation, its train by default
        self.clip_tokenizer = open_clip.get_tokenizer(model_name)
        
        # Save paths
        self.save_path = save_path 
        self.output_image_path = output_image_path 

        self.model_manager = model_manager # For querying models        
        self.logger = logger # For logging
        self.prompts = {} # To store final and buffer prompts

        if config_path:
            with open(config_path, 'r') as file:
                self.config = json.load(file)

        # Not checking save_path because it is created in infoExtractor.py
        if not os.path.exists(self.output_image_path):
            os.mkdir(self.output_image_path)

    def get_image_prompt(self, segment, segment_index):
        input_dict = {
            'descriptions': self.info_extractor.format_info(segment_index),
            'segment': segment['fragment'],
            'initial': segment['prompt']
        }

        with open(f'{self.save_path}/img_{segment_index}_gen.json', 'w') as json_file:
            json.dump(input_dict, json_file, indent=4)

        compose_prompt = self.info_extractor.format_prompt('finalP', input_dict)
        compose_raw = self.model_manager.text_query(compose_prompt, 'You are a story analyzer.')
        image_prompt = self.info_extractor.parse_final_prompt(compose_raw.lower())
        if not image_prompt:
            image_prompt = segment['prompt']
        return image_prompt
    
    def get_image_refined_prompt(self, last_prompt, coherence, segment_index, image):
        min_char = ''
        min_coherence = 1.0
        for char, score in coherence.items():
            if score < min_coherence:
                min_coherence = score
                min_char = char
        
        if self.logger:
            self.logger.log(f'Character with lowest coherence: {min_char} ({min_coherence})')
        
        char_description = self.info_extractor.DC[self.info_extractor.get_dict_version(segment_index)]['dc']['characters'].get(min_char, '')
        
        input_dict = {
            'description': char_description,
            'prompt': last_prompt
        }

        compose_prompt = self.info_extractor.format_prompt('refineImage', input_dict)
        compose_raw = self.model_manager.text_query(compose_prompt, 'You are a story analyzer.', image)
        image_prompt = self.info_extractor.parse_final_prompt(compose_raw.lower())

        return image_prompt if image_prompt else last_prompt

    def generate_image(self, prompt, save_path, img_format, segment, segment_index, threshold=0.3, max_generations=0):
        self.model_manager.image_query(prompt, save_path, img_format)
        print(f'Image query done {save_path}')
        image = Image.open(save_path)

        if self.logger:
            self.logger.log(f'validating {save_path} .....')
        
        coherence = self.get_coherence(image, segment_index, prompt)
        if self.logger:
            self.logger.log(f'Avg Coherence: {coherence["avg"]}')
            self.logger.log(f'Min Coherence: {coherence["min"]}')

        generations = 0
        while coherence["min"] < threshold and generations < max_generations:
            if self.logger:
                self.logger.log("Regenerating ...")

            #prompt = "A dog walking in the park"  # Placeholder for testing
            with open(save_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            prompt = self.get_image_refined_prompt(prompt, coherence, segment_index, base64_image)

            # Generates and saves the image
            self.model_manager.image_query(prompt, save_path, img_format)

            image = Image.open(save_path)
            
            coherence = self.get_coherence(image, segment_index, prompt)
            if self.logger:
                self.logger.log(f'Similarity: {coherence["avg"]}')
            generations += 1
        return prompt

    def get_similarity(self, image_features, prompt):
            text = self.clip_tokenizer(prompt).to(self.device)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                text_features = self.clip_model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                sim = cosine_similarity(image_features, text_features).cpu().item()
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                return sim, text_probs
            
    def get_coherence(self, image, segment_index, prompt):
        scores = {}
        local_DC = self.info_extractor.DC[self.info_extractor.get_dict_version(segment_index)]['dc']

        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            image_features = self.clip_model.encode_image(image)
        
        for char, desc in local_DC['characters'].items():
            char_sim, _ = self.get_similarity(image_features, desc)
            if char_sim > 0.2:
                scores[char] = char_sim

        character_scores = scores.copy()
        
        scores['scene'], _ = self.get_similarity(image_features, local_DC['scene'])
        scores['prompt'], _ = self.get_similarity(image_features, prompt)
        scores['avg'] = sum(scores.values()) / len(scores)
        scores['min'] = min(character_scores.values())
        self.logger.log(f'Scores for segment {segment_index}: {scores}')
        return scores

    def _scene_worker(self, task_queue, result_prompts, buffer_prompts, img_format, similarity_threshold, max_generations):
        print("Trhead started")
        while True:
            item = task_queue.get()
            if item is None:
                print("Thread exiting because of sentinel value")
                break
            i, segment = item

            save_path = f'{self.output_image_path}/{str(i).zfill(3)}.{img_format}'
            if os.path.exists(save_path):
                print("Thread exiting because of existing file")
                task_queue.task_done()
                continue

            dummy_img = Image.open('./black.png')
            
            try:
                prompt = self.get_image_prompt(segment, i)
                #prompt = "A dog walking in the park"  # Placeholder for testing
                buffer_prompts[i] = prompt  # Use index to avoid race conditions
                if self.logger:
                    self.logger.log(f'[Segment-{i}] Generating: {prompt}')
                    print(f'[Segment-{i}] Generating: {prompt}')
                prompt = self.generate_image(prompt, save_path, img_format, segment, i, threshold=similarity_threshold, max_generations=max_generations)
                print(f'[Segment-{i}] Generated: {prompt}')
                result_prompts[i] = prompt
            except Exception as e:
                dummy_img.save(save_path, format=img_format)
                if self.logger:
                    self.logger.log(f'[Segment-{i}] Error: {e}')
            finally:
                print("Thread finished processing segment", i)
                task_queue.task_done()

    def generate_scenes(self, num_threads=4, img_format='png', similarity_threshold=0.3, max_generations=1):
        task_queue = Queue()
        num_segments = len(self.info_extractor.segments)

        # Shared prompt lists using index to preserve order
        result_prompts = [None] * num_segments
        buffer_prompts = [None] * num_segments

        # Enqueue tasks
        for i, segment in enumerate(self.info_extractor.segments):
            task_queue.put((i, segment))

        # Add sentinel values to stop workers
        for _ in range(num_threads):
            task_queue.put(None)

        threads = []
        for _ in range(num_threads):
            t = Thread(target=self._scene_worker, args=(task_queue, result_prompts, buffer_prompts, img_format, similarity_threshold, max_generations))
            t.start()
            threads.append(t)

        # Wait for all threads to finish
        print("Waiting for all threads to complete...")
        for t in threads:
            t.join()

        # Save final results
        self.logger.log(f'result prompts: {result_prompts}')
        self.logger.log(f'buffer prompts: {buffer_prompts}')
        self.prompts['final'] = [p for p in result_prompts if p]
        self.prompts['buffer'] = [p for p in buffer_prompts if p]

    def save_prompts(self):
        if self.prompts['final']:
            with open(self.save_path + '/final_prompts.txt', 'w', encoding='utf-8') as file:
                file.writelines(s + '\n' for s in self.prompts['final'])

        if self.prompts['buffer']:
            with open(self.save_path + '/buffer_prompts.txt', 'w', encoding='utf-8') as file:
                file.writelines(s + '\n' for s in self.prompts['buffer'])
