import infoExtractor as IE
from PIL import Image
import myapi
import json
import os
from torch.nn.functional import cosine_similarity
import torch
import open_clip

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

    def get_image_prompt(self, segment, segment_number):
        input_dict = {
            'descriptions': self.info_extractor.format_info(segment_number),
            'segment': segment['fragment'],
            'initial': segment['prompt']
        }

        with open(f'{self.save_path}/img_{id}_gen.json', 'w') as json_file:
            json.dump(input_dict, json_file, indent=4)

        compose_prompt = self.info_extractor.format_prompt('finalP', input_dict)
        compose_raw = self.model_manager.text_query(compose_prompt, 'You are a story analyzer.')
        image_prompt = self.info_extractor.parse_final_prompt(compose_raw.lower())
        if not image_prompt:
            image_prompt = segment['prompt']
        return image_prompt

    def generate_image(self, prompt, save_path, img_format, segment, id, threshold=0.7, max_generations=1):
        image_url = self.model_manager.image_query(prompt, save_path, img_format)

        image = Image.open(save_path)

        if self.logger:
            self.logger.log(f'validating {save_path} .....')
        
        sim, probs = self.get_similarity(image, prompt)
        if self.logger:
            self.logger.log(f'Similarity: {sim}')

        generations = 0
        while sim < threshold and generations < max_generations:
            if self.logger:
                self.logger.log("Regenerating ...")
            prompt = self.get_image_prompt(segment, id)

            image_url = self.model_manager.image_query(prompt, save_path, img_format)

            image = Image.open(save_path)
            
            sim, probs = self.get_similarity(image, prompt)
            if self.logger:
                self.logger.log(f'Similarity: {sim}')
            generations += 1
        return prompt

    def get_similarity(self, image, prompt):
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text = self.clip_tokenizer(prompt).to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            sim = cosine_similarity(image_features, text_features).cpu().item()
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            return sim, text_probs
    
    def generate_scenes(self, img_format='png', similarity_threshold=0.7, max_generations=1):
        image_prompts = []
        buffer_prompts = []

        for i in range(len(self.info_extractor.segments)):
            save_path = f'{self.output_image_path}/{str(i).zfill(3)}.{img_format}'
            if os.path.exists(save_path):
                continue

            if i == 0:
                dummy_img = Image.open('./black.png')
            else:
                dummy_img = Image.open(f'{self.output_image_path}/{str(i-1).zfill(3)}.{img_format}')

            prompt = self.get_image_prompt(self.info_extractor.segments[i], i)
            buffer_prompts.append(prompt)

            if self.logger:
                self.logger.log(f'Generating image {i}: {prompt}')
            try:
                prompt = self.generate_image(prompt, save_path, img_format, self.info_extractor.segments[i], i, threshold=similarity_threshold, max_generations=max_generations)
                image_prompts.append(prompt)
            except Exception as e:
                dummy_img.save(save_path, format=img_format)
                if self.logger:
                    self.logger.log(f"Error generating image for segment {i}: {e}")
                continue
        
        self.logger.log(f'prompts: {image_prompts}')
        self.logger.log(f'buffer prompts: {buffer_prompts}')
        self.prompts['final'] = image_prompts
        self.prompts['buffer'] = buffer_prompts

    def save_prompts(self):
        if self.prompts['final']:
            with open(self.save_path + '/final_prompts.txt', 'w') as file:
                file.writelines(s + '\n' for s in self.prompts['final'])

        if self.prompts['buffer']:
            with open(self.save_path + '/buffer_prompts.txt', 'w') as file:
                file.writelines(s + '\n' for s in self.prompts['buffer'])
