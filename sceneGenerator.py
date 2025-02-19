import infoExtractor as IE
import requests
from PIL import Image
from io import BytesIO
import myapi
import json
import os
from torch.nn.functional import cosine_similarity
import torch

class SceneGenerator:
    def __init__(self, config_path=None, save_path=None, output_image_path=None, info_extractor:IE.InfoExtractor=None, image_captioning_model=None, image_captioning_processor=None, sentence_model=None, model_manager:myapi.ModelManager=None, logger=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.info_extractor = info_extractor
        self.image_captioning_model = image_captioning_model.to(self.device)
        self.image_captioning_processor = image_captioning_processor
        self.sentence_model = sentence_model
        self.prompts = {}
        self.save_path = save_path
        self.model_manager = model_manager
        self.output_image_path = output_image_path
        self.logger = logger

        if config_path:
            with open(config_path, 'r') as file:
                self.config = json.load(file)

        # Not checking save_path because it is created in infoExtractor.py
        if not os.path.exists(self.output_image_path):
            os.mkdir(self.output_image_path)

    def get_image_prompt(self, segment, id, update_dc=True):
        if update_dc:
            if segment['appearance_change'] or segment['scene_change']:
                if self.logger:
                    self.logger.log('Updating ....')
                input_dict = {
                'descriptions': self.info_extractor.format_info(),
                'segment': segment['fragment']
                }

                update_prompt = self.info_extractor.format_prompt('updateDC', input_dict)
                updatedDC_raw = self.model_manager.text_query(update_prompt, 'You are a story analyzer.')

                updatedDC = self.info_extractor.parse_info(updatedDC_raw)
                for name, description in updatedDC['characters'].items():
                    if len(description) > 5:
                        self.info_extractor.DC['characters'][name] = description
                if 'scene' in updatedDC and len(updatedDC['scene']) > 5:
                    self.info_extractor.DC['scene'] = updatedDC['scene']
        
        input_dict = {
            'descriptions': self.info_extractor.format_info(),
            'segment': segment['fragment'],
            'initial': segment['prompt']
        }

        with open(f'{self.save_path}/img_{id}_gen.json', 'w') as json_file:
            json.dump(input_dict, json_file, indent=4)

        compose_prompt = self.info_extractor.format_prompt('finalP', input_dict)
        compose_raw = self.model_manager.text_query(compose_prompt, 'You are a story analyzer.')
        image_prompt = self.info_extractor.parse_final_prompt(compose_raw)
        return image_prompt

    def generate_image(self, prompt, save_path, img_format, segment, id, threshold=0.7, max_generations=1):
        image_url = self.model_manager.image_query(prompt)
        if self.logger:
            self.logger.log(f'Image url: {image_url}')

        self.save_image(image_url, save_path, img_format)
        image = Image.open(save_path)

        if self.logger:
            self.logger.log(f'validating {save_path} .....')
        
        sim = self.get_similarity(image, prompt, self.image_captioning_model, self.image_captioning_processor)
        if self.logger:
            self.logger.log(f'Similarity: {sim}')

        generations = 0
        while sim < threshold and generations < max_generations:
            if self.logger:
                self.logger.log("Regenerating ...")
            prompt = self.get_image_prompt(segment, id, update_dc=False)

            image_url = self.model_manager.image_query(prompt)
            self.save_image(image_url, save_path, img_format)

            image = Image.open(save_path)
            
            sim = self.get_similarity(image, prompt, self.image_captioning_model, self.image_captioning_processor)
            if self.logger:
                self.logger.log(f'Similarity: {sim}')
            generations += 1
        return prompt

    def save_image(self, image_url, save_path, img_format):
        image_data = requests.get(image_url).content
        image = Image.open(BytesIO(image_data))
        image.save(save_path, format=img_format)

    def get_similarity(self, image, prompt, model, processor):

        inputs = processor(image, return_tensors="pt").to(self.device)

        out = model.generate(**inputs)
        predicted = processor.decode(out[0], skip_special_tokens=True)

        embedding1 = self.sentence_model.encode(prompt, convert_to_tensor=True)
        embedding2 = self.sentence_model.encode(predicted, convert_to_tensor=True)

        cos_sim = cosine_similarity(embedding1, embedding2, dim=-1)

        return cos_sim.item()
    
    def generate_scenes(self, update_dc=True, img_format='png', similarity_threshold=0.7, max_generations=1):
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

            prompt = self.get_image_prompt(self.info_extractor.segments[i], i, update_dc)
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
        
        self.prompts['final'] = image_prompts
        self.prompts['buffer'] = buffer_prompts

    def save_prompts(self):
        with open(self.save_path + '/final_prompts.txt', 'w') as file:
            file.writelines(s + '\n' for s in self.prompts['final'])

        with open(self.save_path + '/buffer_prompts.txt', 'w') as file:
            file.writelines(s + '\n' for s in self.prompts['buffer'])
