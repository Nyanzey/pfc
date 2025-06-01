from openai import OpenAI
import torch
from transformers import pipeline
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from diffusers import FluxPipeline, StableDiffusionPipeline, DiffusionPipeline, DDIMScheduler
from PIL import Image
from io import BytesIO
import requests
import base64

# Available text models: gpt, llama
# Available image models: dall-e, stable diffusion
class ModelManager:
    def __init__(self, config_path=None, logger=None):
        self.logger = logger
        with open(config_path, 'r') as file:
            self.config = json.load(file)
            self.text_model_pipe = None
            self.api_text_key = None
            self.api_image_key = None
            self.openai_client = None
            self.ner_tokenizer = None
            self.ner_model = None
            self.diffusion_api_link = None
        self.setup_models()

    def setup_models(self):

        # Getting API keys
        if self.config['Text-to-Text']['requires_key']:
            self.api_text_key = input(f"An API key is required for {self.config['Text-to-Text']['model']} by {self.config['Text-to-Text']['source']}: ")

        if self.config['Text-to-Image']['requires_key']:
            self.api_image_key = input(f"An API key is required for {self.config['Text-to-Image']['model']} by {self.config['Text-to-Image']['source']}: ")

        # Setting text models
        if self.config['Text-to-Text']['source'] == 'meta':
            model_id = self.config['Text-to-Text']['model']
            self.text_model_pipe = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=self.api_text_key,
            )
        if self.config['Text-to-Text']['source'] == 'deepseek':
            model_id = self.config['Text-to-Text']['model']
            self.text_model_pipe = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=self.api_text_key,
            )
        elif self.config['Text-to-Text']['source'] == 'openai':
            self.openai_client = OpenAI(api_key=self.api_text_key)

        # Setting image models
        if self.config['Text-to-Image']['source'] == 'openai' and not self.openai_client:
            self.openai_client = OpenAI(api_key=self.api_image_key)
        
        if self.config['Text-to-Image']['source'] == 'diffusion':
            self.set_diffusion_pipeline(self.config['Text-to-Image']['model'])

        if self.config['Text-to-Image']['source'] == 'diffusion-api':
            self.diffusion_api_link = self.config['Text-to-Image']['diffusion_api_link']
            if not self.diffusion_api_link:
                raise Exception("Diffusion API link is not set in the configuration")
        
        # Setting NER model
        if self.config['Segment Method'] == "feature-based":
            self.ner_tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
            self.ner_model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")

    def set_diffusion_pipeline(self, model):
        if model == 'black-forest-labs/FLUX.1-schnell':
            self.image_pipe = FluxPipeline.from_pretrained(
                model, 
                torch_dtype=torch.bfloat16, 
                token=self.api_image_key
            )
            self.image_pipe.enable_sequential_cpu_offload()
            self.image_model = "flux"
        elif model == "sd-legacy/stable-diffusion-v1-5":
            self.image_pipe = StableDiffusionPipeline.from_pretrained(
                model,
                torch_dtype=torch.float32
            ).to("cuda")
            self.image_pipe.scheduler = DDIMScheduler.from_config(self.image_pipe.scheduler.config)
            self.image_model = "sd15"
        elif model == "stabilityai/stable-diffusion-xl-base-1.0":
            self.image_pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                torch_dtype=torch.float16, 
                use_safetensors=True, 
                variant="fp16"
            ).to("cuda")
            self.image_model = "sdxl"
        else:
            # In case of invalid model, stop execution
            raise Exception("Invalid model for stable diffusion")


    # Not in config
    def recognize_entities(self, text):
        nlp = pipeline("ner", model=self.ner_model, tokenizer=self.ner_tokenizer, device=0)
        example = text
        entities = nlp(example)

        people = set()
        places = set()
        misc = set()
        orgs = set()
        for i in range(len(entities)):
            if entities[i]['entity'] == 'B-PER':
                person = entities[i]['word']
                while i+1 < len(entities) and entities[i+1]['entity'] == 'I-PER':
                    person += " " + entities[i+1]['word']
                    i += 1
                people.add(person)
            elif entities[i]['entity'] == 'B-LOC':
                place = entities[i]['word']
                while i+1 < len(entities) and entities[i+1]['entity'] == 'I-LOC':
                    place += " " + entities[i+1]['word']
                    i += 1
                places.add(place)
            elif entities[i]['entity'] == 'B-MISC':
                misci = entities[i]['word']
                while i+1 < len(entities) and entities[i+1]['entity'] == 'I-MISC':
                    misci += " " + entities[i+1]['word']
                    i += 1
                misc.add(misci)
            else:
                org = entities[i]['word']
                while i+1 < len(entities) and entities[i+1]['entity'] == 'I-ORG':
                    org += " " + entities[i+1]['word']
                    i += 1
                orgs.add(org)
        return {"people": people, "places": places, "misc": misc, "orgs": orgs}
    
    def query_text_openai(self, model, user_prompt, system_prompt):
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
        
        completion = self.openai_client.chat.completions.create(
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
        response = completion.choices[0].message.content
        self.logger.log(f'Text generated by openai: {response}')
        return response

    def query_image_openai(self, prompt, model, size="256x256"):
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
        
        response = self.openai_client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size=size,
            response_format="url"
        )
        self.logger.log(f'Image url generated by openai: {response.data[0].url}')
        return response.data[0].url

    def query_text_llama(self, user_prompt, system_prompt, max_new_tokens=2048):
        if not self.text_model_pipe:
            raise Exception("Huggingface pipeline not initialized for llama")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        outputs = self.text_model_pipe(
            messages,
            max_new_tokens=max_new_tokens,
        )

        response = outputs[0]["generated_text"][-1]['content']
        self.logger.log(f'Text generated by llama: {response}')
        return response
    
    def query_text_deepseek(self, user_prompt, system_prompt, max_new_tokens=2048):
        if not self.text_model_pipe:
            raise Exception("Huggingface pipeline not initialized for deepseek")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        outputs = self.text_model_pipe(
            messages,
            max_new_tokens=max_new_tokens,
        )

        response = outputs[0]["generated_text"][-1]['content']
        self.logger.log(f'Text generated by deepseek: {response}')
        return response

    def summarize_text(self, text, max_length=100, min_length=30):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device="cuda")

        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        out = summary[0]['summary_text']
        return out

    def query_image_sd(self, prompt):     
        if self.config['Text-to-Image']['Summarize Prompt']:
            prompt = self.summarize_text(prompt)
            self.logger.log(f'Summarized Prompt: {prompt}')
        # this is stable diffusion 1.5, faster but quality is not good (summarizing recommended, so clip can encode the prompt without missing to much information)
        if self.image_model in ["sd15", "sdxl"]:	 
            image = self.image_pipe(prompt, num_inference_steps=10).images[0]
            return image
        # flux is really heavy and takes a while to run (this is the schnell version, probably more testing to do)
        elif self.image_model == "flux":
            image = self.image_pipe(
                prompt,
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]
            return image
        else:
            print("Invalid model for stable diffusion")
            image = Image.open('./black.png')
            return image
        
    def query_image_diffusion_api(self, prompt):
        if not self.diffusion_api_link:
            raise Exception("Diffusion API link is not set in the configuration")
        
        res = requests.post(self.diffusion_api_link, json={"prompt": prompt})

        data = res.json()

        if "image_base64" in data:
            image_data = base64.b64decode(data["image_base64"])
            image = Image.open(BytesIO(image_data))
            return image
        else:
            print("Error:", data)

        return None

    def text_query(self, user_prompt, system_prompt):
        if self.config['Text-to-Text']['source'] == 'openai':
            return self.query_text_openai(self.config['Text-to-Text']['model'], user_prompt, system_prompt)
        elif self.config['Text-to-Text']['source'] == 'meta':
            return self.query_text_llama(user_prompt, system_prompt)
        elif self.config['Text-to-Text']['source'] == 'deepseek':
            return self.query_text_deepseek(user_prompt, system_prompt)
        return None
    
    def save_image_from_url(self, image_url, save_path, img_format):
        image_data = requests.get(image_url).content
        image = Image.open(BytesIO(image_data))
        image.save(save_path, format=img_format)

    def image_query(self, prompt, save_path, img_format):
        if self.config['Text-to-Image']['source'] == 'openai':
            img_url = self.query_image_openai(prompt, self.config['Text-to-Image']['model'])
            self.save_image_from_url(img_url, save_path, img_format)
            return True
        elif self.config['Text-to-Image']['source'] == 'diffusion':
            print(f'Querying image in sd with prompt: {prompt}')
            img = self.query_image_sd(prompt)
            img.save(save_path, format=img_format)
            return True
        elif self.config['Text-to-Image']['source'] == 'diffusion-api':
            print(f'Querying image in diffusion API with prompt: {prompt}')
            img = self.query_image_diffusion_api(prompt)
            img.save(save_path, format=img_format)
            return True
        return False
