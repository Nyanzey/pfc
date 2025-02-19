from openai import OpenAI
import torch
from transformers import pipeline
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Available text models: gpt, llama
# Available image models: dall-e, stable diffusion
class ModelManager:
    def __init__(self, config_path=None):
        with open(config_path, 'r') as file:
            self.config = json.load(file)
            self.text_model_pipe = None
            self.api_text_key = None
            self.api_image_key = None
            self.openai_client = None
            self.ner_tokenizer = None
            self.ner_model = None
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

        # Setting NER model
        if self.config['Segment Method'] == "feature-based":
            self.ner_tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
            self.ner_model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")

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
        return completion.choices[0].message.content

    def query_image_openai(self, prompt, model, size="1024x1024"):
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
        
        response = self.openai_client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size=size,
            response_format="url"
        )
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

        return outputs[0]["generated_text"][-1]['content']
    
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

        return outputs[0]["generated_text"][-1]['content']

    def query_image_sd(self, prompt, model):
        # TODO: Implement stable diffusion image generation
        return 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Osu%21_2024.png/800px-Osu%21_2024.png'

    def text_query(self, user_prompt, system_prompt):
        if self.config['Text-to-Text']['source'] == 'openai':
            return self.query_text_openai(self.config['Text-to-Text']['model'], user_prompt, system_prompt)
        elif self.config['Text-to-Text']['source'] == 'meta':
            return self.query_text_llama(user_prompt, system_prompt)
        elif self.config['Text-to-Text']['source'] == 'deepseek':
            return self.query_text_deepseek(user_prompt, system_prompt)
        return None
    
    def image_query(self, prompt):
        if self.config['Text-to-Image']['source'] == 'openai':
            return self.query_image_openai(prompt, self.config['Text-to-Image']['model'])
        elif self.config['Text-to-Image']['source'] == 'sd':
            return self.query_image_sd(prompt, self.config['Text-to-Image']['model'])
        return None
