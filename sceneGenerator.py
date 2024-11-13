import infoExtractor as IE
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import myapi
import json

# Step 3
def get_image_prompt(DC, segment, id, update=True):
    if update:
        if segment['appearance_change'] or segment['scene_change']:
           print('Updating ....')
           input_dict = {
               'descriptions': IE.DC_to_descriptions(DC),
               'segment': segment['fragment']
           }

           update_prompt = IE.get_txt_prompt('updateDC', input_dict)
           updatedDC_raw = myapi.query_openai_api('gpt-4o', update_prompt, 'You are a story analyzer.')
           print(updatedDC_raw)

           #updatedDC_raw = Path("./dynamicPrompts/updatedDict.txt").read_text() # replace with response from query

           updatedDC = IE.parse_DC(updatedDC_raw)
           for name, description in updatedDC['characters'].items():
               DC['characters'][name] = description
           if 'scene' in updatedDC:
               DC['scene'] = updatedDC['scene']
    
    input_dict = {
        'descriptions': IE.DC_to_descriptions(DC),
        'segment': segment['fragment'],
        'initial': segment['prompt']
    }

    with open(f'./dynamicPrompts/img_{id}_gen.json', 'w') as json_file:
        json.dump(input_dict, json_file, indent=4)

    compose_prompt = IE.get_txt_prompt('finalP', input_dict)
    compose_raw = myapi.query_openai_api('gpt-4o', compose_prompt, 'You are a story analyzer.')
    image_prompt = IE.parse_final_prompt(compose_raw)
    return DC, image_prompt

def generate_image(prompt, save_path, img_format, DC, segment, id, threshold=0.5, max_generations=0):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_url = myapi.query_image_api(prompt, 'dall-e-2')
    print(f'image: {image_url}') # just in case something bad happens
    save_image(image_url, save_path, img_format)

    image = Image.open(save_path)
    print(f'validating {save_path} .....')
    sim = get_similarity(image, prompt, model, processor)
    print(sim)
    generations = 0
    while sim < threshold and generations < max_generations:
        print("regenerating ...")
        dc, prompt = get_image_prompt(DC, segment, id, update=False)

        #image_url = myapi.query_image_api(prompt, 'dall-e-2')
        save_image(image_url, save_path, img_format)

        image = Image.open(save_path)
        
        sim = get_similarity(image, prompt, model, processor)
        print(sim)
        generations += 1
    return prompt

def save_image(image_url, save_path, img_format):
    image_data = requests.get(image_url).content
    image = Image.open(BytesIO(image_data))
    image.save(save_path, format=img_format)

def get_similarity(image, prompt, model, processor):
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    image_embeddings = outputs.image_embeds
    text_embeddings = outputs.text_embeds

    # Normalize the embeddings
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity
    similarity = (image_embeddings @ text_embeddings.T).squeeze().item()
    return similarity
