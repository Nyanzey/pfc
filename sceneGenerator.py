import infoExtractor as IE
import requests
from PIL import Image
from io import BytesIO
import myapi
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

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

           updatedDC = IE.parse_DC(updatedDC_raw)
           for name, description in updatedDC['characters'].items():
                if len(description) > 5:
                    DC['characters'][name] = description
           if 'scene' in updatedDC and len(updatedDC['scene']) > 5:
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

def generate_image(prompt, save_path, img_format, DC, segment, id, threshold=0.7, max_generations=1):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")  

    image_url = myapi.query_image_api(prompt, 'dall-e-3')
    print(f'Image url: {image_url}')

    save_image(image_url, save_path, img_format)
    image = Image.open(save_path)

    print(f'validating {save_path} .....')
    
    sim = get_similarity(image, prompt, model, processor)
    print(f'Similarity: {sim}')
    generations = 0
    while sim < threshold and generations < max_generations:
        print("Regenerating ...")
        dc, prompt = get_image_prompt(DC, segment, id, update=False)

        image_url = myapi.query_image_api(prompt, 'dall-e-3')
        save_image(image_url, save_path, img_format)

        image = Image.open(save_path)
        
        sim = get_similarity(image, prompt, model, processor)
        print(f'Similarity: {sim}')
        generations += 1
    return prompt

def save_image(image_url, save_path, img_format):
    image_data = requests.get(image_url).content
    image = Image.open(BytesIO(image_data))
    image.save(save_path, format=img_format)

def get_similarity(image, prompt, model, processor):
    stc_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

    inputs = processor(image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    predicted = processor.decode(out[0], skip_special_tokens=True)

    embedding1 = stc_model.encode(prompt, convert_to_tensor=True)
    embedding2 = stc_model.encode(predicted, convert_to_tensor=True)

    cos_sim = cosine_similarity(embedding1, embedding2, dim=-1)

    return cos_sim.item()
