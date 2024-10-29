import infoExtractor as IE
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel

# Step 3
def get_image_prompt(DC, segment, update=True):
    if update:
        if segment['appearance_change'] or segment['scene_change']:
           input_dict = {
               'descriptions': IE.DC_to_descriptions(DC),
               'segment': segment['fragment']
           }
           update_prompt = IE.get_txt_prompt('updateDC', input_dict)
           # query_openai_api(update_prompt, ...)
           updatedDC_raw = Path("./dynamicPrompts/updatedDict.txt").read_text() # replace with response from query
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
    compose_prompt = IE.get_txt_prompt('finalP', input_dict)
    #image_prompt = query_openai_api(compose_prompt, ...)
    compose_prompt = 'final prompt: "In a rustic kitchen with a wooden table and brick fireplace, Little Red Riding Hood, a petite girl with wavy brown hair under a red velvet cap, listens attentively to her mother, a woman with dark brown hair in a bun, wearing a green cotton dress with a flour-smudged apron, as she hands her a piece of cake and a bottle of wine."' # for testing
    image_prompt = IE.parse_final_prompt(compose_prompt)
    return DC, image_prompt

def generate_image(prompt, save_path, img_format, DC, segment, threshold=0.5, max_generations=1):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_url = IE.query_image_api(prompt, 'dall-e3')
    save_image(image_url, save_path, img_format)
    image = Image.open(save_path)
    print(f'validating {save_path} .....')
    sim = get_similarity(image, prompt, model, processor)
    print(sim)
    generations = 0
    while sim < threshold and generations < max_generations:
        print("regenerating ...")

        image_url = IE.query_image_api(prompt, 'dall-e3')
        save_image(image_url, save_path, img_format)
        image = Image.open(save_path)
        dc, prompt = get_image_prompt(DC, segment, update=False)

        sim = get_similarity(image, prompt, model, processor)
        print(sim)
        generations += 1

def save_image(image_url, save_path, img_format):
    return
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
