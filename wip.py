# File for testing and experimenting with stuff

import os

import torch.amp
os.environ['HF_HOME'] = 'F:\\modelscache'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers import FluxPipeline
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline as hf_pipeline
from infoExtractor import InfoExtractor
from myapi import ModelManager
import re
import spacy
from PIL import Image
import open_clip
from torch.nn.functional import cosine_similarity
import threading
from queue import Queue

device = "cuda" if torch.cuda.is_available() else "cpu"

def long_prompt(model, prompt, neg_prompt):
    pipe = DiffusionPipeline.from_pretrained(
        model,
        custom_pipeline="lpw_stable_diffusion",
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)

    pipe.text2img(prompt, negative_prompt=neg_prompt, width=512, height=512, max_embeddings_multiples=3).images[0].show()

def generate_sdxl(prompt, neg_prompt):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to(device)

    images = pipe(prompt=prompt, negative_prompt=neg_prompt).images[0].show()

# This is expensive as hell and will take a long time to run, best model tho competes with dall-e results
def generate_flux(prompt, neg_prompt):
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, token="")
    pipe.enable_sequential_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    image.save("flux-dev.png")

def get_paragraph_similarities(input):
    with open(input, 'r', encoding='utf-8') as file:
        input = file.read()
    paragraphs = input.split(".")
    paragraphs = [p for p in paragraphs if p != '']
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    similarities = []
    print(f'number of paragraphs: {len(paragraphs)}')
    print(paragraphs)
    for i in range(1, len(paragraphs)):
        embedding1 = model.encode(paragraphs[i-1], convert_to_tensor=True)
        embedding2 = model.encode(paragraphs[i], convert_to_tensor=True)
        sim = util.cos_sim(embedding1, embedding2).item()
        similarities.append(sim)
    return similarities

def ner(input):
    tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")

    nlp = hf_pipeline("ner", model=model, tokenizer=tokenizer, device=0)
    example = input

    ner_results = nlp(example)
    return ner_results

def extract_entities(entities):
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

def divide_paragraphs_by_entities(paragraphs):
    people = set()
    places = set()
    i = 0
    for paragraph in paragraphs:
        entities = ner(paragraph)
        extracted = extract_entities(entities)
        p, pl = extracted["people"], extracted["places"]

        diff = p - people
        if diff:
            print(f"Paragraph {i} has the following people: {diff} with the previous paragraph")
        diff = pl - places
        if diff:
            print(f"Paragraph {i} has the following places: {diff} with the previous paragraph")

        people.update(p)
        places.update(pl)
        i += 1

# Assuming deepseek
def divide_paragraphs_by_entities_llm(paragraphs):
    model_manager = ModelManager(config_path="./config.json")
    info_extractor = InfoExtractor(config_path="./config.json", model_manager=model_manager)
    i = 0
    for paragraph in paragraphs:
        input_dict = {"paragraph": paragraph}
        prompt = info_extractor.format_prompt("identifyCharsInP", input_dict)
        response = model_manager.query_text_deepseek("", prompt)
        characters = parse_response(response)
        print(f"Paragraph {i} has the following characters: {characters}")

def parse_response(response):
    # the format is: [character1, character2, ...]
    expr = r'\[(.*?)\]'
    characters = re.findall(expr, response)
    return characters

def extract_active_characters(doc, character_list):
    """Extract characters who are actively performing actions in the scene."""
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

def segment_scenes(text, character_list):
    """Segments text into scenes based on active character changes."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    scenes = []
    current_scene = []
    last_active_characters = set()
    
    for sent in doc.sents:
        active_characters = extract_active_characters(sent, character_list)
        
        # If there's a significant shift in active characters, start a new scene
        if active_characters and active_characters != last_active_characters:
            if current_scene:
                scenes.append(" ".join(current_scene))
                current_scene = []
            last_active_characters = active_characters
        
        current_scene.append(sent.text)
    
    # Add the last scene
    if current_scene:
        scenes.append(" ".join(current_scene))
    
    return scenes

def multiplicate_image(n, img_path, output_path):
    img = Image.open(img_path)
    for i in range(n):
        img.save(f'{output_path}/{str(i).zfill(3)}.png')

def summarize_text(text, max_length=100, min_length=30):
    summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn", device=device)

    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    out = summary[0]['summary_text']
    return out

def get_similarity(image_path, prompt, model_name):
  model, _, preprocess = open_clip.create_model_and_transforms(model_name, cache_dir='F:\\modelscache', device=device)
  model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
  tokenizer = open_clip.get_tokenizer(model_name)
  image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
  text = tokenizer(prompt).to(device)
  with torch.no_grad(), torch.amp.autocast('cuda'):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    sim = cosine_similarity(image_features, text_features).cpu().item()
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return sim, text_probs

# Load model once
pipe = StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5").to("cuda")

def worker(prompt_queue, result_queue):
    while True:
        prompt = prompt_queue.get()
        if prompt is None:
            break
        with torch.no_grad():
            image = pipe(prompt).images[0]
        result_queue.put(image)

# Setup
prompt_queue = Queue()
result_queue = Queue()
threads = [threading.Thread(target=worker, args=(prompt_queue, result_queue)) for _ in range(1)]

# Start threads
for t in threads:
    t.start()

# Add prompts
for prompt in ["a castle on a hill", "a futuristic city"]:
    prompt_queue.put(prompt)

# Shutdown
for _ in threads:
    prompt_queue.put(None)
for t in threads:
    t.join()
