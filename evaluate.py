from sentence_transformers import SentenceTransformer, util
from brisque import BRISQUE
from jiwer import wer
import numpy as np
import sceneGenerator as SG
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

alpha, beta, gamma = 0.4, 0.3, 0.3

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)  
sentence_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')  # Modelo transformer para VAC
brisque = BRISQUE()

def calculate_vac(story_segments, prompts, images, audios):
    Cpi = []
    for prompt, image in zip(prompts, images):
        Cpi.append(SG.get_similarity(image, prompt, model, processor))

    Csa = []
    total_length = sum(len(seg) for seg in story_segments)
    for seg, audio_text in zip(story_segments, audios):
        seg_embedding = sentence_model.encode(seg, convert_to_tensor=True)
        audio_embedding = sentence_model.encode(audio_text, convert_to_tensor=True)
        similarity = util.cos_sim(seg_embedding, audio_embedding).item()
        weight = len(seg) / total_length
        Csa.append(weight * similarity)

    print('Image coherence scores')
    print(Cpi)
    print('Audio coherence scores')
    print(Csa)

    VAC = ((sum(Cpi) / (len(prompts)) + sum(Csa)) / 2)
    return VAC

def calculate_iqs(images):
    scores = []
    for image in images:
        scores.append(brisque.get_score(np.asarray(image)))
    IQS = 1 - (sum(scores) / (len(scores) * 100))
    return IQS

def calculate_aqs(story_segments, audios):
    scores = [wer(seg, audio) for seg, audio in zip(story_segments, audios)]
    AQS = 1 - (sum(scores) / len(scores))
    return AQS

def calculate_cm(story_segments, prompts, images, audios):
    print('Calculating VAC ...')
    VAC = calculate_vac(story_segments, prompts, images, audios)
    print('Calculating IQS ...')
    IQS = calculate_iqs(images)
    print('Calculating AQS ...')
    AQS = calculate_aqs(story_segments, audios)
    print(f'VAC:{VAC}')
    print(f'IQS:{IQS}')
    print(f'AQS:{AQS}')

    CM = alpha * VAC + beta * IQS + gamma * AQS
    return CM
