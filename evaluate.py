import torch
import clip
from sentence_transformers import SentenceTransformer, util
from brisque import BRISQUE
from jiwer import wer
import numpy as np
from PIL import Image
from pathlib import Path
import infoExtractor as IE
import sceneGenerator as SG
from transformers import CLIPProcessor, CLIPModel

# Configuración de pesos
alpha, beta, gamma = 0.4, 0.3, 0.3  # Ajustables según experimentos

# Modelos
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo transformer para VAC
brisque = BRISQUE()  # Métrica IQS

# Función para calcular VAC
def calculate_vac(story_segments, prompts, images, audios):
    # Cálculo de Cpi (similitud entre prompt e imagen)
    Cpi = []
    for prompt, image in zip(prompts, images):
        Cpi.append(SG.get_similarity(image, prompt, clip_model, clip_processor))

    # Cálculo de Csa (similitud entre segmento de texto y audio)
    Csa = []
    total_length = sum(len(seg) for seg in story_segments)
    for seg, audio_text in zip(story_segments, audios):
        seg_embedding = sentence_model.encode(seg, convert_to_tensor=True)
        audio_embedding = sentence_model.encode(audio_text, convert_to_tensor=True)
        similarity = util.cos_sim(seg_embedding, audio_embedding).item()
        weight = len(seg) / total_length
        Csa.append(weight * similarity)

    print(Cpi)
    print(Csa)

    # Cálculo de VAC
    VAC = ((sum(Cpi) / (len(prompts)) + sum(Csa)) / 2)
    return VAC

# Función para calcular IQS
def calculate_iqs(images):
    scores = []
    for image in images:
        scores.append(brisque.get_score(np.asarray(image)))
    print(scores)
    IQS = 1 - (sum(scores) / (len(scores) * 100))
    return IQS

# Función para calcular AQS
def calculate_aqs(story_segments, audios):
    scores = [wer(seg, audio) for seg, audio in zip(story_segments, audios)]
    AQS = 1 - (sum(scores) / len(scores))
    return AQS

# Función para calcular la métrica compuesta CM
def calculate_cm(story_segments, prompts, images, audios):
    # Calcular cada métrica
    print('Calculating VAC ...')
    VAC = calculate_vac(story_segments, prompts, images, audios)
    print('Calculating IQS ...')
    IQS = calculate_iqs(images)
    print('Calculating AQS ...')
    AQS = calculate_aqs(story_segments, audios)
    print(f'VAC:{VAC}')
    print(f'IQS:{IQS}')
    print(f'AQS:{AQS}')

    # Métrica compuesta
    CM = alpha * VAC + beta * IQS + gamma * AQS
    return CM


