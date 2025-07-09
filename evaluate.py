from sentence_transformers import SentenceTransformer, util
from brisque import BRISQUE
from jiwer import wer
import numpy as np
import sceneGenerator as SG
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class Evaluation:
    def __init__(self, alpha, beta, gamma, sceneGenerator, logger):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.set_up_models()
        self.SG = sceneGenerator
    
    def set_up_models(self): 
        self.sentence_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')  # Modelo transformer para VAC
        self.brisque = BRISQUE()
        
    def calculate_vac(self, story_segments, prompts, images, audios):
        Cpi = []
        i = 0
        self.logger.log('Processing images for VAC ...')
        for prompt, image in zip(prompts, images):
            coherence = self.SG.get_coherence(image, i, prompt)
            self.logger.log(f'Avg Coherence for image {i}: {coherence['avg']}')
            Cpi.append(coherence["avg"])
            i += 1

        Csa = []
        i = 0
        total_length = sum(len(seg) for seg in story_segments)
        self.logger.log('Processing audios for VAC ...')
        for seg, audio_text in zip(story_segments, audios):
            seg_embedding = self.sentence_model.encode(seg, convert_to_tensor=True)
            audio_embedding = self.sentence_model.encode(audio_text, convert_to_tensor=True)
            similarity = util.cos_sim(seg_embedding, audio_embedding).item()
            weight = len(seg) / total_length
            local_csa = weight * similarity
            Csa.append(local_csa)
            self.logger.log(f'Csa for audio {i}: {local_csa}')
            i += 1

        self.logger.log('Image coherence scores')
        self.logger.log(Cpi)
        self.logger.log('Audio coherence scores')
        self.logger.log(Csa)

        VAC = ((sum(Cpi) / (len(prompts)) + sum(Csa)) / 2)
        return VAC

    # Will be replaced with a composition of more metrics, it will be done in Matlab
    def calculate_iqs(self, images):
        scores = []
        for image in images:
            scores.append(self.brisque.get_score(np.asarray(image)))
        scores = [score if score else 0 for score in scores]
        IQS = 1 - (sum(scores) / (len(scores) * 100))
        return IQS

    def calculate_aqs(self, story_segments, audios):
        scores = [wer(seg, audio) for seg, audio in zip(story_segments, audios)]
        AQS = 1 - (sum(scores) / len(scores))
        return AQS

    def calculate_cm(self, story_segments, prompts, images, audios):
        self.logger.log('Calculating VAC ...')
        VAC = self.calculate_vac(story_segments, prompts, images, audios)
        self.logger.log('Calculating IQS ...')
        #IQS = self.calculate_iqs(images)
        IQS = 0.75 # IQS is calculated in MatLab
        self.logger.log('Calculating AQS ...')
        AQS = self.calculate_aqs(story_segments, audios)
        self.logger.log(f'VAC:{VAC}')
        self.logger.log(f'IQS:{IQS}')
        self.logger.log(f'AQS:{AQS}')

        CM = self.alpha * VAC + self.beta * IQS + self.gamma * AQS
        return CM