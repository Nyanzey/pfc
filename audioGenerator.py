import torch
import numpy as np
from TTS.api import TTS
from jiwer import wer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class AudioGenerator:
    def __init__(self, model_paths=[], output_path='./audios'):
        self.model_paths = model_paths
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_path = output_path
        self.quality_matrix = None
        self.chosen_model = None

        modelASR_id = "openai/whisper-large-v3-turbo"
        self.modelASR = AutoModelForSpeechSeq2Seq.from_pretrained(modelASR_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True).to(self.device)
        self.processorASR = AutoProcessor.from_pretrained(modelASR_id)

        self.pipeASR = pipeline(
            "automatic-speech-recognition",
            model=self.modelASR,
            tokenizer=self.processorASR.tokenizer,
            feature_extractor=self.processorASR.feature_extractor,
            torch_dtype=torch.float16,
            device=self.device,
            return_timestamps=True
        )

    def generate_audio(self, model_path, text, output_path):
        tts = TTS(model_name=model_path, progress_bar=False).to(self.device)

        tts.tts_to_file(text=text, file_path=output_path)

    def transcribe_audio(self, audio_path):
        result = self.pipeASR(audio_path)
        transcribed_text = result["text"]
        return transcribed_text

    def evaluate_audio_quality(self, audio_path, reference_text):
        result = self.pipeASR(audio_path)
        transcribed_text = result["text"]

        quality_score = wer(reference_text, transcribed_text)
        
        return quality_score

    def select_best_tts_model(self, fragments):
        num_fragments = len(fragments)
        num_models = len(self.model_paths)
        
        print('Generating audios ...')
        audio_matrix = []
        for i, fragment in enumerate(fragments):
            audios_for_segment = []
            for model_path in self.model_paths:
                output_audio_path = f"{self.output_path}/{model_path.split('/')[-1]}/audio_fragment_{i}.wav"
                self.generate_audio(model_path, fragment, output_audio_path)
                audios_for_segment.append(output_audio_path)
            audio_matrix.append(audios_for_segment)

        print('Evaluating audio quality ...')
        quality_matrix = np.zeros((num_fragments, num_models))
        for i, audios_for_fragment in enumerate(audio_matrix):
            for j, audio_path in enumerate(audios_for_fragment):
                quality_matrix[i, j] = self.evaluate_audio_quality(audio_path, fragments[i])

        self.quality_matrix = quality_matrix

        rankings_matrix = np.argsort(quality_matrix, axis=1) + 1

        ffragment_lengths = [len(ffragment.split()) for ffragment in fragments]
        total_length = sum(ffragment_lengths)
        weights = [length / total_length for length in ffragment_lengths]

        weighted_scores = np.zeros(num_models)
        print('Final weighted scores')
        for j in range(num_models):
            weighted_scores[j] = sum(weights[i] * rankings_matrix[i, j] for i in range(num_fragments))
            print(f'{self.model_paths[j]} : {weighted_scores[j]}')

        print('Discarding models ...')
        remaining_models = list(range(num_models))
        while len(remaining_models) > 1:
            worst_model = remaining_models[np.argmax([weighted_scores[j] for j in remaining_models])]
            remaining_models.remove(worst_model)
            weighted_scores[worst_model] = float('inf')

        best_model_index = remaining_models[0]
        best_model_path = self.model_paths[best_model_index]
        print(f"Selected best model: {best_model_path}")

        self.chosen_model = best_model_path
        return best_model_path
    
    
