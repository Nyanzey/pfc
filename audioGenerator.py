import torch
import numpy as np
from TTS.api import TTS
from jiwer import wer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
modelASR_id = "openai/whisper-large-v3-turbo"

modelASR = AutoModelForSpeechSeq2Seq.from_pretrained(
modelASR_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
modelASR.to(device)
processorASR = AutoProcessor.from_pretrained(modelASR_id)

pipeASR = pipeline(
    "automatic-speech-recognition",
    model=modelASR,
    tokenizer=processorASR.tokenizer,
    feature_extractor=processorASR.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True
)

def generate_audio(model_path, text, output_path):
    tts = TTS(model_name=model_path, progress_bar=False).to(device)

    tts.tts_to_file(text=text, file_path=output_path)

def transcribe_audio(audio_path):
    result = pipeASR(audio_path)
    transcribed_text = result["text"]
    return transcribed_text

def evaluate_audio_quality(audio_path, reference_text):
    result = pipeASR(audio_path)
    transcribed_text = result["text"]

    quality_score = wer(reference_text, transcribed_text)
    
    return quality_score

def select_best_tts_model(segments, model_paths):
    num_segments = len(segments)
    num_models = len(model_paths)
    
    print('Generating audios ...')
    audio_matrix = []
    for i, segment in enumerate(segments):
        audios_for_segment = []
        for model_path in model_paths:
            output_audio_path = f"./audios/{model_path.split('/')[-1]}/audio_segment_{i}.wav"
            generate_audio(model_path, segment, output_audio_path)
            audios_for_segment.append(output_audio_path)
        audio_matrix.append(audios_for_segment)

    print('Evaluating quality ...')
    quality_matrix = np.zeros((num_segments, num_models))
    for i, audios_for_segment in enumerate(audio_matrix):
        for j, audio_path in enumerate(audios_for_segment):
            quality_matrix[i, j] = evaluate_audio_quality(audio_path, segments[i])

    print('Quality matrix')
    print(quality_matrix)

    rankings_matrix = np.argsort(quality_matrix, axis=1) + 1

    segment_lengths = [len(segment.split()) for segment in segments]
    total_length = sum(segment_lengths)
    weights = [length / total_length for length in segment_lengths]

    weighted_scores = np.zeros(num_models)
    print('Final weighted scores')
    for j in range(num_models):
        weighted_scores[j] = sum(weights[i] * rankings_matrix[i, j] for i in range(num_segments))
        print(f'{model_paths[j]} : {weighted_scores[j]}')

    print('Discarding models ...')
    remaining_models = list(range(num_models))
    while len(remaining_models) > 1:
        worst_model = remaining_models[np.argmax([weighted_scores[j] for j in remaining_models])]
        remaining_models.remove(worst_model)
        weighted_scores[worst_model] = float('inf')

    best_model_index = remaining_models[0]
    best_model_path = model_paths[best_model_index]
    print(f"Selected best model: {best_model_path}")

    return best_model_path
