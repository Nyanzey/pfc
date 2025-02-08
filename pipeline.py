import os
os.environ['HF_HOME'] = 'F:\\modelscache'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Pipeline components
import infoExtractor as IE
import sceneGenerator as SG
import audioGenerator as AG
import videoAssembler as VA
import evaluate as eval
from myapi import ModelManager

# Utilities
from PIL import Image
import json
import os
import re

# For image captioning and similarity
from transformers import BlipProcessor, BlipForConditionalGeneration

# Sentence transformer
from sentence_transformers import SentenceTransformer

# Step 1 and 2
print('Entering step 1 and 2')

input_path = "./input/test.txt"
config_path = "./config.json"
save_path = "./dynamicPrompts"
output_audio_path = "./audios"

model_manager = ModelManager(config_path)

info_extractor = IE.InfoExtractor(input_path, config_path, save_path, model_manager)
info_extractor.get_characteristics(regenerate_always=False)
info_extractor.segment_story(regenerate_always=False)
info_extractor.save_all()

print(len(info_extractor.DC))
print(len(info_extractor.segments))

print('Finished step 1 and 2')

print('Entering step 3 and 4')

# Step 3 and 4

img_captioning_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
img_captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
stc_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

scene_generator = SG.SceneGenerator(config_path, save_path, info_extractor, img_captioning_model, img_captioning_processor, stc_model, model_manager)

scene_generator.generate_scenes()
scene_generator.save_prompts()
scene_generator.info_extractor.save_all()

print('Finished step 3 and 4')

print('Entering step 5')

# Step 5
fragments = []
pattern = r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ.,!?;:\' ]' # Need to adjust when generating for different languages (english or spanish) before executing the pipeline
for segment in scene_generator.info_extractor.segments:
    fragments.append(re.sub(pattern, '', segment['fragment']))

# for spanish: "tts_models/spa/fairseq/vits"
audio_models = ['tts_models/en/ljspeech/neural_hmm', 'tts_models/en/ljspeech/overflow']

audio_generator = AG.AudioGenerator(audio_models, output_audio_path)

print(fragments)

best_tts = audio_generator.select_best_tts_model(fragments)
best_tts_name = best_tts.split('/')[-1]
print(f'Best tts: {best_tts_name}')

print('Finished step 5')

print('Entering step 6')

# Step 6
images = []  
audios = []  
output_path = f'./output/' + input("Name for output video: ") + '.mp4'

for i in range(len(info_extractor.segments)):
    images.append(f'./images/{str(i).zfill(3)}.jpeg')
    audios.append(f'./audios/{best_tts_name}/audio_segment_{i}.wav')

VA.create_narrative_video(images, audios, output_path)

print('Finished step 6')

print('Evaluating CM ...')

# Evaluation
val_images = []
for i in range(len(info_extractor.segments)):
    img_path = f'./images/{str(i).zfill(3)}.jpeg'
    val_images.append(Image.open(img_path))

val_transcriptions = []
audios_dir = f"./audios/{best_tts.split('/')[-1]}/"
for i in range(len(info_extractor.segments)):
    audio_path = audios_dir + f'audio_segment_{i}.wav'
    val_transcriptions.append(AG.transcribe_audio(audio_path))

score = eval.calculate_cm(fragments, scene_generator.prompts['buffer'], val_images, val_transcriptions)
print(f'Final metric score: {score}')