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
from Logger import Logger

# Utilities
from PIL import Image
import os
import re

# For image captioning and similarity
from transformers import BlipProcessor, BlipForConditionalGeneration

# Sentence transformer
from sentence_transformers import SentenceTransformer

log_dir = './logs'
logger = Logger(log_dir)

# Step 1 and 2
logger.log('Entering step 1 and 2')

input_path = "./input/test2.txt"
config_path = "./config.json"
save_path = "./dynamicPrompts"
output_audio_path = "./audios"
output_image_path = "./images"

model_manager = ModelManager(config_path, logger=logger)

info_extractor = IE.InfoExtractor(input_path, config_path, save_path, model_manager, logger)
info_extractor.get_characteristics(regenerate_always=False)
info_extractor.segment_story(regenerate_always=False, segment_method=info_extractor.llm_part_segment)
info_extractor.save_all()

logger.log(len(info_extractor.DC))
logger.log(len(info_extractor.segments))

logger.log('Finished step 1 and 2')

if (input("continue? (y/n): ") != 'y'):
    exit()

logger.log('Entering step 3 and 4')

# Step 3 and 4

img_captioning_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
img_captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
stc_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

scene_generator = SG.SceneGenerator(config_path, save_path, output_image_path, info_extractor, model_manager, logger)

image_format = 'png'
scene_generator.generate_scenes(img_format=image_format, similarity_threshold=0.4) # Similarity value of 0.3 and above seems to be considered good based on some papers, looking for a standard metric would be ideal though
scene_generator.save_prompts()
scene_generator.info_extractor.save_all()

logger.log('Finished step 3 and 4')

if (input("continue? (y/n): ") != 'y'):
    exit()

logger.log('Entering step 5')

# Step 5
fragments = []
pattern = r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ.,!?;:\' ]' # Need to adjust when generating for different languages (english or spanish) before executing the pipeline
for segment in scene_generator.info_extractor.segments:
    fragments.append(re.sub(pattern, '', segment['fragment']))

# for spanish: "tts_models/spa/fairseq/vits"
audio_models = ['tts_models/en/ljspeech/neural_hmm', 'tts_models/en/ljspeech/overflow']

audio_generator = AG.AudioGenerator(audio_models, output_audio_path, logger)

logger.log(fragments)

best_tts = audio_generator.select_best_tts_model(fragments, regenerate_always=False)
best_tts_name = best_tts.split('/')[-1]
logger.log(f'Best tts: {best_tts_name}')

logger.log('Finished step 5')

if (input("continue? (y/n): ") != 'y'):
    exit()

logger.log('Entering step 6')

# Step 6
images = []  
audios = []  
output_path = f'./output/' + input("Name for output video: ") + '.mp4'

for i in range(len(info_extractor.segments)):
    images.append(f'./images/{str(i).zfill(3)}.{image_format}')
    audios.append(f'./audios/{best_tts_name}/audio_fragment_{i}.wav')

VA.create_narrative_video(images, audios, output_path)

logger.log('Finished step 6')

logger.log('Evaluating CM ...')

# Evaluation

eval_prompts = scene_generator.prompts['final']
if len(eval_prompts) < len(info_extractor.segments):
    eval_prompts = scene_generator.prompts['buffer']
if not eval_prompts:
    with open(scene_generator.save_path + '/buffer_prompts.txt', 'r') as file:
        eval_prompts = file.readlines()

evaluator = eval.Evaluation(0.5, 0.25, 0.25, scene_generator, scene_generator.logger)
val_images = []
for i in range(len(info_extractor.segments)):
    img_path = f'./images/{str(i).zfill(3)}.{image_format}'
    val_images.append(Image.open(img_path))

val_transcriptions = []
audios_dir = f"./audios/{best_tts.split('/')[-1]}/"
for i in range(len(info_extractor.segments)):
    audio_path = audios_dir + f'audio_fragment_{i}.wav'
    val_transcriptions.append(audio_generator.transcribe_audio(audio_path))

score = evaluator.calculate_cm(fragments, eval_prompts, val_images, val_transcriptions)
logger.log(f'Final metric score: {score}')