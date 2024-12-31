import infoExtractor as IE
import sceneGenerator as SG
import audioGenerator as AG
import videoAssembler as VA
import evaluate as eval
from PIL import Image
import json
import os
import re

# Step 1 and 2
print('Entering step 1 and 2')

with open('config.json', 'r') as file:
    config = json.load(file)

DC = IE.get_characteristics("./input/test.txt", regenerate_always=False)
SEGMENTS = IE.segment_story("./input/test.txt", regenerate_always=False)

print(len(DC))
print(len(SEGMENTS))

print('Finished step 1 and 2')

print('Entering step 3 and 4')

# Step 3 and 4
image_prompts = []
buffer_prompts = []

for i in range(len(SEGMENTS)):
    save_path = f'./images/{str(i).zfill(3)}.jpeg'
    if os.path.exists(save_path):
        continue

    if i == 0:
        dummy_img = Image.open('./black.png')
    else:
        dummy_img = Image.open(f'./images/{str(i-1).zfill(3)}.jpeg')

    DC, prompt = SG.get_image_prompt(DC, SEGMENTS[i], i)
    buffer_prompts.append(prompt)
    print(f'Generating image {i}: {prompt}')

    try:
        prompt = SG.generate_image(prompt, save_path, 'jpeg', DC, SEGMENTS[i], i, threshold=0.6, max_generations=1)
        image_prompts.append(prompt)
    except Exception as e:
        dummy_img.save(save_path, format='jpeg')
        print(f"Error generating image for segment {i}: {e}")
        continue

with open('./dynamicPrompts/prompts.txt', 'w') as file:
    file.writelines(s + '\n' for s in image_prompts)

with open('./dynamicPrompts/buffer_prompts.txt', 'w') as file:
    file.writelines(s + '\n' for s in buffer_prompts)

with open('./dynamicPrompts/updatedDict.txt', 'w') as file:
    file.write(IE.DC_to_string(DC))

print('Finished step 3 and 4')

print('Entering step 5')

# Step 5
fragments = []
pattern = r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ.,!?;:\' ]' # Need to adjust when generating for different languages (english or spanish) before executing the pipeline
for segment in SEGMENTS:
    fragments.append(re.sub(pattern, '', segment['fragment']))

# for spanish: "tts_models/spa/fairseq/vits"
audio_models = ['tts_models/en/ljspeech/neural_hmm', 'tts_models/en/ljspeech/overflow']

print(fragments)

best_tts = AG.select_best_tts_model(fragments, audio_models)
best_tts_name = best_tts.split('/')[-1]
print(f'Best tts: {best_tts_name}')

print('Finished step 5')

print('Entering step 6')

# Step 6
images = []  
audios = []  
output_path = f'./output/' + input("Name for output video: ") + '.mp4'

for i in range(len(SEGMENTS)):
    images.append(f'./images/{str(i).zfill(3)}.jpeg')
    audios.append(f'./audios/{best_tts_name}/audio_segment_{i}.wav')

VA.create_narrative_video(images, audios, output_path)

print('Finished step 6')

print('Evaluating CM ...')

# Evaluation
val_images = []
for i in range(len(SEGMENTS)):
    img_path = f'./images/{str(i).zfill(3)}.jpeg'
    val_images.append(Image.open(img_path))

val_transcriptions = []
audios_dir = f"./audios/{best_tts.split('/')[-1]}/"
for i in range(len(SEGMENTS)):
    audio_path = audios_dir + f'audio_segment_{i}.wav'
    val_transcriptions.append(AG.transcribe_audio(audio_path))

score = eval.calculate_cm(fragments, image_prompts, val_images, val_transcriptions)
print(f'Final metric score: {score}')