import infoExtractor as IE
import sceneGenerator as SG
import audioGenerator as AG
import videoAssembler as VA
import evaluate as eval
from pathlib import Path
from PIL import Image
import myapi
import os
import re

# Step 1 and 2
def get_info(input_path, regenerate_always=False):
    with open(input_path, 'r') as file:
        story = file.read()
    
    input_dict = {'narrative': story}
    system_prompt = "You are a story analyzer."

    dictionary_path = './dynamicPrompts/dictionary.txt'
    segments_path = './dynamicPrompts/segments.txt'

    if os.path.exists(dictionary_path) and not regenerate_always:
        print('Using buffered DC')
        raw_DC = Path(dictionary_path).read_text()
    else:
        print('Creating DC .....')
        createDC_prompt = IE.get_txt_prompt('createDC', input_dict)
        raw_DC = myapi.query_openai_api('gpt-4o', createDC_prompt, system_prompt)

        with open(dictionary_path, 'w') as f:
            f.write(raw_DC)

    DC = IE.parse_DC(raw_DC)

    if os.path.exists(segments_path) and not regenerate_always:
        print('Using buffered segments')
        raw_segments = Path(segments_path).read_text()
    else:
        print('Segmenting story .....')
        segments_prompt = IE.get_txt_prompt('segment', input_dict)
        raw_segments = myapi.query_openai_api('gpt-4o', segments_prompt, system_prompt)
        
        with open(segments_path, 'w') as f:
            f.write(raw_segments)
    SEGMENTS = IE.parse_segment(raw_segments)

    return DC, SEGMENTS

print('Entering step 1 and 2')
# Steps 1 and 2
DC, SEGMENTS = get_info("./input/hansel.txt", regenerate_always=True)

print(len(DC))
print(len(SEGMENTS))

print('Finished step 1 and 2')
sup = input('continue ? ')
if sup == 'n':
    exit()

print('Entering step 3 and 4')
# Step 3 and 4

image_prompts = []

"""
with open('./dynamicPrompts/prompts.txt', 'r') as file:
    image_prompts = [line.strip() for line in file.readlines()]
"""

for i in range(len(SEGMENTS)):
    DC, prompt = SG.get_image_prompt(DC, SEGMENTS[i], i)
    print(f'Generating: {prompt}')
    prompt = SG.generate_image(prompt, f'./images/{str(i).zfill(3)}.jpeg', 'jpeg', DC, SEGMENTS[i], i)
    image_prompts.append(prompt)

with open('./dynamicPrompts/prompts.txt', 'w') as file:
    file.writelines(s + '\n' for s in image_prompts)

with open('./dynamicPrompts/updatedDict.txt', 'w') as file:
    file.write(IE.DC_to_string(DC))

print('Finished step 3 and 4')
sup = input('continue ? ')
if sup == 'n':
    exit()

print('Entering step 5')
# Step 5

fragments = []
pattern = r'[^a-zA-Z.,!?;:\' ]'
for segment in SEGMENTS:
    fragments.append(re.sub(pattern, '', segment['fragment']))

audio_models = ['tts_models/en/ljspeech/neural_hmm', 'tts_models/en/ljspeech/overflow']

print(fragments)

best_tts = AG.select_best_tts_model(fragments, audio_models)
#best_tts = 'tts_models/en/ljspeech/overflow'

print('Finished step 5')
sup = input('continue ? ')
if sup == 'n':
    exit()

print('Entering step 6')
# Step 6

# delete after
"""
best_tts = 'tts_models/en/ljspeech/overflow'
with open('./dynamicPrompts/segments.txt') as f:
    raw = f.read()
SEGMENTS = IE.parse_segment(raw)

fragments = []
pattern = r'[^a-zA-Z.,!?;:\' ]'
for segment in SEGMENTS:
    fragments.append(re.sub(pattern, '', segment['fragment']))
"""
# delete after

images = []  
audios = []  
output_path = f'./output/' + input("Name for output video") + '.mp4'

for i in range(len(SEGMENTS)):
    images.append(f'./images/{str(i).zfill(3)}.jpeg')
    audios.append(f'./audios/overflow/audio_segment_{i}.wav')

VA.create_narrative_video(images, audios, output_path)

print('Finished step 6')

# Evaluation

val_images = []
for i in range(len(SEGMENTS)):
    img_path = f'./images/{str(i).zfill(3)}.jpeg'
    val_images.append(Image.open(img_path))

val_transcriptions = []
audios_dir = f'./audios/{best_tts.split('/')[-1]}/'
for i in range(len(SEGMENTS)):
    audio_path = audios_dir + f'audio_segment_{i}.wav'
    val_transcriptions.append(AG.transcribe_audio(audio_path))

score = eval.calculate_cm(fragments, image_prompts, val_images, val_transcriptions)
print(f'Final metric score: {score}')