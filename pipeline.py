import infoExtractor as IE
import sceneGenerator as SG
from pathlib import Path
import myapi
import os



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
        #raw_DC = myapi.query_openai_api('gpt-4o', createDC_prompt, system_prompt)

        with open(dictionary_path, 'w') as f:
            f.write(raw_DC)

    DC = IE.parse_DC(raw_DC)

    if os.path.exists(segments_path) and not regenerate_always:
        print('Using buffered segments')
        raw_segments = Path(segments_path).read_text()
    else:
        print('Segmenting story .....')
        segments_prompt = IE.get_txt_prompt('segment', input_dict)
        #raw_segments = myapi.query_openai_api('gpt-4o', segments_prompt, system_prompt)
        
        with open(segments_path, 'w') as f:
            f.write(raw_segments)
    SEGMENTS = IE.parse_segment(raw_segments)

    return DC, SEGMENTS

# Steps 1 and 2
DC, SEGMENTS = get_info("./input/redhoodsmall.txt")

print(len(DC))
print(len(SEGMENTS))

# Step 3 and 4

image_prompts = []

with open('./dynamicPrompts/prompts.txt', 'r') as file:
    image_prompts = [line.strip() for line in file.readlines()]

for i in range(len(SEGMENTS)):
    #DC, prompt = SG.get_image_prompt(DC, SEGMENTS[i], i)
    prompt = image_prompts[i]
    print(f'Generating: {prompt}')
    prompt = SG.generate_image(prompt, f'./images/{str(i).zfill(3)}.jpeg', 'jpeg', DC, SEGMENTS[i], i)
    image_prompts.append(prompt)

# Step 5


