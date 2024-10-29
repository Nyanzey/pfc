import infoExtractor as IE
import sceneGenerator as SG
from pathlib import Path

# Step 1 and 2
def get_info(input_path):
    with open(input_path, 'r') as file:
        story = file.read()
    
    input_dict = {'narrative': story}

    createDC_prompt = IE.get_txt_prompt('createDC', input_dict)
    # query_openai_api(createDC_prompt, ...)
    raw_DC = Path("./dynamicPrompts/dictionary.txt").read_text() # replace with response from query
    DC = IE.parse_DC(raw_DC)

    segments_prompt = IE.get_txt_prompt('segment', input_dict)
    # query_openai_api(segments_prompt, ...)
    raw_segments = Path("./dynamicPrompts/segments.txt").read_text() # replace with response from query
    SEGMENTS = IE.parse_segment(raw_segments)

    return DC, SEGMENTS

# Steps 1 and 2
DC, SEGMENTS = get_info("./input/redhood.txt")

# Step 3 and 4
image_prompts = []
for segment in SEGMENTS:
    DC, prompt = SG.get_image_prompt(DC, segment)
    image_prompts.append(prompt)

for i in range(len(image_prompts)):
    SG.generate_image(image_prompts[i], f'./images/{str(i).zfill(3)}.jpg', 'jpg', DC, segment)

# Step 5