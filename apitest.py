from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64
from PIL import Image
import threading

app = Flask(__name__)

inference_lock = threading.Lock()

# Load model once at startup (use fp16 for G4dn and CUDA)
pipe = StableDiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Run inference
    with inference_lock:
        image = pipe(prompt).images[0]

    # Encode as base64
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({"image_base64": img_str})

@app.route("/health", methods=["GET"])
def health_check():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=False)