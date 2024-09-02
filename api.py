from dotenv import load_dotenv
import os, base64, requests

# Load environment variables from .env file
load_dotenv()

classes = ["plastic", "wood", "metal", "food", "electronics", "other"]

# Fetch the API token from environment variables
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch16"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(data):
    # open the image
    with open(data["image_path"], "rb") as f:
        img = f.read()
    
    # data for API
    payload = {
        "parameters": data["parameters"],
        "inputs": base64.b64encode(img).decode("utf-8")
    }

    # get response
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "image_path": "bottle.jpg",
    "parameters": {"candidate_labels": classes},
})

print(output)

for category in output:
    print(category['label'], round(category['score'], 2))