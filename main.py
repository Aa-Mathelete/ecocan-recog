import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import pyfirmata
import time

# Define classes
classes = ["plastic", "wood", "metal", "food", "electronics", "other"]

# Initialize CLIP model and processor outside the loop for performance
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

def classify_image(image):
    """
    Classify the given PIL image using the CLIP model and return the class with the highest probability.
    """
    inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # This is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # We can take the softmax to get the label probabilities

    # Get the most likely class
    max_prob, predicted_class = probs.max(dim=1)
    return classes[predicted_class.item()], max_prob.item()

def capture_and_classify():
    """
    Capture image from webcam, classify it, and print the result.
    """
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize the image if needed (adjust dimensions as required)
        image = image.resize((224, 224)) 

        # Classify the image
        try:
            predicted_class, probability = classify_image(image)
            print(f"Predicted Class: {predicted_class} (Probability: {probability:.4f})")

        except Exception as e:
            print(f"Error: {e}")

        # Add a delay to avoid overwhelming the system
        time.sleep(1)

# Run the capture and classify function
capture_and_classify()
