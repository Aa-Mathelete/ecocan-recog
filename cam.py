from dotenv import load_dotenv
import os
import base64
import requests
import cv2  # OpenCV for camera access and image processing
import time

# Load environment variables from .env file
load_dotenv()

classes = ["plastic", "wood", "metal", "food", "electronics", "other"]

# Fetch the API token from environment variables
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not API_TOKEN:
    raise ValueError("Please set the HUGGINGFACE_API_TOKEN environment variable.")

API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch16"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(image_path):
    # Open the image file in binary mode
    with open(image_path, "rb") as f:
        img = f.read()

    # Prepare data for API
    payload = {
        "inputs": base64.b64encode(img).decode("utf-8"),
        "parameters": {"candidate_labels": classes}
    }

    # Send request to Hugging Face API
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def capture_and_classify():
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the webcam feed
        cv2.imshow('Webcam Feed', frame)

        # Take a picture every 5 seconds
        if int(time.time()) % 5 == 0:
            # Save frame as a temporary image
            image_path = "temp_image.jpg"
            cv2.imwrite(image_path, frame)

            # Classify the image
            output = query(image_path)
            print(output)
            print("Classification Results:")
            for category in output:
                print(category['label'], round(category['score'], 2))

            # Wait for a second to avoid multiple captures within the same second
            time.sleep(1)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the capture and classify function
capture_and_classify()
