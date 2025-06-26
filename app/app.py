import re
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.spatial import distance
from collections import deque, Counter
import base64
import platform
import sys
from scipy.spatial import distance
import json
from flask import Flask, jsonify, request, render_template
from io import BytesIO
from PIL import Image

# load story
with open("story_flow.json", "r") as f:
    story = json.load(f)
    
# configuration
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# predict app route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        print("Empty or invalid JSON received.")
        return jsonify({"error": "Invalid JSON"}), 400

    if "image" not in data:
        return jsonify({"error": "No image data found"}), 400
        
     # extract base64 part pf the image
    image_data = re.sub('^data:image/.+;base64,', '', data["image"])
    if not image_data.strip():
        print("Empty base64 string received.")
        return jsonify({"error": "Empty image data"}), 400
        
    try:
        decoded_image = base64.b64decode(image_data)
        if not decoded_image or len(decoded_image) < 100:
            return jsonify({"error": "corrupted image"}), 400
        
        try:
            image = Image.open(BytesIO(decoded_image))
            image.verify()  # validate it's a real image
            image = Image.open(BytesIO(decoded_image))  # re-open after verify()
            image = image.convert("RGB")
        except Exception as e:
            print("image verification failed:", e)
            return jsonify({"error": "invalid image format"}), 400

        frame = np.array(image)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 1)

        step_index = data.get("step_index", 0)
        if step_index >= len(story):
            return jsonify({
                "predicted_label": "BRAVO",
                "confidence": 1.0,
                "is_correct": True,
                "next_step_index": step_index,
                "next_prompt": "Histoire terminée ! - Story completed! - To restart, turn the camera on again",
                "next_image": "static/images/bravo.gif",
                "expected": None,
                "feedback": "BRAVO!!!! (Bonus sign 'BRAVO')",
                "left_box": None,
                "right_box": None,
                "story_completed": True
    })

        result = sign_detector(frame, step_index)    

        return jsonify(result)

    except Exception as e:
        print("Error processing image:", e)
        return jsonify({"error": "Image decoding or prediction failed"}), 500

# load sign list
with open("label_mapping_V2.json", "r") as f:
    label_map = json.load(f)
    ACTIONS = [label_map[str(i)] for i in range(len(label_map))]

SEQUENCE_LENGTH = 64
NUM_FEATURES = 1629
MODEL_PATH = "model/fiveSigns_V5_model.keras"

# load model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

except (OSError, IOError):
    print("Model file not found at path: %s", MODEL_PATH)
    sys.exit(1)

#### THIS HIDDEN/ DELETED IS MOST IMPORTANT PART OF MAIN LOGIC AND IT WILL NEVER BE PUBLIC. CONTACT ME THEN :D.

    prediction_history.append(predicted_label)

    left_box = get_hand_box(frame, results.left_hand_landmarks)
    right_box = get_hand_box(frame, results.right_hand_landmarks)

    story_step = story[step_index]
    story_image = story_step['image']
    story_prompt = story_step['prompt']
    expected = story_step['expected']
    is_correct = predicted_label in expected and confidence > 0.65
    feedback = ""
    story_completed = False

    if is_correct:
        step_index += 1
        if step_index < len(story):
            story_step = story[step_index]
            story_image = story_step['image']
            story_prompt = story_step['prompt']
            expected = story_step['expected']
            feedback = "Correct!"
            story_completed = False
        else:
            story_image = "static/images/bravo.gif"
            story_prompt = "BRAVO!!!! (Bonus sign 'BRAVO')"
            feedback = "Histoire terminée ! - Story completed! - To restart, turn the camera on again"
            expected = None
            story_completed = True
    else: 
        feedback = "Essayez encore! - Try again!"

    return {
        "predicted_label": predicted_label,
        "confidence": float(confidence),
        "is_correct": bool(is_correct),
        "next_step_index": step_index,
        "next_prompt": story_prompt,
        "next_image": story_image,
        "expected": expected or "None",
        "feedback": feedback or "", 
        "left_box": left_box,
        "right_box": right_box,
        "story_completed": story_completed
        }

print("Running the app...")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
   

