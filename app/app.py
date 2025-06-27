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
from flask_socketio import SocketIO, emit


# load story
with open("story_flow.json", "r") as f:
    story = json.load(f)
    
# configuration
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def home():
    return render_template("index.html")

@socketio.on("connect")
def on_connect():
    print("Client connected")

# main predict app route
@socketio.on('predict_frame')
def handle_prediction(data):
    if not data:
        print("Empty or invalid JSON received.")
        emit("prediction_response", {"error": "Invalid JSON"})
        return

    if "image" not in data:
        emit("prediction_response", {"error": "No image data found"})
        return
        
     # extract base64 part of the image
    image_data = re.sub('^data:image/.+;base64,', '', data["image"])
    if not image_data.strip():
        print("Empty base64 string received.")
        emit("prediction_response", {"error": "Empty image data"})
        return
        
    try:
        decoded_image = base64.b64decode(image_data)
        if not decoded_image or len(decoded_image) < 100:
            emit("prediction_response", {"error": "corrupted image"})
            return
        
        try:
            image = Image.open(BytesIO(decoded_image))
            image.verify()  # validate it's a real image
            image = Image.open(BytesIO(decoded_image))  # re-open after verify()
            image = image.convert("RGB")
        except Exception as e:
            print("image verification failed:", e)
            emit("prediction_response", {"error": "invalid image format"})
            return

        frame = np.array(image)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 1)

        step_index = data.get("step_index", 0)
        if step_index >= len(story):
            emit("prediction_response", {
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
            return

        result = sign_detector(frame, step_index)    

        print("Received frame, emitting result...")
        emit("predict_response", result, to=request.sid)

        return

    except Exception as e:
        print("Error processing image:", e)
        emit("prediction_response", {"error": "Image decoding or prediction failed"})
        return

# fallback predict app route
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

# helper functions
def is_idle(results):
    return results.left_hand_landmarks is None and results.right_hand_landmarks is None

def get_hand_box(image, hand_landmarks):
    if hand_landmarks:
        h, w, _ = image.shape
        # x_coords = [int((1 - lm.x) * w) for lm in hand_landmarks.landmark]
        x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        x = x_min - 10
        y = y_min - 10
        width = (x_max - x_min) + 25
        height = (y_max - y_min) + 25

        return [x, y, width, height]
        

def get_backend():
    if platform.system() == 'Darwin':
        return cv2.CAP_AVFOUNDATION
    elif platform.system() == 'Windows':
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY

def interpolate_missing(data):
    for i in range(data.shape[1]):
        mask = data[:, i] != 0
        if np.sum(mask) > 0:
            data[:, i] = np.interp(np.arange(len(data)), np.where(mask)[0], data[mask, i])
        else:
            data[:, i] = 0
    return data

def preprocess_single_frame(frame_landmarks):
    frame = np.expand_dims(frame_landmarks, axis=0)       # shape: (1, 258)
    frame = interpolate_missing(frame)                     # fill missing                   # center nose
    return frame[0]      

def pad_with_last_frame(data, maxlen):
    if len(data) == 0:
        return np.zeros((maxlen, data.shape[1]))
    last_frame = data[-1]
    pad_length = maxlen - len(data)
    if pad_length > 0:
        padding = np.repeat(last_frame[np.newaxis, :], pad_length, axis=0)
        data = np.vstack([data, padding])
    return data[:maxlen]


def calc_distance(landmark1, landmark2): # landmark objects only
    if landmark1 is None or landmark2 is None:
        return float('inf')
    return distance.euclidean([landmark1.x, landmark1.y, landmark1.z], [landmark2.x, landmark2.y, landmark2.z])

def calc_distance_from_list(p1, p2): # list or array
    return distance.euclidean(p1, p2)

def check_fingers(hand_landmarks):
    if hand_landmarks is None:
        return 0
    finger_tips = [8, 12, 16, 20]  
    wrist = hand_landmarks.landmark[0]
    extended = 0
    for tip_idx in finger_tips:
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[tip_idx - 2]  # proximal interphalangeal joint
        if calc_distance(tip, wrist) > calc_distance(pip, wrist):
            extended += 1
    return extended

def fingers_are_curled(hand_landmarks):
    curled = 0
    fingertip_ids = [8, 12, 16, 20]  # index, middle, ring, pinky tips
    pip_ids = [6, 10, 14, 18] # proximal interphalangeal joint

    for tip_id, pip_id in zip(fingertip_ids, pip_ids):
        if hand_landmarks.landmark[tip_id].y > hand_landmarks.landmark[pip_id].y:
            curled += 1
    return curled >= 2

def check_thumb_extended(hand_landmarks):
    if hand_landmarks is None:
        return False
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]  # interphalangeal joint
    wrist = hand_landmarks.landmark[0]
    return calc_distance(thumb_tip, wrist) > calc_distance(thumb_ip, wrist)

def is_hand_visible(hand_landmarks):
    return hand_landmarks and len(hand_landmarks.landmark) == 21

def validate_au_revoir(results, index_tip_positions):
    left_hand = results.left_hand_landmarks
    right_hand = results.right_hand_landmarks
    pose = results.pose_landmarks

    if pose is None:
        return False

    if is_hand_visible(right_hand) and is_hand_visible(left_hand):
        return False  # au_revoir should be done with only one hand

    # use whichever hand is present
    if right_hand:
        hand = right_hand
        index_tip = hand.landmark[8]
        reference_ear = pose.landmark[8]  # right ear
    elif left_hand:
        hand = left_hand
        index_tip = hand.landmark[8]
        reference_ear = pose.landmark[7]  # left ear
    else:
        return False

    # check fingers extended
    fingers = check_fingers(hand)
    if fingers < 3:
        return False

    # check distance of index_tip to key facial landmarks
    nose = pose.landmark[0]
    mouth = pose.landmark[9]
    dist_to_ear = calc_distance(index_tip, reference_ear)
    dist_to_mouth = calc_distance(index_tip, mouth)
    dist_to_nose = calc_distance(index_tip, nose)

    if dist_to_mouth < 0.16:  # relaxed a bit
        print("Au_revoir: Too close to mouth, might be bonjour")
        return False

    if dist_to_ear < dist_to_nose and dist_to_ear < dist_to_mouth:
        print("Likely au_revoir: hand near ear")
    else:
        print("Hand not near ear (au_revoir)")
        return False

    # relax motion range
    if len(index_tip_positions) >= 3:
        x_positions = [p[0] for p in index_tip_positions]
        motion_range = max(x_positions) - min(x_positions)
        if motion_range < 0.01:  # was 0.004
            print("No significant waving motion")
            return False
        else:
            print(f"Waving motion detected: range={motion_range:.4f}")

    return True


def is_drinking_position(hand, results):
    if hand is None or results.pose_landmarks is None:
        return False
    wrist = hand.landmark[0]
    mouth = results.pose_landmarks.landmark[0]  # nose area
    return wrist.y < mouth.y  # hand up near mouth level

def validate_bonjour(index_tip_positions, results, hand_landmarks):
    if not results.pose_landmarks or not hand_landmarks or len(index_tip_positions) < 3:
        return False
    
    left_hand = results.left_hand_landmarks
    right_hand = results.right_hand_landmarks

    if not is_hand_visible(right_hand) or is_hand_visible(left_hand): # forced to do right hand only
        print("Bonjour must be right hand only")
        return False
    
    pose = results.pose_landmarks
    mouth = pose.landmark[9]
    nose = pose.landmark[0]

    index_tip_positions = list(index_tip_positions)[-5:]

    mouth_coords = [mouth.x, mouth.y, mouth.z]
    distances = [
        calc_distance_from_list(index_tip, mouth_coords)
        for index_tip in index_tip_positions
    ]

    # movement toward mouth
    if index_tip_positions[0][1] > mouth.y and index_tip_positions[-1][1] < mouth.y:
        return True

    # large change in distance to mouth (movement)
    if max(distances) - min(distances) > 0.02:
        return True
    
    x_positions = [pos[0] for pos in index_tip_positions]
    x_range = max(x_positions) - min(x_positions)
    if x_range > 0.2:
        print("Skip bonjour: too much waving motion")
        return False

    # fallback: close to nose = maybe bonjour
    last_tip = index_tip_positions[-1]
    if calc_distance_from_list(last_tip, [nose.x, nose.y, nose.z]) < 0.08:
        print("Bonjour: fallback (close to nose)")
        return True

    return False


# def palm_facing_camera(hand_landmarks):
#     wrist = hand_landmarks.landmark[0]
#     index_base = hand_landmarks.landmark[5]
#     pinky_base = hand_landmarks.landmark[17]
#     v1 = np.array([index_base.x - wrist.x, index_base.y - wrist.y, index_base.z - wrist.z])
#     v2 = np.array([pinky_base.x - wrist.x, pinky_base.y - wrist.y, pinky_base.z - wrist.z])
#     normal = np.cross(v1, v2)
#     return normal[2] < 0.02

# main program
sequence = deque(maxlen=SEQUENCE_LENGTH)
prediction_history = deque(maxlen=10)
index_tip_positions = deque(maxlen=10)
motion_frame_count = 0
prev_tips = None

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
   
def sign_detector(frame, step_index):
    global index_tip_positions
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    frame_landmarks = np.zeros(NUM_FEATURES, dtype=np.float32)

    predicted_label = "..still predicting"
    confidence = 0.0

    # pose landmarks (33 * 3 = 99)
    if results.pose_landmarks:
        for i, l in enumerate(results.pose_landmarks.landmark):
            frame_landmarks[i*3:i*3+3] = [l.x, l.y, l.z]

    # face landmarks (468 * 3 = 1404)
    if results.face_landmarks:
        for i, l in enumerate(results.face_landmarks.landmark):
            start = 99 + i*3
            frame_landmarks[start:start+3] = [l.x, l.y, l.z]

    # left hand (21 * 3 = 63)
    if results.left_hand_landmarks:
        for i, l in enumerate(results.left_hand_landmarks.landmark):
            start = 1503 + i*3  # 99 + 1404 = 1503
            frame_landmarks[start:start+3] = [l.x, l.y, l.z]

    # right hand (21 * 3 = 63)
    if results.right_hand_landmarks:
        for i, l in enumerate(results.right_hand_landmarks.landmark):
            start = 1566 + i*3  # 1503 + 63 = 1566
            frame_landmarks[start:start+3] = [l.x, l.y, l.z]


    if is_idle(results):
        print("No hands detected")
        sequence.clear()
        predicted_label = "Idle / No Gesture"
        confidence = 0.0

    else:
        processed = preprocess_single_frame(frame_landmarks)
        sequence.append(processed)

        if len(sequence) >= 10: 
            if len(sequence) < SEQUENCE_LENGTH:
                padded_seq = pad_with_last_frame(np.array(sequence), SEQUENCE_LENGTH)
            else:
                padded_seq = np.array(sequence)[-SEQUENCE_LENGTH:]

            input_data = np.expand_dims(padded_seq, axis=0)
            prediction = model.predict(input_data, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            predicted_label = ACTIONS[predicted_class]
            print("original predicted label", predicted_label)
            print("original confidence", confidence)

            if predicted_class < len(ACTIONS):
        
                # hand presence and fingers
                left_hand = results.left_hand_landmarks
                right_hand = results.right_hand_landmarks
                left_fingers = check_fingers(left_hand)
                right_fingers = check_fingers(right_hand)
                print("Fingers left/right:", left_fingers, right_fingers)

                # both two hands required: ca_va , quoi
                if predicted_label == "quoi":
                    if not (left_hand and right_hand 
                            and left_fingers > 3 and right_fingers > 3):
                        predicted_label = "ambiguous"
                        confidence = 0.0

                if left_hand and right_hand:
                    if left_fingers <= 3 and right_fingers <= 3:
                        predicted_label = "ca_va"
                        confidence = 0.75
                        print(f"overriden confidence: {confidence}")
                        print(f"overriden label: {predicted_label}")
                    else:
                        if predicted_label in ["bonjour", "au_revoir", "boire"]:
                            confidence = 0.0
                            predicted_label = "ambiguous"
                            print(f"overriden confidence: {confidence}")
                            print(f"overriden label: {predicted_label}")
                    
                # only one hand allowed
                if (left_hand is not None) != (right_hand is not None):
                    single_hand = left_hand if left_hand else right_hand
                    fingers = left_fingers if left_hand else right_fingers
                    mouth_landmark = results.pose_landmarks.landmark[9]

                    # check if thumb up like drinking water
                    if check_thumb_extended(single_hand) and fingers <= 2 and is_drinking_position(single_hand, results):
                        predicted_label = "boire"
                        confidence = 0.75
                        print(f"overriden confidence: {confidence}")
                        print(f"overriden label: {predicted_label}")
                            
                    elif fingers >= 4:
                        if right_hand:
                            index_tip = right_hand.landmark[8]
                        elif left_hand:
                            index_tip = left_hand.landmark[8]
                        else:
                            index_tip = None

                        if index_tip:
                            index_tip_positions.append([index_tip.x, index_tip.y, index_tip.z])

                            # bonjour - right hand only
                            if validate_bonjour(index_tip_positions, results, right_hand) and mouth_landmark:
                                predicted_label = "bonjour"
                                confidence = 0.75
                                print(f"overriden confidence: {confidence}")
                                print(f"overriden label: {predicted_label}")
                                            
                            # au revoir
                            elif validate_au_revoir(results, index_tip_positions): 
                                if predicted_label == "au_revoir":
                                    confidence = 0.75
                                    print(f"overriden confidence: {confidence}")

                                else: 
                                    predicted_label = "au_revoir"
                                    confidence = 0.75
                                    print(f"overriden confidence: {confidence}")
                                    print(f"overriden label: {predicted_label}")
                         
                            elif predicted_label in ["ca_va", "quoi"] and single_hand:
                                confidence = 0.0
                                predicted_label = "Unknown"
                                print(f"overriden confidence: {confidence}")
                                print(f"overriden label: {predicted_label}")


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

if __name__ == "__main__":
    print("Running the app...")
    socketio.run(app, host="0.0.0.0", port=7860)
   

