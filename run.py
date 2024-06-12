from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import cv2
import numpy as np
import face_recognition
import base64
from io import BytesIO
from PIL import Image
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

app = Flask(__name__)
CORS(app)

USER_DATA_FOLDER = 'user_data'
USER_DATA_FILE = os.path.join(USER_DATA_FOLDER, 'users.json')
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

if not os.path.exists(USER_DATA_FOLDER):
    os.makedirs(USER_DATA_FOLDER)

if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump([], f)

def load_user_data():
    with open(USER_DATA_FILE, 'r') as f:
        return json.load(f)

def save_user_data(data):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f)

def decode_image(img_str):
    img_data = base64.b64decode(img_str.split(",")[1])
    img = Image.open(BytesIO(img_data))
    return np.array(img)

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def is_blinking(face_image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) == 0:
        print("No face detected")
        return False

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
        rightEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
        if ear < 0.25:
            print("Blink detected")
            return True

    print("No blink detected")
    return False

def head_movement(face_image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) == 0:
        print("No face detected")
        return False

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        left_eye_center = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]].mean(axis=0)
        right_eye_center = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]].mean(axis=0)
        mouth_center = shape[face_utils.FACIAL_LANDMARKS_IDXS["mouth"][0]:face_utils.FACIAL_LANDMARKS_IDXS["mouth"][1]].mean(axis=0)

        eye_center = (left_eye_center + right_eye_center) / 2.0
        dY = mouth_center[1] - eye_center[1]
        dX = mouth_center[0] - eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 90

        if -20 < angle < 20:
            print("Head movement detected")
            return True

    print("No head movement detected")
    return False

@app.route('/profile', methods=['GET'])
def get_profile():
    # Mock user data, replace with actual data retrieval from database
    user = {'username': 'john_doe', 'email': 'john_doe@example.com'}
    return jsonify(user)

@app.route('/profile', methods=['PUT'])
def update_profile():
    data = request.get_json()
    # Update user profile in database
    # Mock implementation
    print(f"Updating user profile: {data}")
    return jsonify({'status': 'Profile updated successfully'})

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data['username']
    face_data = decode_image(data['face_data'])

    # Encode the face data
    face_encoding = face_recognition.face_encodings(face_data)[0].tolist()

    # Load existing users
    users = load_user_data()

    # Check if the user already exists
    if any(user['username'] == username for user in users):
        return jsonify({'status': 'User already exists'}), 400

    # Add new user
    users.append({'username': username, 'face_encoding': face_encoding})
    save_user_data(users)

    return jsonify({'status': 'User registered successfully'}), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    face_data = decode_image(data['face_data'])

    # Anti-spoofing check
    if not is_blinking(face_data):
        print("Anti-spoofing failed: Blink")
        return jsonify({'status': 'Spoofing detected'}), 400
    if not head_movement(face_data):
        print("Anti-spoofing failed: Head movement")
        return jsonify({'status': 'Spoofing detected'}), 400

    # Encode the face data
    face_encoding = face_recognition.face_encodings(face_data)[0]

    # Load existing users
    users = load_user_data()

    # Perform face recognition
    for user in users:
        known_face_encoding = np.array(user['face_encoding'])
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        if matches[0]:
            print("Face recognized")
            return jsonify({'status': 'Login successful', 'username': user['username']}), 200

    print("Face not recognized")
    return jsonify({'status': 'Login failed'}), 400

@app.route('/delete_user', methods=['DELETE'])
def delete_user():
    data = request.get_json()
    username = data.get('username')

    if not username:
        return jsonify({'status': 'Username is required'}), 400

    users = load_user_data()
    users = [user for user in users if user['username'] != username]

    save_user_data(users)
    print(f"User {username} deleted successfully")
    return jsonify({'status': 'User deleted successfully'}), 200

@app.route('/users', methods=['GET'])
def get_users():
    users = load_user_data()
    return jsonify(users)

if __name__ == '__main__':
    app.run(debug=True)
