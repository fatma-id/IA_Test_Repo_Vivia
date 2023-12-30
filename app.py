from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import joblib

app = Flask(__name__)
# Replace 'your_model_path.joblib' with the actual path and filename of your saved model
rf_classifier = joblib.load('model.joblib')

# Initialize MediaPipe Face Detection and Facial Landmarks
mp_face = mp.solutions.face_detection
mp_landmarks = mp.solutions.face_mesh

face_detection = mp_face.FaceDetection()
face_mesh = mp_landmarks.FaceMesh()

# Function to calculate blink stress
def calculate_blink_stress(landmarks):
    # use the eye landmarks to detect blinking
    left_eye = landmarks[42:48]
    right_eye = landmarks[36:42]

    # Calculate the average vertical position of the eyes
    avg_y_left = np.mean(left_eye[:, 1])
    avg_y_right = np.mean(right_eye[:, 1])

    # Calculate blink stress based on the difference in vertical position
    blink_stress = abs(avg_y_left - avg_y_right)

    return blink_stress

# Function to calculate eyebrow stress
def calculate_eyebrow_stress(landmarks):
    # Implement your logic to calculate eyebrow stress based on landmarks
    # For example, you can use the eyebrow landmarks to detect raised or lowered eyebrows
    left_eyebrow = landmarks[22:27]
    right_eyebrow = landmarks[17:22]

    # Calculate eyebrow stress based on the difference in vertical position
    eyebrow_stress = abs(np.mean(left_eyebrow[:, 1]) - np.mean(right_eyebrow[:, 1]))

    return eyebrow_stress

# Function to calculate lip stress
def calculate_lip_stress(landmarks):
    # Implement your logic to calculate lip stress based on landmarks
    # For example, you can use the upper and lower lip landmarks to detect lip movement
    upper_lip = landmarks[50:53]
    lower_lip = landmarks[56:59]

    # Calculate lip stress based on the difference in vertical position
    lip_stress = abs(np.mean(upper_lip[:, 1]) - np.mean(lower_lip[:, 1]))

    return lip_stress

# Function to calculate head movement stress
def calculate_head_movement_stress(landmarks):
    # For example, you can use the position of the head in the frame
    head_position = landmarks[0]  # Assuming the first landmark represents the top of the head

    # Calculate head movement stress based on the position of the head
    head_movement_stress = head_position[1]  # Use the vertical position as an example

    return head_movement_stress

# Function to calculate mouth openness stress
def calculate_mouth_openness_stress(landmarks):
    # For example, you can use the mouth landmarks to detect the degree of mouth openness
    mouth_openness = landmarks[60:68]

    # Calculate mouth openness stress based on the difference in vertical position
    mouth_openness_stress = abs(np.mean(mouth_openness[:, 1]) - landmarks[152][1])  # Assuming landmark 152 represents the chin

    return mouth_openness_stress

# Function to calculate red face color feature
def calculate_red_face_color(landmarks, frame):
    # Extract the region around the nose
    nose_region = landmarks[27:36]

    # Calculate the average color in the region
    avg_color = np.mean(frame[nose_region[:, 1].astype(int), nose_region[:, 0].astype(int)], axis=0)

    # Calculate the redness of the face
    redness = avg_color[2]

    return redness

# Function to calculate eyebrow raise feature
def calculate_eyebrow_raise(landmarks):
    # Use the eyebrow landmarks to detect raised eyebrows
    left_eyebrow = landmarks[22:27]
    right_eyebrow = landmarks[17:22]

    # Calculate the average vertical position of the eyebrows
    avg_y_left = np.mean(left_eyebrow[:, 1])
    avg_y_right = np.mean(right_eyebrow[:, 1])

    # Calculate eyebrow raise feature based on the difference in vertical position
    eyebrow_raise = avg_y_left - avg_y_right

    return eyebrow_raise

def calculate_smile_detection(landmarks):
    # Use the lip landmarks to detect smile
    upper_lip = landmarks[50:53]
    lower_lip = landmarks[56:59]

    # Calculate the average vertical position of the lips
    avg_y_upper = np.mean(upper_lip[:, 1])
    avg_y_lower = np.mean(lower_lip[:, 1])

    # Calculate smile detection feature based on the absolute difference in vertical position
    smile_detection = abs(avg_y_upper - avg_y_lower)

    return smile_detection

@app.route('/get_stress_prediction', methods=['POST'])
def get_stress_prediction():
    data = request.json
    step_count = data['step_count']
    temperature = data['temperature']
    humidity = data['humidity']
    
    # Prepare features for prediction
    features = np.array([step_count, temperature, humidity]).reshape(1, -1)

    # Make prediction using the pre-trained model
    predicted_stress = rf_classifier.predict(features)

    return render_template('index.html', predicted_stress=int(predicted_stress[0]))

@app.route('/get_stress_data', methods=['POST'])
def process_frame_route():
    try:
        # Assuming the frame is sent as binary data
        frame = request.get_data()
        frame = np.frombuffer(frame, dtype=np.uint8).reshape(480, 640, 3)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face and facial landmarks
        results_face = face_detection.process(frame)
        results_landmarks = face_mesh.process(rgb_frame)

        if results_face.detections:
            # Get facial landmarks
            landmarks = np.array([[landmark.x, landmark.y] for landmark in results_landmarks.multi_face_landmarks[0].landmark])

            # Prepare features for prediction
            features = np.array([
                calculate_blink_stress(landmarks),
                calculate_eyebrow_stress(landmarks),
                calculate_lip_stress(landmarks),
                calculate_head_movement_stress(landmarks),
                calculate_mouth_openness_stress(landmarks),
                calculate_red_face_color(landmarks, frame),
                calculate_eyebrow_raise(landmarks),
                calculate_smile_detection(landmarks)
            ]).reshape(1, -1)

            # Make prediction using the pre-trained model
            predicted_stress = rf_classifier.predict(features)

            return jsonify({'predicted_stress': int(predicted_stress[0])})

        return jsonify({'predicted_stress': None, 'message': 'No face detected'})

    except Exception as e:
        print(f"Error in process_frame_route: {e}")
        return jsonify({'predicted_stress': -1, 'error': str(e)})

@app.route('/')
def index():
    return render_template('index.html', predicted_stress=None)

if __name__ == '__main__':
