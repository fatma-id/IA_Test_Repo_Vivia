import cv2
import mediapipe as mp
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize MediaPipe Face Detection and Facial Landmarks
mp_face = mp.solutions.face_detection
mp_landmarks = mp.solutions.face_mesh

face_detection = mp_face.FaceDetection()
face_mesh = mp_landmarks.FaceMesh()
# Function to calculate blink stress
def calculate_blink_stress(landmarks):
    #use the eye landmarks to detect blinking
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
# Function to calculate blink stress
def calculate_blink_stress(landmarks):
    #use the eye landmarks to detect blinking
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


# Initialize lists to store stress levels for each factor
blink_stress = []
eyebrow_stress = []
lip_stress = []
head_movement_stress = []
mouth_openness_stress = []
red_face_color = []
eyebrow_raise = []
smile_detection = []
final_stress = []

# Function to process each frame in real-time
def process_frame(frame):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face and facial landmarks
    results_face = face_detection.process(frame)
    results_landmarks = face_mesh.process(rgb_frame)

    if results_face.detections:
        # Get facial landmarks
        landmarks = np.array([[landmark.x, landmark.y] for landmark in results_landmarks.multi_face_landmarks[0].landmark])

        # Calculate stress levels
        stress_blink = calculate_blink_stress(landmarks)
        stress_eyebrow = calculate_eyebrow_stress(landmarks)
        stress_lip = calculate_lip_stress(landmarks)
        stress_head_movement = calculate_head_movement_stress(landmarks)
        stress_mouth_openness = calculate_mouth_openness_stress(landmarks)
        stress_red_face_color = calculate_red_face_color(landmarks, frame)
        stress_eyebrow_raise = calculate_eyebrow_raise(landmarks)
        stress_smile_detection = calculate_smile_detection(landmarks)

        # Append stress levels to lists
        blink_stress.append(stress_blink)
        eyebrow_stress.append(stress_eyebrow)
        lip_stress.append(stress_lip)
        head_movement_stress.append(stress_head_movement)
        mouth_openness_stress.append(stress_mouth_openness)
        red_face_color.append(stress_red_face_color)
        eyebrow_raise.append(stress_eyebrow_raise)
        smile_detection.append(stress_smile_detection)

        # Calculate final aggregated stress based on equal weights for each factor
        final_stress.append(
            0.15 * stress_blink +
            0.15 * stress_eyebrow +
            0.15 * stress_lip +
            0.15 * stress_head_movement +
            0.15 * stress_mouth_openness +
            0.15 * stress_red_face_color +
            0.15 * stress_eyebrow_raise +
            0.15 * stress_smile_detection
        )

        # Update the real-time plots
        update_plots()

# Function to update real-time plots
def update_plots():
    # Update plots for each factor
    axes[0, 0].clear()
    axes[0, 0].plot(blink_stress)
    axes[0, 0].set_title('Blink Stress')

    axes[0, 1].clear()
    axes[0, 1].plot(eyebrow_stress)
    axes[0, 1].set_title('Eyebrow Stress')

    axes[0, 2].clear()
    axes[0, 2].plot(lip_stress)
    axes[0, 2].set_title('Lip Stress')

    axes[1, 0].clear()
    axes[1, 0].plot(head_movement_stress)
    axes[1, 0].set_title('Head Movement Stress')

    axes[1, 1].clear()
    axes[1, 1].plot(mouth_openness_stress)
    axes[1, 1].set_title('Mouth Openness Stress')

    axes[1, 2].clear()
    axes[1, 2].plot(red_face_color)
    axes[1, 2].set_title('Red Face Color')

    axes[2, 0].clear()
    axes[2, 0].plot(eyebrow_raise)
    axes[2, 0].set_title('Eyebrow Raise')

    axes[2, 1].clear()
    axes[2, 1].plot(smile_detection)
    axes[2, 1].set_title('Smile Detection')

    axes[2, 2].clear()
    axes[2, 2].plot(final_stress)
    axes[2, 2].set_title('Final Aggregated Stress')

    # Adjust layout
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

# Open the video stream
cap = cv2.VideoCapture(0)

# Plotting stress levels over time
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

# Run the real-time processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    process_frame(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all windows
cv2.destroyAllWindows()

