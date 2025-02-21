import cv2
import dlib
import numpy as np
import pygame
import os
import streamlit as st
import time
from scipy.spatial import distance
from drowsiness_model import get_trained_model

# 🔹 Initialize pygame for alert sound
# 🔹 Use Streamlit Audio Instead of Pygame
def play_alert_sound():
    st.audio("alert.mp3", format="audio/mp3", autoplay=True)


# 🔹 Load Dlib face detector & landmark predictor
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError("⚠️ Dlib model file not found!")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# 🔹 Load trained LSTM model
model = get_trained_model()

# 🔹 Constants
EYE_AR_THRESH = 0.22  # Eye closed threshold
HEAD_TILT_THRESH = 15  # Head tilt threshold (degrees)
DROWSY_TIME = 6  # 🔹 Changed from 10 sec to 6 sec
NOD_TIME = 5  # Time to check for head nodding
drowsy_start_time = None  
nod_start_time = None  
drowsy_alert_triggered = False  

# 🔹 Function to compute Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# 🔹 Function to compute Head Tilt Angle
def head_tilt_angle(landmarks):
    left_eye_center = np.array([(landmarks.part(36).x + landmarks.part(39).x) / 2,
                                (landmarks.part(36).y + landmarks.part(39).y) / 2])
    right_eye_center = np.array([(landmarks.part(42).x + landmarks.part(45).x) / 2,
                                 (landmarks.part(42).y + landmarks.part(45).y) / 2])

    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    angle = np.arctan2(dy, dx) * (180.0 / np.pi)  # Convert to degrees
    return abs(angle)

# 🔹 Streamlit UI
def main():
    st.title("🚦 Real-Time Drowsiness Detection")
    st.write("Press **Start Detection** to analyze drowsiness using webcam.")

    run = st.button("▶ Start Detection")

    if run:
        cap = cv2.VideoCapture(0)
        sequence = []
        global drowsy_start_time, nod_start_time, drowsy_alert_triggered  

        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Could not access the webcam!")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)

                left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
                right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
                
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                head_tilt = head_tilt_angle(landmarks)

                sequence.append([ear, head_tilt])
                if len(sequence) > 10:
                    sequence.pop(0)

                # ✅ Fix: Only predict when we have exactly 10 frames
                if len(sequence) == 10:
                    input_data = np.array(sequence).reshape((1, 10, 2))
                    prediction = model.predict(input_data)[0][0]
                else:
                    prediction = 0  # Default to 'not drowsy'

                # ✅ Detect Continuous Eye Closure (for 6 seconds)
                eye_closed = ear < EYE_AR_THRESH
                head_nodding = head_tilt > HEAD_TILT_THRESH

                if eye_closed:
                    if drowsy_start_time is None:
                        drowsy_start_time = time.time()
                    elif time.time() - drowsy_start_time >= DROWSY_TIME:
                        drowsy_alert_triggered = True
                else:
                    drowsy_start_time = None  

                # ✅ Detect Head Nodding (for 5 seconds)
                if head_nodding:
                    if nod_start_time is None:
                        nod_start_time = time.time()
                    elif time.time() - nod_start_time >= NOD_TIME:
                        drowsy_alert_triggered = True
                else:
                    nod_start_time = None  

                # 🔔 Trigger Alert if Either Condition Happens
                if drowsy_alert_triggered:
                    cv2.putText(frame, "🚨 DROWSY ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    play_alert_sound()
                    drowsy_alert_triggered = False  # Reset after alert

                cv2.putText(frame, f"EAR: {ear:.2f} Head Tilt: {head_tilt:.2f}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
