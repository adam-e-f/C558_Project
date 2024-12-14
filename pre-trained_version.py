import cv2
import dlib
import threading
import time
import numpy as np
from deepface import DeepFace

# Load the pre-trained face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Open the video capture device (use 0 for webcam)
video_path = 0
cap = cv2.VideoCapture(video_path)

# Check if the video capture object is initialized
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Global variable to store the predicted emotion
predicted_emotion = "No Prediction"
lock = threading.Lock()

# Function to extract features and predict emotion
def predict_emotion():
    global predicted_emotion
    print("Emotion prediction thread has started.")  # Confirm that the thread has started
    
    while True:
        # Capture a frame every 0.5 seconds
        time.sleep(0.5)
        ret, frame = cap.read()
        if not ret:
            continue  # Skip if the frame is not available

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame using the dlib face detector
        faces = detector(gray)

        if len(faces) > 0:
            # Consider only the first detected face for simplicity
            face = faces[0]
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

            # Extract the face region of interest (ROI)
            face_roi = frame[y:y + h, x:x + w]

            try:
                # Perform emotion analysis on the face ROI
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                # Determine the dominant emotion
                with lock:
                    predicted_emotion = result[0]['dominant_emotion']
            except:
                print("Error in emotion analysis. Skipping this frame.")
                continue

# Start a background thread to predict emotions
emotion_thread = threading.Thread(target=predict_emotion, daemon=True)
emotion_thread.start()

# Process the video frame by frame for displaying landmarks
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if no more frames are available

    frame = cv2.resize(frame, (640, 480))  # Resize image for faster processing if laggy

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using the dlib detector
    faces = detector(gray)

    # Loop through each face detected
    for face in faces:
        # Get the landmarks for the face using dlib
        landmarks = predictor(gray, face)

        # Draw landmarks on the face
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw green dots for landmarks

    # Display the predicted emotion at a fixed location at the top of the screen
    with lock:
        cv2.putText(frame, f"Emotion: {predicted_emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Landmarks and Emotion Prediction", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()