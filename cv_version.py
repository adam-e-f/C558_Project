import time
import joblib
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

# Load the Haar cascade for facial detection (Viola-Jones detector)
viola_jones = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the saved classifier and scaler
classifier = joblib.load("emotion_classifier.pkl")
scaler = joblib.load("scaler.pkl")

# Parameters for LBP and HOG
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# Helper function that takes an image as input and returns a concatenated array of
# all of its LBP and HOG features. (Copied from train_classifier.py)
def extract_features(image):
    # Compute Local Binary Pattern (LBP)
    lbp = local_binary_pattern(image, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(),
                               bins=np.arange(0, LBP_POINTS + 3),
                               range=(0, LBP_POINTS + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize the histogram

    # Compute Histogram of Oriented Gradients (HOG)
    hog_features = hog(image, orientations=HOG_ORIENTATIONS,
                       pixels_per_cell=HOG_PIXELS_PER_CELL,
                       cells_per_block=HOG_CELLS_PER_BLOCK, feature_vector=True)
    
    # Combine LBP and HOG features
    combined_features = np.hstack([lbp_hist, hog_features])
    return combined_features

def preprocess_face(face):
    # Resize to match classifier's input dimensions
    face = cv2.resize(face, (48, 48))
    
    # Apply bilateral filtering for edge-preserving smoothing
    face = cv2.bilateralFilter(face, d=9, sigmaColor=75, sigmaSpace=75)
    
    return face


def predict_emotion(image):
    # Convert image to grayscale, 48x48 to match the data the classifier
    # was trained on
    image = cv2.resize(image, (48, 48))
    
    # Extract features and normalize
    features = extract_features(image)
    features = scaler.transform([features])
    
    # Predict emotion
    emotion_label = classifier.predict(features)[0]
    return emotion_label


emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

# Use 0 for webcam
T = 0.5
video_feed = cv2.VideoCapture(0)
last_classification_time = -T

while True:
    ret, frame = video_feed.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = viola_jones.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        # Use the first detected face. Ideally there will only be one on screen.
        (x, y, w, h) = faces[0]
        
        # Extract and preprocess the face ROI
        face_roi = gray_frame[y:y + h, x:x + w]
        face_roi = preprocess_face(face_roi)

        # Predict emotion at specified intervals
        current_time = time.time()
        if current_time - last_classification_time >= T:
            last_classification_time = current_time
            emotion_label = predict_emotion(face_roi)
            emotion_name = emotion_map[emotion_label]

    # Display predicted emotion in red on the frame
    cv2.putText(frame, f"Emotion: {emotion_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Show the video feed
    cv2.imshow("Emotion Recognition", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_feed.release()
cv2.destroyAllWindows()