import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, accuracy_score

import joblib
import kagglehub

# Download latest version of facial emotion recognition dataset from Kaggle
path = kagglehub.dataset_download("msambare/fer2013")

train_dir = os.path.join(path, 'train')
test_dir = os.path.join(path, 'test')

emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}

def load_images_from_folder(folder):
    images = []
    labels = []
    for emotion, label in emotion_map.items():
        emotion_folder = os.path.join(folder, emotion)
        for img_name in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_name)
            # Load the image in grayscale
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # Resize to 48x48 (if necessary)
                image = cv2.resize(image, (48, 48))
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load the training and testing data
X_train, y_train = load_images_from_folder(train_dir)
X_test, y_test = load_images_from_folder(test_dir)

print(f"Training data: {X_train.shape}, Training labels: {y_train.shape}")
print(f"Testing data: {X_test.shape}, Testing labels: {y_test.shape}")

# Normalize to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0


# Pre-processing of the dataset is complete. Now we extract LBP and HOG features.

# Parameters:
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# Helper function that takes an image as input and returns a concatenated array of
# all of its LBP and HOG features.
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

# Function to get all images' features from a dataset
def extract_features_from_dataset(images):
    features = []
    for image in images:
        features.append(extract_features(image))
    return np.array(features)


print("Extracting features from training data...")
X_train_features = extract_features_from_dataset(X_train)
print("Extracting features from testing data...")
X_test_features = extract_features_from_dataset(X_test)

print(f"Training features shape: {X_train_features.shape}")
print(f"Testing features shape: {X_test_features.shape}")

# Normalize the features
scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_test_features = scaler.transform(X_test_features)

# Train a SVM classifier. We choose SVC but limited the degree of the kernel
# to reduce training time.
print("Training the SVM classifier...")
# classifier = LinearSVC(C=0.1, random_state=42, max_iter=20000)
classifier = SVC(kernel='poly', degree=3, random_state=42)
classifier.fit(X_train_features, y_train)

# Evaluate
print("Evaluating...")
y_pred = classifier.predict(X_test_features)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=list(emotion_map.keys())))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


# Save the trained classifier and the scaler
joblib.dump(classifier, "emotion_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Classifier and scaler saved successfully!")