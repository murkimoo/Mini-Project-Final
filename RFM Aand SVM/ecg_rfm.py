import os
import numpy as np
import cv2
import joblib
import pywt
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')

# Constants
dataset_path = "ecg"
classes = ["normal", "HB", "MI"]
image_size = (128, 128)
FEATURE_SIZE = 8100

# Feature extraction function
def extract_features(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping invalid image: {image_path}")
            return None

        img = cv2.resize(img, image_size)
        img = cv2.equalizeHist(img)
        img = img.astype(np.float32) / 255.0

        # HOG Features
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys',
                           feature_vector=True)

        # Wavelet Transform Features
        coeffs = pywt.wavedec2(img, 'db1', level=2)
        wavelet_features = np.hstack([c.flatten() if isinstance(c, np.ndarray) 
                                      else np.hstack([x.flatten() for x in c]) 
                                      for c in coeffs])
        
        required_length = FEATURE_SIZE - len(hog_features)
        wavelet_features = wavelet_features[:required_length]
        
        return np.hstack((hog_features, wavelet_features))
    except Exception as e:
        print(f"Feature extraction error for {image_path}: {e}")
        return None

# Load dataset
def load_dataset():
    X, y = [], []
    min_samples = min([len(os.listdir(os.path.join(dataset_path, c))) or 0 for c in classes])

    if min_samples == 0:
        raise ValueError("Dataset folder is empty. Check `ecg/` directory.")

    with ThreadPoolExecutor(max_workers=8) as executor:
        for label, class_name in enumerate(classes):
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Folder {class_path} does not exist. Skipping.")
                continue

            image_files = os.listdir(class_path)[:min_samples]
            args = [(os.path.join(class_path, img), label) for img in image_files]
            results = list(executor.map(lambda x: (extract_features(x[0]), x[1]), args))
            valid_results = [r for r in results if r[0] is not None]

            X.extend([f for f, _ in valid_results])
            y.extend([l for _, l in valid_results])

    return shuffle(np.array(X), np.array(y), random_state=42)

# Load dataset
print("Loading dataset...")
X, y = load_dataset()
print(f"Dataset loaded: {len(X)} samples")

# Feature scaling
print("Scaling features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest classifier
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=classes))

# Save model and scaler
print("Saving model...")
joblib.dump(rf_model, "randomforest.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model saved as randomforest.pkl")
