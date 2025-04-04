import os
import numpy as np
import cv2
import joblib
import pywt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')

# Constants
dataset_path = "ecg"
classes = ["normal", "HB", "MI"]
image_size = (128, 128)
FEATURE_SIZE = 8100

def extract_features(image_path):
    """
    Extracts HOG + Wavelet features ensuring exactly 8100 features.
    """
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

        # Ensure feature vector is exactly 8100
        required_length = FEATURE_SIZE - len(hog_features)
        wavelet_features = wavelet_features[:required_length]
        
        return np.hstack((hog_features, wavelet_features))
    
    except Exception as e:
        print(f"Feature extraction error for {image_path}: {e}")
        return None

def process_image(args):
    img_path, label = args
    features = extract_features(img_path)
    return (features, label) if features is not None else None

def load_dataset():
    """
    Loads and balances ECG dataset.
    """
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

            image_files = os.listdir(class_path)[:min_samples]  # Balance dataset
            args = [(os.path.join(class_path, img), label) for img in image_files]
            results = list(executor.map(process_image, args))
            valid_results = [r for r in results if r is not None]

            X.extend([f for f, _ in valid_results])
            y.extend([l for _, l in valid_results])

    if len(X) == 0:
        raise ValueError("No valid images found. Check dataset path and image formats.")

    return shuffle(np.array(X), np.array(y), random_state=42)

# Load & Preprocess Dataset
print("Loading dataset...")
X, y = load_dataset()
print(f"Dataset loaded: {len(X)} samples")

print("Scaling features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM with Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_svm():
    best_model, best_score = None, 0
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced')
        model.fit(X_tr, y_tr)
        val_score = model.score(X_val, y_val)

        if val_score > best_score:
            best_model, best_score = model, val_score

    return best_model

# Train and evaluate SVM
print("Training SVM...")
svm_model = train_svm()

print("Evaluating model...")
y_pred = svm_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=classes))

# Save model & scaler
print("Saving model...")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model saved as svm_model.pkl")
