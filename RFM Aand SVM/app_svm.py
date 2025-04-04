import os
import cv2
import numpy as np
import pywt
import joblib
from skimage.feature import hog
from flask import Flask, request, render_template

app = Flask(__name__)

# Constants
MODEL_PATH = "svm_model.pkl"
SCALER_PATH = "scaler.pkl"
UPLOAD_FOLDER = "uploads"
FEATURE_SIZE = 8100

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model & scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def extract_features(image_path):
    """
    Extracts HOG + Wavelet features ensuring exactly 8100 features.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        img = cv2.resize(img, (128, 128))
        img = cv2.equalizeHist(img)
        img = img.astype(np.float32) / 255.0

        # Extract HOG features
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys',
                           feature_vector=True)

        # Extract Wavelet features
        coeffs = pywt.wavedec2(img, 'db1', level=2)
        wavelet_features = np.hstack([c.flatten() if isinstance(c, np.ndarray) 
                                      else np.hstack([x.flatten() for x in c]) 
                                      for c in coeffs])

        # Ensure feature vector is exactly 8100
        required_length = FEATURE_SIZE - len(hog_features)
        wavelet_features = wavelet_features[:required_length]

        return np.hstack((hog_features, wavelet_features))
    
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("result.html", result="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template("result.html", result="No file selected")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    features = extract_features(file_path)
    if features is None or len(features) != FEATURE_SIZE:
        return render_template("result.html", result="Feature extraction failed")

    features_scaled = scaler.transform([features])
    probabilities = model.predict_proba(features_scaled)[0]  # Get probabilities
    prediction = np.argmax(probabilities)  # Get the class with the highest probability
    confidence = probabilities[prediction] * 100  # Convert to percentage
    
    labels = {0: "Normal", 1: "Abnormal", 2: "Myocardial Infarction"}
    return render_template("result.html", result=labels.get(prediction, "Unknown"), confidence=f"{confidence:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)
