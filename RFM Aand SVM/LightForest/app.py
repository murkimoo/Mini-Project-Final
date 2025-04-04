import os
import numpy as np
import cv2
import joblib
from flask import Flask, request, render_template, flash
from skimage.feature import hog
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flash messages

# Constants
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
UPLOAD_FOLDER = 'uploads'
IMAGE_SIZE = (128, 128)
CLASSES = ["Normal", "HB", "MI"]

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class LightForest:
    def __init__(self, rf_model, lgbm_model, rf_acc, lgbm_acc):
        self.rf_model = rf_model
        self.lgbm_model = lgbm_model
        self.scaler = None  # Will be loaded from the model
        total_acc = rf_acc + lgbm_acc
        self.rf_weight = rf_acc / total_acc
        self.lgbm_weight = lgbm_acc / total_acc

    def predict(self, X):
        try:
            if self.scaler:
                X = self.scaler.transform(X)
            rf_preds = self.rf_model.predict_proba(X)
            lgbm_preds = self.lgbm_model.predict_proba(X)
            final_preds = (rf_preds * self.rf_weight + lgbm_preds * self.lgbm_weight)
            return np.argmax(final_preds, axis=1)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to load image")

        # Preprocessing
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        img = cv2.equalizeHist(img)
        img = img.astype(np.float32) / 255.0

        # Extract HOG features
        features, _ = hog(img, 
                         orientations=8,
                         pixels_per_cell=(16, 16),
                         cells_per_block=(2, 2),
                         block_norm='L2-Hys',
                         feature_vector=True,
                         visualize=True)
        
        return np.array(features).reshape(1, -1)
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return None

# Load the model
try:
    MODEL_PATH = "lightforest_model.pkl"
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash('Model not loaded properly', 'error')
        return render_template('index.html')

    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return render_template('index.html')

    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return render_template('index.html')

    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a PNG or JPG image.', 'error')
        return render_template('index.html')

    try:
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Extract features
        features = extract_features(file_path)
        if features is None:
            raise ValueError("Feature extraction failed")

        # Make prediction
        prediction = model.predict(features)
        if prediction is None:
            raise ValueError("Prediction failed")

        result = CLASSES[prediction[0]]
        
        # Clean up uploaded file
        os.remove(file_path)

        return render_template('result.html', 
                             result=result, 
                             confidence=100)  # Add confidence score if available

    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'error')
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)