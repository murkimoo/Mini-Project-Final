import os
import numpy as np
import cv2
import joblib
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils import shuffle
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Constants
DATASET_PATH = "ecg"
CLASSES = ["Normal", "HB", "MI"]
IMAGE_SIZE = (128, 128)


def extract_features(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        img = cv2.equalizeHist(img)
        img = img.astype(np.float32) / 255.0
        features, _ = hog(img, orientations=8,
                         pixels_per_cell=(16, 16),
                         cells_per_block=(2, 2),
                         block_norm='L2-Hys',
                         feature_vector=True,
                         visualize=True)
        return features
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def process_image(args):
    img_path, label = args
    features = extract_features(img_path)
    return (features, label) if features is not None else None


def load_dataset():
    X, y = [], []
    min_samples = float('inf')
    print("Analyzing dataset structure...")
    for class_name in CLASSES:
        class_path = os.path.join(DATASET_PATH, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))])
            min_samples = min(min_samples, count)
            print(f"Found {count} images in class {class_name}")
    print(f"\nBalancing dataset to {min_samples} samples per class...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        for label, class_name in enumerate(CLASSES):
            class_path = os.path.join(DATASET_PATH, class_name)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) 
                             if f.endswith(('.jpg', '.png'))][:min_samples]
                args = [(os.path.join(class_path, img), label) for img in image_files]
                results = list(executor.map(process_image, args))
                valid_results = [r for r in results if r is not None]
                X.extend([f for f, _ in valid_results])
                y.extend([l for _, l in valid_results])
                print(f"Processed {len(valid_results)} images for class {class_name}")
    return shuffle(np.array(X), np.array(y), random_state=42)


print("Loading and preprocessing dataset...")
X, y = load_dataset()

print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining Random Forest classifier...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"Random Forest Accuracy: {rf_acc:.2f}")

print("\nTraining LightGBM classifier...")
lgbm = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=31,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
lgbm.fit(X_train, y_train)
lgbm_preds = lgbm.predict(X_test)
lgbm_acc = accuracy_score(y_test, lgbm_preds)
print(f"LightGBM Accuracy: {lgbm_acc:.2f}")


class LightForest:
    def __init__(self, rf_model, lgbm_model, rf_acc, lgbm_acc):
        self.rf_model = rf_model
        self.lgbm_model = lgbm_model
        self.scaler = scaler
        total_acc = rf_acc + lgbm_acc
        self.rf_weight = rf_acc / total_acc
        self.lgbm_weight = lgbm_acc / total_acc

    def predict(self, X_scaled):
        rf_preds = self.rf_model.predict_proba(X_scaled)
        lgbm_preds = self.lgbm_model.predict_proba(X_scaled)
        final_preds = (rf_preds * self.rf_weight + lgbm_preds * self.lgbm_weight)
        return np.argmax(final_preds, axis=1)

    def predict_proba(self, X_scaled):
        rf_preds = self.rf_model.predict_proba(X_scaled)
        lgbm_preds = self.lgbm_model.predict_proba(X_scaled)
        return (rf_preds * self.rf_weight + lgbm_preds * self.lgbm_weight)


print("\nCreating and evaluating ensemble model...")
lightforest = LightForest(rf, lgbm, rf_acc, lgbm_acc)
y_pred = lightforest.predict(X_test)
ensemble_acc = accuracy_score(y_test, y_pred)
print(f"LightForest Accuracy: {ensemble_acc:.2f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

print("\nPlotting AUC-ROC curves...")
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
probs = lightforest.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(CLASSES)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 7))
colors = ['blue', 'green', 'red']
for i, color in zip(range(len(CLASSES)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve for {CLASSES[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve for LightForest')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png")  
plt.show()

# Print macro and micro AUC
macro_auc = roc_auc_score(y_test_bin, probs, average='macro')
micro_auc = roc_auc_score(y_test_bin, probs, average='micro')
print(f"\nMacro-Averaged AUC: {macro_auc:.2f}")
print(f"Micro-Averaged AUC: {micro_auc:.2f}")

print("\nSaving model...")
joblib.dump(lightforest, "lightforest_model.pkl")
print("Model saved as 'lightforest_model.pkl'")
