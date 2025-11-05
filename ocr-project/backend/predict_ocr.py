import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from Trainer import AttentionLayer  # Absolute import (trainer is in the same folder)

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "ocr_models/ocr_model_production.keras"  # Path to the saved model
IMAGE_PATH = "test_images"  # Folder containing test/unlabeled images
IMG_HEIGHT = 64  # Height expected by the model (adjust based on training)
IMG_WIDTH = 160  # Width expected by the model (adjust based on training)

# ===============================
# LOAD MODEL
# ===============================
print("[INFO] Loading OCR model...")
model = load_model(MODEL_PATH, compile=False, custom_objects={"AttentionLayer": AttentionLayer})
print("[INFO] Model loaded successfully.")

# ===============================
# CHARACTER SET (must match your training setup)
# ===============================
characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
char_to_num = tf.keras.layers.StringLookup(vocabulary=list(characters), oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), invert=True
)

# ===============================
# IMAGE PREPROCESSING
# ===============================
def preprocess_image(image_path):
    """Preprocess a single image to match model input shape."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Resize image to 160x64 as expected by the model
    img = cv2.resize(img, (160, 64))  # Resize to (160, 64)
    img = img.astype(np.float32) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (H, W, 1)
    img = np.expand_dims(img, axis=0)   # Add batch dimension (1, H, W, 1)
    return img

# ===============================
# CTC DECODING
# ===============================
def decode_ctc(prediction):
    """Decode the CTC output into readable text (greedy decoding)."""
    pred_indices = np.argmax(prediction, axis=-1)[0]  # Get most probable indices
    prev_idx = -1
    decoded_text = ""
    for idx in pred_indices:
        if idx != prev_idx and idx < len(characters):  # Skip duplicates (CTC blank label)
            decoded_text += characters[idx]
        prev_idx = idx
    return decoded_text

# ===============================
# PREDICT TEXT FROM IMAGE
# ===============================
def predict_text(image_path):
    """Predict text from a single image."""
    img = preprocess_image(image_path)
    prediction = model.predict(img)  # Get prediction from the model
    decoded = decode_ctc(prediction)  # Decode the output to text
    return decoded

# ===============================
# RUN OCR ON FOLDER OF IMAGES
# ===============================
def run_ocr_on_folder(folder_path):
    """Process all images in the folder and predict text."""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Process image files
            img_path = os.path.join(folder_path, filename)
            try:
                text = predict_text(img_path)
                print(f"Predicted text for {filename}: {text}")
            except Exception as e:
                print(f"❌ Error on {filename}: {e}")

# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Folder not found: {IMAGE_PATH}")
    else:
        print("[INFO] Running OCR on test images...")
        run_ocr_on_folder(IMAGE_PATH)
