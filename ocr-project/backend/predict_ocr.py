import os
import cv2
import numpy as np
import tensorflow as tf

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "ocr_modelsv2/ocr_model_production_fp16.tflite"  # TFLite model
IMAGE_PATH = "test_images"  # Folder containing test/unlabeled images
IMG_HEIGHT = 64  # Height expected by the model
IMG_WIDTH = 160  # Width expected by the model

# ===============================
# CHARACTER SET (must match your training setup)
# ===============================
characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
char_to_num = tf.keras.layers.StringLookup(vocabulary=list(characters), oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), invert=True
)

# ===============================
# LOAD TFLITE MODEL
# ===============================
print("[INFO] Loading TFLite OCR model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"[INFO] Input details: {input_details[0]['shape']}")
print(f"[INFO] Output details: {output_details[0]['shape']}")
print("[INFO] Model loaded successfully.")

# ===============================
# IMAGE PREPROCESSING
# ===============================
def preprocess_image(image_path):
    """Preprocess a single image to match model input shape."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Resize image to 160x64 as expected by the model
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (H, W, 1)
    img = np.expand_dims(img, axis=0)   # Add batch dimension (1, H, W, 1)
    return img

# ===============================
# CTC DECODING
# ===============================
def decode_ctc(prediction):
    """Decode the CTC output into readable text (greedy decoding)."""
    # For TFLite, the output might be in a different format
    if len(prediction.shape) == 3:
        pred_indices = np.argmax(prediction, axis=-1)[0]  # Get most probable indices
    else:
        pred_indices = np.argmax(prediction, axis=-1)
    
    prev_idx = -1
    decoded_text = ""
    for idx in pred_indices:
        if idx != prev_idx and idx < len(characters) and idx > 0:  # Skip duplicates and blank (0)
            decoded_text += characters[idx]
        prev_idx = idx
    return decoded_text.strip()

# ===============================
# PREDICT TEXT FROM IMAGE (TFLite)
# ===============================
def predict_text(image_path):
    """Predict text from a single image using TFLite."""
    img = preprocess_image(image_path)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)
    
    # Run inference
    interpreter.invoke()
    
    # Get prediction output
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    decoded = decode_ctc(prediction)
    return decoded

# ===============================
# RUN OCR ON FOLDER OF IMAGES
# ===============================
def run_ocr_on_folder(folder_path):
    """Process all images in the folder and predict text."""
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    if not image_files:
        print(f"❌ No image files found in: {folder_path}")
        return
    
    for filename in image_files:
        img_path = os.path.join(folder_path, filename)
        try:
            text = predict_text(img_path)
            print(f"📄 {filename}: '{text}'")
        except Exception as e:
            print(f"❌ Error on {filename}: {e}")

# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Folder not found: {IMAGE_PATH}")
    elif not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
    else:
        print("[INFO] Running OCR on test images...")
        run_ocr_on_folder(IMAGE_PATH)