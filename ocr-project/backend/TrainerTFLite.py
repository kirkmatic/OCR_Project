import os
import shutil
import string
import datetime
import random
import json
import difflib
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense,   
    BatchNormalization, SpatialDropout2D, Lambda, GRU
)
from tensorflow.keras.callbacks import (
    TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
)
from tensorflow.keras import backend as K
from tensorflow_addons.optimizers import AdamW
from PIL import Image, ImageDraw, ImageFont

CHARACTERS = string.ascii_letters + string.digits + " -'.,:"
NUM_CHARS = len(CHARACTERS)
BLANK_TOKEN = NUM_CHARS
IMAGE_WIDTH, IMAGE_HEIGHT = 160, 64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_dir = os.path.join(BASE_DIR, 'ocr_dataset')
logs_dir = os.path.join(BASE_DIR, 'ocr_logs')
models_dir = os.path.join(BASE_DIR, 'ocr_modelsv2')
sample_logs_dir = os.path.join(BASE_DIR, 'sample_logs')

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(sample_logs_dir, exist_ok=True)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def prepare_data(samples):
    images = []
    texts = []
    feature_width = IMAGE_WIDTH // 4
    for img_path, text in samples:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img = (img / 255.0).astype(np.float32)
        img = np.expand_dims(img, axis=-1)
        text_labels = [CHARACTERS.index(c) for c in text if c in CHARACTERS]
        if len(text_labels) > feature_width:
            print(f"Skipping sample: label too long ({len(text_labels)} > {feature_width}) for {img_path}")
            continue
        images.append(img)
        texts.append(text_labels)
    max_text_len = max(len(t) for t in texts)
    padded_texts = np.ones((len(texts), max_text_len), dtype='int32') * BLANK_TOKEN
    for i, text in enumerate(texts):
        padded_texts[i, :len(text)] = text
    feature_width = IMAGE_WIDTH // 4
    input_length = np.ones((len(images), 1), dtype='int32') * feature_width
    label_length = np.array([[len(t)] for t in texts], dtype='int32')
    return np.array(images), padded_texts, input_length, label_length

def build_ocr_model_tflite_compatible():
    """
    Simplified model architecture that's TFLite compatible
    Removes custom layers and complex operations
    """
    input_img = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name='input_image')
    
    # CNN Feature Extractor
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)  # Only pool vertically
    
    # Prepare for RNN - TFLite compatible reshape
    conv_shape = K.int_shape(x)
    time_steps = conv_shape[2]  # This should be 40 (160/4)
    features = conv_shape[1] * conv_shape[3]  # 8 * 128 = 1024
    
    x = Reshape((time_steps, features))(x)
    
    # RNN - Use GRU instead of LSTM for better TFLite compatibility
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.2))(x)
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.2))(x)
    
    # Output layer
    output = Dense(NUM_CHARS + 1, activation='softmax', name='output')(x)
    
    # Training model with CTC
    labels = Input(name='labels', shape=(None,), dtype='int32')
    input_length = Input(name='input_length', shape=(1,), dtype='int64')
    label_length = Input(name='label_length', shape=(1,), dtype='int64')
    
    loss_out = Lambda(ctc_lambda_func, name='ctc')([output, labels, input_length, label_length])
    
    train_model = Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=loss_out
    )
    
    train_model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss={'ctc': lambda y_true, y_pred: y_pred}
    )
    
    # Prediction model (for inference)
    pred_model = Model(inputs=input_img, outputs=output)
    
    return train_model, pred_model, time_steps

def convert_to_tflite_with_flex(pred_model):
    """
    Convert model to TFLite with Flex ops for compatibility
    """
    # Save the model first
    keras_model_path = os.path.join(models_dir, "ocr_model_keras.keras")
    pred_model.save(keras_model_path)
    print(f"Saved Keras model to: {keras_model_path}")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(pred_model)
    
    # Enable experimental features for better compatibility
    converter.experimental_new_converter = True
    converter.experimental_enable_resource_variables = True
    
    # Try different optimization strategies
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Allow custom ops (important for CTC decoding if needed)
    converter.allow_custom_ops = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS,    # TensorFlow ops (flex)
    ]
    
    try:
        tflite_model = converter.convert()
        
        # Save the TFLite model
        tflite_path = os.path.join(models_dir, "ocr_model_production_fp16.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ Successfully converted to TFLite: {tflite_path}")
        print(f"Model size: {len(tflite_model) / (1024*1024):.2f} MB")
        
        return tflite_path
        
    except Exception as e:
        print(f"❌ TFLite conversion failed: {e}")
        print("Trying alternative conversion...")
        return convert_to_tflite_simple(pred_model)

def convert_to_tflite_simple(pred_model):
    """
    Simple conversion without optimizations
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(pred_model)
    converter.allow_custom_ops = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(models_dir, "ocr_model_simple.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Saved simple TFLite model: {tflite_path}")
    
    return tflite_path

def test_tflite_model(tflite_path, test_images):
    """
    Test the TFLite model to ensure it works
    """
    print("\n🧪 Testing TFLite model...")
    
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"📊 Input details: {input_details[0]}")
    print(f"📊 Output details: {output_details[0]}")
    
    # Test with a sample image
    if len(test_images) > 0:
        test_image = test_images[0:1]  # Take first image
        print(f"🧪 Test image shape: {test_image.shape}")
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], test_image)
        
        # Run inference
        interpreter.invoke()
        
        # Get prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"✅ TFLite model works! Output shape: {output_data.shape}")
        
        return True
    else:
        print("❌ No test images available")
        return False

# ... (keep the existing TerminalLogger, ImageLogger, ValidationCallback classes) ...

class TerminalLogger(tf.keras.callbacks.Callback):
    def __init__(self, validation_callback=None):
        super().__init__()
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_filename = f"{now}-result.txt"
        self.log_filepath = os.path.join(logs_dir, self.log_filename)
        self.validation_callback = validation_callback

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_line = (
            f"Epoch {epoch+1:2d}/{self.params['epochs']} "
            f"- accuracy: {logs.get('accuracy', 0):.4f} "
            f"- loss: {logs.get('loss', 0):.4f} "
            f"- val_accuracy: {logs.get('val_accuracy', 0):.4f} "
            f"- val_loss: {logs.get('val_loss', 0):.4f}"
        )
        if self.validation_callback is not None:
            log_line += (
                f" - val_word_accuracy: {getattr(self.validation_callback, 'last_word_acc', 0):.4f}"
                f" - val_precision: {getattr(self.validation_callback, 'last_precision', 0):.4f}"
                f" - val_recall: {getattr(self.validation_callback, 'last_recall', 0):.4f}"
                f" - val_f1_score: {getattr(self.validation_callback, 'last_f1_score', 0):.4f}"
            )
        with open(self.log_filepath, "a") as f:
            f.write(log_line + "\n")


class ImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, images, log_dir):
        super().__init__()
        self.images = images
        self.log_dir = log_dir

    def on_train_begin(self, logs=None):
        imgs = self.images[:10]
        imgs = np.array(imgs)
        if imgs.ndim == 3:
            imgs = np.expand_dims(imgs, -1)
        file_writer = tf.summary.create_file_writer(self.log_dir)
        with file_writer.as_default():
            tf.summary.image("Training samples", imgs, step=0)


class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, pred_model, X_val, y_val, log_dir):
        super().__init__()
        self.pred_model = pred_model
        self.X_val = X_val
        self.y_val = y_val
        self.writer = tf.summary.create_file_writer(log_dir)
        self.best_accuracy = 0.0
        self.best_char_accuracy = 0.0

    def decode_prediction(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        decoded, _ = K.ctc_decode(pred, input_length=input_len, greedy=True, beam_width=5)
        decoded_text = ''.join([CHARACTERS[idx] for idx in K.get_value(decoded[0][0]) if idx != -1 and idx < NUM_CHARS])
        return decoded_text

    def on_epoch_end(self, epoch, logs=None):
        sample_images = []
        correct = 0
        TP, FP, FN = 0, 0, 0
        total = min(10, len(self.X_val))
        char_accs = []
        for i in range(total):
            img = (self.X_val[i] * 255).astype(np.uint8).squeeze()
            pred = self.pred_model.predict(np.expand_dims(self.X_val[i], axis=0))
            pred_text = self.decode_prediction(pred)
            true_indices = self.y_val[i]
            true_text = ''.join([CHARACTERS[idx] for idx in true_indices if idx < NUM_CHARS])
            if pred_text == true_text:
                correct += 1
                TP += 1
            else:
                FP += 1
                FN += 1
            img_rgb = np.stack([img]*3, axis=-1)
            pil_img = Image.fromarray(img_rgb)
            canvas = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT + 30), "white")
            canvas.paste(pil_img, (0, 0))
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()
            draw.text((5, IMAGE_HEIGHT + 5), f"True: {true_text}", fill=(0, 0, 0), font=font)
            color = (0, 200, 0) if pred_text == true_text else (200, 0, 0)
            draw.text((5, IMAGE_HEIGHT + 15), f"Pred: {pred_text}", fill=color, font=font)
            img_arr = np.array(canvas).astype(np.float32) / 255.0
            sample_images.append(img_arr)
            char_accs.append(char_accuracy(pred_text, true_text))
        avg_char_acc = np.mean(char_accs)
        word_acc = correct / total if total > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        with self.writer.as_default():
            tf.summary.image("Image + Text OCR Results", np.stack(sample_images), step=epoch)
            tf.summary.scalar('val_word_accuracy', word_acc, step=epoch)
            tf.summary.scalar('val_precision', precision, step=epoch)
            tf.summary.scalar('val_recall', recall, step=epoch)
            tf.summary.scalar('val_f1_score', f1_score, step=epoch)
            tf.summary.scalar('val_char_accuracy', avg_char_acc, step=epoch)
            self.writer.flush()

def load_samples(label_csv_path, images_folder):
    df = pd.read_csv(label_csv_path)
    samples = []
    for _, row in df.iterrows():
        img_filename = row['IMAGE']
        label = str(row['MEDICINE_NAME'])
        img_path = os.path.join(images_folder, img_filename)
        if os.path.exists(img_path):
            samples.append((img_path, label))
        else:
            print("Missing image:", img_path)
    return samples

# ... (previous imports and setup code remains the same) ...

if __name__ == "__main__":
    # Load data
    base_dir = dataset_dir
    all_samples = load_samples(
        os.path.join(dataset_dir, "Training", "training_labels.csv"),
        os.path.join(dataset_dir, "Training", "training_words")
    )
    print(f"Total samples loaded: {len(all_samples)}")
    
    if len(all_samples) == 0:
        print("❌ No training samples found!")
        exit(1)
        
    random.shuffle(all_samples)
    num_total = len(all_samples)
    num_train = int(0.8 * num_total)
    num_val = int(0.1 * num_total)
    
    train_samples = all_samples[:num_train]
    val_samples = all_samples[num_train:num_train+num_val]
    test_samples = all_samples[num_train+num_val:]
    
    # Prepare data
    X_train, y_train, il_train, ll_train = prepare_data(train_samples)
    X_val, y_val, il_val, ll_val = prepare_data(val_samples)
    X_test, y_test, il_test, ll_test = prepare_data(test_samples)
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    
    # Build TFLite compatible model
    train_model, pred_model, time_steps = build_ocr_model_tflite_compatible()
    
    print(f"📐 Model time steps: {time_steps}")
    print(f"📐 Number of characters: {NUM_CHARS}")
    print(f"📐 Expected output shape: (batch_size, {time_steps}, {NUM_CHARS + 1})")
    
    # Train the model
    checkpoint_path = os.path.join(models_dir, "best_model.keras")
    
    # FIXED: Add TensorBoard callback and fix validation callback
    callbacks = [
        TensorBoard(log_dir=logs_dir, histogram_freq=1),  # ✅ Added this line
        TerminalLogger(),
        ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            save_weights_only=False,
            monitor='val_loss',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
    ]
    
    print("🚀 Starting training...")
    print(f"📊 Logs will be saved to: {logs_dir}")
    
    history = train_model.fit(
        [X_train, y_train, il_train, ll_train],
        np.zeros(len(X_train)),
        validation_data=([X_val, y_val, il_val, ll_val], np.zeros(len(X_val))),
        epochs=100,  # Reduced for testing
        batch_size=32,  # Smaller batch size
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model
    if os.path.exists(checkpoint_path):
        print("📥 Loading best model...")
        pred_model = load_model(checkpoint_path, custom_objects={'tf': tf})
    
    # Convert to TFLite
    print("🔄 Converting to TFLite...")
    tflite_path = convert_to_tflite_with_flex(pred_model)
    
    # Test the TFLite model
    test_tflite_model(tflite_path, X_test)
    
    print(f"✅ Training completed! TFLite model saved to: {tflite_path}")
    print(f"📊 Training logs saved to: {logs_dir}")