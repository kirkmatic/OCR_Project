import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
)
from TrainerComponent.model_builder import build_ocr_model_tflite_compatible
from TrainerComponent.data_utils import prepare_data, load_samples, NUM_CHARS, IMAGE_WIDTH, IMAGE_HEIGHT
from TrainerComponent.callbacks import TerminalLogger, ValidationCallback
from TrainerComponent.tflite_utils import convert_to_tflite_with_flex, test_tflite_model
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(BASE_DIR, 'ocr_dataset')
logs_dir = os.path.join(BASE_DIR, 'ocr_logs')
models_dir = os.path.join(BASE_DIR, 'ocr_modelsv2')
sample_logs_dir = os.path.join(BASE_DIR, 'sample_logs')

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(sample_logs_dir, exist_ok=True)

if os.path.exists(logs_dir):
    shutil.rmtree(logs_dir)
os.makedirs(logs_dir, exist_ok=True)

if __name__ == "__main__":
    # Load data
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

    # Before defining callbacks
    validation_callback = ValidationCallback(pred_model, X_val, y_val, logs_dir)

    # Define callbacks
    checkpoint_path = os.path.join(models_dir, "best_model.keras")
    callbacks = [
        TensorBoard(log_dir=logs_dir, histogram_freq=1),
        TerminalLogger(validation_callback=validation_callback),  # <-- Pass the callback here
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
        ),
        validation_callback  # <-- Add this to callbacks if you want TensorBoard metrics
    ]

    print("🚀 Starting training...")
    print(f"📊 Logs will be saved to: {logs_dir}")

    history = train_model.fit(
        [X_train, y_train, il_train, ll_train],
        np.zeros(len(X_train)),
        validation_data=([X_val, y_val, il_val, ll_val], np.zeros(len(X_val))),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    if os.path.exists(checkpoint_path):
        print("📥 Loading best model...")
        pred_model.load_weights(checkpoint_path)

    # Convert to TFLite
    print("🔄 Converting to TFLite...")
    tflite_path = convert_to_tflite_with_flex(pred_model, models_dir)

    # Test the TFLite model
    test_tflite_model(tflite_path, X_test)

    print(f"✅ Training completed! TFLite model saved to: {tflite_path}")
    print(f"📊 Training logs saved to: {logs_dir}")