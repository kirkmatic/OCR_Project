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
    BatchNormalization, SpatialDropout2D, Lambda, Attention
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
models_dir = os.path.join(BASE_DIR, 'ocr_models')
sample_logs_dir = os.path.join(BASE_DIR, 'sample_logs')

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(sample_logs_dir, exist_ok=True)


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention = Attention()

    def call(self, inputs):
        return self.attention([inputs, inputs])


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


def build_ocr_model():
    input_img = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    x = Conv2D(64, (3, 3), activation='swish', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)
    x = Conv2D(128, (3, 3), activation='swish', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)
    x = Conv2D(256, (3, 3), activation='swish', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)
    x = SpatialDropout2D(0.3)(x)
    print(x.shape)
    conv_shape = K.int_shape(x)
    if conv_shape is None:
        conv_shape = tf.shape(x)
    time_steps = conv_shape[2]
    features = conv_shape[1] * conv_shape[3]
    x = Reshape((time_steps, features))(x)
    x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)
    x = AttentionLayer()(x)
    x = Bidirectional(LSTM(96, return_sequences=True, dropout=0.3))(x)
    output = Dense(NUM_CHARS + 1, activation='softmax')(x)
    labels = Input(name='labels', shape=(None,), dtype='int32')
    input_length = Input(name='input_length', shape=(1,), dtype='int64')
    label_length = Input(name='label_length', shape=(1,), dtype='int64')
    loss_out = Lambda(ctc_lambda_func, name='ctc')([output, labels, input_length, label_length])
    train_model = Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=loss_out
    )
    train_model.compile(
        optimizer=AdamW(learning_rate=2e-4, weight_decay=1e-5, beta_1=0.9, beta_2=0.999),
        loss={'ctc': lambda y_true, y_pred: y_pred}
    )
    pred_model = Model(inputs=input_img, outputs=output)
    return train_model, pred_model


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


if os.path.exists(logs_dir):
    shutil.rmtree(logs_dir)
os.makedirs(logs_dir, exist_ok=True)


def log_training_samples(samples, log_path):
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Training samples used for this run:\n")
        for img_path, label in samples:
            f.write(f"{img_path}, {label}\n")
    print(f"Training samples logged to {log_path}")


def copy_training_images(samples, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for img_path, label in samples:
        try:
            shutil.copy(img_path, dest_folder)
        except Exception as e:
            print(f"Failed to copy {img_path}: {e}")
    print(f"Copied {len(samples)} training images to {dest_folder}")


def char_accuracy(pred, true):
    matcher = difflib.SequenceMatcher(None, pred, true)
    return matcher.ratio()


def save_production_model(pred_model, train_model):
    prod_model_path = os.path.join(models_dir, "ocr_model_production.keras")
    pred_model.save(prod_model_path)
    config = {
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'characters': CHARACTERS,
        'num_chars': NUM_CHARS,
        'model_version': '1.0',
        'final_word_accuracy': validation_callback.best_accuracy,
        'final_char_accuracy': validation_callback.best_char_accuracy
    }
    config_path = os.path.join(models_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dumps(config, f)
    print(f"Saved production model to {prod_model_path}")
    print(f"Saved model configuration to {config_path}")


if __name__ == "__main__":
    base_dir = dataset_dir
    all_samples = load_samples(
        os.path.join(dataset_dir, "Training", "training_labels.csv"),
        os.path.join(dataset_dir, "Training", "training_words")
    )
    print(f"Total samples loaded: {len(all_samples)}")
    random.shuffle(all_samples)
    num_total = len(all_samples)
    num_train = int(0.8 * num_total)
    num_val = int(0.1 * num_total)
    num_test = num_total - num_train - num_val
    train_samples = all_samples[:num_train]
    val_samples = all_samples[num_train:num_train+num_val]
    test_samples = all_samples[num_train+num_val:]
    train_samples_log_path = os.path.join(sample_logs_dir, "training_samples.txt")
    log_training_samples(train_samples, train_samples_log_path)
    training_images_dest = os.path.join(sample_logs_dir, "training_images")
    copy_training_images(train_samples, training_images_dest)
    X_train, y_train, il_train, ll_train = prepare_data(train_samples)
    X_val, y_val, il_val, ll_val = prepare_data(val_samples)
    X_test, y_test, il_test, ll_test = prepare_data(test_samples)
    final_weights_path = os.path.join(models_dir, "ocr_model_final_weights.keras")
    checkpoint_weights_path = os.path.join(models_dir, "ocr_model_best_weights.keras")
    best_accuracy_path = os.path.join(models_dir, "ocr_model_best_weights.keras")
    best_char_path = os.path.join(models_dir, "ocr_model_best_char_weights.keras")
    train_model, pred_model = build_ocr_model()
    tensorboard_callback = TensorBoard(log_dir=logs_dir)
    validation_callback = ValidationCallback(pred_model, X_train, y_train, logs_dir)
    terminal_logger = TerminalLogger(validation_callback=validation_callback)
    if os.path.exists(best_accuracy_path):
        print("Loading best word accuracy weights...")
        train_model.load_weights(best_accuracy_path)
        print("Current best word accuracy:", validation_callback.best_accuracy)
    elif os.path.exists(best_char_path):
        print("Loading best character accuracy weights...")
        train_model.load_weights(best_char_path)
        print("Current best character accuracy:", validation_callback.best_char_accuracy)
    else:
        print("No previous weights found. Starting fresh training...")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_weights_path,
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=50,
        min_lr=1e-6,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    history = train_model.fit(
        [X_train, y_train, il_train, ll_train],
        np.zeros(len(X_train)),
        validation_data=([X_test, y_test, il_test, ll_test], np.zeros(len(X_test))),
        epochs=500,
        batch_size=256,
        callbacks=[
            tensorboard_callback,
            terminal_logger,
            checkpoint_callback,
            validation_callback,
            reduce_lr_callback,
            early_stop
        ]
    )
    train_model.save_weights(final_weights_path)
    save_production_model(pred_model, train_model)