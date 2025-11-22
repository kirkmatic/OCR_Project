import datetime
import os
import numpy as np
import difflib
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from .data_utils import CHARACTERS, NUM_CHARS, IMAGE_WIDTH, IMAGE_HEIGHT

def char_accuracy(pred, true):
    matcher = difflib.SequenceMatcher(None, pred, true)
    return matcher.ratio()

class TerminalLogger(tf.keras.callbacks.Callback):
    def __init__(self, validation_callback=None, logs_dir="logs"):
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
        decoded, _ = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True, beam_width=5)
        decoded_text = ''.join([CHARACTERS[idx] for idx in tf.keras.backend.get_value(decoded[0][0]) if idx != -1 and idx < NUM_CHARS])
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
        self.last_word_acc = word_acc
        self.last_precision = precision
        self.last_recall = recall
        self.last_f1_score = f1_score
        self.last_char_accuracy = avg_char_acc