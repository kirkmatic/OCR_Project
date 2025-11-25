import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization,
    Dropout, Reshape, Bidirectional, GRU, Dense, Lambda
)
from tensorflow.keras import backend as K
from tensorflow_addons.optimizers import AdamW
from .data_utils import NUM_CHARS, IMAGE_WIDTH, IMAGE_HEIGHT, ctc_lambda_func

def build_ocr_model_tflite_compatible():
    input_img = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name='input_image')
    x = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.15)(x)

    x = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)
    x = Dropout(0.3)(x)

    # Column-wise max pooling
    x = Lambda(lambda x: tf.reduce_max(x, axis=1))(x)

    x = Reshape((x.shape[1], x.shape[2]))(x)
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.2))(x)
    x = Bidirectional(GRU(48, return_sequences=True, dropout=0.2))(x)

    output = Dense(NUM_CHARS + 1, activation='softmax', name='output')(x)
    labels = Input(name='labels', shape=(None,), dtype='int32')
    input_length = Input(name='input_length', shape=(1,), dtype='int64')
    label_length = Input(name='label_length', shape=(1,), dtype='int64')
    loss_out = Lambda(ctc_lambda_func, name='ctc')([output, labels, input_length, label_length])
    train_model = tf.keras.models.Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=loss_out
    )
    train_model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipvalue=1.0),
        loss={'ctc': lambda y_true, y_pred: y_pred}
    )
    pred_model = tf.keras.models.Model(inputs=input_img, outputs=output)
    return train_model, pred_model, x.shape[1]