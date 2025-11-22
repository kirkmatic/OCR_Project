import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Reshape, Bidirectional, GRU, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow_addons.optimizers import AdamW
from .data_utils import NUM_CHARS, IMAGE_WIDTH, IMAGE_HEIGHT, ctc_lambda_func

def build_ocr_model_tflite_compatible():
    input_img = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name='input_image')
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)
    conv_shape = K.int_shape(x)
    time_steps = conv_shape[2]
    features = conv_shape[1] * conv_shape[3]
    x = Reshape((time_steps, features))(x)
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.2))(x)
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.2))(x)
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
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss={'ctc': lambda y_true, y_pred: y_pred}
    )
    pred_model = tf.keras.models.Model(inputs=input_img, outputs=output)
    return train_model, pred_model, time_steps