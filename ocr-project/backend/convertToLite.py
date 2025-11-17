# ...existing code...
import tensorflow as tf 
from Trainer import AttentionLayer
 
# Load the saved model file 
model_path = "ocr_models/ocr_model_production_fp16.tflite" 
ocr_model = tf.keras.models.load_model(model_path, custom_objects={"AttentionLayer": AttentionLayer}) 

#TensorflowLite Conversion 
converter = tf.lite.TFLiteConverter.from_keras_model(ocr_model) 
# keep default optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
# allow fp16 quantization for weights
converter.target_spec.supported_types = [tf.float16] 

# allow SELECT_TF_OPS so TensorList/other TF ops remain supported during conversion
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# disable experimental lowering of tensor-list ops (prevents the TensorListReserve static-shape requirement)
# this attribute is still "private" on some TF builds but recommended by the error message
try:
    converter._experimental_lower_tensor_list_ops = False
except Exception:
    # older/newer TF might not expose the attribute — ignore if not present
    pass

tflite_fp16 = converter.convert() 
out_path = "ocr_models/ocr_model_production_fp16.tflite"

with open(out_path, "wb") as f:
    f.write(tflite_fp16) 

print("Saved converted lite model successfully", out_path)
# ...existing code...