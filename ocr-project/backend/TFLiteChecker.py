import tensorflow as tf

tflite_path = "ocr_modelsv2/ocr_model_production_fp16.tflite"  # update if different
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("INPUT DETAILS:", input_details)
print("OUTPUT DETAILS:", output_details)
for d in input_details:
    print("input index, shape, dtype:", d['index'], d['shape'], d['dtype'])
for d in output_details:
    print("output index, shape, dtype:", d['index'], d['shape'], d['dtype'])
