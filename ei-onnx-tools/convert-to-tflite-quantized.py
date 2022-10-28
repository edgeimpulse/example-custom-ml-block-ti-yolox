import numpy as np
import argparse, math, shutil, os, json, time
import onnx
from onnx_tf.backend import prepare # https://github.com/onnx/onnx-tensorflow
import tensorflow as tf

parser = argparse.ArgumentParser(description='ONNX model to quantized TFLite')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--onnx-file', type=str, required=True)
parser.add_argument('--out-file', type=str, required=True)

args = parser.parse_args()

def get_concrete_function(keras_model, input_shape):
    input_shape_with_batch = (1,) + input_shape
    run_model = tf.function(lambda x: keras_model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(input_shape_with_batch, tf.float32))
    return concrete_func

# Declare a generator that can feed the TensorFlow Lite converter during quantization
def representative_dataset_generator(validation_dataset):
    def gen():
        for data, _ in validation_dataset.take(-1).as_numpy_iterator():
            yield [tf.convert_to_tensor([data])]
    return gen

def convert_int8_io_int8(concrete_func, keras_model, dataset_generator,
                         disable_per_channel = False):
    converter_quantize = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], keras_model)
    if disable_per_channel:
        converter_quantize._experimental_disable_per_channel = disable_per_channel
        print('Note: Per channel quantization has been automatically disabled for this model. '
                'You can configure this in Keras (expert) mode.')
    converter_quantize.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_quantize.representative_dataset = dataset_generator
    # Force the input and output to be int8
    converter_quantize.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Restrict the supported types to avoid ops that are not TFLM compatible
    converter_quantize.target_spec.supported_types = [tf.dtypes.int8]
    converter_quantize.inference_input_type = tf.int8
    converter_quantize.inference_output_type = tf.int8
    tflite_quant_model = converter_quantize.convert()
    return tflite_quant_model

X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')
Y_test = None
validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

dataset_generator = representative_dataset_generator(validation_dataset)

# Load the ONNX model
onnx_model = onnx.load(args.onnx_file)

# Check that the IR is well formed
onnx.checker.check_model(onnx_model)

model_input_shape = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in onnx_model.graph.input][0]
print('model_input_shape', model_input_shape)
print('X_test shape', X_test.shape)

# Now do ONNX => TF
tf_model_path = '/tmp/savedmodel'
tf_rep = prepare(onnx_model, device='cpu')
tf_rep.export_graph(tf_model_path)

converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)

def representative_dataset():
    for i, sample in enumerate(X_test):
        yield [np.expand_dims(sample, axis=0)]
        if i >= 1000: # We only need a small portion of the dataset to do the quantization
            break

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # We only want to use int8 kernels
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset

tflite_model = converter.convert()

# Save the model
with open(args.out_file, 'wb') as f:
    f.write(tflite_model)
