# Edge Impulse ONNX tools

Tools to make it easier to work with ONNX files (especially ones created by PyTorch).

* `inject-mul-255.py` - Edge Impulse expects the input pixels in image models to be scaled from 0..1. If your model has input pixels scaled from 0..255 instead, then this script can rewrite the graph to automatically inject a `Mul` operation between the input and the first node.
* `convert-to-nhwc.py` - Edge Impulse expects the input to be in NHWC format, not NCHW which PyTorch uses. This script changes the input shape and then adds a `Transpose` op at the beginning of the graph. When converting NCHW graphs to TFLite this will create a ton of transpose operations before every convolutional layer though, so if your training pipeline can output NHWC natively (e.g. by converting to a Keras SavedModel first, like ultralytics YOLOv5) then that should always be used.
* `convert-to-tflite.py` - Turns the ONNX graph into a TFLite graph (unquantized). This requires https://github.com/onnx/onnx-tensorflow .

## Attribution

The graph rewriting logic (in `onnx_*.py`) is derived from https://github.com/PINTO0309/simple-onnx-processing-tools (MIT Licensed).
