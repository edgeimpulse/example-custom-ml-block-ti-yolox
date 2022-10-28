from onnx_remove_node import remove
from onnx_operation_adder import add
import onnx
import numpy as np

onnx_graph = onnx.load('/Users/janjongboom/Downloads/ei_model_before.onnx')

# onnx_graph = remove(
#     onnx_graph=onnx_graph,
#     remove_node_names=[
#         'sequential/model/out_relu/Relu6;sequential/model/Conv_1_bn/FusedBatchNormV3;sequential/model/Conv_1/Conv2D__361',
#         'Reshape__94',
#         'sequential/dense/Tensordot/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd;sequential/dense/Tensordot/shape;sequential/dense/Tensordot;dense/bias',
#         'Relu__92',
#         'sequential/flatten/Reshape',
#         'sequential/dense_1/MatMul;sequential/dense_1/BiasAdd_Gemm__99',
#     ],
#     non_verbose=True
# )

input_shape = (1, 1280, 3, 3)

input_name = "Relu6__90:0"

onnx_graph = add(
  onnx_graph=onnx_graph,
  connection_src_op_output_names=[
    ["Relu6__90", input_name, "dummy_mul", "inp1"],
  ],
  connection_dest_op_input_names=[
    ["dummy_mul", "out1", "sequential/model/out_relu/Relu6;sequential/model/Conv_1_bn/FusedBatchNormV3;sequential/model/Conv_1/Conv2D__361", input_name],
  ],
  add_op_type="Mul",
  add_op_name="dummy_mul",
  add_op_input_variables={
    "inp1": [np.float32, input_shape],
    "inp2_const": [np.float32, [255]],
  },
  add_op_output_variables={
    "out1": [np.float32, input_shape],
  },
  non_verbose=True
)

onnx.save(onnx_graph, '/Users/janjongboom/Downloads/ei_model_rewritten.onnx')
