import numpy as np
from onnx_operation_adder import add
from onnx_remove_node import remove
import onnx
import onnx_graphsurgeon as gs
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser(description='Inject a MUL w/ constant value 255 at the beginning of an ONNX graph')
parser.add_argument('--onnx-file', type=str, required=True)
parser.add_argument('--out-file', type=str, required=True)

args, unknown = parser.parse_known_args()

onnx_graph = onnx.load(args.onnx_file)
graph = gs.import_onnx(onnx_graph)
input_name = graph.nodes[0].inputs[0].name
input_shape = graph.nodes[0].inputs[0].shape

onnx_graph = add(
  onnx_graph=onnx_graph,
  connection_src_op_output_names=[
    [graph.nodes[0].name, "images", "dummy_mul", "inp1"],
  ],
  connection_dest_op_input_names=[
    ["dummy_mul", "out1", graph.nodes[0].name, "images"],
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

onnx_graph = remove(
    onnx_graph=onnx_graph,
    remove_node_names=[input_name],
    non_verbose=True
)

# create out directory
Path(os.path.dirname(args.out_file)).mkdir(parents=True, exist_ok=True)

onnx.save(onnx_graph, args.out_file)
