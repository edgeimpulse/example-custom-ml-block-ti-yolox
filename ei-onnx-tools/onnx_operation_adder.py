#! /usr/bin/env python

import os
import sys
import ast
from argparse import ArgumentParser
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.tensor import Variable
import numpy as np
from typing import Optional, List
from onnx_operation_generator import generate

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

AVAILABLE_DTYPES = [
    'float32',
    'float64',
    'int32',
    'int64',
    'str',
]

DTYPES_TO_ONNX_DTYPES = {
    float: onnx.TensorProto.FLOAT,
    int: onnx.TensorProto.INT64,
    str: onnx.TensorProto.STRING,
}

DTYPES_TO_NUMPY_TYPES = {
    'float32': np.float32,
    'float64': np.float64,
    'int32': np.int32,
    'int64': np.int64,
}

NUMPY_TYPES_TO_ONNX_DTYPES = {
    np.dtype('float32'): onnx.TensorProto.FLOAT,
    np.dtype('float64'): onnx.TensorProto.DOUBLE,
    np.dtype('int32'): onnx.TensorProto.INT32,
    np.dtype('int64'): onnx.TensorProto.INT64,
}


def add(
    connection_src_op_output_names: List[List[str]],
    connection_dest_op_input_names: List[List[str]],
    add_op_type: str,
    add_op_name: str,
    add_op_input_variables: Optional[dict] = None,
    add_op_output_variables: Optional[dict] = None,
    add_op_attributes: Optional[dict] = None,
    input_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    output_onnx_file_path: Optional[str] = '',
    non_verbose: Optional[bool] = False,
) -> onnx.ModelProto:
    """
    Parameters
    ----------
    connection_src_op_output_names: List
        Specify the name of the output name from which to connect.\n\n\
        e.g.\n\
        -Before-\n\
            [OpA] outnameA - inpnameB1 [OpB] outnameB\n\
            [OpC] outnameC\n\
        -After-\n\
            [OpA] outnameA - inpname1 [AddOP1] outname1 - inpnameB1 [OpB] outnameB\n\
            [OpC] outnameC - inpname2 [AddOP1]\n\
        When extrapolating a new OP between OpA and OpB.\n\
        connection_src_op_output_names = [\n\
            ["OpA", "outnameA", "AddOP1", "inpname1"],\n\
            ["OpC", "outnameC", "AddOP1", "inpname2"],\n\
        ]\n\n\
        This need not be specified only when the type of the newly added OP is Constant.
    connection_dest_op_input_names: List
        Specify the name of the input name from which to connect.\n\n\
        e.g.\n\
        -Before-\n\
            [OpA] outnameA - inpnameB1 [OpB] outnameB\n\
            [OpC] outnameC\n\
        -After-\n\
            [OpA] outnameA - inpname1 [AddOP1] outname1 - inpnameB1 [OpB] outnameB\n\
            [OpC] outnameC - inpname2 [AddOP1]\n\
        When extrapolating a new OP between OpA and OpB.\n\
        connection_dest_op_input_names = [\n\
            ["AddOP1", "outname1", "OpB", "inpnameB1"],\n\
        ]
    add_op_type: str
        ONNX op type.\n\
        See below for the types of OPs that can be specified.\n\n\
        e.g. "Add", "Div", "Gemm", ...\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md
    add_op_name: str
        Name of OP to be added.\n\n\
        e.g. --add_op_name AddOP1
    add_op_input_variables: Optional[dict]
        Specify input variables for the OP to be generated.\n\
        See below for the variables that can be specified.\n\n\
        {"input_var_name1": [numpy.dtype, shape], "input_var_name2": [dtype, shape], ...}\n\n\
        e.g.\n\
        add_op_input_variables = {\n\
            "inpname1": [np.float32, [1,224,224,3]],\n\
            "inpname2": [np.bool_, [0]],\n\
            ...\n\
        }\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md
    add_op_output_variables: Optional[dict]
        Specify output variables for the OP to be generated.\n\
        See below for the variables that can be specified.\n\n\
        {"output_var_name1": [numpy.dtype, shape], "output_var_name2": [dtype, shape], ...}\n\n\
        e.g.\n\
        add_op_output_variables = {\n\
            "outname1": [np.float32, [1,224,224,3]],\n\
            "outname2": [np.bool_, [0]],\n\
            ...\n\
        }\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md
    add_op_attributes: Optional[dict]
        Specify output add_op_attributes for the OP to be generated.\n\
        See below for the add_op_attributes that can be specified.\n\n\
        {"attr_name1": value1, "attr_name2": value2, "attr_name3": value3, ...}\n\n\
        e.g. add_op_attributes = {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}\n\
        Default: None\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md
    input_onnx_file_path: Optional[str]
        Input onnx file path.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        Default: ''
    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.
    output_onnx_file_path: Optional[str]
        Output onnx file path. If not specified, no ONNX file is output.\n\
        Default: ''
    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n\
        Default: False
    Returns
    -------
    changed_graph: onnx.ModelProto
        Changed onnx ModelProto.
    """

    # Unspecified check for input_onnx_file_path and onnx_graph
    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    if add_op_type not in ['Constant', 'ConstantOfShape']:
        if not add_op_input_variables:
            print(
                f'{Color.RED}ERROR:{Color.RESET} '+
                f'If add_op_type is other than Const or ConstantOfShape, '+
                f'add_op_input_variables must be specified.'
            )
            sys.exit(1)

    # Loading Graphs
    # onnx_graph If specified, onnx_graph is processed first
    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)
    graph = gs.import_onnx(onnx_graph)
    graph.cleanup().toposort()

    # Obtaining the opset of the original ONNX file
    opset = graph.opset

    # Generate an ONNX graph that holds only one OP
    single_op_graph = generate(
        op_type=add_op_type,
        opset=opset,
        op_name=add_op_name,
        input_variables=add_op_input_variables,
        output_variables=add_op_output_variables,
        attributes=add_op_attributes,
        non_verbose=True,
    )
    gs_single_op_graph = gs.import_onnx(single_op_graph)

    single_op_graph_node = None
    single_op_graph_node_inputs = None
    single_op_graph_node_outputs = None
    single_op_graph_node = gs_single_op_graph.nodes[0]
    if add_op_type not in ['Constant', 'ConstantOfShape']:
        single_op_graph_node_inputs = single_op_graph_node.inputs
    single_op_graph_node_outputs = single_op_graph_node.outputs

    graph.nodes.append(single_op_graph_node)

    # Search for the output OPs of the connection source
    src_ops = [graph.nodes[0]]
    src_ops = []
    for graph_node in graph.nodes:
        for srcop_name, _, _, _ in connection_src_op_output_names:
            if graph_node.name == srcop_name:
                src_ops.append(graph_node)

    # Search for the input OPs of the connection dest
    dest_ops = []
    for graph_node in graph.nodes:
        for _, _, destop_name, _ in connection_dest_op_input_names:
            if graph_node.name == destop_name:
                dest_ops.append(graph_node)

    # print('src_ops', src_ops)
    # print('dest_ops', dest_ops)

    # Rewrite the input of the connection Gen OP
    if single_op_graph_node_inputs:
        # [N*4] -> [N, 4]
        connection_src_op_output_names = np.asarray(connection_src_op_output_names)
        connection_src_op_output_names = connection_src_op_output_names.reshape(-1, 4)
        for srcop_name, srcop_output_name, addop_name, addop_input_name in connection_src_op_output_names:
            for srcop_graph_node in src_ops:
                if srcop_graph_node.name == srcop_name:
                    found_output = False
                    for srcop_graph_node_output in srcop_graph_node.outputs:
                        # print('srcop_graph_node_output.name', srcop_graph_node_output.name,
                        #     'srcop_output_name', srcop_output_name)
                        if srcop_graph_node_output.name == srcop_output_name:
                            for idxs, single_op_graph_node_input in enumerate(single_op_graph_node_inputs):
                                if single_op_graph_node_input.name == addop_input_name:
                                    found_output = True
                                    single_op_graph_node.inputs[idxs] = srcop_graph_node_output
                                    break
                                else:
                                    continue
                            break
                        else:
                            continue

                    # if not found_output:
                    #     print('not found', srcop_output_name, 'in',
                    #         [ x.name for x in srcop_graph_node.outputs ])

                    break
                else:
                    continue

    # Rewrite the input of the destination OP
    if single_op_graph_node_outputs:
        # [N*4] -> [N, 4]
        connection_dest_op_input_names = np.asarray(connection_dest_op_input_names)
        connection_dest_op_input_names = connection_dest_op_input_names.reshape(-1, 4)
        for addop_name, addop_output_name, destop_name, destop_input_name in connection_dest_op_input_names:
            for destop_graph_node in dest_ops:
                if destop_graph_node.name == destop_name:
                    found_input = False
                    for idxd, destop_graph_node_input in enumerate(destop_graph_node.inputs):
                        # print('destop_graph_node_input.name', destop_graph_node_input.name,
                        #     'destop_input_name', destop_input_name)

                        if destop_graph_node_input.name == destop_input_name:
                            for single_op_graph_node_output in single_op_graph_node_outputs:
                                # print('single_op_graph_node_output.name', single_op_graph_node_output.name,
                                #     'addop_output_name', addop_output_name)
                                if single_op_graph_node_output.name == addop_output_name:
                                    found_input = True
                                    destop_graph_node.inputs[idxd] = single_op_graph_node_output
                                    break
                                else:
                                    continue
                                break
                        else:
                            continue

                    if not found_input:
                        print('not found', destop_input_name, 'in',
                            [ x.name for x in destop_graph_node.inputs ])
                    break
            else:
                continue

    graph.cleanup().toposort()

    # Add unconnected input and output variables to the input/output OP of a graph
    graph_input_variables = []
    graph_output_variables = []

    # Extraction of input variables
    for graph_node in graph.nodes:
        try:
            for input in graph_node.inputs:
                if isinstance(input, Variable) and input not in graph.inputs:
                    graph_input_variables.append(input)
        except:
            pass

    # Extraction of output variables
    for graph_node in graph.nodes:
        try:
            for output in graph_node.outputs:
                if isinstance(output, Variable) and output not in graph.outputs:
                    graph_output_variables.append(output)
        except:
            pass

    graph_node_input_names = [
        graph_node_input.name for graph_node in graph.nodes for graph_node_input in graph_node.inputs
    ]
    graph_node_output_names = [
        graph_node_output.name for graph_node in graph.nodes for graph_node_output in graph_node.outputs
    ]

    # Extract unused input variables and assign them to graph inputs
    for graph_input_variable in graph_input_variables:
        if graph_input_variable.name in graph_node_output_names:
            pass
        else:
            graph.inputs.append(graph_input_variable)

    # Extract unused output variables and assign them to graph output
    for graph_output_variable in graph_output_variables:
        if graph_output_variable.name in graph_node_input_names:
            pass
        else:
            graph.outputs.append(graph_output_variable)

    graph.cleanup().toposort()

    # Shape Estimation
    changed_graph = None
    try:
        changed_graph = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
    except:
        changed_graph = gs.export_onnx(graph)
        if not non_verbose:
            print(
                f'{Color.YELLOW}WARNING:{Color.RESET} '+
                'The input shape of the next OP does not match the output shape. '+
                'Be sure to open the .onnx file to verify the certainty of the geometry.'
            )

    # Save
    if output_onnx_file_path:
        onnx.save(changed_graph, output_onnx_file_path)

    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Finish!')

    # Return
    return changed_graph
