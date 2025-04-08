# onnx_flops_profiler.py
import onnx
import numpy as np
import sys

def extract_shapes(model):
    shape_dict = {}

    for i in model.graph.input:
        shape = [d.dim_value for d in i.type.tensor_type.shape.dim]
        shape_dict[i.name] = shape

    for vi in model.graph.value_info:
        shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shape_dict[vi.name] = shape

    for o in model.graph.output:
        shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
        shape_dict[o.name] = shape

    for init in model.graph.initializer:
        shape_dict[init.name] = list(init.dims)
    
    return shape_dict

def calculate_node_flops(node, shapes, initializers=None):
    def get_shape(name):
        return shapes.get(name) or shapes.get(name.split('/')[-1])

    op = node.op_type
    inputs = node.input
    if op == "Gemm":
        A = get_shape(inputs[0])
        B = get_shape(inputs[1])

        if A is None or B is None:
            return 0
        M,K = A[-1],A[-2]
        N = B[-2]
        return 2 * M * N * K  # Multiply + Add

    #elif op in ["Relu", "Sigmoid", "Tanh"]:
    elif op == "Relu":
        A = get_shape(inputs[0])
        if A is None:
            return 0
        return np.prod(A)

    elif op == "Conv":
        input_tensor = get_shape(inputs[0])
        weight_tensor = get_shape(inputs[1])

        # fallback to initializers if weight shape not found
        if weight_tensor is None and initializers:
            for init in initializers:
                if init.name == inputs[1]:
                    weight_tensor = list(init.dims)
                    shapes[inputs[1]] = weight_tensor
                    break

        if input_tensor is None or weight_tensor is None:
            return 0

        batch_size = input_tensor[0]
        in_c = input_tensor[1]
        in_h, in_w = input_tensor[2], input_tensor[3]

        out_c, _, k_h, k_w = weight_tensor

        # Default stride and padding
        stride = [1, 1]
        pads = [0, 0, 0, 0]  # [pad_top, pad_left, pad_bottom, pad_right]

        for attr in node.attribute:
            if attr.name == "strides":
                stride = list(attr.ints)
            elif attr.name == "pads":
                pads = list(attr.ints)

        out_h = int((in_h + pads[0] + pads[2] - k_h) / stride[0]) + 1
        out_w = int((in_w + pads[1] + pads[3] - k_w) / stride[1]) + 1

        return batch_size * out_c * out_h * out_w * (in_c * k_h * k_w * 2)

    elif op == "MaxPool":
        input_tensor = get_shape(inputs[0])
        if input_tensor is None:
            return 0

        batch_size, in_c, in_h, in_w = input_tensor

        # Default pool size and stride
        kernel_shape = [1, 1]
        stride = [1, 1]
        pads = [0, 0, 0, 0]

        for attr in node.attribute:
            if attr.name == "kernel_shape":
                kernel_shape = list(attr.ints)
            elif attr.name == "strides":
                stride = list(attr.ints)
            elif attr.name == "pads":
                pads = list(attr.ints)

        out_h = int((in_h + pads[0] + pads[2] - kernel_shape[0]) / stride[0]) + 1
        out_w = int((in_w + pads[1] + pads[3] - kernel_shape[1]) / stride[1]) + 1

        return batch_size * in_c * out_h * out_w * (kernel_shape[0] * kernel_shape[1] - 1)  # Max: comparisons
    
    #Reshape, Constant,...
    else:
        return 0


def profile_onnx_flops(onnx_path):
    model = onnx.load(onnx_path)
    model = onnx.shape_inference.infer_shapes(model)
    shapes = extract_shapes(model)

    total_flops = 0
    print("\n* Node-wise FLOP breakdown:")
    for node in model.graph.node:
        flops = calculate_node_flops(node, shapes)
        print(f"[{node.op_type:<10}] FLOPs: {flops:,}")
        total_flops += flops

    print(f"\n* Total Estimated FLOPs: {total_flops:,} (~{total_flops/1e6:.2f} MFLOPs)")

if __name__ == "__main__":
    profile_onnx_flops(sys.argv[1])

