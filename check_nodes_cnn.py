import onnx
from onnx import numpy_helper, shape_inference

def get_tensor_shapes(graph):
    tensor_shape_map = {}

    # Check inputs
    for value in graph.input:
        name = value.name
        shape = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        tensor_shape_map[name] = shape

    # Check outputs
    for value in graph.output:
        name = value.name
        shape = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        tensor_shape_map[name] = shape

    # Check intermediate tensors (value_info)
    for value in graph.value_info:
        name = value.name
        shape = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        tensor_shape_map[name] = shape

    return tensor_shape_map


# 🔁 Step 1: Load model and infer shapes
original_model = onnx.load("simple_cnn.onnx")
inferred_model = shape_inference.infer_shapes(original_model)
onnx.save(inferred_model, "simple_cnn_with_shapes.onnx")  # 저장해도 되고 안 해도 됨

# 🔁 Step 2: Extract graph and shape map
graph = inferred_model.graph
shape_map = get_tensor_shapes(graph)

# 🔁 Step 3: Print all node info with shapes
for i, node in enumerate(graph.node):
    print(f"Node {i}: {node.op_type}")

    for idx, input_name in enumerate(node.input):
        shape = shape_map.get(input_name, "Unknown")
        print(f"  Input {idx}: {input_name}, shape: {shape}")

    for idx, output_name in enumerate(node.output):
        shape = shape_map.get(output_name, "Unknown")
        print(f"  Output {idx}: {output_name}, shape: {shape}")

    print()

