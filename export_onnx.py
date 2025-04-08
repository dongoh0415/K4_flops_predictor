# export_onnx.py

import torch
from simple_mlp_model import SimpleMLP

model = SimpleMLP()
model.eval()

dummy_input = torch.randn(1, 784)

torch.onnx.export(
    model,
    dummy_input,
    "simple_mlp.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print(" ONNX: simple_mlp.onnx exporting completed!")

