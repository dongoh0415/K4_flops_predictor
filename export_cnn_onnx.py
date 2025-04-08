# export_cnn_to_onnx.py
import torch
from simple_cnn_model import SimpleCNN

model = SimpleCNN()
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)  # 1 channel, 28x28 image
torch.onnx.export(model, dummy_input, "simple_cnn.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
)

print("✅ simple_cnn.onnx exported.")

