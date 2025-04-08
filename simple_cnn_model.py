# simple_cnn_model.py
import torch
import torch.nn as nn
from ptflops.flops_counter import get_model_complexity_info

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(16 * 7 * 7, 10)  # Assuming input is 1x28x28

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool(x)
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    # (channels, height, width)
    input_res = (1, 28, 28)

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, input_res, as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print(f'\nTotal FLOPs (approx.): {macs}  (note: FLOPs ≈ 2 × MACs)')
        print(f'Total Parameters: {params}')

