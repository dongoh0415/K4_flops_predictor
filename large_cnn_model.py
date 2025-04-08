import torch
import torch.nn as nn

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

# ======= GPU 적용 + 실행 코드 =======

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)  # 모델 GPU로
dummy_input = torch.randn(1, 1, 28, 28).to(device)  # 입력도 GPU로

# Warm-up (CUDA 초기화 강제)
_ = model(dummy_input)

# 여러 번 돌려서 프로파일링 시 GPU 커널 잘 잡히게
for _ in range(100):
    _ = model(dummy_input)

