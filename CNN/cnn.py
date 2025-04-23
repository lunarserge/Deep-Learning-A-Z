import torch
from torch import nn
from torchvision.transforms import v2

transforms_base = v2.Compose([
    v2.Resize((64, 64)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3), # 62x62x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 31x31x32
            nn.Conv2d(32, 32, kernel_size=3), # 29x29x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x14x32
            nn.Flatten(),
            nn.Linear(32*14*14, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.cnn(x)
