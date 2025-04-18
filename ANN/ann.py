from torch import nn

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ann = nn.Sequential(
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.ann(x)
