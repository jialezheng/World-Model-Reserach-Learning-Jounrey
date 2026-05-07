import torch
import torch.nn as nn

class ActionWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [Height, Velocity, Thrust] -> 3 neurons
        self.net = nn.Sequential(
            nn.Linear(3, 64), 
            nn.ReLU(),
            nn.Linear(64, 64), # The Latent Space
            nn.ReLU(),
            nn.Linear(64, 2)  # Output: [Predicted Y, Predicted V]
        )

    def forward(self, x):
        return self.net(x)