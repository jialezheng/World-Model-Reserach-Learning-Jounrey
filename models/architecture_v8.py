import torch
import torch.nn as nn

class LatentWorldModel(nn.Module):
    def __init__(self):
        super(LatentWorldModel, self).__init__()
        
        # 1. ENCODER: Compresses [Height, Velocity, Thrust] into a "Hidden Concept"
        self.encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 2) # Crushed down to just 2 "Secret" numbers
        )
        
        # 2. DECODER: Expands the "Secret" numbers back into [Next Height, Next Velocity]
        self.decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        # Pack it
        latent_code = self.encoder(x)
        # Unpack it
        prediction = self.decoder(latent_code)
        return prediction