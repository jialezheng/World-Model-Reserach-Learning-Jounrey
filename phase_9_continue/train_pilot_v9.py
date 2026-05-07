import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

sys.path.insert(0, os.getcwd()) 
from models.architecture_v8 import LatentWorldModel

class PilotNetwork(nn.Module):
    def __init__(self):
        super(PilotNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )
        
    def forward(self, state):
        return self.net(state) * 20.0 

world_model = LatentWorldModel()
world_model.load_state_dict(torch.load("models/weights/latent_brain_v8.pth"))
world_model.eval() 
for param in world_model.parameters():
    param.requires_grad = False

pilot = PilotNetwork()
optimizer = optim.Adam(pilot.parameters(), lr=0.005) # Lower learning rate
loss_fn = nn.HuberLoss() # Prevents exploding gradients

epochs = 1500
print("🧠 Training Pilot in the Dream (Fixed)...")

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Start closer to the ground for a 10-second simulation
    current_state = torch.tensor([[100.0, -5.0]], dtype=torch.float32)
    total_loss = 0
    
    for step in range(10):
        thrust = pilot(current_state)
        action_state = torch.cat((current_state, thrust), dim=1)
        next_state = world_model(action_state)
        
        target_state = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        total_loss += loss_fn(next_state, target_state)
        current_state = next_state
        
    total_loss.backward()
    optimizer.step()
    
    if epoch % 250 == 0:
        print(f"Epoch {epoch} | Dream Flight Loss: {total_loss.item():.2f}")

torch.save(pilot.state_dict(), "models/weights/pilot_v9.pth")
print("✅ Phase 9 Pilot Saved!")