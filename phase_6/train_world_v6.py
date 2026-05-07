import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sys
import os

# Bulletproof path fix
sys.path.insert(0, os.getcwd()) 
from models.architecture import ActionWorldModel

# 1. Load Noisy Data
df = pd.read_csv("data/noisy_training_v6.csv")
X = torch.tensor(df[['y_noisy', 'v_noisy', 'thrust']].values, dtype=torch.float32)
y = torch.tensor(df[['y_next', 'v_next']].values, dtype=torch.float32)

# 2. Train with a focus on "Denoising"
model = ActionWorldModel() # Using our Phase 5 architecture
optimizer = optim.Adam(model.parameters(), lr=0.002) # Slower LR for noise
loss_fn = nn.MSELoss()

print("🛡️ Training Robust World Model (Phase 6)...")
for e in range(600): # More epochs to handle noise
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    if (e+1) % 100 == 0:
        print(f"Epoch {e+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "models/action_brain_v6.pth")
print("💾 Robust weights saved to models/action_brain_v6.pth")