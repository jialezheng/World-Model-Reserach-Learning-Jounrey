import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sys
import os

# This tells Python to look at your terminal's current folder for the 'models' folder
sys.path.insert(0, os.getcwd()) 

from models.architecture import ActionWorldModel


# 1. Setup Data
df = pd.read_csv("data/action_training_v5.csv")
# Inputs: [Height, Velocity, Thrust]
X = torch.tensor(df[['y', 'v', 'thrust']].values, dtype=torch.float32)
y = torch.tensor(df[['y_next', 'v_next']].values, dtype=torch.float32)


model = ActionWorldModel()
optimizer = optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

print("🧠 Training Action-Aware World Model...")
for e in range(300):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    if (e+1) % 100 == 0: print(f"Epoch {e+1}, Loss: {loss.item():.4f}")

# Save the brain
torch.save(model.state_dict(), "action_brain_v5.pth")
print("💾 Model saved as action_brain_v5.pth")