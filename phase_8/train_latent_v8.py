import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import sys

# Tell Python to look in the root folder for our architecture
sys.path.insert(0, os.getcwd()) 
from models.architecture_v8 import LatentWorldModel

# 1. SETUP: Load your Phase 6 data
data_path = "data/noisy_training_v6.csv"
data = pd.read_csv(data_path)

print(f"✅ Data loaded with columns: {list(data.columns)}")

# Inputs: Current noisy position, noisy velocity, and the thrust applied
inputs = torch.tensor(data[['y_noisy', 'v_noisy', 'thrust']].values, dtype=torch.float32)

# Targets: The "Next" state the AI is trying to predict correctly
targets = torch.tensor(data[['y_next', 'v_next']].values, dtype=torch.float32)

# 2. INITIALIZE: The "Compressed" Brain
model = LatentWorldModel()
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. TRAINING LOOP
print("🚀 Training the Latent World Model (Phase 8)...")
for epoch in range( ):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.6f}")

# 4. SAVE: Store this new brain
torch.save(model.state_dict(), "latent_brain_v8.pth")
print("✅ Saved: latent_brain_v8.pth")