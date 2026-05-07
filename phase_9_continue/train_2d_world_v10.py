"""
Phase 10: 2D Feed-Forward World Model (RETRAINED)
==================================================
Changes from original:
  1. Angle properly included in training data
  2. Wider angle range: -60 to +60 degrees
  3. More samples: 30000
  4. Target landing zone: X=75, Y=0
     (natural physics landing from X=0,Y=300,Vx=5,Vy=-10)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os

class LatentWorldModel2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 4))
    def forward(self, state_action):
        return self.decoder(self.encoder(state_action))

def generate_2d_data(samples=30000):
    print(f"⚙️  Generating {samples} samples with full angle range...")
    x   = np.random.uniform(-200, 200, samples)
    y   = np.random.uniform(0, 600, samples)
    vx  = np.random.uniform(-30, 30, samples)
    vy  = np.random.uniform(-50, 50, samples)
    # Wider angle range: -60 to +60 degrees
    angle  = np.random.uniform(-math.pi/3, math.pi/3, samples)
    thrust = np.random.uniform(0, 20, samples)

    dt, g = 1.0, -9.8
    ax = thrust * np.sin(angle)
    ay = g + thrust * np.cos(angle)

    next_x  = x  + vx * dt + 0.5 * ax * dt**2
    next_y  = y  + vy * dt + 0.5 * ay * dt**2
    next_vx = vx + ax * dt
    next_vy = vy + ay * dt

    inputs  = torch.tensor(np.column_stack((x, y, vx, vy, angle, thrust)),
                           dtype=torch.float32)
    targets = torch.tensor(np.column_stack((next_x, next_y, next_vx, next_vy)),
                           dtype=torch.float32)
    return inputs, targets

if __name__ == '__main__':
    inputs, targets = generate_2d_data(samples=30000)

    model     = LatentWorldModel2D()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=100)
    loss_fn   = nn.MSELoss()
    EPOCHS    = 1000
    best_loss = float('inf')

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    print("🌍 Training 2D World Model (with proper angle)...")
    print(f"   Epochs: {EPOCHS} | Samples: 30000 | Target landing: X=75, Y=0")
    print("-" * 55)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_in, batch_tgt in loader:
            optimizer.zero_grad()
            preds      = model(batch_in)
            loss       = loss_fn(preds, batch_tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:10.4f} | LR: {lr:.6f}")

    os.makedirs("models/weights", exist_ok=True)
    torch.save(model.state_dict(), "models/weights/latent_brain_2d_v10.pth")
    print("-" * 55)
    print(f"✅ Done! Best loss: {best_loss:.4f}")
    print("💾 Saved: models/weights/latent_brain_2d_v10.pth")