"""
Phase 11: 2D Pilot (RETRAINED)
================================
Changes from original:
  1. Target: X=75, Y=0 (natural physics landing spot)
  2. Loss penalizes distance from X=75 not X=0
  3. Angle properly rewarded — pilot must steer horizontally
  4. Step-weighted loss + velocity penalty + gradient clipping
  5. Randomized starting positions for robustness
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

class Pilot2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, state):
        raw    = self.net(state)
        # Angle: -60 to +60 degrees (wider than before for X=75 steering)
        angle  = torch.tanh(raw[:, 0:1]) * (math.pi / 3)
        thrust = torch.sigmoid(raw[:, 1:2]) * 20.0
        return torch.cat((angle, thrust), dim=1)

if __name__ == '__main__':
    # Load frozen Phase 10 World Model
    world = LatentWorldModel2D()
    world.load_state_dict(torch.load("models/weights/latent_brain_2d_v10.pth",
                                     map_location="cpu"))
    world.eval()
    for p in world.parameters():
        p.requires_grad = False

    pilot     = Pilot2D()
    optimizer = optim.Adam(pilot.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=200)
    loss_fn   = nn.HuberLoss()

    EPOCHS   = 3000
    SEQ_LEN  = 15
    # NEW TARGET: X=75, Y=0, Vx=0, Vy=0
    TARGET_X = 75.0
    TARGET   = torch.tensor([[TARGET_X, 0.0, 0.0, 0.0]], dtype=torch.float32)
    MAX_ALT  = 600.0
    best_loss = float('inf')

    print("🚀 Training 2D Pilot → Target X=75, Y=0...")
    print(f"   Epochs: {EPOCHS} | Seq: {SEQ_LEN}s | Target: ({TARGET_X}, 0)")
    print("-" * 60)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        # Randomized start near original conditions
        start_x  = float(np.random.uniform(-20,  20))
        start_y  = float(np.random.uniform(200, 400))
        start_vx = float(np.random.uniform(  3,   8))
        start_vy = float(np.random.uniform(-15,  -5))

        state      = torch.tensor([[start_x, start_y, start_vx, start_vy]],
                                  dtype=torch.float32)
        total_loss = 0.0

        for step in range(SEQ_LEN):
            action     = pilot(state)
            state_act  = torch.cat((state, action), dim=1)
            next_state = world(state_act)

            step_weight = (step + 1) / SEQ_LEN

            # Position loss toward X=75, Y=0
            pos_loss  = loss_fn(next_state[:, :2], TARGET[:, :2])
            # Velocity loss — soft landing
            vel_loss  = loss_fn(next_state[:, 2:], TARGET[:, 2:]) * 0.3
            # Altitude ceiling penalty
            ceil_pen  = torch.relu(next_state[:, 1:2] - MAX_ALT).mean() * 2.0

            total_loss += step_weight * (pos_loss + vel_loss + ceil_pen)
            state = next_state

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(pilot.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()

        if epoch % 300 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | Loss: {total_loss.item():8.2f} "
                  f"| Best: {best_loss:8.2f} | LR: {lr:.5f}")

    os.makedirs("models/weights", exist_ok=True)
    torch.save(pilot.state_dict(), "models/weights/pilot_2d_v11.pth")
    print("-" * 60)
    print(f"✅ Done! Best loss: {best_loss:.2f}")
    print("💾 Saved: models/weights/pilot_2d_v11.pth")