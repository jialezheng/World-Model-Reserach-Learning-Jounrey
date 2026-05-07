"""
Phase 13: LSTM Pilot - IMPROVED
=================================
Changes from original:
  1. Step-by-step distance penalty (not just final crash penalty)
     - Original: loss only accumulated over all 15 steps equally
     - New: penalty INCREASES each step (later steps penalized more)
     - This gives the gradient a "direction" to follow earlier in the sequence
  2. Added velocity penalty (penalize Vx, Vy at landing, not just X, Y)
  3. Gradient clipping to prevent exploding gradients
  4. More epochs: 800 -> 2000
  5. Better noise decay schedule
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import os

# ── Load frozen v12 World Model ───────────────────────────────────────────────
# We import the class definition directly to keep demo self-contained
class LSTMWorldModel_v12(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=6, hidden_size=32, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))
    def forward(self, seq):
        out, _ = self.lstm(seq)
        return self.decoder(out)

# ── LSTM Pilot (same architecture as original) ────────────────────────────────
class PilotLSTM_v13(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=4, hidden_size=16, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(16, 2))

    def forward(self, state_seq):
        out, _  = self.lstm(state_seq)
        raw     = self.decoder(out[:, -1, :])
        angle   = torch.tanh(raw[:, 0:1]) * (math.pi / 4)
        thrust  = torch.sigmoid(raw[:, 1:2]) * 20.0
        return torch.cat((angle, thrust), dim=1)

if __name__ == '__main__':
    # 1. Load frozen World Model
    world = LSTMWorldModel_v12()
    world.load_state_dict(torch.load("models/weights/lstm_world_v12.pth",
                                     map_location="cpu"))
    world.eval()
    for p in world.parameters():
        p.requires_grad = False

    # 2. Initialize Pilot
    pilot     = PilotLSTM_v13()
    optimizer = optim.Adam(pilot.parameters(), lr=0.003)
    loss_fn   = nn.HuberLoss()

    EPOCHS    = 2000
    SEQ_LEN   = 15
    START     = [0.0, 200.0, 5.0, -10.0]   # X, Y, Vx, Vy
    TARGET    = [0.0,   0.0, 0.0,   0.0]

    print("🚀 Training LSTM Pilot (Improved Reward Shaping)...")
    print(f"   Epochs: {EPOCHS} | Seq: {SEQ_LEN}s | Start: {START}")
    print("-" * 55)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        state_history  = torch.tensor([[[*START]]], dtype=torch.float32)
        action_history = None
        total_loss     = 0.0

        # Noise decays from 100% -> 5% over training
        noise_factor = max(0.05, 1.0 - (epoch / 1200))

        for step in range(SEQ_LEN):
            # Pilot chooses action
            action      = pilot(state_history)

            # Add exploration noise
            noise       = torch.cat((
                torch.randn(1, 1) * (0.2  * noise_factor),
                torch.randn(1, 1) * (2.0  * noise_factor)
            ), dim=1)
            action      = action + noise
            action      = torch.cat((
                torch.clamp(action[:, 0:1], -math.pi/4, math.pi/4),
                torch.clamp(action[:, 1:2], 0.0, 20.0)
            ), dim=1)

            action_exp     = action.unsqueeze(1)
            action_history = action_exp if action_history is None else \
                             torch.cat((action_history, action_exp), dim=1)

            # World Model imagines next state
            sa_history  = torch.cat((state_history, action_history), dim=2)
            next_states = world(sa_history)
            newest      = next_states[:, -1:, :]

            # KEY IMPROVEMENT: step weight increases over time
            # Early steps matter less, final steps matter most
            step_weight = (step + 1) / SEQ_LEN

            target_t    = torch.tensor([[[*TARGET]]], dtype=torch.float32)

            # Position loss (X, Y)
            pos_loss    = loss_fn(newest[:, :, :2], target_t[:, :, :2])
            # Velocity loss (Vx, Vy) — we want soft landing, not just position
            vel_loss    = loss_fn(newest[:, :, 2:], target_t[:, :, 2:]) * 0.3

            total_loss += step_weight * (pos_loss + vel_loss)

            state_history = torch.cat((state_history, newest), dim=1)

        total_loss.backward()

        # KEY IMPROVEMENT: clip gradients so they don't explode or vanish
        torch.nn.utils.clip_grad_norm_(pilot.parameters(), max_norm=1.0)

        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss: {total_loss.item():8.2f} "
                  f"| Noise: {noise_factor:.3f}")

    os.makedirs("models/weights", exist_ok=True)
    torch.save(pilot.state_dict(), "models/weights/lstm_pilot_v13.pth")
    print("-" * 55)
    print("✅ Done! Saved: models/weights/lstm_pilot_v13.pth")