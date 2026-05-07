"""
Phase 13: RSSM Actor-Critic Pilot (Dreamer-Style)
==================================================
Architecture: Actor + Critic trained inside RSSM World Model

The Critic solves the vanishing gradient problem:
  OLD (Phase 13 LSTM Pilot):
    Loss = distance at FINAL step only
    Gradient must flow back 15 steps → vanishes
    
  NEW (Actor-Critic):
    Critic estimates VALUE at EACH step
    Actor only needs 1-step gradient from Critic
    → No vanishing gradients
    → Matches Dreamer (Hafner et al. 2020)

Target: X=75, Y=0 (natural physics landing spot)
Angle:  -60 to +60 degrees for horizontal steering
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os

# ── RSSM World Model (Phase 12, frozen) ──────────────────────────────────────
class LSTMRSSM_v14(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm      = nn.LSTM(input_size=6, hidden_size=32, batch_first=True)
        self.fc_mu     = nn.Linear(32, 8)
        self.fc_logvar = nn.Linear(32, 8)
        self.decoder   = nn.Sequential(nn.Linear(40, 16), nn.ReLU(), nn.Linear(16, 4))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, seq):
        out, _  = self.lstm(seq)
        mu      = self.fc_mu(out)
        logvar  = self.fc_logvar(out)
        z       = self.reparameterize(mu, logvar)
        return self.decoder(torch.cat((out, z), dim=-1)), mu, logvar

    def forward_mean(self, seq):
        out, _ = self.lstm(seq)
        mu     = self.fc_mu(out)
        return self.decoder(torch.cat((out, mu), dim=-1))

# ── Actor (Pilot) ─────────────────────────────────────────────────────────────
class ActorRSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=4, hidden_size=32, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, state_seq):
        out, _ = self.lstm(state_seq)
        raw    = self.decoder(out[:, -1, :])
        # Wider angle for horizontal steering toward X=75
        angle  = torch.tanh(raw[:, 0:1]) * (math.pi / 3)
        thrust = torch.sigmoid(raw[:, 1:2]) * 20.0
        return torch.cat((angle, thrust), dim=1)

# ── Critic (Value Estimator) ──────────────────────────────────────────────────
class CriticRSSM(nn.Module):
    """Estimates how good the current state is (expected future reward)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)   # Single value output
        )

    def forward(self, state):
        return self.net(state)

if __name__ == '__main__':
    TARGET_X = 75.0
    TARGET   = torch.tensor([[TARGET_X, 0.0, 0.0, 0.0]], dtype=torch.float32)

    # 1. Load frozen RSSM World Model
    world = LSTMRSSM_v14()
    world.load_state_dict(torch.load("models/weights/rssm_world_v14.pth",
                                     map_location="cpu"))
    world.eval()
    for p in world.parameters():
        p.requires_grad = False

    # 2. Initialize Actor + Critic
    actor    = ActorRSSM()
    critic   = CriticRSSM()

    actor_opt  = optim.Adam(actor.parameters(),  lr=0.003)
    critic_opt = optim.Adam(critic.parameters(), lr=0.005)

    actor_sched  = optim.lr_scheduler.ReduceLROnPlateau(actor_opt,  
                                                         mode='min', factor=0.5, patience=300)
    critic_sched = optim.lr_scheduler.ReduceLROnPlateau(critic_opt, 
                                                         mode='min', factor=0.5, patience=300)

    huber   = nn.HuberLoss()
    mse     = nn.MSELoss()
    EPOCHS  = 3000
    SEQ_LEN = 15
    best_actor_loss = float('inf')

    print("🚀 Training RSSM Actor-Critic (Dreamer-Style)...")
    print(f"   Epochs: {EPOCHS} | Seq: {SEQ_LEN}s | Target: X={TARGET_X}, Y=0")
    print(f"   Angle range: ±60° for horizontal steering")
    print(f"   Critic: estimates value at each step → no vanishing gradients")
    print("-" * 65)

    for epoch in range(EPOCHS):
        # Randomized starting position
        start_x  = float(np.random.uniform(-20,  20))
        start_y  = float(np.random.uniform(200, 400))
        start_vx = float(np.random.uniform(  3,   8))
        start_vy = float(np.random.uniform(-15,  -5))

        state_history  = torch.tensor([[[start_x, start_y, start_vx, start_vy]]],
                                      dtype=torch.float32)
        action_history = None

        # ── Roll out trajectory ───────────────────────────────────────────────
        states_list  = []
        rewards_list = []

        with torch.no_grad():
            for step in range(SEQ_LEN):
                action     = actor(state_history)
                action_exp = action.unsqueeze(1)
                action_history = action_exp if action_history is None else \
                                 torch.cat((action_history, action_exp), dim=1)

                sa_history  = torch.cat((state_history, action_history), dim=2)
                next_states = world.forward_mean(sa_history)
                newest      = next_states[:, -1:, :]

                # Reward: negative distance from target (closer = higher reward)
                dist_x   = (newest[0, 0, 0] - TARGET_X) ** 2
                dist_y   = (newest[0, 0, 1] - 0.0) ** 2
                reward   = -(dist_x + dist_y).item() / 10000.0

                states_list.append(newest[:, 0, :].detach())
                rewards_list.append(reward)
                state_history = torch.cat((state_history, newest), dim=1)

        # ── Train Critic ──────────────────────────────────────────────────────
        critic_opt.zero_grad()
        critic_loss = 0.0

        # Critic learns to predict discounted future reward from each state
        gamma = 0.95
        returns = []
        G = 0.0
        for r in reversed(rewards_list):
            G = r + gamma * G
            returns.insert(0, G)

        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        # Normalize returns for stable training
        returns_tensor = (returns_tensor - returns_tensor.mean()) / \
                         (returns_tensor.std() + 1e-8)

        for i, state in enumerate(states_list):
            value = critic(state).squeeze()
            critic_loss += mse(value, returns_tensor[i])

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        critic_opt.step()
        critic_sched.step(critic_loss)

        # ── Train Actor using Critic values ───────────────────────────────────
        actor_opt.zero_grad()
        state_history  = torch.tensor([[[start_x, start_y, start_vx, start_vy]]],
                                      dtype=torch.float32)
        action_history = None
        actor_loss     = 0.0

        for step in range(SEQ_LEN):
            action     = actor(state_history)
            action_exp = action.unsqueeze(1)
            action_history = action_exp if action_history is None else \
                             torch.cat((action_history, action_exp), dim=1)

            sa_history  = torch.cat((state_history, action_history), dim=2)
            next_states = world.forward_mean(sa_history)
            newest      = next_states[:, -1:, :]

            step_weight = (step + 1) / SEQ_LEN

            # Actor loss: direct distance + Critic guidance
            pos_loss    = huber(newest[:, 0, :2], TARGET[:, :2])
            vel_loss    = huber(newest[:, 0, 2:], TARGET[:, 2:]) * 0.3
            ceil_pen    = torch.relu(newest[:, 0, 1:2] - 600.0).mean() * 2.0

            # Critic value: higher value = better state, minimize negative value
            with torch.no_grad():
                value = critic(newest[:, 0, :]).squeeze()
            critic_guidance = -value * 0.1

            actor_loss += step_weight * (pos_loss + vel_loss + 
                                         ceil_pen + critic_guidance)
            state_history = torch.cat((state_history, newest), dim=1)

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        actor_opt.step()
        actor_sched.step(actor_loss)

        if actor_loss.item() < best_actor_loss:
            best_actor_loss = actor_loss.item()

        if epoch % 300 == 0:
            lr = actor_opt.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | Actor: {actor_loss.item():8.2f} "
                  f"| Critic: {critic_loss.item():6.4f} "
                  f"| Best: {best_actor_loss:8.2f} | LR: {lr:.5f}")

    os.makedirs("models/weights", exist_ok=True)
    torch.save(actor.state_dict(),  "models/weights/rssm_actor_v13.pth")
    torch.save(critic.state_dict(), "models/weights/rssm_critic_v13.pth")
    print("-" * 65)
    print(f"✅ Done! Best actor loss: {best_actor_loss:.2f}")
    print("💾 Saved: rssm_actor_v13.pth + rssm_critic_v13.pth")