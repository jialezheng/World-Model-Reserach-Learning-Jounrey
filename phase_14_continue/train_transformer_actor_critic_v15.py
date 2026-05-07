"""
Phase 15: Transformer Actor-Critic Pilot
==========================================
Same Actor-Critic architecture as Phase 13 RSSM
BUT uses Causal Transformer as the World Model.

Key difference from Phase 13:
  Phase 13: RSSM World Model
    → Stochastic uncertainty (mu, sigma)
    → LSTM-based memory
    
  Phase 15: Transformer World Model  
    → Attention over full sequence simultaneously
    → No recurrence, no vanishing gradients in World Model
    → Causal mask ensures no future peeking

This comparison shows whether the World Model architecture
affects how well the Pilot can learn.

Target: X=75, Y=0
Angle:  ±60 degrees
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os

# ── Positional Encoding ───────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=32, max_len=50):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

# ── Transformer World Model (Phase 14, frozen) ────────────────────────────────
class TransformerWorldModel_v15(nn.Module):
    def __init__(self, input_dim=6, output_dim=4, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embedding   = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        enc_layer        = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                      batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.decoder     = nn.Linear(d_model, output_dim)

    def generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

    def forward(self, src):
        x    = self.pos_encoder(self.embedding(src))
        mask = self.generate_causal_mask(src.size(1)).to(src.device)
        return self.decoder(self.transformer(x, mask=mask, is_causal=True))

# ── Actor (Pilot) ─────────────────────────────────────────────────────────────
class ActorTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=4, hidden_size=32, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, state_seq):
        out, _ = self.lstm(state_seq)
        raw    = self.decoder(out[:, -1, :])
        angle  = torch.tanh(raw[:, 0:1]) * (math.pi / 3)
        thrust = torch.sigmoid(raw[:, 1:2]) * 20.0
        return torch.cat((angle, thrust), dim=1)

# ── Critic (Value Estimator) ──────────────────────────────────────────────────
class CriticTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, state):
        return self.net(state)

if __name__ == '__main__':
    TARGET_X = 75.0
    TARGET   = torch.tensor([[TARGET_X, 0.0, 0.0, 0.0]], dtype=torch.float32)

    # 1. Load frozen Transformer World Model
    world = TransformerWorldModel_v15()
    world.load_state_dict(torch.load("models/weights/transformer_world_v15.pth",
                                     map_location="cpu"))
    world.eval()
    for p in world.parameters():
        p.requires_grad = False

    # 2. Initialize Actor + Critic
    actor    = ActorTransformer()
    critic   = CriticTransformer()

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

    print("🚀 Training Transformer Actor-Critic Pilot...")
    print(f"   Epochs: {EPOCHS} | Seq: {SEQ_LEN}s | Target: X={TARGET_X}, Y=0")
    print(f"   World Model: Causal Transformer (frozen, attention-based)")
    print(f"   Angle range: ±60° for horizontal steering")
    print("-" * 65)

    for epoch in range(EPOCHS):
        start_x  = float(np.random.uniform(-20,  20))
        start_y  = float(np.random.uniform(200, 400))
        start_vx = float(np.random.uniform(  3,   8))
        start_vy = float(np.random.uniform(-15,  -5))

        state_history  = torch.tensor([[[start_x, start_y, start_vx, start_vy]]],
                                      dtype=torch.float32)
        action_history = None
        states_list    = []
        rewards_list   = []

        # ── Roll out trajectory ───────────────────────────────────────────────
        with torch.no_grad():
            for step in range(SEQ_LEN):
                action     = actor(state_history)
                action_exp = action.unsqueeze(1)
                action_history = action_exp if action_history is None else \
                                 torch.cat((action_history, action_exp), dim=1)

                sa_history  = torch.cat((state_history, action_history), dim=2)
                next_states = world(sa_history)
                newest      = next_states[:, -1:, :]

                dist_x  = (newest[0, 0, 0] - TARGET_X) ** 2
                dist_y  = (newest[0, 0, 1] - 0.0) ** 2
                reward  = -(dist_x + dist_y).item() / 10000.0

                states_list.append(newest[:, 0, :].detach())
                rewards_list.append(reward)
                state_history = torch.cat((state_history, newest), dim=1)

        # ── Train Critic ──────────────────────────────────────────────────────
        critic_opt.zero_grad()
        critic_loss = 0.0
        gamma   = 0.95
        returns = []
        G = 0.0
        for r in reversed(rewards_list):
            G = r + gamma * G
            returns.insert(0, G)

        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / \
                         (returns_tensor.std() + 1e-8)

        for i, state in enumerate(states_list):
            value = critic(state).squeeze()
            critic_loss += mse(value, returns_tensor[i])

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        critic_opt.step()
        critic_sched.step(critic_loss)

        # ── Train Actor ───────────────────────────────────────────────────────
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
            next_states = world(sa_history)
            newest      = next_states[:, -1:, :]

            step_weight = (step + 1) / SEQ_LEN
            pos_loss    = huber(newest[:, 0, :2], TARGET[:, :2])
            vel_loss    = huber(newest[:, 0, 2:], TARGET[:, 2:]) * 0.3
            ceil_pen    = torch.relu(newest[:, 0, 1:2] - 600.0).mean() * 2.0

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
    torch.save(actor.state_dict(),  "models/weights/transformer_actor_v15.pth")
    torch.save(critic.state_dict(), "models/weights/transformer_critic_v15.pth")
    print("-" * 65)
    print(f"✅ Done! Best actor loss: {best_actor_loss:.2f}")
    print("💾 Saved: transformer_actor_v15.pth + transformer_critic_v15.pth")