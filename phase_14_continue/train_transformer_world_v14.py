"""
Phase 15: Causal Transformer World Model (CPU-OPTIMIZED)
=========================================================
Changes from previous improved version:
  - Mini-batch training (256 at a time) instead of full dataset at once
  - Reduced to 2000 samples, seq_len=15, 1500 epochs
  - Estimated time: 30-60 min on CPU laptop
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=32, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

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

def generate_trajectory_data(batches=2000, seq_len=15):
    print(f"⚙️  Generating {batches} trajectories (seq_len={seq_len})...")
    dt, g   = 1.0, -9.8
    inputs  = torch.zeros((batches, seq_len, 6))
    targets = torch.zeros((batches, seq_len, 4))

    for b in range(batches):
        x,  y  = np.random.uniform(-100, 100), np.random.uniform(500, 1000)
        vx, vy = np.random.uniform(-10,  10),  np.random.uniform(-20, 0)
        for t in range(seq_len):
            angle  = np.random.uniform(-math.pi/4, math.pi/4)
            thrust = np.random.uniform(0, 20)
            inputs[b, t] = torch.tensor([x, y, vx, vy, angle, thrust])
            ax = thrust * math.sin(angle)
            ay = g + thrust * math.cos(angle)
            nx, ny   = x + vx*dt + 0.5*ax*dt**2, y + vy*dt + 0.5*ay*dt**2
            nvx, nvy = vx + ax*dt, vy + ay*dt
            targets[b, t] = torch.tensor([nx, ny, nvx, nvy])
            x, y, vx, vy  = nx, ny, nvx, nvy
    return inputs, targets

if __name__ == '__main__':
    EPOCHS     = 1500
    BATCH_SIZE = 256

    inputs, targets = generate_trajectory_data(batches=2000, seq_len=15)
    model     = TransformerWorldModel_v15()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=80)
    loss_fn   = nn.MSELoss()
    best_loss = float('inf')

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("🧠 Training Causal Transformer World Model (CPU-Optimized)...")
    print(f"   Epochs: {EPOCHS} | Samples: 2000 | Mini-batch: {BATCH_SIZE}")
    print(f"   Estimated time: ~30-60 min on CPU")
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
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:10.4f} | LR: {current_lr:.6f}")

    os.makedirs("models/weights", exist_ok=True)
    torch.save(model.state_dict(), "models/weights/transformer_world_v15.pth")
    print("-" * 55)
    print(f"✅ Done! Best loss: {best_loss:.4f}")
    print("💾 Saved: models/weights/transformer_world_v15.pth")