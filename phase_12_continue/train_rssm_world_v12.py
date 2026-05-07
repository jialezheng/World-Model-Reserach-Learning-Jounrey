import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os

class LSTMRSSM_v14(nn.Module):
    def __init__(self):
        super(LSTMRSSM_v14, self).__init__()
        # 1. Deterministic Memory (The Rules)
        self.lstm = nn.LSTM(input_size=6, hidden_size=32, batch_first=True)
        
        # 2. Stochastic State (The Uncertainty)
        # We split the 32 memory features into a Mean (mu) and Variance (logvar)
        self.fc_mu = nn.Linear(32, 8)
        self.fc_logvar = nn.Linear(32, 8)
        
        # 3. The Decoder
        # It takes BOTH the deterministic memory (32) and stochastic sample (8) to guess reality
        self.decoder = nn.Sequential(
            nn.Linear(32 + 8, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def reparameterize(self, mu, logvar):
        # The Reparameterization Trick: z = mu + sigma * epsilon
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std) # Random noise
        return mu + epsilon * std

    def forward(self, seq_state_action):
        # Step 1: Process deterministic memory over time
        lstm_out, _ = self.lstm(seq_state_action)
        
        # Step 2: Calculate uncertainty
        mu = self.fc_mu(lstm_out)
        logvar = self.fc_logvar(lstm_out)
        
        # Step 3: Sample the stochastic latent state
        z = self.reparameterize(mu, logvar)
        
        # Step 4: Combine memory + uncertainty and decode
        combined_state = torch.cat((lstm_out, z), dim=-1)
        prediction = self.decoder(combined_state)
        
        return prediction, mu, logvar

# DATA GENERATOR (Same as Phase 12)
def generate_trajectory_data(batches=3000, seq_len=15):
    print(f"⚙️ Generating {batches} stochastic flight trajectories...")
    dt = 1.0
    g = -9.8
    inputs = torch.zeros((batches, seq_len, 6))
    targets = torch.zeros((batches, seq_len, 4))
    
    for b in range(batches):
        x, y = np.random.uniform(-100, 100), np.random.uniform(500, 1000)
        vx, vy = np.random.uniform(-10, 10), np.random.uniform(-20, 0)
        
        for t in range(seq_len):
            angle = np.random.uniform(-math.pi/4, math.pi/4)
            thrust = np.random.uniform(0, 20)
            
            inputs[b, t] = torch.tensor([x, y, vx, vy, angle, thrust])
            
            ax = thrust * math.sin(angle)
            ay = g + (thrust * math.cos(angle))
            next_x = x + vx * dt + 0.5 * ax * (dt**2)
            next_y = y + vy * dt + 0.5 * ay * (dt**2)
            next_vx = vx + ax * dt
            next_vy = vy + ay * dt
            
            targets[b, t] = torch.tensor([next_x, next_y, next_vx, next_vy])
            x, y, vx, vy = next_x, next_y, next_vx, next_vy
            
    return inputs, targets

# CUSTOM LOSS FUNCTION (MSE + KL Divergence)
def rssm_loss(prediction, target, mu, logvar, beta=0.1):
    # Standard physics error
    mse_loss = nn.MSELoss()(prediction, target)
    
    # KL Divergence error (forcing the network to accept uncertainty)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalize KL by batch size so it doesn't overpower MSE
    kl_loss = kl_loss / target.size(0) 
    
    return mse_loss + (beta * kl_loss), mse_loss, kl_loss

if __name__ == '__main__':
    inputs, targets = generate_trajectory_data(batches=3000, seq_len=15)
    
    model = LSTMRSSM_v14()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print("🧠 Training Stochastic RSSM World Model...")
    for epoch in range(800):
        optimizer.zero_grad()
        
        preds, mu, logvar = model(inputs)
        
        loss, mse, kl = rssm_loss(preds, targets, mu, logvar, beta=0.05)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Total: {loss.item():.2f} (Physics MSE: {mse.item():.2f}, KL: {kl.item():.2f})")
            
    os.makedirs("models/weights", exist_ok=True)
    torch.save(model.state_dict(), "models/weights/rssm_world_v14.pth")
    print("✅ Phase 14 RSSM Model Saved!")