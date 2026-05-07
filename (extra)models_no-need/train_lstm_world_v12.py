import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os

# 1. THE LSTM WORLD MODEL (The "M" Module)
class LSTMWorldModel_v12(nn.Module):
    def __init__(self):
        super(LSTMWorldModel_v12, self).__init__()
        # The Memory Engine: Takes the 6 variables and passes them through time
        # batch_first=True means our data shape is (Batch, Sequence, Features)
        self.lstm = nn.LSTM(input_size=6, hidden_size=32, batch_first=True)
        
        # The Decoder: Takes the 32-number memory state and predicts the 4 next variables
        self.decoder = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, seq_state_action):
        # lstm_out is the memory-infused representation of the data at EVERY time step
        lstm_out, (hidden_state, cell_state) = self.lstm(seq_state_action)
        
        # We decode the LSTM's thoughts into actual physics predictions
        prediction = self.decoder(lstm_out)
        return prediction

# 2. SEQUENCE DATA GENERATOR
def generate_trajectory_data(batches=2000, seq_len=10):
    """Generates continuous flight paths so the LSTM can learn momentum over time."""
    print(f"⚙️ Generating {batches} flight trajectories ({seq_len} steps each)...")
    dt = 1.0
    g = -9.8
    
    # Empty tensors to hold our data: Shape (Batches, Time, Features)
    inputs = torch.zeros((batches, seq_len, 6))
    targets = torch.zeros((batches, seq_len, 4))
    
    for b in range(batches):
        # Random starting state for this flight
        x, y = np.random.uniform(-100, 100), np.random.uniform(500, 1000)
        vx, vy = np.random.uniform(-10, 10), np.random.uniform(-20, 0)
        
        for t in range(seq_len):
            # Random action for this specific second
            angle = np.random.uniform(-math.pi/4, math.pi/4)
            thrust = np.random.uniform(0, 20)
            
            # Save the current state + action to inputs
            inputs[b, t] = torch.tensor([x, y, vx, vy, angle, thrust])
            
            # Calculate next physics state
            ax = thrust * math.sin(angle)
            ay = g + (thrust * math.cos(angle))
            
            next_x = x + vx * dt + 0.5 * ax * (dt**2)
            next_y = y + vy * dt + 0.5 * ay * (dt**2)
            next_vx = vx + ax * dt
            next_vy = vy + ay * dt
            
            # Save the true next state to targets
            targets[b, t] = torch.tensor([next_x, next_y, next_vx, next_vy])
            
            # Update variables for the next loop
            x, y, vx, vy = next_x, next_y, next_vx, next_vy
            
    return inputs, targets

# 3. TRAINING LOOP
if __name__ == '__main__':
    inputs, targets = generate_trajectory_data(batches=3000, seq_len=15)
    
    model = LSTMWorldModel_v12()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    
    print("🧠 Training LSTM Memory World Model...")
    for epoch in range(800):
        optimizer.zero_grad()
        
        # Predict the entire sequence at once
        preds = model(inputs)
        
        # Compare sequence predictions to true sequence targets
        loss = loss_fn(preds, targets)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Sequence Loss: {loss.item():.4f}")
            
    os.makedirs("models/weights", exist_ok=True)
    torch.save(model.state_dict(), "models/weights/lstm_world_v12.pth")
    print("✅ Phase 12 LSTM Model Saved!")  