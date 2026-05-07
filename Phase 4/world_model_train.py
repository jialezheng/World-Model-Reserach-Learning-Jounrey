import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# 1. Load the data
df = pd.read_csv("world_model_training.csv")
X = torch.tensor(df[['y_now', 'v_now']].values, dtype=torch.float32)
y = torch.tensor(df[['y_next', 'v_next']].values, dtype=torch.float32)

# 2. Define the World Model Architecture
class WorldModel(nn.Module):
    def __init__(self):
        super(WorldModel, self).__init__()
        # Input: [y, v] -> Hidden (Latent Space) -> Output: [y_next, v_next]
        self.network = nn.Sequential(
            nn.Linear(2, 32), # Input Layer
            nn.ReLU(),        # Activation
            nn.Linear(32, 32),# Hidden Layer (The "Latent Space")
            nn.ReLU(),
            nn.Linear(32, 2)  # Output Layer
        )

    def forward(self, x):
        return self.network(x)

model = WorldModel()
criterion = nn.MSELoss() # Mean Squared Error (how far is the guess from reality?)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. Training Loop
print("🧠 Training the Latent Space World Model...")
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward() # Backpropagation: Adjusting weights based on error
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.6f}")

# 4. THE ULTIMATE TEST (Time 7 Prediction)
# At Time 6, the state was: Position 0.0, Velocity 0.0 (per your previous data)
# Let's test the state just before it hit the ground to see if it predicts T=6 correctly
# From your data: Time 5 was [y: 2.0, v: -49.0]
test_input = torch.tensor([[2.0, -49.0]], dtype=torch.float32)
prediction = model(test_input).detach().numpy()

print("\n--- NEURAL WORLD MODEL PREDICTION ---")
print(f"Input State (T=5): [Height: 2.0, Velocity: -49.0]")
print(f"Predicted Next State (T=6): Height: {prediction[0][0]:.2f}, Velocity: {prediction[0][1]:.2f}")
print(f"Actual Physics (Ground Truth): Height: -47.0, Velocity: -58.8")