import torch
import math
import os
import sys
import matplotlib.pyplot as plt

# Setup paths
sys.path.insert(0, os.getcwd()) 

# Import architectures (ensure the training loops in these files are commented out or wrapped)
from phase_9_continue.train_2d_world_v10 import LatentWorldModel2D
from phase_9_continue.train_2d_pilot_v11 import Pilot2D

# 1. LOAD THE BRAINS
print("🧠 Loading 2D World Model and Orbital Pilot...")
world_model = LatentWorldModel2D()
world_model.load_state_dict(torch.load("models/weights/latent_brain_2d_v10.pth"))
world_model.eval()

pilot = Pilot2D()
pilot.load_state_dict(torch.load("models/weights/pilot_2d_v11.pth"))
pilot.eval()

# 2. SIMULATE THE FLIGHT
# Starting at X=100, Y=200, moving right at 5m/s and falling at -10m/s
state = torch.tensor([[100.0, 200.0, 5.0, -10.0]], dtype=torch.float32)

x_path = [100.0]
y_path = [200.0]

print("🚀 Initiating 2D Simulation...")
for step in range(15):
    with torch.no_grad():
        # Pilot decides thrust and angle
        action = pilot(state)
        angle_rad = action[0][0].item()
        thrust = action[0][1].item()
        
        # World Model predicts the next position
        state_action = torch.cat((state, action), dim=1)
        next_state = world_model(state_action)
        
        # Extract new X and Y
        new_x = next_state[0][0].item()
        new_y = next_state[0][1].item()
        
        x_path.append(new_x)
        y_path.append(new_y)
        
        # Update state for next loop
        state = next_state
        
        print(f"Sec {step+1}: X={new_x:.1f}m, Y={new_y:.1f}m | Thrust={thrust:.1f}, Angle={math.degrees(angle_rad):.1f}°")

# 3. GRAPH THE TRAJECTORY
plt.figure(figsize=(8, 6))
plt.plot(x_path, y_path, marker='o', linestyle='-', color='b', label='AI Flight Path')
plt.scatter(0, 0, color='red', s=100, label='Target Landing Zone (0,0)', zorder=5)

plt.title("Phase 11: 2D Orbital Pilot Trajectory (MBRL)", fontsize=14)
plt.xlabel("Horizontal Position (X meters)", fontsize=12)
plt.ylabel("Vertical Altitude (Y meters)", fontsize=12)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
