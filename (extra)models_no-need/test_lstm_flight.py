import torch
import math
import os
import sys
import matplotlib.pyplot as plt

# Ensure Python can find your architecture files
sys.path.insert(0, os.getcwd()) 
from train_lstm_world_v12 import LSTMWorldModel_v12
from train_lstm_pilot_v13 import PilotLSTM_v13

# 1. LOAD BOTH MEMORY MODELS
print("🧠 Loading LSTM World Model and LSTM Pilot...")
world_model = LSTMWorldModel_v12()
world_model.load_state_dict(torch.load("models/weights/lstm_world_v12.pth"))
world_model.eval()

pilot = PilotLSTM_v13()
pilot.load_state_dict(torch.load("models/weights/lstm_pilot_v13.pth"))
pilot.eval()

# 2. SIMULATE THE FLIGHT AUTOREGRESSIVELY
# Start state shape: (Batch=1, Time=1, Features=4)
state_history = torch.tensor([[[100.0, 200.0, 5.0, -10.0]]], dtype=torch.float32)
action_history = None

x_path = [100.0]
y_path = [200.0]

print("🚀 Initiating Memory-Driven Simulation...")
with torch.no_grad():
    for step in range(15):
        # 1. Pilot reads the history and decides
        action = pilot(state_history)
        angle_rad = action[0][0].item()
        thrust = action[0][1].item()
        
        # Format action and add to history
        action_expanded = action.unsqueeze(1)
        if action_history is None:
            action_history = action_expanded
        else:
            action_history = torch.cat((action_history, action_expanded), dim=1)
            
        # 2. World Model reads the history and predicts
        state_action_history = torch.cat((state_history, action_history), dim=2)
        next_states_pred = world_model(state_action_history)
        
        # 3. Extract the newest reality
        newest_state = next_states_pred[:, -1:, :]
        
        new_x = newest_state[0][0][0].item()
        new_y = newest_state[0][0][1].item()
        
        x_path.append(new_x)
        y_path.append(new_y)
        
        # Update state history for next second
        state_history = torch.cat((state_history, newest_state), dim=1)
        
        print(f"Sec {step+1}: X={new_x:.1f}m, Y={new_y:.1f}m | Thrust={thrust:.1f}, Angle={math.degrees(angle_rad):.1f}°")

# 3. GRAPH THE TRAJECTORY
plt.figure(figsize=(8, 6))
plt.plot(x_path, y_path, marker='o', linestyle='-', color='purple', label='LSTM AI Flight Path')
plt.scatter(0, 0, color='red', s=100, label='Target Landing Zone (0,0)', zorder=5)

plt.title("Phase 13: LSTM Memory Pilot Trajectory (POMDP)", fontsize=14)
plt.xlabel("Horizontal Position (X meters)", fontsize=12)
plt.ylabel("Vertical Altitude (Y meters)", fontsize=12)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()