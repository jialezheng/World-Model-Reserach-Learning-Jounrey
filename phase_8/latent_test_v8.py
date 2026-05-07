import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Setup paths
sys.path.insert(0, os.getcwd()) 
from models.architecture_v8 import LatentWorldModel

# 1. Load the "Compressed" Brain
model = LatentWorldModel()
model.load_state_dict(torch.load("models/weights/latent_brain_v8.pth"))
model.eval()

def simulate_latent_flight(start_y, start_v, thrust, steps=10):
    # Prepare the starting state
    current_state = torch.tensor([[start_y, start_v, thrust]], dtype=torch.float32)
    
    preds, actuals = [], []
    y_real, v_real = start_y, start_v
    g, dt = -9.8, 1.0
    a = g + thrust

    print(f"\n🚀 Phase 9: Latent Space Simulation...")
    
    for t in range(steps):
        with torch.no_grad():
            # AI predicts next state
            pred = model(current_state).numpy()[0]
            pred_y, pred_v = pred[0], pred[1]
            preds.append(pred_y)
            
        # Real Math
        y_real = y_real + v_real * dt + 0.5 * a * (dt**2)
        v_real = v_real + a * dt
        actuals.append(y_real)
        
        # Update current state for the next loop
        current_state = torch.tensor([[pred_y, pred_v, thrust]], dtype=torch.float32)
        
        print(f"Sec {t+1}: AI: {pred_y:.2f}m | Real: {y_real:.2f}m")

    return preds, actuals

# Run it
ai_y, real_y = simulate_latent_flight(500.0, 0.0, 5.0)

# Graph it
plt.plot(real_y, label="Real Physics", color="black", linestyle="--")
plt.plot(ai_y, label="Latent AI (v8)", color="red")
plt.title("Phase 9: Latent World Model Prediction")
plt.legend()
plt.show()