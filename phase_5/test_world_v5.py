import torch
import pandas as pd
import sys
import os

# This tells Python to look at your terminal's current folder for the 'models' folder
sys.path.insert(0, os.getcwd()) 

from models.architecture import ActionWorldModel

# 1. Load the trained brain
model = ActionWorldModel()
model.load_state_dict(torch.load("action_brain_v5.pth"))
model.eval() # Set to evaluation mode

def run_test(y, v, thrust):
    # Physics Ground Truth (The actual math)
    g = -9.8
    dt = 1.0
    a = g + thrust
    true_v_next = v + a * dt
    true_y_next = y + v * dt + 0.5 * a * (dt**2)

    # AI Prediction
    input_tensor = torch.tensor([[y, v, thrust]], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).numpy()[0]
    
    pred_y_next, pred_v_next = prediction[0], prediction[1]

    print(f"\n--- Scenario: Height={y}, Velocity={v}, Thrust={thrust} ---")
    print(f"REAL PHYSICS: Next Height: {true_y_next:.2f}, Next Velocity: {true_v_next:.2f}")
    print(f"AI PREDICTION: Next Height: {pred_y_next:.2f}, Next Velocity: {pred_v_next:.2f}")
    
    error = abs(true_y_next - pred_y_next)
    print(f"ACCURACY: {100 - (error/abs(true_y_next)*100):.2f}%")

# --- TEST CASES ---
run_test(100.0, 0.0, 0.0)    # Free Fall (No thrust)
run_test(100.0, 0.0, 15.0)   # Slow Ascent (Low thrust)
run_test(100.0, 0.0, 25.0)   # Rocket Launch (High thrust)