import torch
import os
import sys

# Root Path Fix
sys.path.insert(0, os.getcwd()) 
from models.architecture import ActionWorldModel

# 1. Load the Robust Brain
model = ActionWorldModel()
model.load_state_dict(torch.load("action_brain_v6.pth"))
model.eval()

def run_detailed_test(y, v, thrust, label):
    # Physics Ground Truth (The "Perfect" Reality)
    g, dt = -9.8, 1.0
    a = g + thrust
    true_y = y + v * dt + 0.5 * a * (dt**2)
    true_v = v + a * dt

    # AI Prediction (Based on Phase 6 Robust Training)
    input_tensor = torch.tensor([[y, v, thrust]], dtype=torch.float32)
    with torch.no_grad():
        pred = model(input_tensor).numpy()[0]
    
    pred_y, pred_v = pred[0], pred[1]

    # Accuracy Math Breakdown
    error_y = abs(true_y - pred_y)
    accuracy_y = 100 - (error_y / abs(true_y) * 100)

    print(f"\n--- Scenario {label}: Thrust={thrust} ---")
    print(f"INPUTS:  Height={y:.2f}, Velocity={v:.2f}")
    print(f"RESULTS: Actual={true_y:.2f} | Predicted={pred_y:.2f}")
    print(f"MATH:    Error = {error_y:.2f} meters")
    print(f"FINAL ACCURACY: {accuracy_y:.2f}%")

print("🛰️ PHASE 6: Testing Robustness Against Perfect Reality...")
# Test 1: No thrust (Gravity only)
run_detailed_test(100.0, 0.0, 0.0, "Free Fall")
# Test 2: Moderate thrust
run_detailed_test(100.0, 0.0, 15.0, "Low Boost")
# Test 3: High thrust
run_detailed_test(100.0, 0.0, 25.0, "Rocket Launch")