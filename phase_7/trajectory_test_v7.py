import torch
import os
import sys
import matplotlib.pyplot as plt

# 1. SETUP: Tell Python where to find your 'models' folder
sys.path.insert(0, os.getcwd()) 
from models.architecture import ActionWorldModel

# 2. LOAD BRAIN: Use the path found in the root folder
model = ActionWorldModel()
model.load_state_dict(torch.load("action_brain_v6.pth"))
model.eval()

def simulate_flight(start_y, start_v, thrust_value, steps=10):
    # Prepare the starting state for the AI
    current_state = torch.tensor([[start_y, start_v, thrust_value]], dtype=torch.float32)
    
    predictions, actuals = [], []
    y_real, v_real = start_y, start_v
    g, dt = -9.8, 1.0
    a = g + thrust_value

    print(f"\n🚀 Simulating {steps} seconds of flight...")
    
    for t in range(steps):
        # AI PREDICTION: The AI "imagines" the next step
        with torch.no_grad():
            pred = model(current_state).numpy()[0]
            pred_y, pred_v = pred[0], pred[1]
            predictions.append(pred_y)
            
        # REAL PHYSICS: The "Perfect" math for comparison
        y_real = y_real + v_real * dt + 0.5 * a * (dt**2)
        v_real = v_real + a * dt
        actuals.append(y_real)
        
        # FEEDBACK LOOP: The AI uses its own prediction as the next input
        current_state = torch.tensor([[pred_y, pred_v, thrust_value]], dtype=torch.float32)
        
        print(f"Sec {t+1}: AI: {pred_y:.2f}m | Real: {y_real:.2f}m")

    return predictions, actuals

# 3. EXECUTION: Run and Graph
steps = 10
ai_path, real_path = simulate_flight(500.0, 0.0, 5.0, steps=steps)

plt.plot(range(steps), real_path, label="Real Physics", color="black", linestyle="--")
plt.plot(range(steps), ai_path, label="AI Imagination", color="blue")
plt.xlabel("Time (seconds)")
plt.ylabel("Height (meters)")
plt.title("Phase 7: The Compounding Error Effect")
plt.legend()
plt.show()