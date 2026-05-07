import os

import pandas as pd
import random

def generate_action_data(samples=2000):
    data = []
    for _ in range(samples):
        y = random.uniform(50, 500)
        v = random.uniform(-20, 20)
        # Randomly decide to fire thrusters (15.0 m/s^2) or not (0.0)
        thrust = random.choice([0.0, 15.0, 25.0]) 
        g = -9.8
        dt = 1.0
        
        # Physics: Acceleration = Gravity + Thrust
        total_accel = g + thrust
        next_v = v + total_accel * dt
        next_y = y + v * dt + 0.5 * total_accel * (dt**2)
        
        data.append([y, v, thrust, next_y, next_v])
    
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(data, columns=['y', 'v', 'thrust', 'y_next', 'v_next'])
    df.to_csv("data/action_training_v5.csv", index=False)
    print("🚀 Phase 5 Data Generated with Action Vectors.")

if __name__ == "__main__":
    generate_action_data()