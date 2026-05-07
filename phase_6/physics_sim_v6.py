import pandas as pd
import numpy as np
import random
import os

def generate_noisy_data(samples=3000):
    data = []
    for _ in range(samples):
        # Ground Truth (Perfect Physics)
        y_true = random.uniform(50, 500)
        v_true = random.uniform(-20, 20)
        thrust = random.choice([0.0, 15.0, 30.0])
        g, dt = -9.8, 1.0
        
        a = g + thrust
        y_next_true = y_true + v_true * dt + 0.5 * a * (dt**2)
        v_next_true = v_true + a * dt
        
        # --- PHASE 6: ADDING SENSOR NOISE ---
        # We add "Gaussian Noise" to our observations. 
        # The AI never sees the "True" value; it only sees the "Noisy" value.
        noise_level = 2.0 
        y_obs = y_true + np.random.normal(0, noise_level)
        v_obs = v_true + np.random.normal(0, noise_level)
        
        data.append([y_obs, v_obs, thrust, y_next_true, v_next_true])
        
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(data, columns=['y_noisy', 'v_noisy', 'thrust', 'y_next', 'v_next'])
    df.to_csv("data/noisy_training_v6.csv", index=False)
    print("🛰️ Phase 6: Noisy Sensor Data Generated.")

if __name__ == "__main__":
    generate_noisy_data()