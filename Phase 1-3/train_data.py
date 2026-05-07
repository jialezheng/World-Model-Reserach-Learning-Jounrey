import pandas as pd
import random

def generate_bulk_data(num_scenarios=1000):
    all_data = []
    for _ in range(num_scenarios):
        y = random.uniform(50, 500) # Random starting height
        vy = 0.0
        g = -9.8
        dt = 1.0
        
        for _ in range(5): # Record 5 steps per drop
            next_y = y + vy * dt
            next_vy = vy + g * dt
            all_data.append([y, vy, next_y, next_vy])
            y, vy = next_y, next_vy
            if y <= 0: break
            
    return pd.DataFrame(all_data, columns=['y_now', 'v_now', 'y_next', 'v_next'])

df = generate_bulk_data()
df.to_csv("world_model_training.csv", index=False)
print(f"Generated {len(df)} training samples.")