# This was your static Earth simulator and it did improve from 150% to 5% error,
#  but it still is not good enough to be used as a world model.

import pandas as pd

def generate_gravity_data(initial_height=100.0, gravity=-9.8, time_steps=10):
    """Simulates dropping a ball from a height."""
    data = []
    
    y = initial_height # Current height
    vy = 0.0           # Current vertical velocity
    dt = 1.0           # Time step (1 second)
    
    for step in range(time_steps):
        # Record the current state
        data.append({
            'Time': step,
            'Y_Position': round(y, 2),
            'Y_Velocity': round(vy, 2)
        })
        
        # Physics Engine update (Kinematics)
        y = y + (vy * dt)           # Position changes based on velocity
        vy = vy + (gravity * dt)    # Velocity changes based on gravity
        
        # Stop if it hits the ground
        if y <= 0:
            data.append({'Time': step + 1, 'Y_Position': 0.0, 'Y_Velocity': 0.0})
            break
            
    return pd.DataFrame(data)

# Generate the physics data
physics_dataset = generate_gravity_data()
print("--- RAW WORLD STATE DATA ---")
print(physics_dataset)

# Save it to use in Week 3 and 4
physics_dataset.to_csv("gravity_states.csv", index=False)