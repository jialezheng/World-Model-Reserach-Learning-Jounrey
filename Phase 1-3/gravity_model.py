# Gravity Model: Predicting Future States of a Falling 
# Object by using Llama as a World Model to understand 
# the physics of gravity and make predictions based 
# on observed data.
# The reusl is showing llama is not able to understand 
# physical rules and make accurate predictions based on the data provided.


import ollama
import pandas as pd

# 1. Load the data you just generated
df = pd.read_csv("gravity_states.csv")
data_string = df.to_string(index=False)

# 2. Craft the "Physics Intuition" Prompt
prompt = f"""
You are a World Model observing a falling object.
OBSERVED DATA (Time, Position, Velocity):
{data_string}

TASK:
Based on the pattern of acceleration (Gravity), predict the state for Time 6 and Time 7.
Note: At Time 6, the object hit the ground (0.0). If it were to continue 'falling' through the floor, what would the math suggest?

Final Output:
- Predicted Y_Position at Time 7: 
- Predicted Y_Velocity at Time 7:
- Reasoning: (Briefly explain the physics rule you found)
"""

print("🚀 Asking Llama to predict the future state...")
response = ollama.chat(model='llama3.2:1b', messages=[
    {'role': 'user', 'content': prompt},
])

print("\n--- LLM WORLD MODEL PREDICTION ---")
print(response['message']['content'])