"""
DEMO 2 - World Model Grid (Phase 10)
=====================================
9 charts: 3 thrust cases x 3 angle cases
Each chart: real physics (black dashed) vs Phase 10 AI (red)
Time dots on both lines. No start/target markers.
"""

import sys, os, math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "models", "weights")

START_X, START_Y   =  0.0, 300.0
START_VX, START_VY =  5.0, -10.0
STEPS, G, DT       = 15, -9.8, 1.0
X_MIN = START_X - 100
X_MAX = START_X + 200
Y_MIN = START_Y - 150
Y_MAX = START_Y + 100


class LatentWorldModel2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(6,32), nn.ReLU(), nn.Linear(32,3))
        self.decoder = nn.Sequential(nn.Linear(3,32), nn.ReLU(), nn.Linear(32,4))
    def forward(self, sa): return self.decoder(self.encoder(sa))

def real_physics(thrust, angle):
    x, y, vx, vy = START_X, START_Y, START_VX, START_VY
    xs, ys = [x], [y]
    for _ in range(STEPS):
        ax = thrust*math.sin(angle); ay = G+thrust*math.cos(angle)
        vx+=ax*DT; vy+=ay*DT; x+=vx*DT; y+=vy*DT
        xs.append(x); ys.append(y)
    return xs, ys

def rollout_p10(thrust, angle):
    m = LatentWorldModel2D()
    m.load_state_dict(torch.load(os.path.join(WEIGHTS,"latent_brain_2d_v10.pth"), map_location="cpu"))
    m.eval()
    s = torch.tensor([[START_X,START_Y,START_VX,START_VY]], dtype=torch.float32)
    xs, ys = [START_X], [START_Y]
    with torch.no_grad():
        for _ in range(STEPS):
            a  = torch.tensor([[angle,thrust]], dtype=torch.float32)
            ns = m(torch.cat((s,a),dim=1))
            xs.append(ns[0][0].item()); ys.append(ns[0][1].item())
            s = ns
    return xs, ys

def run_demo():
    thrust_cases  = [0.0, 10.0, 20.0]
    thrust_labels = ["Thrust = 0  (Free Fall)",
                     "Thrust = 10  (Moderate)",
                     "Thrust = 20  (Max Boost)"]
    angle_cases   = [-math.pi/6, 0.0, math.pi/6]
    angle_labels  = ["Angle = −30°\n(thrust tilts left)",
                     "Angle =  0°\n(thrust straight up)",
                     "Angle = +30°\n(thrust tilts right)"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(
        "DEMO 2: Phase 10 World Model — Physics Learning\n"
        "9 Test Cases: 3 Thrust × 3 Angle  |  Dots = 1 second intervals\n"
        "Black dashed = Real Physics  |  Red = AI Prediction",
        fontsize=12, fontweight='bold')

    for row, (thrust, tlabel) in enumerate(zip(thrust_cases, thrust_labels)):
        for col, (angle, alabel) in enumerate(zip(angle_cases, angle_labels)):
            ax = axes[row][col]
            rx, ry = real_physics(thrust, angle)
            px, py = rollout_p10(thrust, angle)

            ax.plot(rx, ry, color='black',   lw=2.2, ls='--', zorder=5)
            ax.plot(px, py, color='#e74c3c', lw=2.0, zorder=4)
            ax.scatter(rx, ry, color='black',   s=18, zorder=6)
            ax.scatter(px, py, color='#e74c3c', s=18, marker='s', zorder=6)

            ax.axhline(0, color='green', lw=0.8, ls=':', alpha=0.5)
            ax.set_xlim(X_MIN, X_MAX)
            ax.set_ylim(Y_MIN, Y_MAX)
            ax.grid(True, alpha=0.3)

            # Row label on leftmost column
            if col == 0:
                ax.set_ylabel(f"{tlabel}\nY Altitude (m)", fontsize=8)
            else:
                ax.set_ylabel("Y Altitude (m)", fontsize=8)

            # Column label on top row
            if row == 0:
                ax.set_title(alabel, fontsize=9)

            ax.set_xlabel("X Position (m)", fontsize=8)
            ax.tick_params(labelsize=7)

    # Shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], color='black',   lw=2, ls='--', label='Real Physics'),
        Line2D([0],[0], color='#e74c3c', lw=2,          label='Phase 10 AI Prediction'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.join(ROOT,"demo_output"), exist_ok=True)
    out = os.path.join(ROOT,"demo_output","demo2_world.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {out}")
    plt.show()

if __name__ == "__main__":
    run_demo()