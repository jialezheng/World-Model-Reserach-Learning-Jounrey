"""
DEMO 2 - Phase 11 Pilot (single chart)
========================================
Black dashed = real physics (adaptive angle)
Red line     = Phase 11 Pilot
Start marker (blue triangle) + Target marker (green dot)
Time dots on both lines.
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
TARGET_X, TARGET_Y = 75.0,   0.0
STEPS, G, DT       = 15, -9.8, 1.0
X_MIN, X_MAX       = -20, 160
Y_MIN, Y_MAX       = -30, 380

class LatentWorldModel2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(6,32), nn.ReLU(), nn.Linear(32,3))
        self.decoder = nn.Sequential(nn.Linear(3,32), nn.ReLU(), nn.Linear(32,4))
    def forward(self, sa): return self.decoder(self.encoder(sa))

class Pilot2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,2))
    def forward(self, s):
        r = self.net(s)
        return torch.cat((torch.tanh(r[:,0:1])*(math.pi/3),
                          torch.sigmoid(r[:,1:2])*20.0), dim=1)

def real_adaptive():
    x, y, vx, vy = START_X, START_Y, START_VX, START_VY
    xs, ys = [x], [y]
    for _ in range(STEPS):
        dx  = TARGET_X - x
        dy  = TARGET_Y - y
        ang = math.atan2(dx, max(abs(dy), 1)) * 0.5
        thr = 10.0
        ax_ = thr*math.sin(ang); ay_ = G+thr*math.cos(ang)
        vx+=ax_*DT; vy+=ay_*DT; x+=vx*DT; y+=vy*DT
        xs.append(x); ys.append(y)
    return xs, ys

def rollout_p11():
    world = LatentWorldModel2D()
    world.load_state_dict(torch.load(os.path.join(WEIGHTS,"latent_brain_2d_v10.pth"), map_location="cpu"))
    world.eval()
    for p in world.parameters(): p.requires_grad=False
    pilot = Pilot2D()
    pilot.load_state_dict(torch.load(os.path.join(WEIGHTS,"pilot_2d_v11.pth"), map_location="cpu"))
    pilot.eval()
    s = torch.tensor([[START_X,START_Y,START_VX,START_VY]], dtype=torch.float32)
    xs, ys = [START_X], [START_Y]
    with torch.no_grad():
        for _ in range(STEPS):
            act = pilot(s)
            ns  = world(torch.cat((s,act),dim=1))
            xs.append(ns[0][0].item()); ys.append(ns[0][1].item())
            s = ns
    return xs, ys

def run_demo():
    rx, ry = real_adaptive()
    px, py = rollout_p11()

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(
        "DEMO 2: Phase 11 — 2D Feed-Forward Pilot\n"
        "Start: (X=0, Y=300)  →  Target: (X=75, Y=0)  |  Dots = 1 second intervals",
        fontsize=12, fontweight='bold')

    ax.plot(rx, ry, color='black',   lw=2.5, ls='--', label='Real Physics (adaptive)', zorder=5)
    ax.plot(px, py, color='#e74c3c', lw=2.5, label='Phase 11 Pilot (Feed-Forward)', zorder=4)

    # Time dots
    ax.scatter(rx, ry, color='black',   s=30, zorder=6)
    ax.scatter(px, py, color='#e74c3c', s=30, marker='s', zorder=6)

    # Time labels at t=5, 10, 15
    for t in [4, 9, 14]:
        ax.annotate(f"t={t+1}s",
            xy=(rx[t+1], ry[t+1]), xytext=(-38, 5),
            textcoords='offset points', fontsize=8, color='black')
        ax.annotate(f"t={t+1}s",
            xy=(px[t+1], py[t+1]), xytext=(6, -14),
            textcoords='offset points', fontsize=8, color='#e74c3c')

    # Start and target markers
    ax.scatter(START_X, START_Y, color='#2980b9', s=180, marker='^',
               zorder=8, label=f'Start ({START_X:.0f}, {START_Y:.0f})', edgecolors='black', lw=0.8)
    ax.scatter(TARGET_X, TARGET_Y, color='#27ae60', s=180,
               zorder=8, label=f'Target ({TARGET_X:.0f}, {TARGET_Y:.0f})', edgecolors='black', lw=0.8)

    # Final position of pilot
    final_dist = math.sqrt((px[-1]-TARGET_X)**2 + (py[-1]-TARGET_Y)**2)
    ax.scatter(px[-1], py[-1], color='#e74c3c', s=160, marker='*',
               zorder=9, edgecolors='black', lw=0.8,
               label=f'Pilot ended ({px[-1]:.1f}, {py[-1]:.1f})  dist={final_dist:.1f}m')

    ax.axhline(0, color='green', lw=1, ls=':', alpha=0.6)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("X Position (m)", fontsize=11)
    ax.set_ylabel("Y Altitude (m)", fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.join(ROOT,"demo_output"), exist_ok=True)
    out = os.path.join(ROOT,"demo_output","demo2_pilot.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {out}")
    plt.show()

if __name__ == "__main__":
    run_demo()