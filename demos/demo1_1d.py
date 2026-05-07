"""
DEMO 1: The 1D Era (Phases 6, 8, 9)
=====================================
Same scenario across all charts:
  Start: Height=200m, Velocity=-10 m/s
  Thrust: 0.0, 10.0, 20.0 m/s²
  Duration: 15 seconds

Fixed axis limits for clean comparison.
"""

import sys, os, math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "models", "weights")

START_Y, START_V  = 200.0, -10.0
THRUST_CASES      = [0.0, 10.0, 20.0]
THRUST_LABELS     = ["Thrust = 0  (Free Fall)", "Thrust = 10  (Moderate)", "Thrust = 20  (Max Boost)"]
STEPS, G, DT      = 15, -9.8, 1.0

# ── Axis limits (same across all charts) ─────────────────────────────────────
Y_MIN, Y_MAX = -50, 450
T_MIN, T_MAX =   0,  15

# ── Architectures ─────────────────────────────────────────────────────────────
class ActionWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2))
    def forward(self, x): return self.net(x)

class LatentWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3,16), nn.ReLU(), nn.Linear(16,2))
        self.decoder = nn.Sequential(nn.Linear(2,16), nn.ReLU(), nn.Linear(16,2))
    def forward(self, x): return self.decoder(self.encoder(x))

class PilotNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,16), nn.ReLU(),
            nn.Linear(16,1), nn.Sigmoid())
    def forward(self, s): return self.net(s) * 20.0

# ── Physics ───────────────────────────────────────────────────────────────────
def real_1d(thrust, steps=STEPS):
    y, v = START_Y, START_V
    ys = [y]
    for _ in range(steps):
        a = G + thrust
        v += a * DT
        y += v * DT
        ys.append(y)
    return ys

# ── Rollouts ──────────────────────────────────────────────────────────────────
def rollout_p6(thrust):
    m = ActionWorldModel()
    m.load_state_dict(torch.load(os.path.join(WEIGHTS,"action_brain_v6.pth"), map_location="cpu"))
    m.eval()
    y, v = START_Y, START_V
    ys = [y]
    with torch.no_grad():
        for _ in range(STEPS):
            p = m(torch.tensor([[y,v,thrust]],dtype=torch.float32)).numpy()[0]
            y, v = p[0], p[1]
            ys.append(y)
    return ys

def rollout_p8(thrust):
    m = LatentWorldModel()
    m.load_state_dict(torch.load(os.path.join(WEIGHTS,"latent_brain_v8.pth"), map_location="cpu"))
    m.eval()
    y, v = START_Y, START_V
    ys = [y]
    with torch.no_grad():
        for _ in range(STEPS):
            p = m(torch.tensor([[y,v,thrust]],dtype=torch.float32)).numpy()[0]
            y, v = p[0], p[1]
            ys.append(y)
    return ys

def rollout_p9():
    world = LatentWorldModel()
    world.load_state_dict(torch.load(os.path.join(WEIGHTS,"latent_brain_v8.pth"), map_location="cpu"))
    world.eval()
    pilot = PilotNetwork()
    pilot.load_state_dict(torch.load(os.path.join(WEIGHTS,"pilot_v9.pth"), map_location="cpu"))
    pilot.eval()
    y, v = START_Y, START_V
    ys, thrusts = [y], []
    with torch.no_grad():
        for _ in range(STEPS):
            t  = pilot(torch.tensor([[y,v]],dtype=torch.float32)).item()
            thrusts.append(t)
            p  = world(torch.tensor([[y,v,t]],dtype=torch.float32)).numpy()[0]
            y, v = p[0], p[1]
            ys.append(y)
    return ys, thrusts

# ── Plot ──────────────────────────────────────────────────────────────────────
def run_demo():
    time_axis = list(range(STEPS+1))

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "DEMO 1: The 1D Era — World Model Progression\n"
        "Start: Height=200m, Velocity=−10 m/s  |  Duration: 15 seconds",
        fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

    # ── Row 0: Phase 6 vs Phase 8 at each thrust ──────────────────────────────
    for col, (thrust, label) in enumerate(zip(THRUST_CASES, THRUST_LABELS)):
        ax = fig.add_subplot(gs[0, col])
        real = real_1d(thrust)
        p6   = rollout_p6(thrust)
        p8   = rollout_p8(thrust)

        ax.plot(time_axis, real, color='black',   lw=2.5, ls='--', label='Real Physics', zorder=5)
        ax.plot(time_axis, p6,   color='#e74c3c', lw=2,   label='Phase 6 (Feed-Forward)')
        ax.plot(time_axis, p8,   color='#3498db', lw=2,   label='Phase 8 (Latent)')
        ax.scatter(time_axis, real, color='black',   s=18, zorder=6)
        ax.scatter(time_axis, p6,   color='#e74c3c', s=18, zorder=6)
        ax.scatter(time_axis, p8,   color='#3498db', s=18, zorder=6)

        ax.axhline(0, color='green', lw=1, ls=':', alpha=0.6)
        ax.set_xlim(T_MIN, T_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Height (m)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Row 1 left+center: Phase 9 Pilot trajectory ───────────────────────────
    ax_t = fig.add_subplot(gs[1, 0:2])
    p9_y, p9_thrusts = rollout_p9()
    avg_t  = np.mean(p9_thrusts)
    real_9 = real_1d(avg_t)

    ax_t.plot(time_axis, real_9, color='black',   lw=2.5, ls='--',
              label=f'Real Physics (avg thrust={avg_t:.1f})', zorder=5)
    ax_t.plot(time_axis, p9_y,   color='#9b59b6', lw=2,
              label='Phase 9 Pilot (Actor)')
    ax_t.scatter(time_axis, real_9, color='black',   s=22, zorder=6)
    ax_t.scatter(time_axis, p9_y,   color='#9b59b6', s=22, zorder=6)
    ax_t.axhline(0, color='green', lw=1.5, ls=':', alpha=0.7, label='Ground (Y=0)')
    ax_t.set_xlim(T_MIN, T_MAX)
    ax_t.set_ylim(Y_MIN, Y_MAX)
    ax_t.set_title("Phase 9: First Pilot (Actor)\nDoes it learn to land softly?", fontsize=10)
    ax_t.set_xlabel("Time (seconds)")
    ax_t.set_ylabel("Height (m)")
    ax_t.legend(fontsize=8)
    ax_t.grid(True, alpha=0.3)

    # ── Row 1 right: Thrust decisions ─────────────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 2])
    ax_b.bar(range(1, STEPS+1), p9_thrusts, color='#9b59b6', alpha=0.85)
    ax_b.axhline(avg_t, color='red', lw=1.5, ls='--', label=f'Avg = {avg_t:.1f}')
    ax_b.set_xlim(0.5, STEPS+0.5)
    ax_b.set_ylim(0, 22)
    ax_b.set_title("Phase 9: Thrust Decisions\nper Second", fontsize=10)
    ax_b.set_xlabel("Time (seconds)")
    ax_b.set_ylabel("Thrust (m/s²)")
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.3)

    os.makedirs(os.path.join(ROOT,"demo_output"), exist_ok=True)
    out = os.path.join(ROOT,"demo_output","demo1_1d.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✅ Demo 1 saved: {out}")
    plt.show()

if __name__ == "__main__":
    run_demo()