"""
DEMO 3 - World Model Grid (RSSM vs Transformer)
=================================================
9 charts: 3 thrust cases x 3 angle cases
Each chart: 3 lines
  Black dashed = Real Physics
  Green        = RSSM prediction (Phase 12)
  Blue         = Transformer prediction (Phase 14)
Time dots on all 3 lines. No start/target markers.
"""

import sys, os, math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "models", "weights")

START_X, START_Y   =  0.0, 300.0
START_VX, START_VY =  5.0, -10.0
STEPS, G, DT       = 15, -9.8, 1.0
X_MIN = START_X - 100
X_MAX = START_X + 200
Y_MIN = START_Y - 150
Y_MAX = START_Y + 100


# ── Architectures ─────────────────────────────────────────────────────────────
class LSTMRSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm      = nn.LSTM(input_size=6, hidden_size=32, batch_first=True)
        self.fc_mu     = nn.Linear(32, 8)
        self.fc_logvar = nn.Linear(32, 8)
        self.decoder   = nn.Sequential(nn.Linear(40,16), nn.ReLU(), nn.Linear(16,4))
    def forward_mean(self, seq):
        out,_ = self.lstm(seq)
        return self.decoder(torch.cat((out, self.fc_mu(out)), dim=-1))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=32, max_len=50):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.pe = pe.unsqueeze(0)
    def forward(self, x):
        return x + self.pe[:,:x.size(1),:].to(x.device)

class TransformerWorldModel(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embedding   = nn.Linear(6, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model,nhead,batch_first=True,dropout=0.1), num_layers)
        self.decoder = nn.Linear(d_model, 4)
    def causal_mask(self, sz):
        m = (torch.triu(torch.ones(sz,sz))==1).transpose(0,1)
        return m.float().masked_fill(m==0,float('-inf')).masked_fill(m==1,0.0)
    def forward(self, src):
        x = self.pos_encoder(self.embedding(src))
        return self.decoder(self.transformer(x,
               mask=self.causal_mask(src.size(1)).to(src.device), is_causal=True))

# ── Physics ───────────────────────────────────────────────────────────────────
def real_physics(thrust, angle):
    x, y, vx, vy = START_X, START_Y, START_VX, START_VY
    xs, ys = [x], [y]
    for _ in range(STEPS):
        ax = thrust*math.sin(angle); ay = G+thrust*math.cos(angle)
        vx+=ax*DT; vy+=ay*DT; x+=vx*DT; y+=vy*DT
        xs.append(x); ys.append(y)
    return xs, ys

def rollout_rssm(thrust, angle):
    m = LSTMRSSM()
    m.load_state_dict(torch.load(os.path.join(WEIGHTS,"rssm_world_v14.pth"), map_location="cpu"))
    m.eval()
    x, y, vx, vy = START_X, START_Y, START_VX, START_VY
    xs, ys = [x], [y]; history=[]
    with torch.no_grad():
        for _ in range(STEPS):
            history.append([x,y,vx,vy,angle,thrust])
            pred = m.forward_mean(torch.tensor([history],dtype=torch.float32))
            last = pred[0,-1].numpy()
            x,y,vx,vy = last[0],last[1],last[2],last[3]
            xs.append(x); ys.append(y)
    return xs, ys

def rollout_transformer(thrust, angle):
    m = TransformerWorldModel()
    m.load_state_dict(torch.load(os.path.join(WEIGHTS,"transformer_world_v15.pth"), map_location="cpu"))
    m.eval()
    x, y, vx, vy = START_X, START_Y, START_VX, START_VY
    xs, ys = [x], [y]; history=[]
    with torch.no_grad():
        for _ in range(STEPS):
            history.append([x,y,vx,vy,angle,thrust])
            pred = m(torch.tensor([history],dtype=torch.float32))
            last = pred[0,-1].numpy()
            x,y,vx,vy = last[0],last[1],last[2],last[3]
            xs.append(x); ys.append(y)
    return xs, ys

# ── Plot ──────────────────────────────────────────────────────────────────────
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
        "DEMO 3: RSSM vs Transformer — World Model Comparison\n"
        "9 Test Cases: 3 Thrust × 3 Angle  |  Dots = 1 second intervals\n"
        "Black dashed = Real Physics  |  Green = RSSM  |  Blue = Transformer",
        fontsize=12, fontweight='bold')

    for row, (thrust, tlabel) in enumerate(zip(thrust_cases, thrust_labels)):
        for col, (angle, alabel) in enumerate(zip(angle_cases, angle_labels)):
            ax = axes[row][col]
            rx, ry = real_physics(thrust, angle)
            gx, gy = rollout_rssm(thrust, angle)
            bx, by = rollout_transformer(thrust, angle)

            ax.plot(rx, ry, color='black',   lw=2.2, ls='--', zorder=5)
            ax.plot(gx, gy, color='#27ae60', lw=2.0, zorder=4)
            ax.plot(bx, by, color='#3498db', lw=2.0, zorder=3)

            ax.scatter(rx, ry, color='black',   s=16, zorder=6)
            ax.scatter(gx, gy, color='#27ae60', s=16, marker='s', zorder=6)
            ax.scatter(bx, by, color='#3498db', s=16, marker='^', zorder=6)

            ax.axhline(0, color='green', lw=0.8, ls=':', alpha=0.5)
            ax.set_xlim(X_MIN, X_MAX)
            ax.set_ylim(Y_MIN, Y_MAX)
            ax.grid(True, alpha=0.3)

            if col == 0:
                ax.set_ylabel(f"{tlabel}\nY Altitude (m)", fontsize=8)
            else:
                ax.set_ylabel("Y Altitude (m)", fontsize=8)

            if row == 0:
                ax.set_title(alabel, fontsize=9)

            ax.set_xlabel("X Position (m)", fontsize=8)
            ax.tick_params(labelsize=7)

    legend_elements = [
        Line2D([0],[0], color='black',   lw=2, ls='--', label='Real Physics'),
        Line2D([0],[0], color='#27ae60', lw=2,          label='RSSM (Phase 12)'),
        Line2D([0],[0], color='#3498db', lw=2,          label='Transformer (Phase 14)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=11, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.join(ROOT,"demo_output"), exist_ok=True)
    out = os.path.join(ROOT,"demo_output","demo3_world.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {out}")
    plt.show()

if __name__ == "__main__":
    run_demo()