"""
DEMO 3C - Combined Pilot Comparison
=====================================
Left chart: Trajectory
  Green = RSSM Pilot (Phase 13)
  Blue  = Transformer Pilot (Phase 15)
  NO real physics line
  Start marker + Target marker
  Time dots on both lines

Right chart: Distance to target over time
  Both pilots on same chart
"""

import sys, os, math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "models", "weights")

START_X, START_Y   =  0.0, 300.0
START_VX, START_VY =  5.0, -10.0
TARGET_X, TARGET_Y = 75.0,   0.0
STEPS, G, DT       = 15, -9.8, 1.0
X_MIN, X_MAX       = -20, 160
Y_MIN, Y_MAX       = -30, 380

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

class ActorRSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=4, hidden_size=32, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(32,16), nn.ReLU(), nn.Linear(16,2))
    def forward(self, seq):
        out,_ = self.lstm(seq)
        r = self.decoder(out[:,-1,:])
        return torch.cat((torch.tanh(r[:,0:1])*(math.pi/3),
                          torch.sigmoid(r[:,1:2])*20.0), dim=1)

class ActorTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=4, hidden_size=32, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(32,16), nn.ReLU(), nn.Linear(16,2))
    def forward(self, seq):
        out,_ = self.lstm(seq)
        r = self.decoder(out[:,-1,:])
        return torch.cat((torch.tanh(r[:,0:1])*(math.pi/3),
                          torch.sigmoid(r[:,1:2])*20.0), dim=1)

# ── Rollouts ──────────────────────────────────────────────────────────────────
def rollout_rssm():
    world = LSTMRSSM()
    world.load_state_dict(torch.load(os.path.join(WEIGHTS,"rssm_world_v14.pth"), map_location="cpu"))
    world.eval()
    for p in world.parameters(): p.requires_grad=False
    actor = ActorRSSM()
    actor.load_state_dict(torch.load(os.path.join(WEIGHTS,"rssm_actor_v13.pth"), map_location="cpu"))
    actor.eval()
    sh = torch.tensor([[[START_X,START_Y,START_VX,START_VY]]],dtype=torch.float32)
    ah = None
    xs, ys, dists = [START_X],[START_Y],[]
    with torch.no_grad():
        for _ in range(STEPS):
            action = actor(sh)
            ae = action.unsqueeze(1)
            ah = ae if ah is None else torch.cat((ah,ae),dim=1)
            ns = world.forward_mean(torch.cat((sh,ah),dim=2))
            newest = ns[:,-1:,:]
            nx,ny = newest[0,0,0].item(), newest[0,0,1].item()
            xs.append(nx); ys.append(ny)
            dists.append(math.sqrt((nx-TARGET_X)**2+(ny-TARGET_Y)**2))
            sh = torch.cat((sh,newest),dim=1)
    return xs, ys, dists

def rollout_transformer():
    world = TransformerWorldModel()
    world.load_state_dict(torch.load(os.path.join(WEIGHTS,"transformer_world_v15.pth"), map_location="cpu"))
    world.eval()
    for p in world.parameters(): p.requires_grad=False
    actor = ActorTransformer()
    actor.load_state_dict(torch.load(os.path.join(WEIGHTS,"transformer_actor_v15.pth"), map_location="cpu"))
    actor.eval()
    sh = torch.tensor([[[START_X,START_Y,START_VX,START_VY]]],dtype=torch.float32)
    ah = None
    xs, ys, dists = [START_X],[START_Y],[]
    with torch.no_grad():
        for _ in range(STEPS):
            action = actor(sh)
            ae = action.unsqueeze(1)
            ah = ae if ah is None else torch.cat((ah,ae),dim=1)
            ns = world(torch.cat((sh,ah),dim=2))
            newest = ns[:,-1:,:]
            nx,ny = newest[0,0,0].item(), newest[0,0,1].item()
            xs.append(nx); ys.append(ny)
            dists.append(math.sqrt((nx-TARGET_X)**2+(ny-TARGET_Y)**2))
            sh = torch.cat((sh,newest),dim=1)
    return xs, ys, dists

# ── Plot ──────────────────────────────────────────────────────────────────────
def run_demo():
    r13x, r13y, d13 = rollout_rssm()
    r15x, r15y, d15 = rollout_transformer()
    time_steps      = list(range(1, STEPS+1))

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        "DEMO 3C: RSSM vs Transformer Actor-Critic Pilots — Direct Comparison\n"
        "Start: (X=0, Y=300)  →  Target: (X=75, Y=0)  |  Dots = 1 second intervals",
        fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Left: Trajectory ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(X_MIN, X_MAX)
    ax1.set_ylim(Y_MIN, Y_MAX)

    ax1.plot(r13x, r13y, color='#27ae60', lw=2.5, label='Ph13 RSSM Pilot')
    ax1.plot(r15x, r15y, color='#3498db', lw=2.5, label='Ph15 Transformer Pilot')

    ax1.scatter(r13x, r13y, color='#27ae60', s=28, zorder=6)
    ax1.scatter(r15x, r15y, color='#3498db', s=28, marker='s', zorder=6)

    # Time labels at t=5, 10, 15
    for t in [4, 9, 14]:
        ax1.annotate(f"t={t+1}s",
            xy=(r13x[t+1], r13y[t+1]), xytext=(-38, 5),
            textcoords='offset points', fontsize=8, color='#27ae60')
        ax1.annotate(f"t={t+1}s",
            xy=(r15x[t+1], r15y[t+1]), xytext=(6, -14),
            textcoords='offset points', fontsize=8, color='#3498db')

    # Final positions
    ax1.scatter(r13x[-1], r13y[-1], color='#27ae60', s=160,
                marker='*', zorder=9, edgecolors='black', lw=0.8)
    ax1.scatter(r15x[-1], r15y[-1], color='#3498db', s=160,
                marker='*', zorder=9, edgecolors='black', lw=0.8)

    # Start + target markers
    ax1.scatter(START_X, START_Y, color="#fffb00", s=200, marker='^',
                zorder=10, label=f'Start ({START_X:.0f}, {START_Y:.0f})',
                edgecolors='black', lw=0.8)
    ax1.scatter(TARGET_X, TARGET_Y, color="#f93434", s=200,
                zorder=10, label=f'Target ({TARGET_X:.0f}, {TARGET_Y:.0f})',
                edgecolors='black', lw=0.8)

    ax1.axhline(0, color='green', lw=1, ls=':', alpha=0.6)
    ax1.set_title("Pilot Trajectories", fontsize=11)
    ax1.set_xlabel("X Position (m)", fontsize=11)
    ax1.set_ylabel("Y Altitude (m)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Right: Distance to target ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_steps, d13, color='#27ae60', lw=2.5,
             marker='o', markersize=5, label=f'RSSM Pilot  (final={d13[-1]:.1f}m)')
    ax2.plot(time_steps, d15, color='#3498db', lw=2.5,
             marker='s', markersize=5, label=f'Transformer Pilot  (final={d15[-1]:.1f}m)')
    ax2.fill_between(time_steps, d13, alpha=0.1, color='#27ae60')
    ax2.fill_between(time_steps, d15, alpha=0.1, color='#3498db')

    # Annotate final values
    ax2.annotate(f"{d13[-1]:.1f}m",
        xy=(STEPS, d13[-1]), xytext=(-40, 8),
        textcoords='offset points', fontsize=11,
        color='#27ae60', fontweight='bold')
    ax2.annotate(f"{d15[-1]:.1f}m",
        xy=(STEPS, d15[-1]), xytext=(-40, -20),
        textcoords='offset points', fontsize=11,
        color='#3498db', fontweight='bold')

    winner       = "Transformer" if d15[-1] < d13[-1] else "RSSM"
    winner_color = "#000000" if winner == "Transformer" else '#27ae60'
    ax2.set_title(f"Distance to Target Over Time\n{winner} Pilot gets closer",
                  fontsize=11, color=winner_color)
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel("Distance from Target (m)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.join(ROOT,"demo_output"), exist_ok=True)
    out = os.path.join(ROOT,"demo_output","demo3c_combined.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {out}")
    plt.show()

if __name__ == "__main__":
    run_demo()