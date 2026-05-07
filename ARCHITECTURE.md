# Architecture Notes

A short reference for the neural network architectures used across the 15
phases. Every model in this project is small (CPU-trainable), so this document
focuses on the exact layer shapes, input/output contracts, and the concept that
each architecture is meant to demonstrate.

For a higher-level walkthrough of *what* each phase does and *how* to run it,
see `README.md`.

---

## Shared physics setup

All trajectories come from the same toy 2D rocket simulator:

```
ax = thrust × sin(angle)
ay = gravity + thrust × cos(angle)        # gravity = -9.8 m/s²
x_next  = x  + vx · dt + 0.5 · ax · dt²
y_next  = y  + vy · dt + 0.5 · ay · dt²
vx_next = vx + ax · dt
vy_next = vy + ay · dt
dt = 1.0 s
```

In Demos 2 and 3, angle ∈ [-60°, +60°] and thrust ∈ [0, 20] m/s².
In Demo 1 (1D) the same equations are used with angle = 0°.

---

## Demo 1 — 1D architectures

### `ActionWorldModel` (Phases 5, 6, 7) — `models/architecture.py`
A plain MLP that maps `(y, v, thrust) → (y_next, v_next)`.

```
Linear(3, 64) → ReLU
Linear(64, 64) → ReLU         # this 64-d hidden layer is the "latent"
Linear(64, 2)
```

- Phase 5 trains it on clean data (`data/action_training_v5.csv`).
- Phase 6 trains it on noisy data (`data/noisy_training_v6.csv`) → final loss ≈ 60.
- Phase 7 doesn't retrain — it just feeds the model's own outputs back in for
  10 seconds to expose compounding error.

### `LatentWorldModel` (Phase 8) — `models/architecture_v8.py`
The same job, but with a real bottleneck: only **2 latent numbers** are passed
between the encoder and the decoder.

```
encoder: Linear(3, 16) → ReLU → Linear(16, 2)
decoder: Linear(2, 16) → ReLU → Linear(16, 2)
```

This is the architecture the Phase 9 Pilot dreams inside.

### `PilotNetwork` (Phase 9) — defined inside `phase_9_continue/train_pilot_v9.py`
Maps `(y, v) → thrust ∈ [0, 20]`.

```
Linear(2, 16) → ReLU → Linear(16, 1) → Sigmoid → ×20
```

The pilot is trained against the frozen Phase 8 `LatentWorldModel` for 10
imagined steps per episode, with a Huber loss against `(y=0, v=0)`.

---

## Demo 2 — 2D architectures

### `LatentWorldModel2D` (Phase 10) — defined in `phase_9_continue/train_2d_world_v10.py`
Latent encoder–decoder over the 2D state-action space.

```
input  = (x, y, vx, vy, angle, thrust)         # 6
encoder: Linear(6, 32) → ReLU → Linear(32, 3)  # 3-d latent
decoder: Linear(3, 32) → ReLU → Linear(32, 4)  # (x, y, vx, vy)_next
```

- 30 000 random samples
- Adam, MSE loss
- Final loss ≈ 47.5

### `Pilot2D` (Phase 11) — defined in `phase_9_continue/train_2d_pilot_v11.py`
Maps `(x, y, vx, vy) → (angle, thrust)`.

```
Linear(4, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 2)
  └─ angle   = tanh(out₀) × π/3        # ±60°
  └─ thrust  = sigmoid(out₁) × 20      # [0, 20] m/s²
```

Loss penalises distance from target `(75, 0)` with a step weighting and a
velocity penalty. Trained against the frozen Phase 10 World Model. Has **no
memory** — every second's decision is independent. This is the failure mode
that motivates Demo 3.

---

## Demo 3 — Memory architectures

### `LSTMRSSM_v14` (Phase 12) — RSSM World Model
File: `phase_12_continue/train_rssm_world_v12.py`

```
LSTM(input=6, hidden=32, batch_first=True)        # deterministic h_t
fc_mu     : Linear(32, 8)                          # μ of stochastic z_t
fc_logvar : Linear(32, 8)                          # log σ² of stochastic z_t
z = μ + σ · ε                                      # reparameterisation trick
decoder   : Linear(40, 16) → ReLU → Linear(16, 4)  # combine h_t + z
```

Loss = MSE(prediction, target) + β · KL(N(μ, σ) || N(0, 1)).
The KL term forces the latent to admit when it doesn't know — this is what
"cures hallucinations" in the slide deck. Final training loss ≈ 220.

### `TransformerWorldModel_v15` (Phase 14) — Causal Transformer World Model
File: `phase_15_continue/train_transformer_world_v14.py`

```
embedding   : Linear(6, 32)
positional  : sinusoidal PE (max_len = 50)
transformer : TransformerEncoder(
                 layer = TransformerEncoderLayer(d_model=32, nhead=4,
                                                 batch_first=True, dropout=0.1),
                 num_layers = 2)
decoder     : Linear(32, 4)
```

A causal mask is generated for the input sequence length so the model cannot
attend to future timesteps. Trained on 2 000 trajectories of length 15 with
mini-batches of 256. Final training loss ≈ 48 — the best of the four World
Models.

### Actor–Critic Pilots (Phases 13 and 15)

Both pilots share the same shape:

```
LSTM Actor   : LSTM(input=4, hidden=...) → Linear → (angle ∈ ±60°, thrust ∈ [0, 20])
Value Critic : MLP that estimates V(state) at every step
```

Differences:

|                  | Phase 13 (RSSM Pilot)        | Phase 15 (Transformer Pilot)        |
|------------------|------------------------------|-------------------------------------|
| Frozen World Model | `LSTMRSSM_v14`             | `TransformerWorldModel_v15`         |
| Memory in WM     | LSTM `h_t` + stochastic `z_t` | Self-attention over full sequence  |
| Actor loss       | ≈ 386                        | ≈ 164                               |
| Critic loss      | ≈ 0.009                      | ≈ 0.011                             |
| Final distance to (75, 0) | ≈ 41.9 m            | ≈ **8.9 m**                         |

Returns are computed with discount `γ = 0.95`:

```
G_t = r_t + γ · G_{t+1}
```

The Critic is trained to regress on `G_t`. The Actor is trained to maximise
`-V(next_state)`, which requires only a 1-step gradient — that's what fixes
the vanishing-gradient problem from Demo 1's end-of-episode loss.

---

## Saved checkpoints

Every trained model in this project lands in `models/weights/`:

| File                              | Phase | Architecture                   |
|-----------------------------------|-------|--------------------------------|
| `action_brain_v5.pth`             | 5     | `ActionWorldModel`             |
| `action_brain_v6.pth`             | 6     | `ActionWorldModel` (noisy)     |
| `latent_brain_v8.pth`             | 8     | `LatentWorldModel`             |
| `pilot_v9.pth`                    | 9     | `PilotNetwork` (1D)            |
| `latent_brain_2d_v10.pth`         | 10    | `LatentWorldModel2D`           |
| `pilot_2d_v11.pth`                | 11    | `Pilot2D`                      |
| `rssm_world_v14.pth`              | 12    | `LSTMRSSM_v14`                 |
| `rssm_actor_v13.pth`              | 13    | LSTM Actor (RSSM dream)        |
| `rssm_critic_v13.pth`             | 13    | Value Critic (RSSM dream)      |
| `transformer_world_v15.pth`       | 14    | `TransformerWorldModel_v15`    |
| `transformer_actor_v15.pth`       | 15    | LSTM Actor (Transformer dream) |
| `transformer_critic_v15.pth`      | 15    | Value Critic (Transformer dream) |
| `lstm_world_v12.pth`              | (extra) | Plain LSTM World Model — early baseline |
| `lstm_pilot_v13.pth`              | (extra) | Plain LSTM Pilot — early baseline       |

The two `(extra)` files come from `(extra)models_no-need/` and represent the
pre-Critic LSTM baseline that motivated moving to RSSM and Actor–Critic.
