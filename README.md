# Latent World Model — Building an AI Pilot That Learns Physics From Scratch

**CSCI 490 — Machine Learning | Spring 2026**

A 15-phase research project that progressively builds a Model-Based Reinforcement
Learning system. The AI pilot learns the physics of flight (gravity, thrust,
steering) entirely from synthetic trajectories — no real sensor data — and then
learns to land a rocket inside its own learned "dream" of the world.

The project advances through five architectural eras:

> **1D Physics → 2D World → LSTM → RSSM → Transformer**

The work is organized into **three demos** that each tell one chapter of the story:

| Demo | Phases | What it shows |
|------|--------|--------------------------------|
| Demo 1 | Phases 1 – 9 | Foundation. Can a neural net learn gravity in 1D, and can a Pilot learn to land using only the World Model? |
| Demo 2 | Phases 10 – 11 | Expand to 2D (X, Y, Vx, Vy + angle). Feed-Forward Pilot has no memory and oscillates wildly. |
| Demo 3 | Phases 12 – 15 | Add memory. Compare RSSM (stochastic LSTM) vs Causal Transformer World Models, then train Actor–Critic Pilots inside each dream. |

---

## Repository layout

```
world_Model_Research/
├── Phase 1-3/                 # LLM gravity probe + first synthetic dataset
├── Phase 4/                   # First neural World Model (no actions)
├── phase_5/                   # Action-aware World Model (clean data)
├── phase_6/                   # Robust (noisy) World Model
├── phase_7/                   # Compounding-error trajectory test
├── phase_8/                   # Latent encoder–decoder World Model
├── phase_9_continue/          # Phase 9 (1D Pilot), Phase 10 (2D World), Phase 11 (2D Pilot)
├── phase_12_continue/         # Phase 12 (RSSM World), Phase 13 (RSSM Actor-Critic Pilot)
├── phase_15_continue/         # Phase 14 (Transformer World), Phase 15 (Transformer Actor-Critic Pilot)
├── demos/                     # 5 plotting scripts that produce the slide visuals
├── models/
│   ├── architecture.py        # ActionWorldModel (Phases 5–7)
│   ├── architecture_v8.py     # LatentWorldModel (Phase 8 / 9)
│   └── weights/               # Saved .pth checkpoints for every trained model
├── data/                      # Generated CSV training sets
├── demo_output/               # PNG charts produced by the demo scripts
├── (extra)models_no-need/     # Earlier LSTM Pilot iterations (kept for reference)
├── requirements.txt
└── README.md
```

All phases save their trained weights into `models/weights/` and all demo scripts
load their weights from the same place, so once a phase is trained its model can
be reused by any later phase or any demo.

---

## Setup

The project only needs PyTorch, NumPy, Pandas, and Matplotlib. A virtual
environment is recommended.

```bash
# from project root
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install torch numpy pandas matplotlib
```

`requirements.txt` only pins NumPy because everything else is plain off-the-shelf
PyTorch — any recent CPU build of `torch` works (no GPU required for any phase).

> **Note on working directory.** All training and demo scripts use paths
> relative to the project root (for example `data/...`, `models/weights/...`).
> Always run them from the project root, not from inside the phase folders.

---

# Demo 1 — The 1D Era (Phases 1 – 9)

> **Question.** Can a neural network actually learn gravity, and can a Pilot
> learn to land using only the World Model's predictions?
>
> **Setup.** A rocket only moves up and down (Y, Vy). Start at Y = 200 m,
> Vy = -10 m/s. Thrust ∈ {0, 10, 20} m/s². Duration = 15 seconds.

### Phase 1–3 — LLM probe + synthetic dataset
A first attempt to use a small Llama model (`llama3.2:1b` via `ollama`) as a
"World Model" of gravity. The LLM cannot reliably continue the physics, so the
project pivots to a small neural net trained on synthetic trajectories.
`train_data.py` generates the first 1D free-fall dataset
(`world_model_training.csv`).

```bash
# (optional) requires ollama installed locally with the llama3.2:1b model pulled
python "Phase 1-3/gravity_model.py"

# generate the first free-fall CSV
python "Phase 1-3/train_data.py"
```

### Phase 4 — First neural World Model (free fall, no actions)
A 2-input → 32 → 32 → 2-output MLP that learns `(y, v) → (y_next, v_next)`.
Error drops from ~150 % to ~5 %, but the model has no concept of thrust.

```bash
python "Phase 4/world_model_lightweight.py"   # generate gravity_states.csv
python "Phase 4/world_model_train.py"         # train and print T=6 prediction
```

### Phase 5 — Action-aware World Model
Adds thrust as an input: `(y, v, thrust) → (y_next, v_next)`. This is the first
checkpoint that goes into `models/weights/` (`action_brain_v5.pth`).

```bash
python phase_5/physics_sim_v5.py     # writes data/action_training_v5.csv
python phase_5/train_world_v5.py     # trains and saves action_brain_v5.pth
python phase_5/test_world_v5.py      # smoke-test the trained brain
```

### Phase 6 — Robust (noisy) World Model
Same architecture, but trained on data with Gaussian noise injected on the
inputs. The result is a feed-forward model that holds up to ~10 m/s² thrust but
still fails at the extremes — there is no memory of the trajectory.
Final loss ≈ 60.

```bash
python phase_6/physics_sim_v6.py     # writes data/noisy_training_v6.csv
python phase_6/train_world_v6.py     # trains and saves action_brain_v6.pth
python phase_6/test_world_v6.py
```

### Phase 7 — Compounding-error trajectory test
Lets the Phase 6 model imagine 10 seconds forward by feeding its own predictions
back in as inputs. This is the experiment that proves a feed-forward World Model
cannot be trusted for long sequences — small errors compound at every step.

```bash
python phase_7/trajectory_test_v7.py
```

### Phase 8 — Latent encoder–decoder World Model
Compresses `(y, v, thrust)` into a 2-number latent code, then decodes it back
into `(y_next, v_next)`. Same accuracy as Phase 6 but with a real bottleneck —
this proves the encoder/decoder pattern that every later phase will build on.

```bash
python phase_8/train_latent_v8.py    # saves latent_brain_v8.pth
python phase_8/latent_test_v8.py
```

### Phase 9 — First Pilot (Model-Based RL, 1D)
First demonstration of Model-Based RL: a tiny `PilotNetwork`
(`(y, v) → thrust ∈ [0, 20]`) is trained entirely *inside the dream* of the
frozen Phase 8 World Model. The Pilot never touches a real physics simulator
during training — it only optimizes against what the World Model predicts will
happen. Result: it learns to land softly (avg thrust ≈ 2.1 m/s² with a brake
burst near touchdown).

```bash
python phase_9_continue/train_pilot_v9.py    # saves pilot_v9.pth
```

### Render Demo 1
After Phases 6, 8, and 9 have been trained (or with the checkpoints already in
`models/weights/`), produce the 4-panel chart used in the slide deck:

```bash
python demos/demo1_1d.py
# → demo_output/demo1_1d.png
```

**Key takeaways from Demo 1**
- Phases 1–5 establish the static-physics baseline that the AI has to beat.
- Phase 6 (feed-forward) is accurate at moderate thrust but fails at extremes — no trajectory memory.
- Phase 7 confirms compounding error makes the feed-forward model untrustworthy on long horizons.
- Phase 8 proves the encoder–decoder pattern works at the same accuracy with a tiny latent.
- Phase 9 is the first end-to-end Model-Based RL demonstration: an Actor trained entirely inside the dream.
- Limitation found: 1D is too simple. To steer (Vx, gimbal angle) we need 2D.

---

# Demo 2 — The 2D Era (Phases 10 – 11)

> **Question.** Once the rocket has to *steer* in 2D, can a feed-forward Pilot
> still learn a sensible policy?
>
> **Unified test scenario** (shared with Demo 3):
> Start `(X=0, Y=300, Vx=5, Vy=-10)`, target `(X=75, Y=0)`, duration 15 s,
> thrust ∈ [0, 20] m/s², angle ∈ [-60°, +60°].

### What changes vs Demo 1

|              | Before (1D)   | After (2D)               |
|--------------|---------------|--------------------------|
| State space  | (Y, Vy)       | (X, Y, Vx, Vy)           |
| Action space | thrust only   | angle + thrust           |
| Architecture | 2 inputs      | 6 inputs (encoder)       |
| Challenge    | fight gravity | steer **and** land at X=75 |

How angle is applied:
```
ax = thrust × sin(angle)
ay = gravity + thrust × cos(angle)
angle = 0°  → all thrust UP
angle = +30° → thrust tilts RIGHT
angle = -30° → thrust tilts LEFT
```

### Phase 10 — 2D Feed-Forward World Model
Generates 30 000 random `(x, y, vx, vy, angle, thrust)` samples, trains a
6 → 32 → 3 → 32 → 4 latent encoder–decoder, and saves
`latent_brain_2d_v10.pth`. Loss ≈ 47.5 — comparable to the much bigger
Transformer World Model in Demo 3.

```bash
python phase_9_continue/train_2d_world_v10.py
```

### Phase 11 — 2D Feed-Forward Pilot
Freezes the Phase 10 World Model and trains a Pilot
`(X, Y, Vx, Vy) → (angle, thrust)` to land at (75, 0). The Pilot's trajectory
visually lands near the target, but the angle and thrust signals oscillate
wildly every second. With no memory of the trajectory it has no real strategy —
it stumbles into a landing rather than learning one.

```bash
python phase_9_continue/train_2d_pilot_v11.py    # saves pilot_2d_v11.pth
python phase_9_continue/test_2d_flight.py        # quick flight log in the terminal
```

### Render Demo 2
```bash
python demos/demo2a_world.py    # 3×3 grid of World Model predictions vs real physics
# → demo_output/demo2_world.png

python demos/demo2b_pilot.py    # single chart of the Phase 11 Pilot trajectory
# → demo_output/demo2_pilot.png
```

**Key takeaways from Demo 2**
- The 2D World Model learns the correct *direction* of motion at every angle (left, up, right) and predicts the first ~8 s of free-fall closely.
- The Feed-Forward Pilot has no memory: angle decisions made now don't account for momentum that will accumulate over the next 5 seconds.
- Conclusion: to land deliberately rather than accidentally, both the World Model and the Pilot need *memory*. That motivates Demo 3.

---

# Demo 3 — The Memory Era (Phases 12 – 15)

> **Question.** Does adding memory to the World Model — and adding a Critic to
> the Pilot — fix the compounding-error and erratic-control problems?
>
> Same unified test scenario as Demo 2.

Demo 3 is split into two sub-demos so the World Model results and the Pilot
results can be compared side by side.

## Demo 3A — World Models: RSSM vs Causal Transformer (Phases 12 & 14)

### Phase 12 — RSSM World Model (Hafner, 2020 style)
A Recurrent State-Space Model that splits memory into two parts:
- **Deterministic** `h_t` from an LSTM (the rules)
- **Stochastic** `z_t` sampled from a Gaussian `(μ, σ)` (the uncertainty)

Combined with the reparameterisation trick `z = μ + σ·ε` and a KL-divergence
regulariser that forces honest uncertainty. Training loss ≈ 220 — about
**2× better than a plain LSTM (~452)** because admitting uncertainty cures
hallucinations.

```bash
python phase_12_continue/train_rssm_world_v12.py    # saves rssm_world_v14.pth
```

### Phase 14 — Causal Transformer World Model
Replaces the LSTM entirely with self-attention. Reads all 15 seconds at once,
uses a causal mask so it cannot peek at the future, and uses sinusoidal
positional encoding for time. Trained with mini-batches of 256 over 2 000
trajectories. Final training loss ≈ **48** — the best of the four World Models.

```bash
python phase_15_continue/train_transformer_world_v14.py    # saves transformer_world_v15.pth
```

### Render Demo 3A
```bash
python demos/demo3a_world.py    # 3×3 grid: real physics vs RSSM vs Transformer
# → demo_output/demo3_world.png
```

**Research finding.** The Transformer's ability to attend to *all* previous
timesteps simultaneously cuts compounding prediction error from roughly **9×
(LSTM) down to near 1×** across 15 seconds.

## Demo 3B — Actor–Critic Pilots: RSSM vs Transformer (Phases 13 & 15)

These are full Dreamer-style agents: a frozen World Model, a Critic that
estimates the value of every state, and an Actor (the Pilot) that picks
`(angle, thrust)`.

**Why a Critic?** In Demo 1's Pilot the loss came only from the *final* step,
so the gradient had to flow backwards through 15 LSTM steps and vanished. The
Critic estimates `V(state)` at every step, so the Actor only needs a 1-step
gradient (`-V(next_state)`) — no more vanishing gradients. Returns are
discounted with `γ = 0.95`.

### Phase 13 — RSSM Actor–Critic Pilot
Actor and Critic train inside the frozen RSSM dream from Phase 12.

```bash
python phase_12_continue/train_rssm_actor_critic_v13.py
# saves rssm_actor_v13.pth + rssm_critic_v13.pth
```

### Phase 15 — Transformer Actor–Critic Pilot
Same Actor–Critic architecture as Phase 13, but the dream is the Phase 14
Transformer World Model.

```bash
python phase_15_continue/train_transformer_actor_critic_v15.py
# saves transformer_actor_v15.pth + transformer_critic_v15.pth
```

### Render Demo 3B (combined Pilot comparison)
```bash
python demos/demo3b_pilot.py
# → demo_output/demo3c_combined.png
```

### Phase 13 vs Phase 15 — head-to-head

|                | Ph 13 — RSSM      | Ph 15 — Transformer |
|----------------|-------------------|---------------------|
| World Model    | RSSM (stochastic) | Transformer (attention) |
| Memory         | LSTM `h_t` + `z_t` | Self-Attention      |
| Pilot          | LSTM Actor        | LSTM Actor          |
| Critic         | Value net         | Value net           |
| Actor loss     | ≈ 386             | ≈ 164               |
| Critic loss    | ≈ 0.009           | ≈ 0.011             |
| Final distance to target | 41.9 m  | **8.9 m**           |

**Key takeaways from Demo 3**
- A better *World Model* (Transformer ≪ RSSM ≪ LSTM in loss) translates directly into a better *Pilot*.
- The Critic is the missing ingredient that lets long-horizon Model-Based RL train at all.
- The Transformer Pilot lands within ~9 m of the target; the RSSM Pilot lands ~42 m off — both are dramatic improvements over the erratic Feed-Forward Pilot from Demo 2.

---

## Quick "run everything" recipe

If `models/weights/` is empty and you want to reproduce every checkpoint and
every chart from scratch (CPU only — the Transformer takes the longest, ~30–60
minutes on a laptop):

```bash
# Phase 1–3 (optional LLM probe; required for the first dataset)
python "Phase 1-3/train_data.py"

# Phase 4
python "Phase 4/world_model_lightweight.py"
python "Phase 4/world_model_train.py"

# Phase 5–8 — 1D World Models
python phase_5/physics_sim_v5.py
python phase_5/train_world_v5.py
python phase_6/physics_sim_v6.py
python phase_6/train_world_v6.py
python phase_7/trajectory_test_v7.py
python phase_8/train_latent_v8.py

# Phase 9 — 1D Pilot
python phase_9_continue/train_pilot_v9.py

# Phase 10–11 — 2D World + Pilot
python phase_9_continue/train_2d_world_v10.py
python phase_9_continue/train_2d_pilot_v11.py

# Phase 12–13 — RSSM World + Actor-Critic Pilot
python phase_12_continue/train_rssm_world_v12.py
python phase_12_continue/train_rssm_actor_critic_v13.py

# Phase 14–15 — Transformer World + Actor-Critic Pilot
python phase_15_continue/train_transformer_world_v14.py
python phase_15_continue/train_transformer_actor_critic_v15.py

# All five demo charts
python demos/demo1_1d.py
python demos/demo2a_world.py
python demos/demo2b_pilot.py
python demos/demo3a_world.py
python demos/demo3b_pilot.py
```

Trained checkpoints are already shipped in `models/weights/`, so you can skip
straight to the demo scripts if you only want to regenerate the charts.

---

## What this project demonstrates

1. **Neural networks can learn physics from raw trajectories** — no equations baked in (Phases 4–8).
2. **Feed-forward predictors cannot be trusted on long horizons** because errors compound (Phase 7, Phase 11).
3. **Latent encoder–decoder bottlenecks work** — same accuracy, far smaller representation (Phase 8).
4. **Model-Based RL works** — an Actor trained entirely inside a learned dream can produce competent control (Phase 9).
5. **Memory matters** — RSSM (uncertainty + LSTM) and Transformer (attention) both crush plain LSTM on the World Model task (Demo 3A).
6. **Better dreams make better Pilots** — Actor–Critic agents trained inside a Transformer World Model land within ~9 m of the target vs ~42 m for the RSSM Pilot (Demo 3B).

> **15 phases:** 1D → 2D → LSTM → RSSM → Transformer
> **3 demos:** Feed-Forward · World Models · Actor–Critic Pilots
