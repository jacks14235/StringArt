# String Draw

Turn a photo into string art. Pegs are placed around a circle; each string is a line between two pegs. This repo precomputes how every possible line affects the image, then solves for which lines (and how strongly) best reproduce a target photo.

## How it works

1. **Preprocess** — crop the target to a circle on a white background (`preprocess.py`).
2. **Precalc** — for each peg-to-peg line, rasterize an antialiased “darkness” image and store it as a column in matrix `M` (`matrix.py`, `antialias.py`).
3. **Solve** — find line weights `x` so that `M @ x ≈ target` using one of several methods:
   - **Least squares** — unconstrained L2 fit (`matrix.solve`)
   - **NNLS** — L2 fit with non-negative weights (`matrix.solve_nnls`)
   - **Greedy** — binary 0/1 selection, one line at a time (`greedy.py`)
   - **Color** — multi-thread palette with per-color weights (`color.ipynb`)
4. **Render** — draw the selected lines, or preview the continuous linear reconstruction.

```
preprocess.py ──► matrix.py ◄── antialias.py
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    matrix.ipynb  greedy.py   color.ipynb
                  greedy.ipynb
```

## Root folder map

### Core library

| File | Role |
|---|---|
| `matrix.py` | Main API: `precalc`, solvers (`solve`, `solve_nnls`, `solve_sparse`, `solve_gpu`), `render_image`, `render_exact` |
| `antialias.py` | Antialiased line rasterization (`draw_line_distance`) |
| `preprocess.py` | Circle crop helper (`circle_crop`) |
| `greedy.py` | Greedy binary line selection with vectorized scoring |

### Notebooks (recommended entry points)

| File | Purpose |
|---|---|
| `matrix.ipynb` | Grayscale least-squares workflow — preprocess, solve, preview |
| `greedy.ipynb` | Greedy binary selection — preprocess, precalc, greedy solve, render |
| `color.ipynb` | Color string art with a configurable thread palette |

### Other

| File / folder | Role |
|---|---|
| `PROJECT_MAP.md` | Detailed dependency map and entry-point reference |
| `legacy/` | Older scripts moved out of the root (see below) |
| `rust_version/` | Standalone Rust greedy solver (`cargo run --release`) |
| `images/` | Input photos (gitignored) |
| `precalc/`, `results/` | Generated caches and output (gitignored) |

## Legacy

Superseded experiments and early prototypes live in `legacy/`:

| File | Notes |
|---|---|
| `legacy/main.py` | Original pixel-precalc greedy loop |
| `legacy/main_fast_antialiased.py` | Vectorized version of the old greedy approach |
| `legacy/line_draw.py` | Xiaolin-Wu line drawing experiment |
| `legacy/new.py` | Incomplete stub |

These are not used by the current matrix-based pipeline.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pillow matplotlib tqdm scipy
# optional: torch (GPU solve), opencv-python (greedy live display)
jupyter notebook matrix.ipynb
```

Put target images in `images/` (or update paths in the notebook settings cells).

## Solvers at a glance

| Method | Function | Constraint | Good for |
|---|---|---|---|
| Least squares | `matrix.solve` | none | Smooth continuous approximation |
| NNLS | `matrix.solve_nnls` | `x >= 0` | Non-negative weights, no cancellation |
| Greedy | `greedy.greedy_solve` | binary 0/1 | Actual on/off string selection |
| Sparse LS | `matrix.solve_sparse` | none | Large matrices via iterative solver |
| GPU LS | `matrix.solve_gpu` | none | CUDA-accelerated least squares |

## Dependencies

**Required:** Python 3, NumPy, Pillow, tqdm  
**Notebooks:** matplotlib, Jupyter  
**Optional:** scipy (NNLS, sparse solve), PyTorch (GPU solve), opencv-python (greedy live preview)
