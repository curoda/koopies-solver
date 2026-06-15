# Koopmann Patch-DFT Acoustic Solver (Streamlit)

Interactive front end for Gary Koopmann's patch-block DFT-compressed Green
boundary solver for 3-D acoustic radiation.

## What it does

Solves the closed-body boundary integral equation

    (1/2 I - D_DFT) p = i k S_DFT v_n

for the complex surface pressure `p` given a prescribed normal velocity
`v_n`, then reports surface impedance and radiated power. Far-field patch
interactions are compressed with a local plane-wave / DFT basis; near and
self blocks stay dense.

## Files

- `streamlit_app.py` - interactive UI (entry point for Streamlit Cloud)
- `patch_dft_green_solver.py` - the solver (Gary's code, unmodified)
- `prolate_spheroid.py` - pulsating prolate-spheroid data generator
- `requirements.txt`, `runtime.txt` - Streamlit Community Cloud config

## Run locally

    python3.13 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    streamlit run streamlit_app.py

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On https://share.streamlit.io create a new app.
3. Point it at `streamlit_app.py` as the main file.
4. Community Cloud reads `requirements.txt` and `runtime.txt` automatically.

## Koopmann test case (defaults)

Prolate spheroid, aspect ratio 1:5, pulsating like the sphere (all outward
normal velocities = unity), `a` = the radius at the center slice, `W = 1`.
These are the sidebar defaults.

## Command-line use (no UI)

Generate a prolate-spheroid input and solve it directly:

    python prolate_spheroid.py prolate.csv --N 1440 --a-center 1.0 --ratio 5 --W 1
    python patch_dft_green_solver.py prolate.csv --ka 1.0 --a 1.0 --W 1 --out prolate_out
