# Koopies Solver Web App — Plan

> Status: **Planning** (not yet started). Captured 2026-06-30.
> The build will live in a **new repo** (`curoda/koopies-solver-web`, name TBD).
> This existing Streamlit repo is **not** being modified for the web app; these
> docs are saved here for reference while planning.

## Goal

Move the patch-DFT acoustic solver from a Streamlit app to a standalone web app
that:

1. Runs independently like Streamlit does (user just opens a URL; no local
   install of anything, ever).
2. Saves a snapshot of all inputs and outputs for a run to a database.
3. Groups multiple runs (inputs + outputs) into a **project**.
4. Lets users **compare runs side by side** within a project.
5. Built for eventual **multi-user + auth**, but **single-user for now**
   (auth wired but disabled; `owner_id` nullable). Future auth/multi-user work
   tracked as GitHub issues.

## Confirmed decisions (2026-06-30)

| # | Decision | Choice |
|---|----------|--------|
| Repo | New repo, do not touch existing Streamlit app | **Confirmed** |
| 1 | Web app for eventual multi-user + auth | Yes |
| 2 | Single-user for now; file issues for auth + multi-user | Yes |
| 3 | Database | **Supabase** (matches other apps), **new dedicated project** |
| 4 | Save raw inputs + all outputs; compare runs | Yes to both |
| 5 | Output saving | User **chooses which outputs to save**; do NOT save all by default |
| 6 | Reproducibility | Stamp solver git commit + script name per run |
| 7 | Comparison | Side-by-side run comparison |
| Architecture | Frontend + compute split | **Next.js frontend + separate Python worker** |
| Worker host | Where Python runs | **OPEN — see Phase 2 Exploration doc** (paused) |

## Architecture (chosen)

Two services, one new repo:

- **Frontend + API: Next.js** (matches existing stack: PTCB is Next.js +
  Supabase). Deploys to Vercel. Handles projects, run history, snapshot
  selection, side-by-side comparison, and the 3D pressure viewer
  (Plotly.js, ported from the Streamlit Plotly viewer).
- **Solver worker: Python service** — the existing out-of-process worker,
  lightly adapted. Runs the actual solve, uploads selected artifacts to
  Supabase Storage, writes metrics + status back to Postgres.
- **Supabase**: Postgres (metadata) + Storage (CSV/JSON artifacts) + Auth
  (wired but disabled for single-user now).

**Gary's solver (`patch_dft_green_solver.py` / `_adaptive.py`) stays completely
unmodified.** Only the worker wrapper around it changes.

### Why this split

- The solver is long-running, CPU/memory-heavy Python with native deps
  (numpy/scipy). That rules out Vercel functions / Supabase Edge Functions /
  any serverless runtime for the actual solve.
- Next.js + Supabase matches existing muscle memory (PTCB) and makes the
  projects / comparison / 3D UI straightforward.

### Job flow (queue, no inbound port on worker)

1. Next.js inserts a `run` row with status `queued`.
2. Worker polls Supabase (or listens via Realtime), claims the job.
3. Worker runs the solve, uploads **selected** artifacts to Supabase Storage,
   writes `key_metrics` + status back to Postgres.
4. Next.js renders progress, then the completed report + 3D viewer.

The worker only makes **outbound** calls to Supabase (service-role key). No
public HTTP endpoint on the solver = simpler and more secure than an HTTP
solver API. Scales to multi-user later by running more worker instances against
the same queue.

## Data model (Supabase Postgres)

```
projects        id, owner_id (nullable for now), name, description,
                created_at, updated_at

runs            id, project_id, case_id, status, solver_commit, solver_script,
                created_at, completed_at, input_params (jsonb),
                key_metrics (jsonb)

run_artifacts   id, run_id, kind (enum), storage_path, bytes,
                content_type, saved (bool)
```

- `runs` is **immutable** once completed = the "snapshot."
- `input_params` jsonb = the full ~30-field solver param set + excitation mode
  (frequency Hz vs ka), captured from the submission form.
- `key_metrics` jsonb = `radiated_power_W`, `max_pressure_peak_Pa`,
  `relative_residual`, `method_used`, `ka`, `frequency_hz`, `N_points`
  (pulled from `report.json` for fast table/comparison rendering without
  loading large artifacts).
- `solver_commit` = git hash of the solver script, stamped per run (#6).

## Artifacts and "choose what to save" (#5)

Artifact kinds a run produces (from the current Streamlit job dir):

**Inputs (always saved — required to reproduce a run):**
- `geometry_csv` — uploaded geometry, or generated built-in preset CSV
- `feature_metadata_csv` — optional preprocessor feature metadata
- params — stored in `runs.input_params` jsonb

**Outputs (default OFF; user ticks which to persist):**
- `pressure_csv` — `result_pressure.csv` (the big one; per-node pressure)
- `patch_summary_csv` — `result_patch_summary.csv`
- `feature_summary_csv` — `result_feature_summary.csv`
- `report_json` — `result_report.json` (tiny; recommend default ON because it
  drives metrics + comparison, but user may uncheck)
- `resource_log` — `resources.jsonl`
- `worker_log` — `worker.log`

Un-saved outputs are discarded after the run completes. `key_metrics` is always
extracted from the report into Postgres even if the user doesn't save the full
`report_json` artifact (so the comparison table always works).

Storage layout (private bucket `run-artifacts`):
```
projects/<project_id>/runs/<run_id>/<kind>.<ext>
```
Service-role write from worker; signed-URL read from the Next.js API.

### Output artifact reference (current report.json keys)

`report.json` top-level keys observed in a real run:
`case_id, frequency_hz, N_points, B_patches,
M_plane_wave_terms_max_per_far_block, W_multiplier, ka, a, k,
lambda_acoustic, lambda_velocity, lambda_star, max_patch_diameter_target,
near_blocks_dense, far_blocks_dft_compressed, retained_single_layer_blocks,
far_block_error_control, memory_estimate, solver, geometry_audit,
surface_impedance_normalized, surface_impedance_physical_Pa_s_per_m,
radiated_power_rhoc1, radiated_power_W, max_pressure_peak_Pa, rho_kg_m3,
c_m_s, resource_log, implementation_notes`

`solver` subkeys: `method_used, converged, requested_rtol,
relative_residual, gmres_info, gmres_callback_iterations, gmres_seconds,
gmres_preconditioned_residual_history, preconditioner,
preconditioner_regularized_patches, fallback_krylov, fallback_info,
fallback_iterations, direct_fallback_*, explicit_matrix_inverse_used`

`pressure.csv` columns:
`N, x, y, z, l, m, n, nx, ny, nz, area, vn_real, vn_imag,
p_normalized_real, p_normalized_imag, p_Pa_real, p_Pa_imag, p_Pa_abs_peak,
p_phase_rad, point_active_power_W, patch`

## Input parameters to capture (from current Streamlit form)

Excitation: frequency mode (Hz) **or** ka (dimensionless); `a`, `rho`, `c`.
Geometry source: upload CSV, or built-in pulsating sphere, or prolate spheroid
(with preset params N, radius, b/a_center, ratio, W multiplier).
Solver/compression: `case_id`, `B`, `M`, `W`, `patches_per_wavelength`,
`near_threshold`, `self_d_model`, `far_error_tol`, `max_rank_fraction`,
`max_adapt_far_blocks`, `max_points_per_patch`, `rtol`, `maxiter`,
`gmres_restart`, `memory_budget_mb`.
Feature metadata: separate metadata CSV upload + boundary mode, connectivity
factor, zero-id-is-missing, column overrides, type map, join-key overrides.

(Full flag list is in the existing `streamlit_app.py` `create_job` command
builder — port verbatim so solver behavior is identical.)

## Comparison (#7)

- **Project view:** sortable table of runs with key metrics (power, max
  pressure, ka/freq, N, residual, method, status, date).
- **Compare mode:** select 2+ runs → side-by-side metric table + overlaid
  charts (e.g., radiated power vs ka) + side-by-side 3D pressure viewers.
  Requires `report_json` saved per compared run.

## Supabase project setup (new, dedicated)

- New project named `koopies-solver` (own Postgres + Storage + Auth, isolated
  from PTCB).
- Storage bucket `run-artifacts`, **private** (service-role write, signed-URL
  read).
- Auth enabled at project level but unused for now; `owner_id` nullable.
- Migrations in-repo (`supabase/migrations/`), matching PTCB convention.
- Env keys (PTCB pattern): `NEXT_PUBLIC_SUPABASE_URL`,
  `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`.
- Creating the actual project requires Supabase dashboard login (external
  action). Curt creates it / does the click-through; hand back URL + anon key
  + service-role key.

## GitHub issues to file at kickoff (#2)

- New repo `curoda/koopies-solver-web` (name TBD), log issues per AGENTS.md.
- Issue: "Add Supabase Auth (currently single-user, `owner_id` nullable)."
- Issue: "Multi-user: per-user project ownership + RLS policies."
- Plus a tracking issue per build phase.

## Build phases (product process)

Follow `memory/reference/product-process.md`. Confirm personas + phases with
Curt at kickoff. Each phase: STATUS.md updated, GitHub issue, deliverables in
files (`design/ux/`, `design/visual/`, `design/qa/`, `design/testing/`).

1. **Phase 0 — Scaffold:** new repo, Next.js + Supabase client, migrations for
   the 3 tables + Storage bucket, STATUS.md, CLAUDE.md.
2. **Phase 1 — Worker bridge:** adapt the Python worker to claim a job from the
   DB queue, run solve, upload selected artifacts, write metrics back. Solver
   untouched.
3. **Phase 2 — Run submission UI:** port the form (~30 params + geometry upload
   + built-in presets + output-save checkboxes), submit → create run → trigger
   worker, live progress.
4. **Phase 3 — Run viewer:** report metrics, downloads, 3D Plotly.js pressure
   viewer (port from Streamlit).
5. **Phase 4 — Projects + history:** project CRUD, assign runs to projects,
   run-history table.
6. **Phase 5 — Comparison:** side-by-side compare.

## Open items before build

- **Worker hosting** is unresolved and paused. See
  `docs/WORKER_HOSTING_EXPLORATION.md` (Phase 2 exploration, not a baked plan).
- Kick off the formal product process (Pat confirms personas + phases, then
  premortem) once all questions are resolved.
