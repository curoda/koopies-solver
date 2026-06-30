# Worker Hosting — Phase 2 Exploration

> Status: **EXPLORATION, NOT A BAKED PLAN.** Paused 2026-06-30.
> This documents the options and tradeoffs for *where the Python solver runs*.
> No decision has been made. Do not treat anything here as committed.

## The question

The Next.js web app (frontend) is settled: it deploys to Vercel and users open
a URL, exactly like Streamlit. The **Python solver cannot run serverless**
(long compute, native numpy/scipy deps, ~1.4 GB+ memory budget), so it must run
as a long-lived process on *some* server. This doc explores **which server**.

## Key clarification (resolves a misconception)

**Users never need Python installed locally.** Just like Streamlit, the Python
runs on a *server*, and the user's browser only renders the web UI. "Where the
worker runs" is purely about which machine hosts that server-side Python. It has
nothing to do with the end user's machine.

So in every option below, the app "runs independently like Streamlit." The only
difference between options is **whether runs can be processed when Curt's Mac is
off**, and cost/ops.

## Options

### Option A — VPS worker, DB-polling queue (closest to Streamlit Cloud)

A cheap always-on Linux box hosts the worker as a systemd/Docker service.

- Hosts considered: Fly.io machine, Railway, Hetzner, DigitalOcean droplet.
  Rough cost ~$5-12/mo.
- Worker polls Supabase (or listens via Realtime), claims `queued` runs, solves,
  uploads selected artifacts to Storage, writes metrics/status back.
- **No public endpoint** on the worker (outbound-only to Supabase with
  service-role key). Simpler + more secure than an HTTP solver API.
- Always up, independent of Curt's hardware. Behaves like Streamlit Cloud.
- Scales to multi-user by running more worker instances on the same queue.
- **Recommended host: Fly.io** (good fit for an outbound-only containerized
  worker; deploy the same Docker image we develop locally).

### Option B — Curt's Mac as the host (zero cost, dev-friendly)

Run the identical worker on Curt's Mac (already kept awake via caffeinate for
crons).

- Web app still on Vercel; Gary still just opens a URL. The solve happens on the
  Mac in the background.
- **Catch:** if the Mac is asleep/off, submitted runs sit queued until it wakes.
  Fine for single-user dev; not ideal if Gary must submit anytime.
- Zero hosting cost. Worker code is **identical** to the VPS version, so moving
  to Option A later is a redeploy with no code changes.

### Option C — HTTP solver API (FastAPI on a VPS) — NOT recommended

Next.js calls a FastAPI solver endpoint directly.

- Couples the web layer to solver availability.
- Needs auth on the solver endpoint.
- Handles long jobs worse than a queue (request timeouts, retries, no natural
  backpressure).
- The DB-polling queue (A/B) is strictly better for this workload.

## Recommendation captured at pause time

Build the worker with **Option A architecture** (Supabase-polling, no inbound
port) so it is host-agnostic. Then either:

- **Run on Curt's Mac for now** (Option B deployment) for zero cost during
  single-user dev, OR
- **Go VPS-from-day-one on Fly.io** (Option A deployment) to exactly match
  Streamlit Cloud behavior (always up, independent of Curt's Mac).

Because Curt explicitly wants the app to "run independently like Streamlit
does," the leaning recommendation is **VPS-from-day-one on Fly.io**: develop and
test the worker locally on the Mac, deploy the identical container to Fly.io.
The Mac-vs-VPS choice does **not** affect whether users need anything installed
(they never do); it only affects whether runs process while the Mac is off.

## Tradeoff summary

| Option | Always up? | Cost | User needs Python? | Couples web↔solver? | Verdict |
|--------|-----------|------|--------------------|--------------------|---------|
| A: VPS queue (Fly.io) | Yes | ~$5-12/mo | No | No | Recommended for prod |
| B: Mac queue | Only when Mac awake | $0 | No | No | Fine for single-user dev |
| C: FastAPI HTTP | Depends on host | ~$5-12/mo | No | Yes | Not recommended |

## Decision status

**UNRESOLVED — PAUSED.** Revisit before Phase 1 (worker bridge). The Phase 0
scaffold and data model do not depend on this choice, so build can start without
resolving it; only the worker deployment target does.
