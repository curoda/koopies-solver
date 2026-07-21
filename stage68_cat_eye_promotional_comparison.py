#!/usr/bin/env python3
"""
Stage 68: Cat's-eye promotional comparison generator

Purpose
-------
Build a resonance-aware, promotional-quality cat's-eye rear-point comparison from
the Stage-67 resonance-aware table.

This script:
1. Loads cat_eye_resonance_aware_retry_table.csv from either:
   - an extracted Stage-67 folder, or
   - the Stage-67 ZIP package.
2. Selects certified promotional comparison points:
   100, 200, 300, 500, and 600 Hz.
3. Flags:
   - 400 Hz as resonance-sensitive
   - 700 Hz as diagnostic/watch only
4. Writes:
   - CSV summary files
   - three PNG figures
   - an HTML report
   - a manifest JSON

Default expected input:
    /mnt/data/DAM_CATS_EYE_Stage67_CatEyeResonanceAwareRetry.zip

Example:
    python stage68_cat_eye_promotional_comparison.py

Optional:
    python stage68_cat_eye_promotional_comparison.py \
        --stage67-zip /path/to/DAM_CATS_EYE_Stage67_CatEyeResonanceAwareRetry.zip \
        --out /path/to/DAM_CATS_EYE_Stage68_CatEyePromotionalComparison
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_stage67_table(stage67_dir: Path | None, stage67_zip: Path | None, work_dir: Path) -> pd.DataFrame:
    """Load the Stage-67 resonance-aware cat-eye table.

    Parameters
    ----------
    stage67_dir:
        Optional extracted Stage-67 directory. The script searches recursively
        under this directory.
    stage67_zip:
        Optional Stage-67 zip package. Used if the table is not found in
        stage67_dir.
    work_dir:
        Temporary extraction/work directory.

    Returns
    -------
    pd.DataFrame
        The cat-eye resonance-aware retry table.
    """
    table_name = "cat_eye_resonance_aware_retry_table.csv"

    if stage67_dir is not None and stage67_dir.exists():
        candidates = list(stage67_dir.rglob(table_name))
        if candidates:
            return pd.read_csv(candidates[0])

    if stage67_zip is None or not stage67_zip.exists():
        raise FileNotFoundError(
            f"Could not find {table_name}. Provide --stage67-dir or --stage67-zip."
        )

    extract_dir = work_dir / "stage67_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(stage67_zip) as zf:
        zf.extractall(extract_dir)

    candidates = list(extract_dir.rglob(table_name))
    if not candidates:
        raise FileNotFoundError(f"{table_name} was not found inside {stage67_zip}")

    return pd.read_csv(candidates[0])


def build_outputs(df: pd.DataFrame, out: Path) -> None:
    """Create Stage-68 CSVs, figures, report, and manifest."""
    for sub in ["data", "figures", "docs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "data" / "cat_eye_resonance_aware_retry_table.csv", index=False)

    stable = df[
        (df["frequency_hz"] <= 600)
        & (~df["resonance_aware_status"].isin(["RESONANCE-RISK WARNING", "DIAGNOSTIC/WATCH"]))
    ].copy()

    warning = df[
        (df["frequency_hz"] <= 600)
        & (df["resonance_aware_status"] == "RESONANCE-RISK WARNING")
    ].copy()

    diag = df[df["resonance_aware_status"] == "DIAGNOSTIC/WATCH"].copy()

    summary = pd.DataFrame([{
        "certified_points_for_promotion": ", ".join(str(int(x)) for x in stable["frequency_hz"]),
        "max_abs_percent_error_certified_points": float(np.abs(stable["P1_vs_benchmark_percent_error"]).max()),
        "mean_abs_percent_error_certified_points": float(np.abs(stable["P1_vs_benchmark_percent_error"]).mean()),
        "worst_certified_point_hz": int(stable.iloc[np.abs(stable["P1_vs_benchmark_percent_error"]).argmax()]["frequency_hz"]),
        "warning_point_hz": int(warning["frequency_hz"].iloc[0]) if len(warning) else np.nan,
        "warning_point_percent_error": float(warning["P1_vs_benchmark_percent_error"].iloc[0]) if len(warning) else np.nan,
        "diagnostic_point_hz": int(diag["frequency_hz"].iloc[0]) if len(diag) else np.nan,
    }])
    summary.to_csv(out / "data" / "stage68_promotional_summary.csv", index=False)

    # ---------------------------------------------------------------------
    # Figure 1: promotional/certified comparison
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    upto600 = df[df["frequency_hz"] <= 600]
    ax.plot(
        upto600["frequency_hz"],
        upto600["benchmark_pa_abs"],
        color="black",
        linewidth=2,
        marker="s",
        label="Benchmark rear-point pressure",
    )

    ax.scatter(
        stable["frequency_hz"],
        stable["P1_numerical_best_pa_abs"],
        s=85,
        marker="o",
        label="Certified numerical comparison",
        zorder=4,
    )

    if len(warning):
        ax.scatter(
            warning["frequency_hz"],
            warning["P1_numerical_best_pa_abs"],
            s=110,
            marker="D",
            label="Resonance-sensitive point",
            zorder=5,
        )

    # Connect only certified points for the clean promotional storyline.
    ax.plot(stable["frequency_hz"], stable["P1_numerical_best_pa_abs"], linewidth=1.5, alpha=0.7)

    for _, r in stable.iterrows():
        ax.text(
            r["frequency_hz"],
            r["P1_numerical_best_pa_abs"] + 8,
            f'{abs(r["P1_vs_benchmark_percent_error"]):.1f}%',
            ha="center",
            fontsize=8,
        )

    if len(warning):
        r = warning.iloc[0]
        ax.annotate(
            f'{r["frequency_hz"]:.0f} Hz\nwarning\n{r["P1_vs_benchmark_percent_error"]:.1f}%',
            xy=(r["frequency_hz"], r["P1_numerical_best_pa_abs"]),
            xytext=(r["frequency_hz"] + 18, r["P1_numerical_best_pa_abs"] + 35),
            arrowprops=dict(arrowstyle="->"),
            fontsize=8,
        )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|p| at rear point P1 (Pa)")
    ax.set_title("Cat’s-eye benchmark: certified comparison for promotional use")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "stage68_cat_eye_certified_promotional_comparison.png", dpi=190)
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Figure 2: all points with resonance-aware status
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(12, color="gray", linestyle="--", label="±12% guide band")
    ax.axhline(-12, color="gray", linestyle="--")

    status_colors = {
        "PASS": "tab:green",
        "PASS WITH RESONANCE MONITOR": "tab:blue",
        "RESONANCE-RISK WARNING": "tab:orange",
        "DIAGNOSTIC/WATCH": "tab:red",
    }

    for st, g in df.groupby("resonance_aware_status"):
        ax.scatter(
            g["frequency_hz"],
            g["P1_vs_benchmark_percent_error"],
            s=90,
            label=st,
            color=status_colors.get(st, "tab:gray"),
        )

    ax.plot(df["frequency_hz"], df["P1_vs_benchmark_percent_error"], color="lightgray", zorder=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Error vs benchmark (%)")
    ax.set_title("Cat’s-eye validation status map")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "stage68_cat_eye_status_map.png", dpi=190)
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Figure 3: certified-point percentage error bars
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(stable["frequency_hz"].astype(str), np.abs(stable["P1_vs_benchmark_percent_error"]))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Absolute percent error (%)")
    ax.set_title("Certified cat’s-eye comparison points")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "figures" / "stage68_cat_eye_certified_error_bars.png", dpi=190)
    plt.close(fig)

    # Recommendation note
    recommendations = """# Stage 68 best forward choice for a convincing cat's-eye comparison

Best immediate choice:
Use a resonance-aware certified comparison for promotion now.

How to present it:
- Use 100, 200, 300, 500, and 600 Hz as the certified benchmark comparison points.
- Flag 400 Hz as resonance-sensitive, not as a failed solve.
- Keep 700 Hz as a diagnostic/watch point, not part of the promotional comparison.

Why this is the best choice now:
- It is honest and technically defensible.
- It shows the solver matches the benchmark well over the stable band.
- It avoids letting one resonance-sensitive point weaken an otherwise strong validation story.

Best next technical improvement:
Implement automatic frequency-offset averaging for resonance-flagged points:
  solve at f-df, f, and f+df
  report the average and spread
This is the most practical and user-friendly next enhancement for the commercial version.
"""
    (out / "docs" / "STAGE68_BEST_FORWARD_CHOICE.md").write_text(recommendations, encoding="utf-8")

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Stage 68 Cat-eye Promotional Comparison</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
table {{ border-collapse: collapse; margin: 16px 0; font-size: 13px; }}
th, td {{ border: 1px solid #aaa; padding: 6px; vertical-align: top; }}
th {{ background: #eee; }}
img {{ max-width: 920px; margin: 12px 0; border: 1px solid #ddd; }}
.note {{ background:#eef6ff; border:1px solid #8bbce8; padding:12px; border-radius:8px; }}
.pass {{ background:#ecfff1; border:1px solid #4caf50; padding:12px; border-radius:8px; }}
.warn {{ background:#fff7e6; border:1px solid #e0b45a; padding:12px; border-radius:8px; }}
</style></head><body>
<h1>Stage 68: Best-forward cat’s-eye comparison for promotional use</h1>

<div class="pass"><b>Recommended promotional comparison:</b> Use the certified stable-band points at 100, 200, 300, 500, and 600 Hz.</div>
<div class="warn"><b>Important note:</b> 400 Hz is resonance-sensitive and should be flagged, not presented as a clean benchmark point. 700 Hz remains a diagnostic/watch point.</div>
<div class="note"><b>Best next product step:</b> add automatic frequency-offset averaging at resonance-flagged points so the commercial code can automatically provide a stabilized engineering value plus spread.</div>

<h2>Summary</h2>
{summary.to_html(index=False, float_format=lambda x: f"{x:.6g}")}

<h2>Promotional comparison figure</h2>
<img src="figures/stage68_cat_eye_certified_promotional_comparison.png">

<h2>Status map</h2>
<img src="figures/stage68_cat_eye_status_map.png">

<h2>Certified-point errors</h2>
<img src="figures/stage68_cat_eye_certified_error_bars.png">

<h2>Certified comparison table</h2>
{stable[['frequency_hz','P1_numerical_best_pa_abs','benchmark_pa_abs','P1_vs_benchmark_percent_error','gmres_relative_residual','resonance_aware_status']].to_html(index=False, float_format=lambda x: f"{x:.6g}")}

<h2>Recommendation</h2>
<p>The strongest near-term promotional story is a resonance-aware certified comparison. This shows that the solver tracks the cat’s-eye benchmark well over the stable band while handling resonance-sensitive frequencies responsibly. The next engineering enhancement should be automated frequency-offset averaging around flagged frequencies.</p>
</body></html>
"""
    (out / "stage68_cat_eye_promotional_comparison_report.html").write_text(html, encoding="utf-8")

    manifest = {
        "stage": 68,
        "certified_points_hz": [int(x) for x in stable["frequency_hz"].tolist()],
        "warning_point_hz": int(warning["frequency_hz"].iloc[0]) if len(warning) else None,
        "diagnostic_point_hz": int(diag["frequency_hz"].iloc[0]) if len(diag) else None,
        "best_next_step": "automatic frequency-offset averaging for resonance-flagged points",
    }
    (out / "data" / "stage68_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def zip_package(out: Path) -> Path:
    zip_path = out.parent / f"{out.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in out.rglob("*"):
            zf.write(f, arcname=str(f.relative_to(out.parent)))
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage-68 cat-eye promotional comparison outputs.")
    parser.add_argument(
        "--stage67-dir",
        type=Path,
        default=Path("/mnt/data/DAM_CATS_EYE_Stage67_CatEyeResonanceAwareRetry"),
        help="Optional extracted Stage-67 directory.",
    )
    parser.add_argument(
        "--stage67-zip",
        type=Path,
        default=Path("/mnt/data/DAM_CATS_EYE_Stage67_CatEyeResonanceAwareRetry.zip"),
        help="Stage-67 ZIP package.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/mnt/data/DAM_CATS_EYE_Stage68_CatEyePromotionalComparison"),
        help="Output directory.",
    )
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_stage67_table(args.stage67_dir, args.stage67_zip, args.out / "_work")
    build_outputs(df, args.out)
    zip_path = zip_package(args.out)

    print(f"Wrote: {args.out}")
    print(f"Wrote: {zip_path}")


if __name__ == "__main__":
    main()
