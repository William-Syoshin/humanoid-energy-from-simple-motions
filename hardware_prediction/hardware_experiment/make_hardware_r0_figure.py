"""Build the hardware R0-baseline comparison figure for the ICRA paper.

Reads only the already-written per-trial evaluation summaries under
session_v3_test_10hz/raw. It does not refit, rescore or re-integrate anything;
every number plotted is copied from a summary.json produced by the experiment
pipeline. Trial inclusion follows rules fixed here and recorded in
provenance.json, not the size of the resulting errors.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "session_v3_test_10hz" / "raw"

# (raw directory, evaluation subdirectory, short label, playback note, why this
# evaluation directory and not another one).
TRIALS = [
    ("M014_20260915_152337", "evaluation", "M014", "native",
     "only evaluation present; R3 omitted, time-varying support labels unavailable"),
    ("M015_slow6_pc_20260915_154650", "evaluation", "M015", "x6",
     "evaluation_ground is labelled a sensitivity scenario that was not observed "
     "or user-confirmed, so its R3 row is not reported as a result"),
    ("M016_20260915_145045", "evaluation", "M016 a", "native",
     "only evaluation present"),
    ("M016_20260915_145536", "evaluation_timing_reviewed", "M016 b", "native",
     "evaluation is marked invalid_timing; timing_review/audit.json discards 20 "
     "pre-motion startup-burst rows by a fixed rule and restores jitter to 0.84 ms"),
    ("M016_slow3_pc_20260915_155750", "evaluation", "M016", "x3",
     "only evaluation present"),
    ("M026_20260915_152221", "evaluation", "M026", "native",
     "only evaluation present"),
    ("M026_slow3_pc_20260915_154138", "evaluation", "M026", "x3",
     "only evaluation present"),
    ("M026_slow9_pc_20260915_154254", "evaluation", "M026", "x9",
     "only evaluation present"),
    ("M073_20260915_153043", "evaluation_ground", "M073", "native",
     "support annotated ground throughout the scoring window by the operator, so R3 is available"),
    ("M074_20260915_153316", "evaluation_ground", "M074", "native",
     "support annotated ground throughout the scoring window by the operator, so R3 is available"),
    ("M075_20260915_153356", "evaluation_ground", "M075", "native",
     "support annotated ground throughout the scoring window by the operator, so R3 is available"),
    ("M076_20260915_153430", "evaluation_ground", "M076", "native",
     "support annotated ground throughout the scoring window by the operator, so R3 is available"),
]

EXCLUDED = [
    ("M078_20260915_152913", "meta.json declares purpose="
     "posture_preparation_not_model_evaluation; no evaluation was produced"),
]

MODELS = ["R0", "R1", "R2", "R3"]
COLOR = {"R0": "#6f6f6f", "R1": "#8fbfdd", "R2": "#1f6fb2", "R3": "#d98a2b"}
SELECTED = "R2"  # frozen by validation power RMSE before any test trial was scored


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load():
    rows, prov = [], []
    for name, evaldir, label, playback, why in TRIALS:
        f = RAW / name / evaldir / "summary.json"
        d = json.loads(f.read_text())
        per = {m["model"]: m for m in d["metrics"]}
        rows.append({
            "trial": name,
            "label": label,
            "playback": playback,
            "duration_s": d["scored_duration_s"],
            "actual_J": per["R0"]["actual_J"],
            "err": {m: per[m]["energy_error_pct"] if m in per else None for m in MODELS},
            "rmse": {m: per[m]["power_RMSE_W"] if m in per else None for m in MODELS},
            "pred": {m: per[m]["predicted_J"] if m in per else None for m in MODELS},
        })
        prov.append({
            "trial": name, "evaluation_dir": evaldir, "label": label,
            "playback": playback, "inclusion_reason": why,
            "status": d["status"], "status_reason": d["reason"],
            "mechanical_success": d["mechanical_success"],
            "omitted_models": d.get("omitted_models", []),
            "support_annotation": d.get("support_annotation"),
            "summary_sha256": sha256(f),
        })
    return rows, prov


def main() -> int:
    rows, prov = load()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.9))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    n = len(rows)
    x = np.arange(n, dtype=float)
    ticklab = [f"{r['label']} {r['playback']}" for r in rows]

    def stylex(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(ticklab, fontsize=6.4, rotation=42, ha="right",
                           rotation_mode="anchor")
        ax.set_xlim(-0.6, n - 0.4)
        ax.grid(axis="y", lw=0.3, color="#dddddd", zorder=0)
        ax.set_axisbelow(True)

    # -- (a) per-trial absolute relative energy error -----------------------
    w = 0.2
    for k, m in enumerate(MODELS):
        xs, ys = [], []
        for i, r in enumerate(rows):
            v = r["err"][m]
            if v is None:
                ax_a.text(x[i] + (k - 1.5) * w, 0.075, "n/a", ha="center",
                          va="bottom", fontsize=5.2, rotation=90, color="#999999")
                continue
            xs.append(x[i] + (k - 1.5) * w)
            ys.append(max(v, 0.06))
        ax_a.bar(xs, ys, width=w, color=COLOR[m], label=m,
                 edgecolor="white", linewidth=0.3, zorder=3)
    ax_a.axhline(10, color="#b03030", ls="--", lw=0.8, zorder=4)
    ax_a.set_yscale("log")
    ax_a.set_ylim(0.06, 900)
    ax_a.set_ylabel("Absolute relative energy error (%)")
    stylex(ax_a)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[m]) for m in MODELS]
    handles.append(Line2D([], [], color="#b03030", ls="--", lw=0.8))
    ax_a.legend(handles, MODELS + ["10% target"], ncol=2, frameon=False,
                loc="upper right", handlelength=1.1,
                columnspacing=1.0, handletextpad=0.4)
    ax_a.set_title("(a) Held-out trial energy error", loc="left")

    # -- (b) paired R0 -> selected model -----------------------------------
    better = worse = 0
    ends = []
    for i, r in enumerate(rows):
        a, b = r["err"]["R0"], r["err"][SELECTED]
        improved = b < a
        c = "#1f6fb2" if improved else "#b03030"
        better += improved
        worse += not improved
        ax_b.plot([0, 1], [max(a, 0.06), max(b, 0.06)], "-o", color=c, lw=0.8,
                  ms=2.6, alpha=0.85, zorder=3)
        if not improved:  # only the trials the learned model did not help
            ends.append([np.log10(max(b, 0.06)), ticklab[i], c, max(b, 0.06)])
    ends.sort(key=lambda e: e[0])
    gap = 0.19
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < gap:
            ends[i][0] = ends[i - 1][0] + gap
    for ly, lab, c, yv in ends:
        ax_b.annotate(lab, (1.07, 10 ** ly), fontsize=5.6, color=c, va="center")
        ax_b.plot([1.0, 1.06], [yv, 10 ** ly], color=c, lw=0.4, alpha=0.6, zorder=2)
    m0 = float(np.mean([r["err"]["R0"] for r in rows]))
    m2 = float(np.mean([r["err"][SELECTED] for r in rows]))
    ax_b.plot([0, 1], [m0, m2], "-s", color="black", lw=1.6, ms=4.2,
              zorder=5, label="mean")
    ax_b.axhline(10, color="#b03030", ls="--", lw=0.8, zorder=1)
    ax_b.set_yscale("log")
    ax_b.set_ylim(0.06, 900)
    ax_b.set_xlim(-0.25, 1.70)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["R0\nconstant power", f"{SELECTED}\nselected model"])
    ax_b.set_ylabel("Absolute relative energy error (%)")
    ax_b.set_title(f"(b) Paired change per trial, R0 to {SELECTED} "
                   f"({better} lower, {worse} not lower)", loc="left")
    ax_b.legend([Line2D([], [], color="black", lw=1.6, marker="s", ms=4.2),
                 Line2D([], [], color="#1f6fb2", lw=0.8, marker="o", ms=2.6),
                 Line2D([], [], color="#b03030", lw=0.8, marker="o", ms=2.6)],
                ["mean over 12 trials", f"{SELECTED} lower than R0",
                 f"{SELECTED} not lower"],
                frameon=False, loc="lower left", handlelength=1.6,
                labelspacing=0.35)
    ax_b.grid(axis="y", lw=0.3, color="#dddddd", zorder=0)
    ax_b.set_axisbelow(True)

    # -- (c) power RMSE -----------------------------------------------------
    for k, m in enumerate(MODELS):
        xs, ys = [], []
        for i, r in enumerate(rows):
            v = r["rmse"][m]
            if v is None:
                continue
            xs.append(x[i] + (k - 1.5) * w)
            ys.append(v)
        ax_c.bar(xs, ys, width=w, color=COLOR[m], label=m,
                 edgecolor="white", linewidth=0.3, zorder=3)
    ax_c.set_yscale("log")
    ax_c.set_ylabel("Power RMSE (W)")
    stylex(ax_c)
    ax_c.set_title("(c) Time-resolved power error, same colours as (a)", loc="left")

    # -- (d) measured vs predicted energy -----------------------------------
    act = np.array([r["actual_J"] for r in rows])
    for m in ("R0", SELECTED):
        pr = np.array([r["pred"][m] for r in rows])
        ax_d.scatter(act, pr, s=22, color=COLOR[m], label=m,
                     edgecolor="white", linewidth=0.4, zorder=3)
    lim = [25, 700]
    ax_d.plot(lim, lim, color="black", lw=0.7, ls="-", zorder=2)
    for f, lab in ((1.2, "+20%"), (0.8, "-20%")):
        ax_d.plot(lim, [f * v for v in lim], color="#aaaaaa", lw=0.6, ls=":", zorder=2)
        ax_d.text(lim[1] * 0.93, f * lim[1] * 0.93, lab, fontsize=5.8,
                  color="#888888", va="center", ha="right")
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlim(*lim)
    ax_d.set_ylim(*lim)
    ticks = [30, 50, 100, 200, 300, 500]
    for setter, fmt in ((ax_d.set_xticks, ax_d.set_xticklabels),
                        (ax_d.set_yticks, ax_d.set_yticklabels)):
        setter(ticks)
        fmt([str(t) for t in ticks])
    ax_d.minorticks_off()
    ax_d.set_xlabel("Measured bus energy (J)")
    ax_d.set_ylabel("Predicted energy (J)")
    ax_d.set_title("(d) Predicted vs measured trial energy", loc="left")
    ax_d.legend(frameon=False, loc="upper left", handlelength=1.1)
    ax_d.grid(lw=0.3, color="#dddddd", zorder=0)
    ax_d.set_axisbelow(True)
    handles = [Line2D([], [], color="black", lw=0.7)]
    ax_d.add_artist(ax_d.legend(handles, ["ideal"], frameon=False,
                                loc="lower right", handlelength=1.4))

    fig.tight_layout(pad=0.6, w_pad=1.4, h_pad=1.2)

    out = HERE / "paper_figures" / "hardware_r0_comparison"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "hardware_r0_comparison.pdf", bbox_inches="tight")
    fig.savefig(out / "hardware_r0_comparison.png", dpi=220, bbox_inches="tight")

    # -- numbers behind the figure ------------------------------------------
    lines = ["label,playback,scored_duration_s,measured_J," + ",".join(
        f"{m}_predicted_J,{m}_energy_error_pct,{m}_power_RMSE_W" for m in MODELS)]
    for r in rows:
        cells = [r["label"], r["playback"], f"{r['duration_s']:.3f}",
                 f"{r['actual_J']:.4f}"]
        for m in MODELS:
            for key in ("pred", "err", "rmse"):
                v = r[key][m]
                cells.append("" if v is None else f"{v:.6f}")
        lines.append(",".join(cells))
    (out / "figure_data.csv").write_text("\n".join(lines) + "\n")

    stats = {
        "n_trials": n,
        "selected_model": SELECTED,
        "mean_abs_energy_error_pct": {
            m: float(np.mean([r["err"][m] for r in rows if r["err"][m] is not None]))
            for m in MODELS},
        "worst_abs_energy_error_pct": {
            m: float(np.max([r["err"][m] for r in rows if r["err"][m] is not None]))
            for m in MODELS},
        "median_abs_energy_error_pct": {
            m: float(np.median([r["err"][m] for r in rows if r["err"][m] is not None]))
            for m in MODELS},
        "n_scored_with_model": {
            m: int(sum(r["err"][m] is not None for r in rows)) for m in MODELS},
        "selected_lower_than_R0_count": int(better),
        "selected_not_lower_than_R0_count": int(worse),
        "mean_power_RMSE_W": {
            m: float(np.mean([r["rmse"][m] for r in rows if r["rmse"][m] is not None]))
            for m in MODELS},
    }
    (out / "summary_statistics.json").write_text(json.dumps(stats, indent=2))
    (out / "provenance.json").write_text(json.dumps({
        "source_session": str(RAW),
        "generated_by": pathlib.Path(__file__).name,
        "note": "Every plotted value is copied from the listed summary.json. "
                "No refitting, rescoring or re-integration was performed here.",
        "included": prov,
        "excluded": [{"trial": t, "reason": why} for t, why in EXCLUDED],
    }, indent=2))

    print(f"wrote {out}")
    for m in MODELS:
        print(f"  {m}: mean {stats['mean_abs_energy_error_pct'][m]:7.3f}%  "
              f"median {stats['median_abs_energy_error_pct'][m]:7.3f}%  "
              f"worst {stats['worst_abs_energy_error_pct'][m]:8.3f}%  "
              f"power RMSE {stats['mean_power_RMSE_W'][m]:6.3f} W  "
              f"(n={stats['n_scored_with_model'][m]})")
    print(f"  {SELECTED} lower than R0 on {better}/{n} trials")
    return 0


if __name__ == "__main__":
    sys.exit(main())
