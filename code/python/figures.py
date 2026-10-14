"""Построение рисунков статьи из результатов идентификации.

Читает results/identification_cumulative.json и data/processed/waves.csv,
сохраняет PNG в figures/.

Запуск:
    uv run python code/python/figures.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from fractional_sir import (RESULTS, ROOT, load_waves, rates_from_lambda_tau,  # noqa: E402
                            simulate_sir)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "axes.titlesize": 10,
    "legend.fontsize": 8,
})


def load_results(name: str = "identification_cumulative.json") -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def get_row(res: dict, territory: str, model: str) -> dict | None:
    for r in res["rows"]:
        if r["territory"] == territory and r["model"] == model:
            return r
    return None


def get_curve(res: dict, territory: str, model: str) -> np.ndarray | None:
    key = f"{territory}|{model}"
    if key not in res["curves"]:
        return None
    return np.array(res["curves"][key]["cumulative"])


# ------------------------------------------------------------------ рисунок 1 ---
def fig1_classic(res: dict, waves: dict, territory: str = "Москва") -> None:
    w = waves[territory]
    row = get_row(res, territory, "classic")
    if row is None:
        return
    model = get_curve(res, territory, "classic")
    t = np.arange(w.n)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(5.8, 3.8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(t, w.cumulative, "o", ms=2.4, color="#1f4e79", label="наблюдения")
    ax.plot(t, model, "-", lw=1.6, color="#c00000",
            label=f"классическая SIR (α = 1), R² = {row['r2']:.4f}")
    ax.set_ylabel("накопленное число случаев")
    ax.set_title(f"Классическая SIR-модель: {territory}, волна "
                 f"{w.dates[0]} — {w.dates[-1]}")
    ax.legend(loc="lower right")
    axr.axhline(0, color="k", lw=0.6)
    axr.plot(t, model - w.cumulative, "-", lw=1.0, color="#c00000")
    axr.set_ylabel("невязка")
    axr.set_xlabel("сутки от начала окна")
    fig.savefig(FIG / "fig1_classic_moscow.png")
    plt.close(fig)
    print("figures/fig1_classic_moscow.png")


# ------------------------------------------------------------------ рисунок 2 ---
def fig2_alpha_sensitivity(waves: dict, territory: str = "Москва",
                           alphas=(0.3, 0.5, 0.7, 0.9, 1.0)) -> None:
    w = waves[territory]
    lam, tau, i0, A = 0.15, 7.0, 200.0, 4.0 * float(w.cumulative[-1])
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(alphas)))
    for a, c in zip(alphas, colors):
        beta, gamma, R0 = rates_from_lambda_tau(lam, tau, a)
        C = simulate_sir(a, beta, gamma, A, w.n, i0=i0)
        ax.plot(range(w.n), C, lw=1.6, color=c,
                label=f"α = {a:g}  (R₀ = {R0:.2f})")
    ax.plot(range(w.n), w.cumulative, "o", ms=2.4, color="#1f4e79", label="наблюдения")
    ax.set_xlabel("сутки от начала окна")
    ax.set_ylabel("накопленное число случаев")
    ax.set_title("Роль порядка α (λ = 0,15 сут⁻¹, τ = 7 сут)", fontsize=9)
    ax.legend()
    fig.savefig(FIG / "fig2_alpha_sensitivity.png")
    plt.close(fig)
    print("figures/fig2_alpha_sensitivity.png")


# ------------------------------------------------------------------ рисунок 3 ---
def fig3_comparison(res: dict, waves: dict, territory: str = "Москва") -> None:
    w = waves[territory]
    rc, rf = get_row(res, territory, "classic"), get_row(res, territory, "fractional")
    mc, mf = get_curve(res, territory, "classic"), get_curve(res, territory, "fractional")
    if rf is None:
        return
    t = np.arange(w.n)
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.4),
                             gridspec_kw={"height_ratios": [2, 1]})
    ax, ax2, axr, axr2 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax.plot(t, w.cumulative, "o", ms=2.2, color="#1f4e79", label="наблюдения")
    if mc is not None:
        ax.plot(t, mc, lw=1.5, color="#c00000",
                label=f"α = 1, R² = {rc['r2']:.4f}")
    ax.plot(t, mf, lw=1.5, color="#2e7d32",
            label=f"α = {rf['alpha']:.3f}, R² = {rf['r2']:.4f}")
    ax.set_xlabel("сутки от начала окна")
    ax.set_ylabel("накопленные случаи")
    ax.set_title(f"{territory}: накопленная кривая")
    ax.legend(loc="lower right")

    ax2.plot(t, w.daily_ma7, "-", lw=1.2, color="#1f4e79", label="наблюдения (MA7)")
    if mc is not None:
        ax2.plot(t, np.gradient(mc), lw=1.2, color="#c00000", label="α = 1")
    ax2.plot(t, np.gradient(mf), lw=1.2, color="#2e7d32", label=f"α = {rf['alpha']:.3f}")
    ax2.set_xlabel("сутки от начала окна")
    ax2.set_ylabel("суточные случаи (MA7)")
    ax2.set_title("Суточная кривая")
    ax2.legend()

    axr.axhline(0, color="k", lw=0.6)
    if mc is not None:
        axr.plot(t, mc - w.cumulative, lw=1.0, color="#c00000")
    axr.plot(t, mf - w.cumulative, lw=1.0, color="#2e7d32")
    axr.set_xlabel("сутки от начала окна")
    axr.set_ylabel("невязка")
    axr.set_title("Невязки: α = 1 (красный) и дробная модель (зелёный)")

    axr2.axis("off")
    txt = (f"Классическая SIR (α = 1)\n"
           f"  λ = {rc['lambda']:.4f} сут⁻¹\n"
           f"  R₀ = {rc['R0']:.3f}\n"
           f"  A = {rc['A']:,.0f}\n"
           f"  RMSE = {rc['rmse']:,.1f}\n"
           f"  AIC = {rc['aic']:,.1f}\n\n"
           f"Дробная SIR\n"
           f"  α = {rf['alpha']:.3f} [{rf['ci_alpha_lo']:.3f}; {rf['ci_alpha_hi']:.3f}]\n"
           f"  λ = {rf['lambda']:.4f} сут⁻¹\n"
           f"  R₀ = {rf['R0']:.3f}\n"
           f"  A = {rf['A']:,.0f}\n"
           f"  RMSE = {rf['rmse']:,.1f}\n"
           f"  AIC = {rf['aic']:,.1f}")
    axr2.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_comparison_moscow.png")
    plt.close(fig)
    print("figures/fig3_comparison_moscow.png")


# ------------------------------------------------------------------ рисунок 4 ---
def fig4_alpha(res: dict) -> None:
    rows = [r for r in res["rows"] if r["model"] == "fractional"]
    rows.sort(key=lambda r: r["alpha"])
    names = [r["territory"] for r in rows]
    vals = np.array([r["alpha"] for r in rows])
    lo = np.array([r["ci_alpha_lo"] for r in rows])
    hi = np.array([r["ci_alpha_hi"] for r in rows])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="o", ms=5,
                color="#1f4e79", ecolor="#c00000", capsize=3, lw=1.4)
    ax.axvline(1.0, color="k", ls="--", lw=1.0)
    ax.text(0.995, -0.65, "α = 1  (классическая SIR)", fontsize=8, va="top", ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("порядок дробной производной α (95 % ДИ)")
    ax.set_title("Оценки порядка α по территориям", fontsize=9)
    ax.set_xlim(0, 1.08)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_alpha_by_city.png")
    plt.close(fig)
    print("figures/fig4_alpha_by_city.png")


def main() -> None:
    res = load_results()
    waves = load_waves()
    fig1_classic(res, waves)
    fig2_alpha_sensitivity(waves)
    fig3_comparison(res, waves)
    fig4_alpha(res)


if __name__ == "__main__":
    main()
