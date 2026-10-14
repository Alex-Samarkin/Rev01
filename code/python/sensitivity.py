"""Анализ чувствительности: длительность инфекционного периода tau.

Прогоняет дробную и классическую модели при tau in {5, 7, 10} суток и сохраняет
results/sensitivity_tau.csv.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from fractional_sir import RESULTS, TAU_GRID, fit_wave, load_waves  # noqa: E402


def main() -> None:
    waves = load_waves()
    rows = []
    for tau in TAU_GRID:
        for name, w in waves.items():
            for model in ("classic", "fractional"):
                r = fit_wave(w, model, observable="cumulative", n_starts=16, tau=tau)
                rows.append(dict(tau_fixed=tau, **r.summary()))
                print(f"tau={tau:4.1f}  {name:24s} {model:11s} "
                      f"alpha={r.params['alpha']:.4f}  R0={r.params['R0']:.3f}  "
                      f"rmse={r.rmse:9.1f}  aic={r.aic:8.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "sensitivity_tau.csv", index=False, encoding="utf-8")
    print(f"\nЗаписано: {RESULTS / 'sensitivity_tau.csv'}")

    piv = df[df["model"] == "fractional"].pivot_table(
        index="territory", columns="tau_fixed", values="alpha")
    print("\nОценки alpha при разных tau:")
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
    piv2 = df[df["model"] == "fractional"].pivot_table(
        index="territory", columns="tau_fixed", values="R0")
    print("\nОценки R0 (дробная модель) при разных tau:")
    print(piv2.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
