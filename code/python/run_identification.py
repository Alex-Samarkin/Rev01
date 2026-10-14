"""Идентификация параметров волн: классическая SIR против дробной SIR.

Запуск (через uv):
    uv run python code/python/run_identification.py --territories "Москва"
    uv run python code/python/run_identification.py --observable cumulative
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from fractional_sir import RESULTS, aic_weights, fit_wave, load_waves  # noqa: E402

COLS = ["territory", "model", "alpha", "lambda", "tau", "R0", "A",
        "rmse", "r2", "aic", "test_mape"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--territories", nargs="*", default=None)
    ap.add_argument("--observable", default="daily_ma7", choices=["daily_ma7", "cumulative"])
    ap.add_argument("--starts", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_name = args.out or f"identification_{args.observable}.json"
    waves = load_waves()
    names = args.territories or list(waves)
    rows, curves = [], {}

    for name in names:
        w = waves[name]
        print(f"\n=== {name} (n={w.n}, население {w.population:,.0f}) ===", flush=True)
        for model in ("classic", "fractional"):
            t0 = time.time()
            r = fit_wave(w, model, observable=args.observable, n_starts=args.starts)
            p = r.params
            print(f"  {model:11s} alpha={p['alpha']:.4f}  lambda={p['lambda_']:.4f} сут^-1  "
                  f"tau={p['tau']:.2f} сут  R0={p['R0']:.3f}  A={p['A']:,.0f}  "
                  f"RMSE={r.rmse:9.1f}  R2={r.r2:8.4f}  AIC={r.aic:8.1f}  "
                  f"({time.time() - t0:.1f} с)", flush=True)
            if r.ci.get("alpha"):
                lo_a, hi_a = r.ci["alpha"]
                print(f"              95 % ДИ alpha: [{lo_a:.4f}; {hi_a:.4f}]", flush=True)
            if r.ci.get("R0"):
                print(f"              95 % ДИ R0:    [{r.ci['R0'][0]:.3f}; {r.ci['R0'][1]:.3f}]"
                      f"   MAPE прогноза суточной кривой: {r.test_mape_daily:.1f} %", flush=True)
            rows.append(r.summary())
            curves[f"{name}|{model}"] = {
                "cumulative": r.curve_cumulative.tolist(),
                "daily": r.curve_daily.tolist(),
            }

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / out_name.replace(".json", ".csv"), index=False, encoding="utf-8")
    (RESULTS / out_name).write_text(
        json.dumps({"observable": args.observable, "rows": rows, "curves": curves},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Сводка ({args.observable}) ===")
    print(df[COLS].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\n=== Сравнение ===")
    for name in names:
        sub = df[df["territory"] == name]
        if len(sub) != 2:
            continue
        cl = sub[sub["model"] == "classic"].iloc[0]
        fr = sub[sub["model"] == "fractional"].iloc[0]
        wts = aic_weights([cl["aic"], fr["aic"]])
        print(f"{name:26s} RMSE: {cl['rmse']:9.1f} -> {fr['rmse']:9.1f} "
              f"({100 * (1 - fr['rmse'] / cl['rmse']):5.1f} %)   "
              f"R2: {cl['r2']:7.4f} -> {fr['r2']:7.4f}   "
              f"delta AIC = {cl['aic'] - fr['aic']:7.1f}   вес: {wts[0]:.4f} / {wts[1]:.4f}")
    print(f"\nРезультаты: {RESULTS / out_name}")


if __name__ == "__main__":
    main()
