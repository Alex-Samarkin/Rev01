"""Идентификация параметров SEIR-модели (классической и дробной) по волнам Москвы.

Два метода:

* ``notebook`` — воспроизведение постановки блокнота
  ``02_moscow_fractional_seir_wave*.jl``: sigma и gamma фиксированы по литературе,
  начальное состояние выводится из данных, подгоняются beta, scale и alpha;
* ``pipeline`` — расширенная постановка: sigma, gamma, scale (или E0) оцениваются,
  два набора границ (широкие и физиологичные).

Запуск:
    python code/python/run_seir.py --method notebook
    python code/python/run_seir.py --method pipeline --presets physiological
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from fractional_seir import (DATA, RESULTS, aic_weights, fit_seir,  # noqa: E402
                             fit_seir_notebook, load_wave)

WAVES = {
    "wave1": ("moscow_wave1_autumn_winter_2020.csv", "Волна 1: осень—зима 2020/21"),
    "wave2": ("moscow_wave2_delta_2021.csv", "Волна 2: «дельта», 2021"),
    "wave3": ("moscow_wave3_autumn_2021.csv", "Волна 3: осень 2021"),
}
COLS = ["wave", "model", "observable", "preset", "e0_mode", "alpha", "beta", "sigma",
        "gamma", "scale", "R0", "rmse", "mae", "r2", "aic"]


def _worker(task: tuple) -> dict:
    wkey, model, obs, fit_e0, starts, preset, method = task
    fname, label = WAVES[wkey]
    w = load_wave(DATA / fname, label)
    t0 = time.time()
    if method == "notebook":
        r = fit_seir_notebook(w, model, observable=obs, n_starts=starts)
    else:
        r = fit_seir(w, model, observable=obs, fit_e0=fit_e0, n_starts=starts,
                     preset=preset)
    d = r.summary()
    d["method"] = method
    d["seconds"] = round(time.time() - t0, 1)
    return d


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["notebook", "pipeline"], default="notebook")
    ap.add_argument("--waves", nargs="*", default=["wave1", "wave2", "wave3"])
    ap.add_argument("--observables", nargs="*", default=["daily_ma7", "cumulative"])
    ap.add_argument("--presets", nargs="*", default=["wide", "physiological"],
                    choices=["wide", "physiological"])
    ap.add_argument("--e0", choices=["fixed", "fit", "both"], default="fixed")
    ap.add_argument("--starts", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or f"seir_{args.method}.csv"
    modes = {"fixed": [False], "fit": [True], "both": [False, True]}[args.e0]

    if args.method == "notebook":
        tasks = [(wk, model, obs, False, args.starts, "notebook", "notebook")
                 for wk in args.waves
                 for obs in args.observables
                 for model in ("classic", "fractional")]
    else:
        tasks = [(wk, model, obs, e0, args.starts, preset, "pipeline")
                 for wk in args.waves
                 for preset in args.presets
                 for e0 in modes
                 for obs in args.observables
                 for model in ("classic", "fractional")]

    print(f"метод: {args.method};  прогонов: {len(tasks)};  стартов: {args.starts}",
          flush=True)
    t_start = time.time()

    def report(d: dict) -> None:
        print(f"  {d['wave']:6s} {d['model']:11s} {d['observable']:10s} "
              f"alpha={d['alpha']:.4f} beta={d['beta']:.4f} scale={d['scale']:.4f} "
              f"RMSE={d['rmse']:9.2f} MAE={d['mae']:9.2f} ({d['seconds']} с)", flush=True)

    rows = []
    if args.serial:
        for t in tasks:
            d = _worker(t)
            rows.append(d)
            report(d)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for d in ex.map(_worker, tasks):
                rows.append(d)
                report(d)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / out, index=False, encoding="utf-8")
    (RESULTS / out.replace(".csv", ".json")).write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Сводка (всего {time.time() - t_start:.0f} с) ===")
    print(df[COLS].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\n=== Классическая SEIR против дробной ===")
    keys = ["wave", "observable"] + (["preset", "e0_mode"] if args.method == "pipeline" else [])
    for key, sub in df.groupby(keys, sort=False):
        cl = sub[sub["model"] == "classic"]
        fr = sub[sub["model"] == "fractional"]
        if cl.empty or fr.empty:
            continue
        c, f = cl.iloc[0], fr.iloc[0]
        wts = aic_weights([c["aic"], f["aic"]])
        label = key if isinstance(key, str) else " ".join(str(k) for k in key)
        print(f"{label:60s} RMSE {c['rmse']:9.2f} -> {f['rmse']:9.2f} "
              f"({100 * (1 - f['rmse'] / c['rmse']):5.1f} %)   "
              f"MAE {c['mae']:9.2f} -> {f['mae']:9.2f}   "
              f"dAIC={c['aic'] - f['aic']:8.1f}   вес: {wts[0]:.3f}/{wts[1]:.3f}")

    print(f"\nРезультаты: {RESULTS / out}")


if __name__ == "__main__":
    main()
