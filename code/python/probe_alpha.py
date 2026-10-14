"""Выяснение внутреннего оптимума по alpha: прогон с широкими границами."""
import io
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Alex\Rev01\code\python")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import fractional_sir as fs  # noqa: E402

fs.ALPHA_LO = 0.05

waves = fs.load_waves()
for obs in ("cumulative", "daily_ma7"):
    print(f"\n########## наблюдаемая величина: {obs} ##########")
    for name in ["Москва", "Нью-Йорк", "Милан", "Дели", "Псковская область"]:
        w = waves[name]
        r = fs.fit_wave(w, "fractional", observable=obs, free_gamma=True, n_starts=10)
        p = r.params
        ci = r.ci.get("alpha", [float("nan")] * 2)
        print(f"{name:22s} alpha={p['alpha']:.3f} [{ci[0]:.3f};{ci[1]:.3f}]  "
              f"beta={p['beta']:.3f}  gamma={p['gamma']:.4f}  R0={p['R0']:.2f}  "
              f"A={p['A']:,.0f}  RMSE={r.rmse:9.1f}  R2={r.r2:.4f}", flush=True)
