"""Дробная SEIR-модель (производная Капуто) — реализация постановки из блокнота
``02_moscow_fractional_seir_wave*.jl`` в едином пайплайне проекта.

Модель
------
    D^alpha S = -beta * S * I / N
    D^alpha E =  beta * S * I / N - sigma * E
    D^alpha I =  sigma * E - gamma * I
    D^alpha R =  gamma * I

Наблюдаемые величины:
    суточная заболеваемость     y(t) = q * sigma * E(t)
    накопленная заболеваемость  C(t) = q * ∫_0^t sigma * E(s) ds

где q — коэффициент выявления, sigma — скорость перехода E -> I (1/инкубация),
gamma — скорость выбывания из I. При alpha = 1 модель переходит в классическую SEIR.

Численная схема — та же явная схема L1, что и в ``fractional_sir.py`` (см. там
описание и верификацию), применённая к трём уравнениям (R не влияет на остальные).

Отличия от блокнота (оговорены в отчёте):
  * начальные условия E0 и I0 можно оценивать, а не фиксировать (E0 = I0 = 500);
  * минимизация — trust-region reflective с многократным стартом вместо чистого
    Нелдера—Мида;
  * добавлена наблюдаемая величина ``cumulative`` и информационные критерии.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import gamma as gamma_fn

from fractional_sir import ROOT, RESULTS, l1_weights, logit, sigmoid

DATA = ROOT / "data"

STEPS_PER_DAY = 4            # dt = 0.25 сут
N_MOSCOW = 12_678_000.0      # население Москвы, как в блокноте
E0_NOTEBOOK = 500.0          # начальные условия из блокнота

BETA_LO, BETA_HI = 0.02, 2.00
SIGMA_LO, SIGMA_HI = 1.0 / 14.0, 1.00
GAMMA_LO, GAMMA_HI = 1.0 / 21.0, 1.00
SCALE_LO, SCALE_HI = 1e-4, 20.0
ALPHA_LO, ALPHA_HI = 0.10, 1.00
E0_LO, E0_HI = 1.0, 1.0e5

OBSERVABLES = ("daily_ma7", "cumulative", "daily_raw")

# Два набора границ:
#   wide          — как в блокноте (границы почти не ограничивают);
#   physiological — латентный период 2-14 сут, инфекционный 3-21 сут.
PRESETS = {
    "wide": {
        "alpha": (ALPHA_LO, ALPHA_HI, False),
        "beta": (BETA_LO, BETA_HI, True),
        "sigma": (SIGMA_LO, SIGMA_HI, False),
        "gamma": (GAMMA_LO, GAMMA_HI, False),
        "scale": (SCALE_LO, SCALE_HI, True),
        "e0": (E0_LO, E0_HI, True),
    },
    "physiological": {
        "alpha": (ALPHA_LO, ALPHA_HI, False),
        "beta": (BETA_LO, BETA_HI, True),
        "sigma": (1.0 / 14.0, 1.0 / 2.0, False),   # инкубация 2-14 суток
        "gamma": (1.0 / 21.0, 1.0 / 3.0, False),   # инфекционный период 3-21 суток
        "scale": (SCALE_LO, SCALE_HI, True),
        "e0": (E0_LO, E0_HI, True),
    },
}
BOUNDS = PRESETS["wide"]


# ------------------------------------------------------------------ решатель ---
def simulate_seir(alpha: float, beta: float, sigma: float, gamma: float, N: float,
                  E0: float, I0: float, n_days: int,
                  steps_per_day: int = STEPS_PER_DAY) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (E(t_j), ∫E dt) на суточной сетке. Амплитуда вынесена в scale."""
    n_steps = (n_days - 1) * steps_per_day
    dt = 1.0 / steps_per_day
    b = l1_weights(alpha, n_steps)
    G = gamma_fn(2.0 - alpha) * dt ** alpha

    S = np.empty(n_steps + 1)
    E = np.empty(n_steps + 1)
    I = np.empty(n_steps + 1)
    dS = np.zeros(n_steps + 1)
    dE = np.zeros(n_steps + 1)
    dI = np.zeros(n_steps + 1)
    cum = np.zeros(n_steps + 1)

    S[0] = N - E0 - I0
    E[0] = E0
    I[0] = I0

    for n in range(1, n_steps + 1):
        p = n - 1
        if n > 1:
            w = b[1:n]
            rev = slice(n - 1, 0, -1)
            hS = w @ dS[rev]
            hE = w @ dE[rev]
            hI = w @ dI[rev]
        else:
            hS = hE = hI = 0.0
        inf = beta * S[p] * I[p] / N
        S[n] = S[p] + G * (-inf) - hS
        E[n] = E[p] + G * (inf - sigma * E[p]) - hE
        I[n] = I[p] + G * (sigma * E[p] - gamma * I[p]) - hI
        cum[n] = cum[p] + 0.5 * dt * (E[n] + E[p])
        dS[n] = S[n] - S[p]
        dE[n] = E[n] - E[p]
        dI[n] = I[n] - I[p]

    idx = np.arange(n_days) * steps_per_day
    return E[idx], cum[idx]


# -------------------------------------------------------------------- данные ---
@dataclass
class WaveData:
    name: str
    dates: pd.DatetimeIndex
    cumulative: np.ndarray      # накопленное внутри окна
    daily: np.ndarray           # суточный прирост
    daily_ma7: np.ndarray       # сглаженный суточный прирост

    @property
    def n(self) -> int:
        return len(self.dates)


def load_wave(path: Path, name: str) -> WaveData:
    """CSV со столбцами date,sick,healed,died (накопленные значения, порядок неважен)."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    df = df.sort_values("date").reset_index(drop=True)
    cum_all = df["sick"].astype(float).to_numpy()
    daily = np.concatenate([[0.0], np.diff(cum_all)]).clip(min=0.0)
    dates = pd.DatetimeIndex(df["date"])
    cum_rel = np.cumsum(daily)
    ma7 = pd.Series(daily).rolling(7, center=True, min_periods=3).mean().to_numpy()
    return WaveData(name=name, dates=dates, cumulative=cum_rel, daily=daily, daily_ma7=ma7)


# ------------------------------------------------------------- идентификация ---
@dataclass
class SeirFit:
    wave: str
    model: str
    observable: str
    e0_mode: str
    preset: str
    params: dict
    rmse: float
    mae: float
    r2: float
    aic: float
    bic: float
    n: int
    k: int
    ci: dict = field(default_factory=dict)
    curve: np.ndarray | None = None

    def summary(self) -> dict:
        d = dict(wave=self.wave, model=self.model, observable=self.observable,
                 e0_mode=self.e0_mode, preset=self.preset, rmse=self.rmse, mae=self.mae, r2=self.r2,
                 aic=self.aic, bic=self.bic, n=self.n, k=self.k)
        for key in ("alpha", "beta", "sigma", "gamma", "scale", "q", "e0",
                    "R0", "incubation_days"):
            d[key] = self.params.get(key, float("nan"))
        for key in ("alpha", "beta", "sigma", "gamma", "scale", "q"):
            lo_hi = self.ci.get(key)
            d[f"ci_{key}_lo"] = lo_hi[0] if lo_hi else float("nan")
            d[f"ci_{key}_hi"] = lo_hi[1] if lo_hi else float("nan")
        return d


def _spec(model: str, fit_e0: bool) -> list[str]:
    names = ["beta", "sigma", "gamma", "scale"]
    if model == "fractional":
        names.insert(0, "alpha")
    if fit_e0:
        names.append("e0")
    return names


BOUNDS = PRESETS["wide"]


def fit_seir(w: WaveData, model: str, observable: str = "daily_ma7", N: float = N_MOSCOW,
             fit_e0: bool = False, n_starts: int = 8, seed: int = 20260115,
             steps_per_day: int = STEPS_PER_DAY, preset: str = "wide") -> SeirFit:
    bounds = PRESETS[preset]
    names = _spec(model, fit_e0)
    npar = len(names)

    if observable == "cumulative":
        target, to_obs = w.cumulative, (lambda E, C: C)
    elif observable == "daily_ma7":
        target, to_obs = w.daily_ma7[1:], (lambda E, C: E[1:])
    elif observable == "daily_raw":
        target, to_obs = w.daily[1:], (lambda E, C: E[1:])
    else:
        raise ValueError(observable)

    def unpack(x: np.ndarray) -> dict:
        p = {}
        for i, nm in enumerate(names):
            lo, hi, log_scale = bounds[nm]
            u = sigmoid(x[i])
            p[nm] = lo * (hi / lo) ** u if log_scale else lo + (hi - lo) * u
        p.setdefault("alpha", 1.0)
        if not fit_e0:
            p["e0"] = E0_NOTEBOOK
        return p

    def pack(vals: dict) -> np.ndarray:
        x = []
        for nm in names:
            lo, hi, log_scale = bounds[nm]
            v = min(max(vals[nm], lo * 1.0001 if log_scale else lo + 1e-9),
                    hi * 0.9999 if log_scale else hi - 1e-9)
            u = math.log(v / lo) / math.log(hi / lo) if log_scale else (v - lo) / (hi - lo)
            x.append(logit(u))
        return np.array(x)

    def model_obs(x: np.ndarray) -> np.ndarray:
        p = unpack(x)
        E, C = simulate_seir(p["alpha"], p["beta"], p["sigma"], p["gamma"], N,
                             p["e0"], p["e0"], w.n, steps_per_day)
        if not np.all(np.isfinite(E)):
            return np.full(len(target), 1e12)
        # q * sigma * E  для суточной и q * integral(sigma E) для накопленной
        return p["scale"] * to_obs(E, C)

    def residuals(x: np.ndarray) -> np.ndarray:
        return model_obs(x) - target

    starts = []
    for b0, s0, g0, sc0 in [(0.35, 1 / 5, 1 / 10, 0.5), (0.6, 1 / 4, 1 / 7, 0.3),
                             (0.25, 1 / 6, 1 / 12, 0.7), (0.45, 1 / 3, 1 / 5, 0.4)]:
        starts.append(pack(dict(alpha=0.7, beta=b0, sigma=s0, gamma=g0, scale=sc0,
                                e0=E0_NOTEBOOK)))
    rng = np.random.default_rng(seed)
    while len(starts) < n_starts:
        starts.append(pack(dict(alpha=float(rng.uniform(0.3, 0.95)),
                                beta=float(rng.uniform(0.1, 1.0)),
                                sigma=float(rng.uniform(1 / 10, 1 / 3)),
                                gamma=float(rng.uniform(1 / 14, 1 / 4)),
                                scale=float(rng.uniform(0.02, 3.0)),
                                e0=float(rng.uniform(50, 5000)))))

    best = None
    for x0 in starts:
        try:
            sol = least_squares(residuals, x0, method="trf", xtol=1e-12, ftol=1e-12,
                                gtol=1e-12, max_nfev=2000)
        except Exception:
            continue
        rss = float(np.sum(sol.fun ** 2))
        if not math.isfinite(rss):
            continue
        if best is None or rss < best[0]:
            best = (rss, sol)
    if best is None:
        raise RuntimeError(f"не удалось подобрать параметры: {w.name}/{model}/{observable}")

    rss, sol = best
    p = unpack(sol.x)
    pred = model_obs(sol.x)
    res = pred - target
    n = len(target)
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae = float(np.mean(np.abs(res)))
    r2 = float(1.0 - np.sum(res ** 2) / np.sum((target - target.mean()) ** 2))
    aic = float(n * math.log(rss / n) + 2 * npar)
    bic = float(n * math.log(rss / n) + npar * math.log(n))

    ci: dict[str, list[float]] = {}
    z = 1.959964
    try:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * (rss / max(n - npar, 1))
        se_x = np.sqrt(np.abs(np.diag(cov)))
        for i, nm in enumerate(names):
            lo, hi, log_scale = bounds[nm]
            u = sigmoid(sol.x[i])
            dv_du = p[nm] * math.log(hi / lo) if log_scale else hi - lo
            se_v = abs(u * (1 - u) * dv_du * se_x[i])
            ci[nm] = [float(max(lo, p[nm] - z * se_v)), float(min(hi, p[nm] + z * se_v))]
    except Exception:
        pass

    return SeirFit(
        wave=w.name, model=model, observable=observable,
        e0_mode="оценивается" if fit_e0 else f"фиксировано ({E0_NOTEBOOK:g})",
        preset=preset,
        params=dict(alpha=p["alpha"], beta=p["beta"], sigma=p["sigma"], gamma=p["gamma"],
                    scale=p["scale"], q=p["scale"] / p["sigma"], e0=p["e0"],
                    R0=p["beta"] / p["gamma"],
                    incubation_days=1.0 / p["sigma"]),
        rmse=rmse, mae=mae, r2=r2, aic=aic, bic=bic, n=n, k=npar, ci=ci, curve=pred,
    )


def aic_weights(aics) -> list[float]:
    a = np.asarray(aics, dtype=float)
    w = np.exp(-0.5 * (a - a.min()))
    return list(w / w.sum())


# ---------------------------------------------------------------------------
# Схема идентификации, повторяющая блокнот 02_moscow_fractional_seir_wave*.jl:
#   * sigma и gamma фиксированы по литературе (инкубация 5,5 сут; заразность 10 сут);
#   * начальное состояние выводится из данных: E0 = y_1 / (scale * sigma),
#     I0 = E0 * sigma / gamma — это снимает проблему фазы волны;
#   * подгоняются только beta, scale и (для дробной модели) alpha;
#   * наблюдаемая величина — scale * sigma * E(t), как в блокноте.
# ---------------------------------------------------------------------------
SIGMA_FIXED = 1.0 / 5.5
GAMMA_FIXED = 1.0 / 10.0
NB_BOUNDS = {
    "beta": (0.01, 5.00, True),
    "scale": (1e-3, 1.00, True),
    "alpha": (ALPHA_LO, ALPHA_HI, False),
}


def fit_seir_notebook(w: WaveData, model: str, observable: str = "daily_ma7",
                      N: float = N_MOSCOW, sigma_fixed: float = SIGMA_FIXED,
                      gamma_fixed: float = GAMMA_FIXED, n_starts: int = 12,
                      seed: int = 2024, steps_per_day: int = STEPS_PER_DAY) -> SeirFit:
    """Воспроизведение постановки блокнота; плюс вариант с накопленной кривой."""
    names = ["beta", "scale"] + (["alpha"] if model == "fractional" else [])
    npar = len(names)

    if observable == "daily_ma7":
        target, mode = w.daily_ma7[1:], "daily"
    elif observable == "daily_raw":
        target, mode = w.daily[1:], "daily"
    elif observable == "cumulative":
        target, mode = w.cumulative, "cum"
    else:
        raise ValueError(observable)

    anchor = float(w.daily_ma7[0])

    def unpack(x: np.ndarray) -> dict:
        p = {}
        for i, nm in enumerate(names):
            lo, hi, log_scale = NB_BOUNDS[nm]
            u = sigmoid(x[i])
            p[nm] = lo * (hi / lo) ** u if log_scale else lo + (hi - lo) * u
        p.setdefault("alpha", 1.0)
        p["e0"] = anchor / (p["scale"] * sigma_fixed)
        p["i0"] = p["e0"] * sigma_fixed / gamma_fixed
        return p

    def pack(vals: dict) -> np.ndarray:
        x = []
        for nm in names:
            lo, hi, log_scale = NB_BOUNDS[nm]
            v = min(max(vals[nm], lo * 1.0001 if log_scale else lo + 1e-9),
                    hi * 0.9999 if log_scale else hi - 1e-9)
            u = math.log(v / lo) / math.log(hi / lo) if log_scale else (v - lo) / (hi - lo)
            x.append(logit(u))
        return np.array(x)

    def model_obs(x: np.ndarray) -> np.ndarray:
        p = unpack(x)
        E, Cint = simulate_seir(p["alpha"], p["beta"], sigma_fixed, gamma_fixed, N,
                                p["e0"], p["i0"], w.n, steps_per_day)
        if not np.all(np.isfinite(E)):
            return np.full(len(target), 1e12)
        amp = p["scale"] * sigma_fixed
        return amp * (E[1:] if mode == "daily" else Cint)

    def residuals(x: np.ndarray) -> np.ndarray:
        return model_obs(x) - target

    rng = np.random.default_rng(seed)
    starts = []
    for b0 in np.exp(np.linspace(math.log(0.15), math.log(1.5), n_starts)):
        sc = float(np.exp(rng.uniform(math.log(0.02), math.log(0.5))))
        al = float(rng.uniform(0.5, 0.95))
        starts.append(pack(dict(beta=float(b0), scale=sc, alpha=al)))

    best = None
    for x0 in starts:
        try:
            sol = least_squares(residuals, x0, method="trf", xtol=1e-12, ftol=1e-12,
                                gtol=1e-12, max_nfev=2000)
        except Exception:
            continue
        rss = float(np.sum(sol.fun ** 2))
        if not math.isfinite(rss):
            continue
        if best is None or rss < best[0]:
            best = (rss, sol)
    if best is None:
        raise RuntimeError(f"не удалось подобрать параметры: {w.name}/{model}/{observable}")

    rss, sol = best
    p = unpack(sol.x)
    pred = model_obs(sol.x)
    res = pred - target
    n = len(target)
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae = float(np.mean(np.abs(res)))
    r2 = float(1.0 - np.sum(res ** 2) / np.sum((target - target.mean()) ** 2))
    aic = float(n * math.log(rss / n) + 2 * npar)
    bic = float(n * math.log(rss / n) + npar * math.log(n))

    ci: dict[str, list[float]] = {}
    z = 1.959964
    try:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * (rss / max(n - npar, 1))
        se_x = np.sqrt(np.abs(np.diag(cov)))
        for i, nm in enumerate(names):
            lo, hi, log_scale = NB_BOUNDS[nm]
            u = sigmoid(sol.x[i])
            dv_du = p[nm] * math.log(hi / lo) if log_scale else hi - lo
            se_v = abs(u * (1 - u) * dv_du * se_x[i])
            ci[nm] = [float(max(lo, p[nm] - z * se_v)), float(min(hi, p[nm] + z * se_v))]
    except Exception:
        pass

    return SeirFit(
        wave=w.name, model=model, observable=observable, preset="notebook",
        e0_mode="из данных", params=dict(alpha=p["alpha"], beta=p["beta"], sigma=sigma_fixed,
                                         gamma=gamma_fixed, scale=p["scale"], q=p["scale"],
                                         e0=p["e0"], R0=p["beta"] / gamma_fixed,
                                         incubation_days=1.0 / sigma_fixed),
        rmse=rmse, mae=mae, r2=r2, aic=aic, bic=bic, n=n, k=npar, ci=ci, curve=pred,
    )
