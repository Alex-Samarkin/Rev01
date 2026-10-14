"""Дробная SIR-модель (производная Капуто) и идентификация параметров волны.

Модель
------
    D^alpha S = -beta * S * I / A
    D^alpha I =  beta * S * I / A - gamma * I
    D^alpha R =  gamma * I
    S(0) = A - I0,  I(0) = I0,  R(0) = 0

D^alpha — производная Капуто порядка alpha in (0, 1]; A — эффективная численность
популяции; I0 = 1. При alpha = 1 модель переходит в классическую SIR.

Параметризация
--------------
Вместо (beta, gamma) модель параметризуется эпидемиологически интерпретируемыми
величинами:

    R0 = beta / gamma          — базовое репродуктивное число,
    lambda = (beta - gamma)^(1/alpha)  — начальная скорость экспоненциального роста.

Обратный пересчёт: gamma = lambda^alpha / (R0 - 1), beta = R0 * gamma.
Такая параметризация корректна при любом alpha (величины beta и gamma в дробной
модели имеют размерность сут^-alpha и напрямую несопоставимы между собой и с
классической SIR), а R0 и lambda сопоставимы между моделями и территориями.

Наблюдаемые величины
--------------------
* ``daily_ma7`` — суточная заболеваемость, сглаженная скользящим средним за 7 суток;
* ``cumulative`` — накопленное число случаев внутри окна волны.

Численная схема — L1 (дискретизация Капуто):

    y_n = y_{n-1} + Gamma(2-alpha) * dt^alpha * f(y_{n-1})
          - sum_{k=1}^{n-1} b_k * (y_{n-k} - y_{n-k-1}),
    b_k = (k+1)^(1-alpha) - k^(1-alpha),  b_0 = 1.

Точность проверена в ``verify_scheme.py`` сопоставлением с E_alpha(-lambda t^alpha).

Идентифицируемость
------------------
Скорость инфекции beta и коэффициент регистрации rho раздельно не идентифицируются
(в наблюдение входит только произведение), поэтому наблюдение нормировано на
эффективную численность A: C(t) = A - S(t); A ограничена сверху десятикратным
превышением наблюдённого числа случаев (коэффициент регистрации не ниже 10 %).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import gamma as gamma_fn

ROOT = Path(r"C:\Users\Alex\Rev01")
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

STEPS_PER_DAY = 10                      # dt = 0.1 сут (устойчивость схемы при малых alpha)

# Границы параметров
LAMBDA_LO, LAMBDA_HI = 0.02, 0.90       # сут^-1: начальная скорость роста
TAU_FIXED = 7.0                         # длительность инфекционного периода, сутки
TAU_GRID = (5.0, 7.0, 10.0)             # варианты для анализа чувствительности
ALPHA_LO, ALPHA_HI = 0.10, 1.00         # порядок дробной производной
I0_LO, I0_HI = 0.5, 100000.0            # начальное число инфицированных в момент t0
A_REPORT_MIN = 0.10                     # минимальная доля регистрации -> A <= 10 * C_end
A_POP_MAX = 12.0                        # A <= 12 * численность населения
BETA_MAX = 6.0                          # техническое ограничение (устойчивость схемы)

OBSERVABLES = ("daily_ma7", "cumulative")


# ------------------------------------------------------------------ решатель ---
def l1_weights(alpha: float, n: int) -> np.ndarray:
    """Весовые коэффициенты схемы L1: b_k = (k+1)^(1-alpha) - k^(1-alpha)."""
    k = np.arange(n + 1, dtype=float)
    return np.power(k + 1.0, 1.0 - alpha) - np.power(k, 1.0 - alpha)


def simulate_sir(alpha: float, beta: float, gamma: float, A: float, n_days: int,
                 i0: float = 1.0, steps_per_day: int = STEPS_PER_DAY) -> np.ndarray:
    """Накопленное число инфицированных C(t_j), j = 0..n_days-1 (шаг 1 сутки)."""
    n_steps = (n_days - 1) * steps_per_day
    dt = 1.0 / steps_per_day
    b = l1_weights(alpha, n_steps)
    G = gamma_fn(2.0 - alpha) * dt ** alpha

    S = np.empty(n_steps + 1)
    I = np.empty(n_steps + 1)
    dS = np.zeros(n_steps + 1)
    dI = np.zeros(n_steps + 1)
    S[0] = A - i0
    I[0] = i0

    for n in range(1, n_steps + 1):
        p = n - 1
        if n > 1:
            w = b[1:n]
            hS = w @ dS[n - 1:0:-1]
            hI = w @ dI[n - 1:0:-1]
        else:
            hS = hI = 0.0
        inf = beta * (S[p] / A) * I[p]
        S[n] = S[p] + G * (-inf) - hS
        I[n] = I[p] + G * (inf - gamma * I[p]) - hI
        dS[n] = S[n] - S[p]
        dI[n] = I[n] - I[p]

    idx = np.arange(n_days) * steps_per_day
    return A - S[idx]


def rates_from_lambda_tau(lam: float, tau: float, alpha: float) -> tuple[float, float, float]:
    """(lambda, tau, alpha) -> (beta, gamma, R0).

    tau — характерная длительность инфекционного периода (сутки), gamma = tau^(-alpha);
    lambda — начальная скорость экспоненциального роста (сут^-1), beta - gamma = lambda^alpha.
    Тогда R0 = beta / gamma = 1 + (lambda * tau)^alpha; при alpha = 1 получаем
    классическое соотношение R0 = 1 + lambda * tau.
    """
    gamma = tau ** (-alpha)
    beta = gamma + lam ** alpha
    return beta, gamma, 1.0 + (lam * tau) ** alpha


# ------------------------------------------------------- преобразование шкал ---
def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def logit(x: float) -> float:
    return math.log(x / (1.0 - x))


def _from_unit(u: float, lo: float, hi: float, log_scale: bool) -> float:
    return lo * (hi / lo) ** u if log_scale else lo + (hi - lo) * u


def _to_unit(v: float, lo: float, hi: float, log_scale: bool) -> float:
    if log_scale:
        v = min(max(v, lo * 1.0001), hi * 0.9999)
        return math.log(v / lo) / math.log(hi / lo)
    v = min(max(v, lo + 1e-9), hi - 1e-9)
    return (v - lo) / (hi - lo)


BOUNDS = {
    "lam": (LAMBDA_LO, LAMBDA_HI, True),

    "alpha": (ALPHA_LO, ALPHA_HI, False),
    "i0": (I0_LO, I0_HI, True),
    "A": (None, None, False),   # границы зависят от территории
}


@dataclass
class Wave:
    """Наблюдаемые данные одной волны."""
    territory: str
    country: str
    unit_type: str
    population: float
    dates: np.ndarray
    cumulative: np.ndarray
    daily: np.ndarray
    daily_ma7: np.ndarray

    @property
    def n(self) -> int:
        return len(self.cumulative)


def load_waves() -> dict[str, Wave]:
    import pandas as pd

    df = pd.read_csv(DATA / "waves.csv")
    out: dict[str, Wave] = {}
    for terr, g in df.groupby("territory", sort=False):
        g = g.sort_values("date")
        out[terr] = Wave(
            territory=terr, country=g["country"].iloc[0], unit_type=g["unit_type"].iloc[0],
            population=float(g["population"].iloc[0]), dates=g["date"].to_numpy(),
            cumulative=g["cumulative_rel"].to_numpy(dtype=float),
            daily=g["daily"].to_numpy(dtype=float),
            daily_ma7=g["daily_ma7"].to_numpy(dtype=float),
        )
    return out


# ------------------------------------------------------------- идентификация ---
@dataclass
class FitResult:
    territory: str
    model: str
    observable: str
    params: dict
    rmse: float
    r2: float
    aic: float
    bic: float
    n: int
    k: int
    ci: dict = field(default_factory=dict)
    train_rmse: float = float("nan")
    test_rmse: float = float("nan")
    test_mape_daily: float = float("nan")
    curve_cumulative: np.ndarray | None = None
    curve_daily: np.ndarray | None = None

    def summary(self) -> dict:
        d = dict(territory=self.territory, model=self.model, observable=self.observable,
                 rmse=self.rmse, r2=self.r2, aic=self.aic, bic=self.bic, n=self.n, k=self.k,
                 train_rmse=self.train_rmse, test_rmse=self.test_rmse, test_mape=self.test_mape_daily)
        for key in ("beta", "gamma", "A", "alpha", "R0", "lambda_", "tau", "i0"):
            d[key.rstrip("_")] = self.params.get(key, float("nan"))
        d["infected_fraction"] = self.params.get("infected_fraction", float("nan"))
        for key in ("R0", "lambda", "A", "alpha", "beta", "gamma"):
            lo_hi = self.ci.get(key)
            d[f"ci_{key}_lo"] = lo_hi[0] if lo_hi else float("nan")
            d[f"ci_{key}_hi"] = lo_hi[1] if lo_hi else float("nan")
        return d


def _limits(w: Wave) -> tuple[float, float]:
    """Границы эффективной численности A."""
    c_end = float(w.cumulative[-1])
    return c_end, min(c_end / A_REPORT_MIN, A_POP_MAX * w.population)


def _obs_and_map(w: Wave, observable: str):
    if observable == "daily_ma7":
        return w.daily_ma7[1:], (lambda C: np.diff(C))
    if observable == "cumulative":
        return w.cumulative, (lambda C: C)
    raise ValueError(observable)


def param_names(model: str) -> list[str]:
    return ["lam"] + (["alpha"] if model == "fractional" else []) + ["i0", "A"]


def fit_wave(w: Wave, model: str, observable: str = "daily_ma7", n_starts: int = 16,
             seed: int = 20260115, test_fraction: float = 0.3,
             steps_per_day: int = STEPS_PER_DAY, tau: float = TAU_FIXED) -> FitResult:
    """МНК-идентификация параметров одной волны (trust-region reflective, multi-start)."""
    a_lo, a_hi = _limits(w)
    target, to_obs = _obs_and_map(w, observable)
    names = param_names(model)
    npar = len(names)
    bounds = dict(BOUNDS)
    bounds["A"] = (a_lo, a_hi, False)

    def unpack(x: np.ndarray) -> dict:
        u = {}
        for i, nm in enumerate(names):
            lo, hi, log_scale = bounds[nm]
            u[nm] = _from_unit(sigmoid(x[i]), lo, hi, log_scale)
        alpha = u.get("alpha", 1.0)
        beta, gamma, R0 = rates_from_lambda_tau(u["lam"], tau, alpha)
        return dict(u, alpha=alpha, beta=beta, gamma=gamma, R0=R0, tau=tau)

    def pack(vals: dict) -> np.ndarray:
        return np.array([logit(_to_unit(vals[nm], *bounds[nm])) for nm in names])

    def model_obs(x: np.ndarray) -> np.ndarray:
        p = unpack(x)
        if not math.isfinite(p["beta"]) or p["beta"] > BETA_MAX or p["gamma"] <= 0:
            return np.full(len(target), 1e12)
        C = simulate_sir(p["alpha"], p["beta"], p["gamma"], p["A"], w.n, i0=p["i0"],
                         steps_per_day=steps_per_day)
        if not np.all(np.isfinite(C)):
            return np.full(len(target), 1e12)
        return to_obs(C)

    def residuals(x: np.ndarray, n_points: int | None = None) -> np.ndarray:
        pred = model_obs(x)
        k = n_points if n_points is not None else len(target)
        return pred[:k] - target[:k]

    # Детерминированная сетка стартов по (lambda, R0) плюс случайные добавки:
    # целевая функция многомодальна, одного локального запуска недостаточно.
    # Детерминированная сетка стартов: целевая функция многомодальна.
    grid_lam = [0.04, 0.07, 0.12, 0.20, 0.35]
    grid_i0 = [1.0, 30.0, 300.0, 3000.0]
    grid_afrac = [0.03, 0.15, 0.5]
    starts = []
    for i, v in enumerate(grid_lam):
        for j, i0v in enumerate(grid_i0):
            starts.append(pack(dict(lam=v, alpha=0.6, i0=i0v,
                                    A=a_lo + (a_hi - a_lo) * grid_afrac[(i + j) % 3])))
    rng = np.random.default_rng(seed)
    while len(starts) < n_starts:
        starts.append(pack(dict(
            lam=float(rng.uniform(0.03, 0.5)), alpha=float(rng.uniform(0.15, 0.95)),
            i0=float(rng.uniform(1.0, 3000.0)),
            A=a_lo + (a_hi - a_lo) * float(rng.uniform(0.02, 0.8)))))

    best = None
    for x0 in starts:
        try:
            sol = least_squares(residuals, x0, method="trf", xtol=1e-12, ftol=1e-12,
                                gtol=1e-12, max_nfev=4000)
        except Exception:
            continue
        rss = float(np.sum(sol.fun ** 2))
        if not math.isfinite(rss):
            continue
        if best is None or rss < best[0]:
            best = (rss, sol)
    if best is None:
        raise RuntimeError(f"не удалось подобрать параметры: {w.territory}/{model}")

    rss, sol = best
    p = unpack(sol.x)
    C = simulate_sir(p["alpha"], p["beta"], p["gamma"], p["A"], w.n, i0=p["i0"],
                     steps_per_day=steps_per_day)
    pred = to_obs(C)
    res = pred - target
    n = len(target)
    rmse = float(np.sqrt(np.mean(res ** 2)))
    r2 = float(1.0 - np.sum(res ** 2) / np.sum((target - target.mean()) ** 2))
    aic = float(n * math.log(rss / n) + 2 * npar)
    bic = float(n * math.log(rss / n) + npar * math.log(n))

    # --- линеаризованные 95 % ДИ ---
    ci: dict[str, list[float]] = {}
    z = 1.959964
    try:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * (rss / max(n - npar, 1))
        se_x = np.sqrt(np.abs(np.diag(cov)))
        for i, nm in enumerate(names):
            lo, hi, log_scale = bounds[nm]
            u = sigmoid(sol.x[i])
            dv_du = (p[nm] * math.log(hi / lo)) if log_scale else (hi - lo)
            se_v = abs(u * (1 - u) * dv_du * se_x[i])
            ci[nm] = [float(max(lo, p[nm] - z * se_v)), float(min(hi, p[nm] + z * se_v))]
    except Exception:
        pass

    # --- прогностическая проверка ---
    n_tr = int(round(n * (1 - test_fraction)))
    train_rmse = test_rmse = test_mape = float("nan")
    try:
        sol_tr = least_squares(residuals, sol.x, args=(n_tr,), method="trf",
                               xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=3000)
        p_tr = unpack(sol_tr.x)
        C_tr = simulate_sir(p_tr["alpha"], p_tr["beta"], p_tr["gamma"], p_tr["A"], w.n,
                            i0=p_tr["i0"], steps_per_day=steps_per_day)
        pred_tr = to_obs(C_tr)
        train_rmse = float(np.sqrt(np.mean((pred_tr[:n_tr] - target[:n_tr]) ** 2)))
        test_rmse = float(np.sqrt(np.mean((pred_tr[n_tr:] - target[n_tr:]) ** 2)))
        if observable == "daily_ma7":
            obs_t, pred_t = target[n_tr:], pred_tr[n_tr:]
            mask = obs_t > 5
            if mask.any():
                test_mape = float(np.mean(np.abs(pred_t[mask] - obs_t[mask]) / obs_t[mask]) * 100)
    except Exception:
        pass

    return FitResult(
        territory=w.territory, model=model, observable=observable,
        params=dict(beta=p["beta"], gamma=p["gamma"], A=p["A"], alpha=p["alpha"],
                    R0=p["R0"], lambda_=p["lam"], tau=p["tau"], i0=p["i0"],
                    infected_fraction=float(w.cumulative[-1] / p["A"])),
        rmse=rmse, r2=r2, aic=aic, bic=bic, n=n, k=npar, ci=ci,
        train_rmse=train_rmse, test_rmse=test_rmse, test_mape_daily=test_mape,
        curve_cumulative=C, curve_daily=np.diff(C),
    )


def aic_weights(aics: list[float]) -> list[float]:
    """Вес Акаике: относительная поддержка моделей."""
    a = np.asarray(aics, dtype=float)
    w = np.exp(-0.5 * (a - a.min()))
    return list(w / w.sum())
