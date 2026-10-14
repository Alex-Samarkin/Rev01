"""Верификация схемы L1: решение D^alpha y = -lambda*y против функции Миттаг-Леффлера.

Аналитические контрольные точки:
    alpha = 1/2, lambda = 1, t = 1  ->  y = E_{1/2}(-1) = exp(1)*erfc(1) = 0.4275850...
    alpha = 1,   lambda = 1, t = 1  ->  y = exp(-1) = 0.3678794...
"""
from __future__ import annotations

import io
import math
import sys

import numpy as np
from scipy.special import erfc, gamma as gamma_fn

sys.path.insert(0, r"C:\Users\Alex\Rev01\code\python")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def l1_scalar(alpha: float, lam: float, t_end: float, n_steps: int, y0: float = 1.0,
              implicit: bool = False) -> np.ndarray:
    """Схема L1 для D^alpha y = -lam*y, y(0)=y0.

    implicit=False — f вычисляется на предыдущем слое (явный вариант, порядок O(dt));
    implicit=True  — f вычисляется на текущем слое (неявный вариант, порядок O(dt^(2-alpha))).
    """
    dt = t_end / n_steps
    b = np.power(np.arange(n_steps + 1, dtype=float) + 1.0, 1.0 - alpha) - \
        np.power(np.arange(n_steps + 1, dtype=float), 1.0 - alpha)
    G = gamma_fn(2.0 - alpha) * dt ** alpha
    y = np.empty(n_steps + 1)
    dy = np.zeros(n_steps + 1)
    y[0] = y0
    for n in range(1, n_steps + 1):
        p = n - 1
        h = (b[1:n] @ dy[n - 1:0:-1]) if n > 1 else 0.0
        if implicit:
            y[n] = (y[p] - h) / (1.0 + G * lam)
        else:
            y[n] = y[p] + G * (-lam * y[p]) - h
        dy[n] = y[n] - y[p]
    return y


def mittag_leffler_series(z: float, alpha: float, terms: int = 400) -> float:
    """E_alpha(z) = sum z^k / Gamma(alpha*k + 1) — для контрольных расчётов."""
    s = 0.0
    for k in range(terms):
        s += z ** k / gamma_fn(alpha * k + 1.0)
    return s


def main() -> None:
    print("Проверка 1: alpha = 1/2, lambda = 1, t = 1")
    exact = math.exp(1.0) * erfc(1.0)
    for n in (100, 1000, 10000):
        y = l1_scalar(0.5, 1.0, 1.0, n)
        print(f"  n={n:6d}  y(1)={y[-1]:.8f}  точное={exact:.8f}  |err|={abs(y[-1]-exact):.2e}")

    print("\nПроверка 2: alpha = 1, lambda = 1, t = 1 (должно совпасть с exp(-1))")
    print(f"  точное = {math.exp(-1.0):.8f}")
    for n in (100, 1000):
        y = l1_scalar(1.0, 1.0, 1.0, n)
        print(f"  n={n:6d}  y(1)={y[-1]:.8f}  |err|={abs(y[-1]-math.exp(-1.0)):.2e}")

    print("\nПроверка 3: порядок точности по dt (alpha = 0.5, t = 1)")
    for impl in (False, True):
        errs = []
        for n in (100, 200, 400, 800):
            e = abs(l1_scalar(0.5, 1.0, 1.0, n, implicit=impl)[-1] - exact)
            errs.append(e)
        tag = "неявная" if impl else "явная  "
        orders = [f"{math.log(errs[i - 1] / errs[i]) / math.log(2.0):.3f}" for i in range(1, len(errs))]
        print(f"  {tag}: ошибки {[f'{e:.2e}' for e in errs]}  порядки {orders}  "
              f"(теория: {'2-alpha = 1.5' if impl else '1.0'})")

    print("\nПроверка 4: сравнение с рядом Миттаг-Леффлера, t = 5, alpha = 0.5, lambda = 1")
    ml = mittag_leffler_series(-math.sqrt(5.0), 0.5)
    for impl in (False, True):
        y = l1_scalar(0.5, 1.0, 5.0, 20000, implicit=impl)
        tag = "неявная" if impl else "явная  "
        print(f"  {tag}: L1 = {y[-1]:.8f}   ряд = {ml:.8f}   |err| = {abs(y[-1] - ml):.2e}")


if __name__ == "__main__":
    main()
