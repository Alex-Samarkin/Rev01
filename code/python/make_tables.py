"""Формирование таблиц статьи из результатов идентификации.

Пишет article/tables.md (Markdown) — таблицы 1-3 в том виде, в каком они
вставляются в окончательный текст статьи.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from fractional_sir import RESULTS, ROOT, load_waves  # noqa: E402

ART = ROOT / "article"
ART.mkdir(exist_ok=True)

SOURCE_LABEL = {
    "Москва": "COVID-19 в России [25]",
    "Псковская область": "COVID-19 в России [25]",
    "Калининградская область": "COVID-19 в России [25]",
    "Новгородская область": "COVID-19 в России [25]",
    "Нью-Йорк": "NYC DOHMH [23]",
    "Милан": "Protezione Civile [24]",
    "Дели": "IBM covid19-india-data [26]",
}


def num(v: float, nd: int = 1) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:,.{nd}f}".replace(",", " ")


def table1(waves: dict) -> str:
    rows = [
        "| Территория | Страна | Тип единицы | Население | Источник данных | Окно наблюдения | n, сут |",
        "|---|---|---:|---:|---|---|---:|",
    ]
    order = ["Москва", "Псковская область", "Калининградская область",
             "Новгородская область", "Нью-Йорк", "Милан", "Дели"]
    for name in order:
        w = waves[name]
        rows.append(
            f"| {name} | {w.country} | {w.unit_type} | {w.population:,.0f}".replace(",", " ")
            + f" | {SOURCE_LABEL[name]} | {w.dates[0]} — {w.dates[-1]} | {w.n} |"
        )
    return "\n".join(rows)


def table2(res: dict, waves: dict) -> str:
    rows = [
        "| Территория | Модель | α (95 % ДИ) | λ, сут⁻¹ | R₀ | A | RMSE | R² | AIC | ΔAIC |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["Москва", "Псковская область", "Калининградская область",
             "Новгородская область", "Нью-Йорк", "Милан", "Дели"]
    by = {(r["territory"], r["model"]): r for r in res["rows"]}
    for name in order:
        cl, fr = by[(name, "classic")], by[(name, "fractional")]
        daic = cl["aic"] - fr["aic"]
        alpha_txt = f"{fr['alpha']:.3f} ({fr['ci_alpha_lo']:.3f}–{fr['ci_alpha_hi']:.3f})"
        rows.append(
            f"| {name} | классическая | 1 | {num(cl['lambda'], 3)} | {cl['R0']:.3f} | "
            f"{num(cl['A'], 0)} | {num(cl['rmse'])} | {cl['r2']:.4f} | {num(cl['aic'])} | — |")
        rows.append(
            f"| {name} | дробная | {alpha_txt} | {num(fr['lambda'], 3)} | {fr['R0']:.3f} | "
            f"{num(fr['A'], 0)} | {num(fr['rmse'])} | {fr['r2']:.4f} | {num(fr['aic'])} | "
            f"{'+' if daic > 0 else ''}{num(daic)} |")
    return "\n".join(rows)


def table3(res: dict) -> str:
    rows = [
        "| Территория | RMSE классической | RMSE дробной | Снижение RMSE, % | Вес Акаике: классическая / дробная |",
        "|---|---:|---:|---:|---|",
    ]
    by = {(r["territory"], r["model"]): r for r in res["rows"]}
    order = ["Москва", "Псковская область", "Калининградская область",
             "Новгородская область", "Нью-Йорк", "Милан", "Дели"]
    for name in order:
        cl, fr = by[(name, "classic")], by[(name, "fractional")]
        a = [cl["aic"], fr["aic"]]
        m = max(a)
        wts = [pow(2.718281828, -0.5 * (x - m)) for x in a]
        s = sum(wts)
        wts = [x / s for x in wts]
        red = 100 * (1 - fr["rmse"] / cl["rmse"])
        rows.append(f"| {name} | {num(cl['rmse'])} | {num(fr['rmse'])} | {red:.1f} | "
                    f"{wts[0]:.3f} / {wts[1]:.3f} |")
    return "\n".join(rows)


def table4_sensitivity() -> str:
    path = RESULTS / "sensitivity_tau.csv"
    if not path.exists():
        return "_(данные анализа чувствительности отсутствуют)_"
    df = pd.read_csv(path)
    fr = df[df["model"] == "fractional"]
    piv_a = fr.pivot_table(index="territory", columns="tau_fixed", values="alpha")
    piv_r = fr.pivot_table(index="territory", columns="tau_fixed", values="R0")
    taus = sorted(fr["tau_fixed"].unique())
    head = "| Территория | " + " | ".join(f"α (τ = {t:g})" for t in taus) + \
           " | " + " | ".join(f"R₀ (τ = {t:g})" for t in taus) + " |"
    sep = "|---|" + "---:|" * (2 * len(taus))
    rows = [head, sep]
    for name in piv_a.index:
        cells = [f"{piv_a.loc[name, t]:.3f}" for t in taus] + \
                [f"{piv_r.loc[name, t]:.3f}" for t in taus]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    res = json.loads((RESULTS / "identification_cumulative.json").read_text(encoding="utf-8"))
    waves = load_waves()
    out = [
        "# Таблицы статьи", "",
        "## Таблица 1. Территории, источники данных и окна наблюдения", "",
        table1(waves), "",
        "## Таблица 2. Оценки параметров классической и дробной SIR-моделей", "",
        table2(res, waves), "",
        "## Таблица 3. Сравнение качества подгонки", "",
        table3(res), "",
        "## Таблица 4. Чувствительность оценок к длительности инфекционного периода τ", "",
        table4_sensitivity(), "",
    ]
    path = ART / "tables.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"Записано: {path}")
    print("\n".join(out[3:14]))


if __name__ == "__main__":
    main()
