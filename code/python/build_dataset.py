"""Подготовка рядов: выделение окон волн, приведение к единому виду.

Читает data/raw/*, формирует data/processed/waves.csv и data/processed/windows.json.

Все источники приводятся к единой схеме:
    накопленный ряд -> cummax (устранение технических «откатов» регистрации)
                    -> суточный прирост
                    -> окно волны
                    -> cumulative_rel = накопленная сумма внутри окна.

Формат waves.csv (long table):
    territory, country, unit_type, population, date, cumulative, cumulative_rel, daily
где cumulative      — накопленное число подтверждённых случаев с начала эпидемии
                      (справочно, для контроля);
    cumulative_rel  — накопленная сумма внутри окна, отсчитываемая от нуля
                      (величина, сопоставляемая с моделью: rho * (N - S(t)));
    daily           — суточный прирост внутри окна.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(r"C:\Users\Alex\Rev01")
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- окна волн ---
# Правило: окно охватывает полностью наблюдаемые фазы роста и спада одной волны;
# на границах окна суточная заболеваемость мала по сравнению с пиком.
WINDOWS = {
    # территория: (источник, начало, конец, население, страна, тип единицы)
    "Москва":                  ("ru", "2020-03-26", "2020-08-15", 12_678_079, "Россия", "город"),
    "Псковская область":       ("ru", "2020-03-26", "2020-08-15",    626_115, "Россия", "регион"),
    "Калининградская область": ("ru", "2020-03-26", "2020-08-15",  1_012_512, "Россия", "регион"),
    "Новгородская область":    ("ru", "2020-03-26", "2020-08-15",    596_508, "Россия", "регион"),
    "Нью-Йорк":                ("nyc", "2020-02-29", "2020-07-31", 8_336_817, "США", "город"),
    "Милан":                   ("it", "2020-02-25", "2020-07-31",  3_259_000, "Италия", "провинция"),
    "Дели":                    ("in", "2021-03-01", "2021-07-31", 19_800_000, "Индия", "город"),
}

RU_NAMES = {
    "Москва": "Москва",
    "Псковская область": "Псковская Область",
    "Калининградская область": "Калининградская Область",
    "Новгородская область": "Новгородская Область",
}


def cumulative_to_daily(cum: pd.Series) -> pd.Series:
    """Накопленный ряд -> суточный прирост (монотонизация, первый день = 0)."""
    cum = cum.sort_index()
    cum = cum[~cum.index.duplicated(keep="last")]
    c = cum.cummax()
    d = c.diff()
    d.iloc[0] = 0.0
    return d.clip(lower=0.0)


def load_ru() -> dict[str, pd.Series]:
    """Суточные приросты по субъектам РФ (источник отдаёт накопленные значения)."""
    df = pd.read_csv(RAW / "ru_regions_infected.csv", encoding="utf-8")
    cols = list(df.columns[2:])
    dates = pd.to_datetime([c + ".2020" for c in cols], format="%d.%m.%Y")
    out = {}
    for terr, ru_name in RU_NAMES.items():
        row = df[df["Субъект"] == ru_name]
        if row.empty:
            raise SystemExit(f"не найдена строка для {ru_name}")
        cum = pd.Series(row.iloc[0, 2:].astype(float).to_numpy(), index=dates)
        out[terr] = cumulative_to_daily(cum)
    return out


def load_nyc() -> pd.Series:
    """Суточные подтверждённые случаи по Нью-Йорку (источник уже суточный)."""
    df = pd.read_csv(RAW / "nyc_cases_by_day.csv")
    df["date"] = pd.to_datetime(df["date_of_interest"], format="%m/%d/%Y")
    df = df.sort_values("date")
    s = pd.Series(df["CASE_COUNT"].astype(float).to_numpy(), index=df["date"])
    return s[~s.index.duplicated(keep="last")]


def load_it() -> pd.Series:
    """Провинция Милан: накопленные случаи -> суточные."""
    df = pd.read_csv(RAW / "pcm_dpc_province.csv")
    df = df[df["denominazione_provincia"] == "Milano"].copy()
    df["date"] = pd.to_datetime(df["data"]).dt.normalize()
    cum = pd.Series(df["totale_casi"].astype(float).to_numpy(), index=df["date"])
    return cumulative_to_daily(cum)


def load_in() -> pd.Series:
    """Дели: накопленные случаи -> суточные."""
    df = pd.read_csv(RAW / "in_delhi_cumulative.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["cumulative_positive_cases"]).sort_values("date")
    cum = pd.Series(df["cumulative_positive_cases"].astype(float).to_numpy(), index=df["date"])
    return cumulative_to_daily(cum)


def main() -> None:
    daily_src = {
        "ru": load_ru(),
        "nyc": {"Нью-Йорк": load_nyc()},
        "it": {"Милан": load_it()},
        "in": {"Дели": load_in()},
    }

    rows, meta = [], []
    for terr, (src, d0, d1, pop, country, unit) in WINDOWS.items():
        d = daily_src[src][terr]
        win = d[(d.index >= pd.Timestamp(d0)) & (d.index <= pd.Timestamp(d1))]
        if win.empty:
            raise SystemExit(f"пустое окно для {terr}")
        win = win.sort_index()
        cum_rel = win.cumsum()
        cum_abs = cum_rel + float(d[d.index <= win.index[0]].sum() - win.iloc[0])
        ma7 = win.rolling(7, center=True, min_periods=3).mean()

        for dt in win.index:
            rows.append(dict(
                territory=terr, country=country, unit_type=unit, population=pop,
                date=dt.strftime("%Y-%m-%d"),
                cumulative=float(cum_abs[dt]),
                cumulative_rel=float(cum_rel[dt]),
                daily=float(win[dt]),
                daily_ma7=float(ma7[dt]),
            ))
        meta.append(dict(
            territory=terr, country=country, unit_type=unit, population=pop,
            date_start=d0, date_end=d1, n_days=int(len(win)),
            cumulative_end=float(cum_abs.iloc[-1]),
            cumulative_rel_end=float(cum_rel.iloc[-1]),
            daily_peak=float(win.max()),
            daily_peak_date=win.idxmax().strftime("%Y-%m-%d"),
        ))
        print(f"{terr:26s} {d0}..{d1}  n={len(win):4d}  C_rel={cum_rel.iloc[-1]:9.0f}  "
              f"пик/сутки={win.max():7.0f} ({win.idxmax():%d.%m.%Y})")

    pd.DataFrame(rows).to_csv(OUT / "waves.csv", index=False, encoding="utf-8")
    (OUT / "windows.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nЗаписано: {OUT / 'waves.csv'} ({len(rows)} строк)")
    print(f"Записано: {OUT / 'windows.json'}")


if __name__ == "__main__":
    main()
