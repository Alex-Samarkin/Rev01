"""Диагностика проблемных рядов: Нью-Йорк (суточные) и Дели (разрывы)."""
import io
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
RAW = r"C:\Users\Alex\Rev01\data\raw"

dl = pd.read_csv(f"{RAW}/in_delhi_cumulative.csv")
dl["date"] = pd.to_datetime(dl["date"])
dl = dl.dropna(subset=["cumulative_positive_cases"]).sort_values("date").reset_index(drop=True)
dl["daily"] = dl["cumulative_positive_cases"].diff()
print("=== DELHI: крупнейшие суточные приросты ===")
print(dl.nlargest(8, "daily")[["date", "cumulative_positive_cases", "daily"]].to_string(index=False))
print("\nОтрицательные приросты:", int((dl["daily"] < 0).sum()))
print(dl[dl["daily"] < 0][["date", "cumulative_positive_cases", "daily"]].head(10).to_string(index=False))

print("\n=== DELHI: окно 2021-03-01..2021-07-31 ===")
w = dl[(dl["date"] >= "2021-03-01") & (dl["date"] <= "2021-07-31")]
print("n =", len(w), " C:", w["cumulative_positive_cases"].iloc[0], "->", w["cumulative_positive_cases"].iloc[-1])
print(w[["date", "daily"]].head(10).to_string(index=False))
print(w[["date", "daily"]].tail(5).to_string(index=False))

nyc = pd.read_csv(f"{RAW}/nyc_cases_by_day.csv")
nyc["date"] = pd.to_datetime(nyc["date_of_interest"], format="%m/%d/%Y")
nyc = nyc.sort_values("date")
nyc["cum"] = nyc["CASE_COUNT"].cumsum()
w2 = nyc[(nyc["date"] >= "2020-02-29") & (nyc["date"] <= "2020-07-31")]
print("\n=== NYC окно ===")
print("n =", len(w2), " C:", w2["cum"].iloc[0], "->", w2["cum"].iloc[-1], " пик/сутки:", w2["CASE_COUNT"].max())
