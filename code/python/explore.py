"""Разведочный анализ загруженных рядов: какие волны видны, какие окна брать."""
import io
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
RAW = r"C:\Users\Alex\Rev01\data\raw"

# ---------- Нью-Йорк ----------
nyc = pd.read_csv(f"{RAW}/nyc_cases_by_day.csv")
nyc["date"] = pd.to_datetime(nyc["date_of_interest"], format="%m/%d/%Y")
nyc = nyc[["date", "CASE_COUNT", "PROBABLE_CASE_COUNT"]].sort_values("date")
nyc["total"] = nyc["CASE_COUNT"] + nyc["PROBABLE_CASE_COUNT"]
nyc["avg7"] = nyc["total"].rolling(7, center=True).mean()
print("=== NYC ===", nyc["date"].min().date(), "->", nyc["date"].max().date())
top = nyc.nlargest(12, "avg7")[["date", "avg7"]].sort_values("date")
print(top.to_string(index=False))

# ---------- Италия, провинция Милан ----------
it = pd.read_csv(f"{RAW}/pcm_dpc_province.csv")
it["date"] = pd.to_datetime(it["data"]).dt.normalize()
print("\n=== ITALY provinces ===", it["date"].min().date(), "->", it["date"].max().date())
mi = it[(it["denominazione_provincia"] == "Milano")][["date", "totale_casi"]].sort_values("date")
print("Milano rows:", len(mi), mi["date"].min().date(), "->", mi["date"].max().date())
print("Milano non-null:", mi["totale_casi"].notna().sum(), "last value:", mi["totale_casi"].dropna().iloc[-1])
mi_nz = mi[mi["totale_casi"].fillna(0) > 0]
print("Milano first>0:", mi_nz["date"].iloc[0].date(), " last>0:", mi_nz["date"].iloc[-1].date())

# ---------- Россия ----------
ru = pd.read_csv(f"{RAW}/ru_regions_infected.csv", encoding="utf-8")
print("\n=== RU ===", ru.shape)
names = [n for n in ru["Субъект"] if any(k in n for k in ("Москва", "Псков", "Калининград", "Новгород"))]
print("matched regions:", names)
print("date columns:", list(ru.columns[2:6]), "...", list(ru.columns[-3:]))

# ---------- Индия, Дели ----------
dl = pd.read_csv(f"{RAW}/in_delhi_cumulative.csv")
dl["date"] = pd.to_datetime(dl["date"])
dln = dl.dropna(subset=["cumulative_positive_cases"])
print("\n=== DELHI ===", dl.shape, "non-null:", len(dln))
print("range:", dln["date"].min().date(), "->", dln["date"].max().date())
print(dln.head(3).to_string(index=False))
print(dln.tail(3).to_string(index=False))
