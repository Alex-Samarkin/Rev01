"""Печать формы волн: суточная сглаженная кривая по неделям."""
import io
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
df = pd.read_csv(r"C:\Users\Alex\Rev01\data\processed\waves.csv")

for terr, g in df.groupby("territory", sort=False):
    g = g.sort_values("date").reset_index(drop=True)
    ma = g["daily_ma7"]
    pk = ma.max()
    pk_i = ma.idxmax()
    print(f"\n=== {terr} ===  пик MA7 = {pk:,.0f} ({g['date'][pk_i]}), "
          f"в начале = {ma.iloc[0]:,.1f}, в конце = {ma.iloc[-1]:,.1f} "
          f"({100 * ma.iloc[-1] / pk:.1f} % от пика)")
    step = max(1, len(g) // 12)
    sub = g.iloc[::step]
    print("  " + "  ".join(f"{d[5:]}:{v:,.0f}" for d, v in zip(sub["date"], sub["daily_ma7"])))
    print(f"  доля окна после пика: {100 * (len(g) - pk_i - 1) / len(g):.0f} %")
