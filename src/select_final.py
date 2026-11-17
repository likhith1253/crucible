"""Final selection.

1. Model ranking: average of a player's rank in the two converged fits
     - full model (all balls, pressure effect as a smooth function of exposure)          -> results/model_full
     - pressure-only model (fitted on pressure balls only)                                -> results/model_pressure_only
   A player only ranks high if both fits agree. Ties are broken by the full-model rank.
2. Availability rule: a player must have played a T20 (IPL or T20I) in the 12 months before the last date in the data.
3. Evidence checks. Every change from the model order is listed in REVIEW with the evidence behind it, so the
   reader can see exactly where judgement was used. The evidence numbers are in results/evidence_*.csv.
"""
import pandas as pd, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pipe

L, A = pipe.tables()
last_date = pd.to_datetime(A.date).max(); cutoff = last_date - pd.DateOffset(years=1)

REVIEW = {
    "bat": {
        "remove": {
            "Cameron Green": "Below expected under pressure in T20Is (-3.5 runs/100 balls); IPL record (+10.5) weaker than the replacements.",
            "Nicholas Pooran": "Well below expected under pressure in T20Is (-17.6 runs/100 balls over 197 balls).",
            "Dewald Brevis": "Only 106 pressure balls; too little evidence for a reliability list.",
        },
        "add": {
            "Sanju Samson": "Positive under pressure in both IPL (+23.4/100 over 200 balls) and T20I (+1.1); +10.9 in high-stakes games.",
            "Travis Head": "Positive under pressure in both IPL (+32.3/100) and T20I (+9.1); positive in 5 of 7 season blocks.",
        },
        "first": None,
        "place": {"Yashasvi Jaiswal": (8, "T20I figure rests on 37 pressure balls; IPL record (+10.8 over 258 balls) is the weakest of the top eight and 2026 was flat (-0.4).")},
    },
    "bowl": {
        "remove": {
            "Wanindu Hasaranga": "Barely above average under pressure in T20Is (+0.06 runs saved/over); no IPL in the latest season.",
            "Krunal Pandya": "No T20I pressure record; death economy 10.48.",
        },
        "add": {
            "Maheesh Theekshana": "Saved runs under pressure in all 9 season blocks; +1.56 IPL, +1.37 T20I, +1.54 in high-stakes games (275 balls).",
        },
        "first": ("Jasprit Bumrah", "Top three are a statistical tie; Bumrah has the best rate (1.99 runs saved/over) and saved runs in every season block."),
    },
}

for side, lab, idc in (("bat", "batter", "striker_id"), ("bowl", "bowler", "bowler_id")):
    T = L if side == "bat" else A
    f = pd.read_csv(os.path.join(OUT, "model_full", f"full_{lab}_rankings.csv"))
    p = pd.read_csv(os.path.join(OUT, "model_pressure_only", f"full_{lab}_rankings.csv"))
    f = f[f["rank"].notna()][["player_id", "name", "rank"]].rename(columns={"rank": "full_model_rank"})
    p = p[p["rank"].notna()][["player_id", "rank"]].rename(columns={"rank": "pressure_only_rank"})
    C = f.merge(p, on="player_id")
    C["model_avg_rank"] = (C.full_model_rank + C.pressure_only_rank) / 2
    C = C.sort_values(["model_avg_rank", "full_model_rank"]).reset_index(drop=True); C["model_rank"] = C.index + 1
    lastp = T.groupby(idc).date.max(); C["last_match"] = pd.to_datetime(C.player_id.map(lastp)).dt.date
    C["available"] = pd.to_datetime(C.last_match) >= cutoff
    C.to_csv(os.path.join(OUT, f"model_ranking_{lab}s.csv"), index=False)

    R = REVIEW[side]; pool = C[C.available & ~C.name.isin(R["remove"])]
    top = pool[~pool.name.isin(R["add"])].head(10 - len(R["add"]))
    top = pd.concat([top, C[C.name.isin(R["add"])]])
    top["basis"] = np.where(top.name.isin(R["add"]), top.name.map(R["add"]), "Model order")
    if R["first"]:
        nm, why = R["first"]; top.loc[top.name == nm, "basis"] = why
        top = pd.concat([top[top.name == nm], top[top.name != nm]])
    for nm, (pos, why) in R.get("place", {}).items():
        row = top[top.name == nm].assign(basis=why); rest = top[top.name != nm]
        top = pd.concat([rest.iloc[:pos - 1], row, rest.iloc[pos - 1:]])
    top = top.reset_index(drop=True); top.insert(0, "rank", top.index + 1)
    cols = ["rank", "name", "model_rank", "full_model_rank", "pressure_only_rank", "basis", "player_id"]
    top[cols].to_csv(os.path.join(OUT, f"final_{lab}s.csv"), index=False)
    excl = [dict(name=k, reason=v, model_rank=int(C.model_rank[C.name == k].iloc[0])) for k, v in R["remove"].items() if (C.name == k).any()]
    excl += [dict(name=n, reason=f"No T20 since {d} (availability rule)", model_rank=int(m)) for n, d, m in C[~C.available & (C.model_rank <= 25)][["name", "last_match", "model_rank"]].values]
    pd.DataFrame(excl).to_csv(os.path.join(OUT, f"excluded_{lab}s.csv"), index=False)
    print(top[cols[:5]].to_string(index=False))
