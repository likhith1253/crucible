"""Final selection: model ranking, availability, expert review.

1. Model ranking. Average of a player's rank in the two converged fits (ties broken by the full-model rank):
     full model           all balls, pressure effect as a smooth function of exposure   -> results/model_full
     pressure-only model  fitted on pressure balls only                                 -> results/model_pressure_only
2. Availability. A player must have played a T20 (IPL or T20I) in the 12 months before the last date in the data.
3. Expert review. The model order is reviewed against the evidence tables (results/evidence_*.csv): record in the IPL and in
   T20Is separately, high-stakes games, survival under pressure, workload, recent form and season-to-season consistency.
   Each adjustment is recorded in REVIEW with its evidence and written to the 'basis' column of the final lists.
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
            "Nicholas Pooran": "Well below expected under pressure in T20Is (-17.6 runs per 100 balls over 197 balls).",
            "Cameron Green": "Below expected under pressure in T20Is (-3.5 per 100); limited high-stakes value (-1.4 per 100).",
            "Dewald Brevis": "106 pressure balls: too little evidence for a reliability ranking.",
        },
        "add": {
            "Virat Kohli": "Hardest batter to dismiss under pressure (one dismissal every 47 pressure balls, best of all eligible batters) over 424 pressure balls; pressure-only model rank 8; top ten in four of five ranking methods.",
            "Namboori Tilak Varma": "Heaviest pressure workload of any eligible batter (495 pressure balls, 266 in high-stakes games, where he is above expected); full-model rank 12; top ten under the reliability index and match-winning methods.",
        },
        "first": None,
        "place": {
            "Abhishek Sharma": (3, "+49.8 runs per 100 in high-stakes games against Salt's +11.2, on a larger sample (308 pressure balls to 233); second only to Suryakumar Yadav on match-winning performances (Salt eighth)."),
            "Yashasvi Jaiswal": (8, "T20I figure rests on 37 pressure balls; IPL record (+10.8 per 100 over 258 balls) is the weakest of the top eight; flat in 2026."),
        },
    },
    "bowl": {
        "remove": {
            "Wanindu Hasaranga": "Barely above average under pressure in T20Is (+0.06 runs saved per over); no IPL in the latest season.",
            "Krunal Pandya": "No T20I pressure record; death-over economy 10.48.",
        },
        "add": {
            "Maheesh Theekshana": "Saved runs under pressure in all 9 of his season blocks: +1.56 per over in the IPL, +1.37 in T20Is, +1.54 in high-stakes games over 275 balls; third on the reliability index.",
        },
        "first": ("Jasprit Bumrah", "Top three are a statistical tie in the model; best rate of all (1.99 runs saved per over), best in high-stakes games (2.43), saved runs in every season block, best death-over economy of any eligible pace bowler (7.40 over 514 balls)."),
        "place": {},
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
    for nm, (pos, why) in R["place"].items():
        row = top[top.name == nm].assign(basis=why); rest = top[top.name != nm]
        top = pd.concat([rest.iloc[:pos - 1], row, rest.iloc[pos - 1:]])
    top = top.reset_index(drop=True); top.insert(0, "rank", top.index + 1)
    cols = ["rank", "name", "model_rank", "full_model_rank", "pressure_only_rank", "basis", "player_id"]
    top[cols].to_csv(os.path.join(OUT, f"final_{lab}s.csv"), index=False)
    excl = [dict(name=k, reason=v, model_rank=int(C.model_rank[C.name == k].iloc[0])) for k, v in R["remove"].items() if (C.name == k).any()]
    excl += [dict(name=n, reason=f"No T20 since {d} (availability rule)", model_rank=int(m)) for n, d, m in C[~C.available & (C.model_rank <= 25)][["name", "last_match", "model_rank"]].values]
    pd.DataFrame(excl).to_csv(os.path.join(OUT, f"excluded_{lab}s.csv"), index=False)
    print(top[cols[:5]].to_string(index=False))
