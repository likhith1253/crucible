"""Robustness check: rank every available eligible player under four alternative definitions of 'reliable under pressure'
and compare them with the final lists.

  reliability_index : equal-weight average of five percentiles - model rank (impact), dependability (batters: pressure balls
                      per dismissal; bowlers: pressure economy), high-stakes value shrunk to zero for small samples,
                      season-to-season consistency, and pressure-ball volume
  wpa_above_expected: total win probability added above the player-blind expectation on pressure balls
  match_winning     : stake-weighted count of matches the player's side won in which he moved its win probability by >= 0.20
  process_outcome   : average of the model rank and the match-winning rank
Outputs: results/method_comparison_batters.csv, results/method_comparison_bowlers.csv
"""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pipe

L, A = pipe.tables()
V = pd.read_parquet(os.path.join(WORK, "ball_values_raw.parquet"))
A = A.copy(); A["y_bat"] = V.y_bat.values; A["y_bowl"] = V.y_bowl.values
X = A.sort_values(["match_id", "innings", "innings_ball_num"]).copy()
X["wpA"] = np.where(X.innings == 1, X.wp0, 1 - X.wp0)
X["postA"] = X.groupby("match_id").wpA.shift(-1)
X["postA"] = X.postA.fillna(pd.Series(np.where(X.innings == 1, X.bat_win, 1 - X.bat_win), index=X.index))
X["d_bat"] = (X.postA - X.wpA) * np.where(X.innings == 1, 1, -1)
pc = lambda s: s.rank(pct=True)
shrink = lambda v, n, k=100: v.fillna(0) * n / (n + k)

for side, lab, idc, team_won, stake, u, y in (("bat", "batter", "striker_id", "bat_win", "S_bat", "u_bat", "y_bat"),
                                              ("bowl", "bowler", "bowler_id", None, "S_bowl", "u_bowl", "y_bowl")):
    M = pd.read_csv(os.path.join(OUT, f"model_ranking_{lab}s.csv")); M = M[M.available].set_index("player_id")
    ev = pd.read_csv(os.path.join(OUT, f"evidence_{lab}s.csv")).set_index("name")
    F = pd.read_csv(os.path.join(OUT, f"final_{lab}s.csv"))
    out = pd.DataFrame(index=M.index); out["name"] = M.name; out["model"] = M.model_rank.rank()
    e = ev.reindex(out.name.values); e.index = out.index
    dep = e.balls_per_dismissal if side == "bat" else -e.pressure_economy
    hs = e.high_stakes_vs_expected if side == "bat" else e.high_stakes_saved
    pri = (pc(-out.model) + pc(dep) + pc(shrink(hs, e.high_stakes_balls)) + pc(e.season_blocks_positive / e.season_blocks) + pc(e.pressure_balls)) / 5
    out["reliability_index"] = pri.rank(ascending=False)
    P = A[(A[u] >= 0.8) & (A.legal if side == "bat" else True)]
    out["wpa_above_expected"] = P.groupby(idc)[y].sum().reindex(out.index).rank(ascending=False)
    T = X[X.legal] if side == "bat" else X
    v = T.d_bat if side == "bat" else -T.d_bat
    won = T.bat_win if side == "bat" else 1 - T.bat_win
    pm = pd.DataFrame({"wpa": v.groupby([T[idc], T.match_id]).sum(), "won": won.groupby([T[idc], T.match_id]).first(),
                       "stake": T[stake].groupby([T[idc], T.match_id]).first()})
    w = (((pm.wpa >= 0.20) & (pm.won == 1)) * (1 + pm.stake)).groupby(level=0).sum()
    out["match_winning"] = w.reindex(out.index).fillna(0).rank(ascending=False, method="min")
    out["process_outcome"] = ((out.model + out.match_winning) / 2).rank(method="min")
    out["final_rank"] = out.name.map(dict(zip(F.name, F["rank"])))
    meth = ["model", "reliability_index", "wpa_above_expected", "match_winning", "process_outcome"]
    out["methods_top10"] = (out[meth] <= 10).sum(1); out["median_rank"] = out[meth].median(1)
    out = out.sort_values(["final_rank", "median_rank"], na_position="last")
    out.round(1).to_csv(os.path.join(OUT, f"method_comparison_{lab}s.csv"), index_label="player_id")
    print(out[out.final_rank.notna()][["name"] + meth + ["methods_top10", "median_rank"]].to_string(index=False))
    print("core (top 10 in >= 4 of 5 methods):", out[out.methods_top10 >= 4].name.tolist())
