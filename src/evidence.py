"""Evidence tables behind the final lists, a simple out-of-time check, and the data file for the dashboard.

Raw pressure numbers here are descriptive (not opponent-adjusted). 'vs expected' compares what happened on a player's pressure
balls with what the player-blind outcome model expected from an average player in exactly those match states.
  batters: runs above expected per 100 pressure balls
  bowlers: runs saved per over against expectation (extras included)
High stakes = knockouts, playoffs, or league matches whose result moves the team's top-4 chance by >= 0.3 (virtual knockouts).
"""
import numpy as np, pandas as pd, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pipe

L, A = pipe.tables()
DASH = os.path.join(REPO, "dashboard"); os.makedirs(DASH, exist_ok=True)
r1 = lambda x: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 2)

def bat_stats(P, B):
    x = P.bat_runs - P.xruns
    per100 = lambda m: r1(100 * x[m].mean()) if m.sum() else None
    seas = x.groupby([P.season_year, P.comp]).sum()
    D = B[B.phase == "DEATH"]
    return dict(pressure_balls=len(P), pressure_sr=r1(100 * P.bat_runs.mean()), balls_per_dismissal=r1(len(P) / max(P.striker_out.sum(), 1)),
                vs_expected=per100(P.index == P.index), ipl_balls=int((P.comp == 1).sum()), ipl_vs_expected=per100(P.comp == 1),
                t20i_balls=int((P.comp == 0).sum()), t20i_vs_expected=per100(P.comp == 0),
                high_stakes_balls=int((P.S_bat >= 0.3).sum()), high_stakes_vs_expected=per100(P.S_bat >= 0.3),
                chase_balls=int((P.innings == 2).sum()), chase_vs_expected=per100(P.innings == 2),
                recent_balls=int((P.season_year >= 2025).sum()), recent_vs_expected=per100(P.season_year >= 2025),
                season_blocks_positive=int((seas > 0).sum()), season_blocks=int(len(seas)),
                all_balls=len(B), all_sr=r1(100 * B.bat_runs.mean()), death_sr=r1(100 * D.bat_runs.mean()) if len(D) else None,
                seasons={f"{y}-{'IPL' if c == 1 else 'T20I'}": r1(100 * v / max(n, 1)) for (y, c), v, n in
                         zip(seas.index, seas.values, P.groupby([P.season_year, P.comp]).size().values)})

def bowl_stats(P, B):
    x = -(P.bowler_runs - (P.xruns.where(P.legal, 0) + P.p_x * P.cx)); lg = P.legal
    pov = lambda m: r1(6 * x[m].sum() / max(lg[m].sum(), 1)) if lg[m].sum() else None
    seas = x.groupby([P.season_year, P.comp]).sum(); nb = lg.groupby([P.season_year, P.comp]).sum()
    Pl = P[lg]; D = B[B.phase == "DEATH"]; Dl = D[D.legal]
    return dict(pressure_balls=int(lg.sum()), pressure_economy=r1(6 * P.bowler_runs.sum() / max(lg.sum(), 1)),
                runs_saved_per_over=pov(P.index == P.index), ipl_balls=int(lg[P.comp == 1].sum()), ipl_saved=pov(P.comp == 1),
                t20i_balls=int(lg[P.comp == 0].sum()), t20i_saved=pov(P.comp == 0),
                high_stakes_balls=int(lg[P.S_bowl >= 0.3].sum()), high_stakes_saved=pov(P.S_bowl >= 0.3),
                defending_balls=int(lg[P.innings == 2].sum()), defending_saved=pov(P.innings == 2),
                recent_balls=int(lg[P.season_year >= 2025].sum()), recent_saved=pov(P.season_year >= 2025),
                dot_pct=r1(100 * (Pl.bowler_runs == 0).mean()), boundary_pct=r1(100 * Pl.bat_runs.isin([4, 6]).mean()),
                balls_per_wicket=r1(len(Pl) / max(P.bowler_wkt.sum(), 1)), extras_pct=r1(100 * (P.is_wide | P.is_nb).mean()),
                death_balls=len(Dl), death_economy=r1(6 * D.bowler_runs.sum() / max(len(Dl), 1)) if len(Dl) else None,
                season_blocks_positive=int((seas > 0).sum()), season_blocks=int(len(seas)),
                seasons={f"{y}-{'IPL' if c == 1 else 'T20I'}": r1(6 * v / max(n, 1)) for (y, c), v, n in zip(seas.index, seas.values, nb.values)})

dash = {}
for side, lab, idc, ucol, fn, T in (("bat", "batter", "striker_id", "u_bat", bat_stats, L), ("bowl", "bowler", "bowler_id", "u_bowl", bowl_stats, A)):
    M = pd.read_csv(os.path.join(OUT, f"model_ranking_{lab}s.csv"))
    F = pd.read_csv(os.path.join(OUT, f"final_{lab}s.csv"))
    X = pd.read_csv(os.path.join(OUT, f"excluded_{lab}s.csv"))
    pool = M[M.model_rank <= 30]
    rows = []
    for _, r in pool.iterrows():
        B = T[T[idc] == r.player_id]; P = B[B[ucol] >= 0.8]
        s = fn(P, B); rows.append(dict(name=r["name"], model_rank=int(r.model_rank), full_model_rank=int(r.full_model_rank),
                                       pressure_only_rank=int(r.pressure_only_rank), last_match=str(r.last_match), **s))
    ev = pd.DataFrame(rows)
    ev.drop(columns="seasons").to_csv(os.path.join(OUT, f"evidence_{lab}s.csv"), index=False)
    fin = F.merge(ev.drop(columns=["model_rank", "full_model_rank", "pressure_only_rank"]), on="name", how="left")
    dash[side] = dict(final=json.loads(fin.drop(columns="player_id").to_json(orient="records")),
                      pool=json.loads(ev.to_json(orient="records")), excluded=json.loads(X.to_json(orient="records")))

# out-of-time check: does pressure performance in 2022-24 carry into 2025-26? (players with >= 60 pressure balls in both windows)
oot = {}
for side, idc, ucol in (("bat", "striker_id", "u_bat"), ("bowl", "bowler_id", "u_bowl")):
    T = L if side == "bat" else A; P = T[T[ucol] >= 0.8].copy()
    if side == "bat": P["v"] = P.bat_runs - P.xruns; P["n"] = 1
    else: P["v"] = -(P.bowler_runs - (P.xruns.where(P.legal, 0) + P.p_x * P.cx)); P["n"] = P.legal.astype(int)
    P["w"] = np.where(P.season_year >= 2025, "late", "early")
    g = P.groupby([idc, "w"]).agg(v=("v", "sum"), n=("n", "sum")).unstack()
    ok = (g["n"]["early"] >= 60) & (g["n"]["late"] >= 60); e = g["v"]["early"][ok] / g["n"]["early"][ok]; l = g["v"]["late"][ok] / g["n"]["late"][ok]
    oot[side] = dict(players=int(ok.sum()), rank_correlation=r1(e.rank().corr(l.rank())),
                     top_quartile_stays_above_median=r1(((l > l.median())[e >= e.quantile(0.75)]).mean()))
dump_json(oot, os.path.join(OUT, "out_of_time_check.json"))

# data facts and pressure definition illustrations for the dashboard
leg = A.legal.sum()
facts = dict(matches=int(A.match_id.nunique()), deliveries=int(len(A)), legal_balls=int(leg), extras_balls=int(len(A) - leg),
             ipl_matches=int(A[A.comp == 1].match_id.nunique()), t20i_matches=int(A[A.comp == 0].match_id.nunique()),
             first=str(pd.to_datetime(A.date).min().date()), last=str(pd.to_datetime(A.date).max().date()),
             batters=int(L.striker_id.nunique()), bowlers=int(A.bowler_id.nunique()),
             matchups=int(L.groupby(["striker_id", "bowler_id"]).ngroups),
             pressure_share_high_stakes=r1((A[A.u_bowl >= 0.8].S_bowl >= 0.3).mean()),
             eligible_batters=int(pd.read_csv(os.path.join(OUT, "model_ranking_batters.csv")).shape[0]),
             eligible_bowlers=int(pd.read_csv(os.path.join(OUT, "model_ranking_bowlers.csv")).shape[0]))
P2 = L[L.innings == 2]
heat = P2.assign(ob=(P2.over // 4).astype(int).clip(upper=4), wk=(P2.wkts.clip(upper=8) // 2).astype(int)).groupby(["ob", "wk"]).u_bat.apply(lambda s: r1((s >= 0.8).mean()))
facts["chase_pressure_grid"] = [dict(overs=f"{4*o+1}-{4*o+4 if o < 4 else 20}", wickets=f"{2*w}-{2*w+1}" if w < 4 else "8+", share=v) for (o, w), v in heat.items()]
ph = A[A.u_bowl >= 0.8].groupby("phase").size(); facts["pressure_by_phase_bowl"] = {k: int(v) for k, v in ph.items()}
ph = L[L.u_bat >= 0.8].groupby("phase").size(); facts["pressure_by_phase_bat"] = {k: int(v) for k, v in ph.items()}
# head-to-head network between the two final lists (all legal balls): balls, runs above expected for the batter, dismissals
fb = pd.read_csv(os.path.join(OUT, "final_batters.csv")); fw = pd.read_csv(os.path.join(OUT, "final_bowlers.csv"))
H = L[L.striker_id.isin(fb.player_id) & L.bowler_id.isin(fw.player_id)]
g = H.groupby(["striker_id", "bowler_id"]).agg(balls=("bat_runs", "size"), runs=("bat_runs", "sum"), xr=("xruns", "sum"), outs=("bowler_wkt", "sum")).reset_index()
nb = dict(zip(fb.player_id, fb.name)); nw = dict(zip(fw.player_id, fw.name))
network = [dict(bat=nb[r.striker_id], bowl=nw[r.bowler_id], balls=int(r.balls), runs=int(r.runs), vs_expected=r1(r.runs - r.xr), outs=int(r.outs))
           for r in g.itertuples() if r.balls >= 6]
with open(os.path.join(DASH, "data.js"), "w", encoding="utf-8") as f:
    f.write("window.CRUCIBLE = " + json.dumps(dict(facts=facts, oot=oot, network=network, **dash), default=str) + ";\n")
print(json.dumps(facts, default=str)[:600]); print(oot)
