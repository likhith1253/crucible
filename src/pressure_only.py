"""Pressure-only model. For each side the same joint batter-bowler model is fitted ONLY on that side's pressure balls
(exposure percentile u >= 0.8), so a player's rating comes from pressure evidence alone, adjusted for the opponents faced in
those moments. Ranked by the cautious 10th percentile of the posterior. Consistency = share of a player's pressure matches
with a positive opponent-adjusted contribution. Fits are cached in work/pressure_only_<side>.pkl."""
import numpy as np, pandas as pd, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import jm, pipe, prep

t0 = time.time()
M = pd.read_pickle(os.path.join(WORK, "main_fit.pkl"))
E = pd.read_csv(os.path.join(OUT, "eligibility_audit.csv"), index_col="player_id")
L, A = pipe.tables(); rng = np.random.default_rng(SEED + 11); OD = os.path.join(OUT, "model_pressure_only"); os.makedirs(OD, exist_ok=True)
usual = {s: pipe.player_table(M, L, A, s) for s in ("bat", "bowl")}
for side, lab, ucol in (("bat", "batter", "u_bat"), ("bowl", "bowler", "u_bowl")):
    Lp = L[L[ucol] >= 0.8]; Ap = A[A[ucol] >= 0.8]
    f = os.path.join(WORK, f"pressure_only_{side}.pkl")
    if os.path.exists(f):
        r = pd.read_pickle(f)
    else:   # warm start of the variance components from the full model; same strict convergence rule as every other fit
        r = pipe.fit_cfg(Lp, Ap, dkw=dict(slopes=False, pop_slope=False),
                         init=dict(Sb=M["Sb"], Sw=M["Sw"], sm=M["sm"], phiS=M["phiS"], phiX=M["phiX"], theta=None))
        r["fkind"] = "sq"; pd.to_pickle(r, f)
    print(side, "converged", r["converged"], r["criterion"], r["em_iters"], flush=True)
    Fr = pipe.as_fit(r); Q = prep.reference(L, side, "sq", pressure=True)
    ids = r["bats"] if side == "bat" else r["bows"]
    vals = jm.run_value(Fr, Q, side, draws=jm.draw_effects(Fr, side, 400, rng)); point = jm.run_value(Fr, Q, side)
    e = E[E.side == side]
    out = pd.DataFrame(index=ids)
    out["name"] = e.name.reindex(ids)
    out["pressure_rating"] = point
    out["conservative_q10"] = np.percentile(vals, 10, axis=1)
    out["interval90_lo"] = np.percentile(vals, 5, axis=1); out["interval90_hi"] = np.percentile(vals, 95, axis=1)
    out["usual_rating"] = usual[side].usual_ability.reindex(ids)
    if side == "bat":
        y = (Lp.bat_runs - Lp.xruns) - Lp.VW * (Lp.bowler_wkt.astype(float) - Lp.p_bwkt / (1 - Lp.p_x))
        T = pd.DataFrame({"pid": Lp.striker_id.values, "match_id": Lp.match_id.values, "y": y.values})
    else:
        pl = Ap.p_bwkt / (1 - Ap.p_x)
        y = np.where(Ap.legal, (Ap.xruns - Ap.bat_runs) + Ap.VW * (Ap.bowler_wkt.astype(float) - pl), 0) - (np.where(Ap.is_wide | Ap.is_nb, Ap.bowler_runs, 0) - Ap.p_x * Ap.cx)
        T = pd.DataFrame({"pid": Ap.bowler_id.values, "match_id": Ap.match_id.values, "y": y})
    pm = T.groupby(["pid", "match_id"]).y.sum().reset_index()
    out["pressure_matches"] = pm.groupby("pid").size().reindex(ids)
    out["consistency_share_positive_matches"] = pm.assign(pos=pm.y > 0).groupby("pid").pos.mean().reindex(ids)
    for c in ("n_eff", "ipl_evidence_share", "seasons", "opponents", "eligible_primary"):
        out[c] = e[c].reindex(ids)
    out = out[out.name.notna()]
    el = out.eligible_primary.eq(True)
    X = vals[np.isin(ids, out.index[el])]; sel = ids[np.isin(ids, out.index[el])]
    out.loc[sel, "top10_probability"] = ((-X).argsort(0).argsort(0) < 10).mean(1)
    out.loc[el, "rank"] = out.loc[el, "conservative_q10"].rank(ascending=False, method="min")
    out.sort_values("rank", na_position="last").round(4).to_csv(os.path.join(OD, f"full_{lab}_rankings.csv"), index_label="player_id")
print("total", round(time.time() - t0))
