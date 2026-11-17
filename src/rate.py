"""Eligibility evidence, player ratings over common reference distributions, and ranking helpers."""
import numpy as np, pandas as pd
from common import *
import jm, prep

GATES = {"bat": dict(n_eff=15, n_hi=100, seasons=2, opp=5, ipl_floor=5.0),
         "bowl": dict(n_eff=25, n_hi=100, seasons=2, opp=5, ipl_floor=25 / 3)}
IPL_SHARE = 1 / 3          # POST-HOC rule (introduced after the earlier rankings were seen); applied to every candidate

def _neff(T):
    g = T.groupby(["pid", "match_id"]).w.sum(); s = g.groupby("pid").sum(); q = (g ** 2).groupby("pid").sum()
    return (s ** 2 / q.replace(0, np.nan)).fillna(0)

def evidence(L, A, side):
    """Per-player evidence. Pressure weight w = u^2 of the player's own exposure (the existing match-clustered calculation);
    effective pressure matches n_eff = (sum_m W_m)^2 / sum_m W_m^2 with W_m the player's pressure weight in match m."""
    if side == "bat": T = pd.DataFrame({"pid": L.striker_id.values, "u": L.u_bat.values, "opp_team": L.bowling_team_id.values})
    else: T = pd.DataFrame({"pid": A.bowler_id.values, "u": A.u_bowl.values, "opp_team": A.batting_team_id.values})
    S = L if side == "bat" else A
    T["match_id"] = S.match_id.values; T["season"] = S.season_year.values; T["family"] = S.family.values
    T["w"] = T.u ** 2; T["hi"] = (T.u >= 0.8).astype(int)
    e = pd.DataFrame(index=np.sort(T.pid.unique()))
    e["balls"] = T.groupby("pid").size()
    e["n_hi"] = T.groupby("pid").hi.sum()
    pm = T.groupby(["pid", "match_id"]).agg(hi=("hi", "sum"), season=("season", "first"), family=("family", "first"), opp=("opp_team", "first")).reset_index()
    pmh = pm[pm.hi >= 3]
    e["pressure_matches"] = pmh.groupby("pid").size().reindex(e.index).fillna(0).astype(int)
    e["seasons"] = pmh.groupby("pid").season.nunique().reindex(e.index).fillna(0).astype(int)
    e["opponents"] = T.groupby("pid").opp_team.nunique()
    e["n_eff"] = _neff(T).reindex(e.index).fillna(0)
    for fam in ("IPL", "T20I"):
        Tf = T[T.family == fam]
        e[f"n_eff_{fam}"] = _neff(Tf).reindex(e.index).fillna(0)
        e[f"n_hi_{fam}"] = Tf.groupby("pid").hi.sum().reindex(e.index).fillna(0).astype(int)
        e[f"weight_{fam}"] = Tf.groupby("pid").w.sum().reindex(e.index).fillna(0)
        e[f"opponents_{fam}"] = Tf.groupby("pid").opp_team.nunique().reindex(e.index).fillna(0).astype(int)
        e[f"seasons_{fam}"] = pmh[pmh.family == fam].groupby("pid").season.nunique().reindex(e.index).fillna(0).astype(int)
    e["ipl_evidence_share"] = (e.n_eff_IPL / (e.n_eff_IPL + e.n_eff_T20I)).fillna(0)
    e["ipl_weight_share"] = (e.weight_IPL / (e.weight_IPL + e.weight_T20I)).fillna(0)
    top = T[T.hi == 1].groupby(["pid", "match_id"]).w.sum().reset_index()
    e["top3_share"] = top.groupby("pid").w.apply(lambda z: z.nlargest(3).sum() / z.sum() if z.sum() > 0 else np.nan).reindex(e.index)
    return e

def gate(e, side, share=True):
    g = GATES[side]
    fails = pd.DataFrame({"n_eff": e.n_eff < g["n_eff"], "pressure_deliveries": e.n_hi < g["n_hi"], "seasons": e.seasons < g["seasons"],
                          "opponents": e.opponents < g["opp"], "ipl_floor": e.n_eff_IPL < g["ipl_floor"]}, index=e.index)
    if share: fails["ipl_share"] = e.ipl_evidence_share < IPL_SHARE
    reason = fails.apply(lambda r: ";".join(c for c in fails.columns if r[c]), axis=1)
    return ~fails.any(axis=1), reason

def ratings(F, L, side, fkind="sq", nd=0, rng=None):
    """Usual (non-pressure reference), pressure ability (common pressure reference) and uplift for every player in the fit."""
    D = F["design"]; ids = D.bats if side == "bat" else D.bows
    Qp = prep.reference(L, side, fkind, pressure=True); Qo = prep.reference(L, side, fkind, pressure=False)
    out = pd.DataFrame(index=ids)
    out["pressure_ability"] = jm.run_value(F, Qp, side); out["usual_ability"] = jm.run_value(F, Qo, side)
    out["pressure_uplift"] = out.pressure_ability - out.usual_ability
    if nd:
        dr = jm.draw_effects(F, side, nd, rng)
        return out, jm.run_value(F, Qp, side, draws=dr), jm.run_value(F, Qo, side, draws=dr)
    return out

def raw_pressure(L, A, side):
    """Conventional, unadjusted pressure performance: mean delivery run-value on the player's own pressure deliveries."""
    if side == "bat":
        y = (L.bat_runs - L.xruns) - L.VW * (L.bowler_wkt.astype(float) - L.p_bwkt / (1 - L.p_x))
        T = pd.DataFrame({"pid": L.striker_id.values, "y": y.values, "hi": (L.u_bat >= 0.8).values})
    else:
        pl = A.p_bwkt / (1 - A.p_x)
        y = np.where(A.legal, (A.xruns - A.bat_runs) + A.VW * (A.bowler_wkt.astype(float) - pl), 0) - (np.where(A.is_wide | A.is_nb, A.bowler_runs, 0) - A.p_x * A.cx)
        T = pd.DataFrame({"pid": A.bowler_id.values, "y": y, "hi": (A.u_bowl >= 0.8).values})
    return T[T.hi].groupby("pid").y.mean()
