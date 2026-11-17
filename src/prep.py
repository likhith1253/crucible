"""Delivery tables for the joint model: merges cross-fitted state values, stakes and the two ex-ante pressure exposures."""
import numpy as np, pandas as pd, os
from common import *
import jm

def load_deliveries():
    D = pd.read_parquet(os.path.join(WORK, "deliveries.parquet"))
    return D[~D.is_pen].reset_index(drop=True)

def stakes():
    return pd.read_csv(os.path.join(WORK, "stakes.csv"))

def build(D, V, st, cfg=None):
    """D deliveries (index aligned with V), V cross-fitted values. cfg: stake_a (1.0), ko (None = reconstructed values),
    exposure ('onesided' | 'abs' = symmetric leverage), fkind ('sq')."""
    cfg = dict(dict(stake_a=1.0, ko=None, exposure="onesided"), **(cfg or {}))
    X = D.copy()
    for c in ("xruns", "xvar", "p_bwkt", "p_x", "VW", "cx", "vx", "Pb", "Pw", "LI", "wp0"): X[c] = V[c].values
    s = st.copy()
    if cfg["ko"] is not None: s.loc[s.kind.isin(["ipl_playoff", "t20i_knockout"]), "stake"] = cfg["ko"]
    s = s.set_index(["match_id", "team_id"]).stake
    X["S_bat"] = s.reindex(list(zip(X.match_id, X.batting_team_id))).fillna(0).values
    X["S_bowl"] = s.reindex(list(zip(X.match_id, X.bowling_team_id))).fillna(0).values
    Eb = (X.Pb if cfg["exposure"] == "onesided" else X.LI) * (1 + cfg["stake_a"] * X.S_bat)
    Ew = (X.Pw if cfg["exposure"] == "onesided" else X.LI) * (1 + cfg["stake_a"] * X.S_bowl)
    X["E_bat"] = Eb; X["E_bowl"] = Ew
    lg = X.legal.values
    X["u_bat"] = np.nan; X.loc[lg, "u_bat"] = Eb[lg].rank(pct=True, method="average").values   # one global scale, all innings
    X["u_bowl"] = Ew.rank(pct=True, method="average").values
    X["phase"] = np.where(X.over <= 6, "PP", np.where(X.over <= 15, "MID", "DEATH"))
    X["bat_role"] = np.where(X.s_order <= 2, "opener", np.where(X.s_order <= 5, "top_middle", "lower_finisher"))
    L = X[lg].copy(); A = X
    return L, A

def reference(L, side, fkind="sq", role=None, n=2000, seed=SEED, pressure=True):
    """Common reference distribution Q: population legal balls with the side's exposure in the top 20% (pressure) or below (usual)."""
    u = L.u_bat if side == "bat" else L.u_bowl
    sel = (u >= 0.8) if pressure else (u < 0.8)
    if role is not None: sel &= (L.bat_role == role) if side == "bat" else (L.phase == role)
    Q = L[sel]
    Q = Q.sample(min(n, len(Q)), random_state=seed)
    pl = Q.p_bwkt / np.maximum(1 - Q.p_x, 1e-6)
    return pd.DataFrame({"f_bat": jm.fshape(Q.u_bat.values, fkind), "f_bowl": jm.fshape(Q.u_bowl.values, fkind), "l0": jm.logit(pl.values),
                         "VW": Q.VW.values, "px": Q.p_x.values})
