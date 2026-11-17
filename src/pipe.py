"""Shared helpers: build data under a configuration, fit the joint model, produce a light fit record and player ratings."""
import numpy as np, pandas as pd, os, time
from common import *
import prep, jm, rate

_cache = {}
def base():
    if "D" not in _cache:
        _cache["D"] = prep.load_deliveries(); _cache["V"] = pd.read_parquet(os.path.join(WORK, "ball_values_raw.parquet"))
        _cache["st"] = prep.stakes()
        assert (_cache["V"].match_id.values == _cache["D"].match_id.values).all()
        assert (_cache["V"].ball_no.values == _cache["D"].innings_ball_num.values).all()
    return _cache["D"], _cache["V"], _cache["st"]

def tables(cfg=None, D=None, V=None):
    D0, V0, st = base()
    return prep.build(D0 if D is None else D, V0 if V is None else V, st, cfg)

def light(F):
    D = F["design"]
    return dict(Sb=F["Sb"], Sw=F["Sw"], sm=F["sm"], phiS=F["phiS"], phiX=F["phiX"], theta=F["theta"], ub=F["ub"], uw=F["uw"],
                Cbb=F["Cbb"], Cbw=F["Cbw"], bats=D.bats, bows=D.bows, mats=D.mats, groups=D.groups, p=D.p, ll=F["ll"],
                converged=F["converged"], em_iters=F["em_iters"], criterion=F.get("criterion"), trace=F["trace"], grad_inf=F["grad_inf"], fkind=D.fkind)

class _D:   # minimal design stand-in so rating functions work on a light record
    def __init__(self, r): self.bats, self.bows, self.fkind = r["bats"], r["bows"], r["fkind"]

def as_fit(r):
    return dict(r, design=_D(r))

def fit_cfg(L, A, init=None, dkw=None, tol=1e-4, verbose=False):
    """Every fit (main, refits, audits, out-of-time variants) uses the SAME convergence rule as the main fit: parameter change
    < 1e-4 of scale, or a flat likelihood (< 0.1 gain over 20 EM evaluations); no step cap below 2000. The tol argument is
    accepted for backward compatibility and ignored."""
    t = time.time(); Ds = jm.Design(L, A, **(dkw or {}))
    F = jm.fit(Ds, init=init, tol=1e-4, verbose=verbose, flat_window=10**9, max_em=4000)
    r = light(F); r["seconds"] = round(time.time() - t, 1)
    del F
    return r

def player_table(r, L, A, side, nd=0, rng=None):
    F = as_fit(r)
    if nd:
        pt, dp, do = rate.ratings(F, L, side, r["fkind"], nd=nd, rng=rng); return pt, dp, do
    return rate.ratings(F, L, side, r["fkind"])

def rank_eligible(score, elig):
    s = score.reindex(elig.index)[elig.values]
    return s.rank(ascending=False, method="min")

_FP = None
def fp():
    global _FP
    if _FP is None: _FP = fingerprint()
    return _FP

def cached_fit(key, L, A, init=None, dkw=None, verbose=False):
    """fit_cfg with a resumable checkpoint: work/fitcache/<key>.pkl is reused only if its code+data fingerprint matches;
    a stale file is moved to work/superseded/ (never silently overwritten)."""
    d = os.path.join(WORK, "fitcache"); os.makedirs(d, exist_ok=True); f = os.path.join(d, f"{key}.pkl")
    if os.path.exists(f):
        r = pd.read_pickle(f)
        if r.get("fingerprint") == fp(): return r
        sd = os.path.join(WORK, "superseded", "fitcache"); os.makedirs(sd, exist_ok=True)
        os.replace(f, os.path.join(sd, f"{key}.{r.get('fingerprint')}.pkl"))
    r = fit_cfg(L, A, init=init, dkw=dkw, verbose=verbose); r["fingerprint"] = fp(); r["key"] = key
    atomic_pickle(r, f)
    return r
