"""Main joint fit on all data + all-candidate eligibility audit (printed BEFORE any resampling) + point ratings."""
import numpy as np, pandas as pd, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import jm, pipe, rate

t0 = time.time()
L, A = pipe.tables()
_w = os.path.join(WORK, "strict_local.pkl")   # warm start of the variance components only (speed); the strict rule decides convergence
_i = pd.read_pickle(_w) if os.path.exists(_w) else None
r = pipe.cached_fit("main", L, A, verbose=True, init=None if _i is None else dict(Sb=_i["Sb"], Sw=_i["Sw"], sm=_i["sm"], phiS=_i["phiS"], phiX=_i["phiX"], theta=None))
pd.to_pickle(r, os.path.join(WORK, "main_fit.pkl"))
print("main fit", r["converged"], r["em_iters"], r["seconds"], flush=True)
NAMES = {}
for idc, nc in (("striker_id", "striker_name"), ("bowler_id", "bowler_name"), ("non_striker_id", "non_striker_name")):
    if nc in A:
        g = A.groupby([idc, nc]).size().reset_index(name="n").sort_values("n", ascending=False).drop_duplicates(idc)
        for i, n in zip(g[idc], g[nc]): NAMES.setdefault(i, n)
pd.to_pickle(NAMES, os.path.join(WORK, "names.pkl"))
el = []
for side in ("bat", "bowl"):
    e = rate.evidence(L, A, side)
    ok, why = rate.gate(e, side, share=True); ok2, why2 = rate.gate(e, side, share=False)
    e["eligible_primary"] = ok; e["exclusion_reason_primary"] = why
    e["eligible_without_share_rule"] = ok2; e["exclusion_reason_without_share_rule"] = why2
    e["affected_by_share_rule"] = ok2 & ~ok
    e.insert(0, "side", side); e.insert(1, "name", [NAMES.get(i, str(i)) for i in e.index])
    el.append(e)
E = pd.concat(el); E.index.name = "player_id"
E.round(6).to_csv(os.path.join(OUT, "eligibility_audit.csv"))
pd.set_option("display.width", 250)
for side in ("bat", "bowl"):
    e = E[E.side == side]
    print(f"\n=== {side}: candidates {len(e)}, eligible primary {int(e.eligible_primary.sum())}, eligible without 1/3 share {int(e.eligible_without_share_rule.sum())}")
    near = e[(e.n_eff >= 0.6 * rate.GATES[side]["n_eff"])].sort_values("n_eff", ascending=False)
    print(near[["name", "n_eff", "n_eff_IPL", "n_eff_T20I", "ipl_evidence_share", "ipl_weight_share", "n_hi", "n_hi_IPL", "n_hi_T20I", "seasons", "opponents",
                "eligible_primary", "exclusion_reason_primary"]].round(2).to_string())
    print("excluded ONLY by the 1/3 IPL share rule:", e[e.affected_by_share_rule].name.tolist())
print("total", round(time.time() - t0))
