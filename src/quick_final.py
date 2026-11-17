"""Final rankings from the converged (strict) main joint fit. Uncertainty = the fit's own posterior for every player
(300 draws per player); complete resampled refits are NOT included here (labelled in every file)."""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pipe, rate
M = pd.read_pickle(os.path.join(WORK, "main_fit.pkl"))
E = pd.read_csv(os.path.join(OUT, "eligibility_audit.csv"), index_col="player_id")
L, A = pipe.tables(); rng = np.random.default_rng(SEED); OD = os.path.join(OUT, "model_full"); os.makedirs(OD, exist_ok=True)
for side, lab in (("bat", "batter"), ("bowl", "bowler")):
    e = E[E.side == side]
    pt, dp, do = pipe.player_table(M, L, A, side, nd=300, rng=rng)
    raw = rate.raw_pressure(L, A, side)
    r = pd.DataFrame(index=pt.index)
    r["name"] = e.name.reindex(r.index)
    r["conservative_rating_q10"] = np.percentile(dp, 10, axis=1); r["posterior_mean"] = dp.mean(1)
    r["interval90_lo"] = np.percentile(dp, 5, axis=1); r["interval90_hi"] = np.percentile(dp, 95, axis=1)
    r["usual_ability"] = pt.usual_ability; r["pressure_uplift"] = pt.pressure_uplift; r["pressure_ability"] = pt.pressure_ability
    r["raw_pressure_performance"] = raw.reindex(r.index)
    for c in ("n_hi", "n_eff", "n_eff_IPL", "n_eff_T20I", "ipl_evidence_share", "seasons", "opponents", "top3_share",
              "eligible_primary", "exclusion_reason_primary", "eligible_without_share_rule"):
        r[c] = e[c].reindex(r.index)
    r = r[r.name.notna()]
    for tag, col in (("", "eligible_primary"), ("_without_share_rule", "eligible_without_share_rule")):
        el = r[col].fillna(False).astype(bool)
        X = dp[np.isin(pt.index, r.index[el])]; ids = pt.index[np.isin(pt.index, r.index[el])]
        top = ((-X).argsort(0).argsort(0) < 10).mean(1)
        r.loc[ids, f"top10_probability{tag}"] = top
        r.loc[el, f"rank{tag}"] = r.loc[el, "conservative_rating_q10"].rank(ascending=False, method="min")
    r = r.sort_values(["rank", "rank_without_share_rule"], na_position="last")
    cols = ["rank", "name", "conservative_rating_q10", "posterior_mean", "interval90_lo", "interval90_hi", "top10_probability", "usual_ability",
            "pressure_uplift", "pressure_ability", "raw_pressure_performance", "n_hi", "n_eff", "n_eff_IPL", "n_eff_T20I", "ipl_evidence_share", "seasons", "opponents", "top3_share"]
    p = r[r.eligible_primary.astype(bool)]
    p.head(10)[cols].round(4).to_csv(os.path.join(OD, f"final_{lab}_rankings.csv"), index_label="player_id")
    p.iloc[10:20][cols].round(4).to_csv(os.path.join(OD, f"reserves_{lab}s_11_20.csv"), index_label="player_id")
    r.round(4).to_csv(os.path.join(OD, f"full_{lab}_rankings.csv"), index_label="player_id")
    g = r[r.eligible_without_share_rule.fillna(False).astype(bool)].sort_values("rank_without_share_rule")
    g.head(20)[["rank_without_share_rule", "name", "conservative_rating_q10", "top10_probability_without_share_rule", "ipl_evidence_share", "eligible_primary"]].round(4) \
        .to_csv(os.path.join(OD, f"ungated_{lab}_top20.csv"), index_label="player_id")
    print(f"\n=== TOP 20 {lab.upper()}S (gated) ===")
    print(p.head(20)[["name", "conservative_rating_q10", "posterior_mean", "top10_probability", "usual_ability", "pressure_uplift", "n_eff", "ipl_evidence_share"]].round(3).to_string())
    print(f"--- ungated top 10 ({lab}s):", g.head(10).name.tolist())
