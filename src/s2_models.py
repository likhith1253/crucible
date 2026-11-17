"""Step 2: player-blind state models and ball-level values.

Models (all player-blind: no batter/bowler identity enters any of them; features are pre-delivery):
  WP1 / WP2 : win probability of the batting side, innings 1 / innings 2 (gradient boosting, monotone in resources)
  OUT       : multinomial next-delivery outcome distribution {0,1,2,3,4,6,W(bowler wicket),R(run-out etc.),X(wide/no-ball)}
  RE        : expected remaining runs in an innings from the state (trained on innings 1 only: uncensored)
Every delivery's predictions come from models trained WITHOUT that delivery's match (5-fold cross-fitting by match).
Per delivery we store: pre-ball win probability, outcome probabilities, leverage (expected |change in WP|), one-sided downside
leverages for each side, the realised and expected next-state win probability (-> win-probability-added-above-expected, WPAE) and the
same in a run currency (change in projected innings total). Pressure exposure uses ONLY pre-delivery quantities.
Outputs: work/ball_values.parquet, out/validation_*.csv
"""
import numpy as np, pandas as pd, os, sys, time
from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from common import *

D = pd.read_parquet(os.path.join(WORK, "deliveries.parquet"))
D = D[~D.is_pen].reset_index(drop=True)
D["pace"] = D.pace_or_spin.eq("Pace").astype(np.float32)
n = len(D)
rng = np.random.default_rng(SEED)
mids = np.array(sorted(D.match_id.unique())); rng.shuffle(mids)
fold_of = {mm: i % 5 for i, mm in enumerate(mids)}
D["fold"] = D.match_id.map(fold_of).astype(int)
INN1_TOT = pd.read_csv(os.path.join(WORK, "matches_clean.csv")).set_index("match_id").inn1_total

# ---------- outcome class label (true event) ----------
cls = np.full(n, 0)
bat = D.bat_runs.values.clip(0, 6)
bat = np.where(bat == 5, 4, bat)
cmap = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5}
cls = np.array([cmap[int(x)] for x in bat])
cls[D.other_wkt.values] = 7
cls[D.bowler_wkt.values] = 6
cls[(D.is_wide | D.is_nb).values] = 8
D["cls"] = cls
RUNS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6}  # class idx -> runs for classes 0..5

SCOLS = ["inn", "score", "wkts", "legal_used", "target", "s_order", "s_balls", "s_runs", "ns_order", "ns_balls", "ns_runs",
         "max_order", "pos_in_over", "comp", "venue_par", "elo_diff"]

def state(df):
    S = {c: df[c].values.astype(np.float64).copy() for c in SCOLS if c != "inn"}
    S["inn"] = df.innings.values.astype(np.int8)
    return S

def advance(S, runs_add, bat_add, wk, consume):
    """Next pre-delivery state after an event. Arrays are broadcast to len(S)."""
    T = {k: v.copy() for k, v in S.items()}
    T["score"] = S["score"] + runs_add
    T["legal_used"] = S["legal_used"] + consume
    T["s_balls"] = S["s_balls"] + consume
    T["s_runs"] = S["s_runs"] + bat_add
    T["pos_in_over"] = (S["pos_in_over"] + consume) % 6
    wk = np.asarray(wk, dtype=bool)
    # wicket: striker replaced by a fresh batter
    T["wkts"] = S["wkts"] + wk
    T["max_order"] = S["max_order"] + wk
    T["s_order"] = np.where(wk, S["max_order"] + 1, S["s_order"])
    T["s_balls"] = np.where(wk, 0, T["s_balls"]); T["s_runs"] = np.where(wk, 0, T["s_runs"])
    # strike rotation: odd bat runs, then end of over
    sw = (np.asarray(bat_add) % 2 == 1) & ~wk
    sw2 = (consume == 1) & (S["pos_in_over"] == 5)
    a = ["s_order", "s_balls", "s_runs"]; bb = ["ns_order", "ns_balls", "ns_runs"]
    for do in (sw, sw2):
        for x, y in zip(a, bb):
            tx = np.where(do, T[y], T[x]); ty = np.where(do, T[x], T[y]); T[x], T[y] = tx, ty
    return T

def terminal(S):
    t = (S["wkts"] >= 10) | (S["legal_used"] >= 120) | ((S["inn"] == 2) & (S["score"] >= S["target"]))
    return t

# Feature sets chosen by 5-fold match-grouped cross-validation on pre-2025 data, not by ranking outcomes:
#  * win probability: resources only. Match-constant context (Elo gap, venue par, target) and batter state made grouped-CV log loss WORSE
#    (they let the model memorise matches); competition flag and batter state gave no gain.
#  * outcome and expected-runs models: batter state (batting position, balls faced) and recent wickets help out of sample.
F1 = ["score", "wkts", "legal_used"]
F2 = ["runs_needed", "balls_left", "wkts", "rrr"]
FRE = ["legal_used", "wkts", "comp", "s_order", "s_balls", "ns_order", "ns_balls"]   # no runs already scored by the current batters
# The outcome baseline must not contain the current batters' own realised scoring (their runs, partnership runs, recent-overs runs),
# otherwise a fast scorer is judged against a bar raised by his own earlier balls in the same innings.
FOUT = ["legal_used", "wkts", "comp", "inn", "pos_in_over", "runs_needed", "rrr", "s_order", "s_balls", "ns_order", "ns_balls",
        "last12_w", "part_balls", "pace"]
BASE2 = ["runs_needed", "balls_left", "wkts"]

def col(S, c):
    if c == "balls_left": return 120 - S["legal_used"]
    if c == "runs_needed": return S["target"] - S["score"]
    if c == "rrr": return (S["target"] - S["score"]) / np.maximum(120 - S["legal_used"], 1) * 6
    return S[c]

def mat(S, cols, idx=None):
    X = np.column_stack([col(S, c) for c in cols])
    return X if idx is None else X[idx]

def mono(cols, spec):
    return [spec.get(c, 0) for c in cols]

P_WP = dict(max_iter=100, learning_rate=0.06, max_leaf_nodes=8, min_samples_leaf=1500, l2_regularization=5.0,
            early_stopping=False, random_state=SEED)
def fit_wp(S, y, innsel, cols, spec):
    X = mat(S, cols, innsel)
    return HGC(monotonic_cst=mono(cols, spec), **P_WP).fit(X, y[innsel])

M1 = {"score": 1, "wkts": -1}
M2 = {"runs_needed": -1, "balls_left": 1, "wkts": -1, "rrr": -1}
def out_matrix(S, df):
    S = dict(S); S["last12_w"] = df.last12_w.values.astype(float); S["last12_r"] = df.last12_r.values.astype(float)
    S["part_runs"] = df.part_runs.values.astype(float); S["part_balls"] = df.part_balls.values.astype(float); S["pace"] = df.pace.values.astype(float)
    return mat(S, FOUT).astype(np.float64)

def fit_models(tr):
    """Fit WP1, WP2, OUT, RE on the rows of D selected by boolean mask tr."""
    sub = D[tr]; S = state(sub)
    y = sub.bat_win.values
    keep = y != 0.5
    mods = {}
    i1 = (S["inn"] == 1) & keep; i2 = (S["inn"] == 2) & keep
    mods["wp1"] = fit_wp(S, np.where(y > 0.5, 1, 0), i1, F1, M1)
    mods["wp2"] = fit_wp(S, np.where(y > 0.5, 1, 0), i2, F2, M2)
    Xo = out_matrix(S, sub)
    mods["out"] = HGC(max_iter=120, learning_rate=0.08, max_leaf_nodes=8, min_samples_leaf=600, l2_regularization=5.0,
                      early_stopping=False, random_state=SEED).fit(Xo, sub.cls.values)
    rem = sub.match_id.map(INN1_TOT).values - S["score"]
    sel = S["inn"] == 1
    mods["re"] = HGR(max_iter=100, learning_rate=0.06, max_leaf_nodes=8, min_samples_leaf=800, l2_regularization=5.0,
                     early_stopping=False, random_state=SEED).fit(mat(S, FRE, sel), rem[sel])
    il = (sub.is_wide | sub.is_nb).values
    mods["cx"] = float(sub.bowler_runs.values[il].mean()); mods["vx"] = float(sub.bowler_runs.values[il].var())
    return mods

def wp_eval(mods, S):
    """Win probability of the batting side for arbitrary states (terminal handled by rule)."""
    n_ = len(S["score"]); out = np.zeros(n_)
    term = terminal(S)
    inn = S["inn"]
    # inn2 terminal
    t2 = term & (inn == 2)
    win = S["score"] >= S["target"]
    tie = (S["score"] == S["target"] - 1)
    out[t2] = np.where(win[t2], 1.0, np.where(tie[t2], 0.5, 0.0))
    m2 = (~term) & (inn == 2)
    if m2.any(): out[m2] = mods["wp2"].predict_proba(mat(S, F2, m2))[:, 1]
    m1 = (~term) & (inn == 1)
    if m1.any(): out[m1] = mods["wp1"].predict_proba(mat(S, F1, m1))[:, 1]
    t1 = term & (inn == 1)
    if t1.any():  # chase starts: opposing side bats from zero needing score+1
        z = np.zeros(t1.sum())
        C = {"inn": np.full(t1.sum(), 2, dtype=np.int8), "score": z, "wkts": z, "legal_used": z, "target": S["score"][t1] + 1,
             "s_order": z + 1, "s_balls": z, "s_runs": z, "ns_order": z + 2, "ns_balls": z, "ns_runs": z, "max_order": z + 2,
             "pos_in_over": z, "comp": S["comp"][t1], "venue_par": S["venue_par"][t1], "elo_diff": -S["elo_diff"][t1]}
        out[t1] = 1 - mods["wp2"].predict_proba(mat(C, F2))[:, 1]
    return out

def re_eval(mods, S):
    term = terminal(S)
    out = np.zeros(len(S["score"]))
    m_ = ~term
    if m_.any(): out[m_] = np.maximum(mods["re"].predict(mat(S, FRE, m_)), 0)
    return out

# transitions: state index -> (runs_add, bat_add, wk, consume)
TR = {0: (0, 0, 0, 1), 1: (1, 1, 0, 1), 2: (2, 2, 0, 1), 3: (3, 3, 0, 1), 4: (4, 4, 0, 1), 5: (6, 6, 0, 1), 6: (0, 0, 1, 1), 7: (1, 0, 0, 0)}
# 8 distinct next states: 0,1,2,3,4,6 runs, W (wicket), X (extra: +1 run, ball not consumed)

def block_values(mods, df):
    S = state(df); k = len(df)
    P = mods["out"].predict_proba(out_matrix(S, df))
    pcls = np.zeros((k, 9)); pcls[:, mods["out"].classes_] = P
    wp0 = wp_eval(mods, S); re0 = re_eval(mods, S)
    wps = np.zeros((k, 8)); projs = np.zeros((k, 8))
    for s_i, (ra, ba, w, c) in TR.items():
        T = advance(S, np.full(k, ra), np.full(k, ba), np.full(k, w), np.full(k, c))
        wps[:, s_i] = wp_eval(mods, T); projs[:, s_i] = T["score"] + re_eval(mods, T)
    # class -> state probabilities: classes 0..5 -> states 0..5 ; W(6)->6 ; R(7)-> wicket state for LI / batter ; X(8)->7
    pst_full = np.column_stack([pcls[:, 0], pcls[:, 1], pcls[:, 2], pcls[:, 3], pcls[:, 4], pcls[:, 5], pcls[:, 6] + pcls[:, 7], pcls[:, 8]])
    d = wps - wp0[:, None]
    LI = (pst_full * np.abs(d)).sum(1)
    Pb = (pst_full * np.maximum(-d, 0)).sum(1)    # expected downside for the batting side
    Pw = (pst_full * np.maximum(d, 0)).sum(1)     # expected upside for batting side (= downside for the bowling side)
    # batter expectation: legal deliveries only, run-outs counted as dismissal
    pb = pst_full.copy(); pb[:, 7] = 0; pb = pb / pb.sum(1, keepdims=True)
    E_bat = (pb * wps).sum(1); E_bat_proj = (pb * projs).sum(1)
    # bowler expectation: all deliveries, run-out (not credited) treated as a dot
    pw = np.column_stack([pcls[:, 0] + pcls[:, 7], pcls[:, 1], pcls[:, 2], pcls[:, 3], pcls[:, 4], pcls[:, 5], pcls[:, 6], pcls[:, 8]])
    E_bowl = (pw * wps).sum(1); E_bowl_proj = (pw * projs).sum(1)
    # realised next states, per perspective
    ba = df.bat_runs.values.astype(float)
    Ab = advance(S, ba, ba, df.striker_out.values, np.ones(k))                        # batter-attributable (legal rows only used)
    Aw = advance(S, df.bowler_runs.values.astype(float), df.bat_runs.values.astype(float), df.bowler_wkt.values,
                 df.legal.values.astype(float))
    wp_ab = wp_eval(mods, Ab); wp_aw = wp_eval(mods, Aw)
    proj_ab = Ab["score"] + re_eval(mods, Ab); proj_aw = Aw["score"] + re_eval(mods, Aw)
    xruns = (pb[:, :6] * np.array([0, 1, 2, 3, 4, 6])).sum(1)
    # v3: immediate-runs variance on a legal ball, and the wicket's FUTURE resource value measured AFTER the ball in both branches
    # (dot-ball next state vs wicket next state), so the current ball's runs are not inside it (no double counting with xruns).
    xr2 = (pb[:, :6] * np.array([0, 1, 4, 9, 16, 36])).sum(1)
    VW = projs[:, 0] - projs[:, 6]
    res = pd.DataFrame({
        "match_id": df.match_id.values, "innings": df.innings.values, "ball_no": df.innings_ball_num.values,
        "wp0": wp0, "re0": re0, "p_wkt": pcls[:, 6] + pcls[:, 7], "p_bwkt": pcls[:, 6], "p_x": pcls[:, 8],
        "p_dot": pcls[:, 0], "p_bnd": pcls[:, 4] + pcls[:, 5],
        "LI": LI, "Pb": Pb, "Pw": Pw,
        "y_bat": wp_ab - E_bat, "y_bowl": -(wp_aw - E_bowl),
        "yr_bat": proj_ab - E_bat_proj, "yr_bowl": -(proj_aw - E_bowl_proj),
        "xruns": xruns, "wp_ab": wp_ab, "wp_aw": wp_aw, "wkt_loss_wp": wp0 - wps[:, 6],
        "wkt_loss_runs": (S["score"] + re0) - projs[:, 6],
        "xvar": np.maximum(xr2 - xruns ** 2, 0.05), "VW": VW, "cx": np.full(k, mods["cx"]), "vx": np.full(k, mods["vx"]),
    })
    return res

def crossfit(df, seed, k=5):
    """Match-blocked k-fold cross-fitting on an arbitrary delivery table (match ids must identify clusters)."""
    global D
    r = np.random.default_rng(seed); ms = np.array(sorted(df.match_id.unique())); r.shuffle(ms)
    fo = pd.Series(np.arange(len(ms)) % k, index=ms); f = df.match_id.map(fo).values
    D0 = D; D = df; outs = []
    try:
        for i in range(k):
            mods = fit_models(f != i); outs.append(block_values(mods, df[f == i]).set_index(df.index[f == i]))
    finally:
        D = D0
    return pd.concat(outs).loc[df.index]

if __name__ == "__main__":
    t0 = time.time()
    V = crossfit(D, SEED).reset_index(drop=True)
    V.to_parquet(os.path.join(WORK, "ball_values_raw.parquet"), index=False)
    print("crossfit done", round(time.time() - t0), flush=True)

    # ---------- temporal validation (train < 2025, test >= 2025) ----------
    tr = (D.date < "2025-01-01").values
    mods = fit_models(tr); te = D[~tr]; S = state(te)
    y = te.bat_win.values; ok = y != 0.5
    rows = []
    for name, innn, cols in (("wp1", 1, F1), ("wp2", 2, F2)):
        sel = (S["inn"] == innn) & ok
        p = mods[name].predict_proba(mat(S, cols, sel))[:, 1]; yy = (y[sel] > 0.5).astype(int)
        # baseline: resources only (inn2: runs needed, balls left, wickets; inn1: score, wickets, balls)
        Str = state(D[tr]); ytr = D[tr].bat_win.values; seltr = (Str["inn"] == innn) & (ytr != 0.5)
        bcols = BASE2 if innn == 2 else ["score", "wkts", "legal_used"]
        base = HGC(max_iter=150, learning_rate=0.08, max_leaf_nodes=16, min_samples_leaf=150, random_state=SEED).fit(mat(Str, bcols, seltr), (ytr[seltr] > 0.5).astype(int))
        pb_ = base.predict_proba(mat(S, bcols, sel))[:, 1]  # simple resource-only baseline (tiny-leaf model)
        rows.append(dict(model=name, n=int(sel.sum()), logloss=log_loss(yy, p), brier=brier_score_loss(yy, p), auc=roc_auc_score(yy, p),
                         base_logloss=log_loss(yy, pb_), base_brier=brier_score_loss(yy, pb_), base_auc=roc_auc_score(yy, pb_),
                         const_logloss=log_loss(yy, np.full(len(yy), yy.mean()))))
        cal = pd.DataFrame({"p": p, "y": yy}); cal["bin"] = pd.cut(cal.p, np.linspace(0, 1, 11), include_lowest=True)
        cal.groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).to_csv(os.path.join(OUT, f"validation_calibration_{name}.csv"))
    Pt = mods["out"].predict_proba(out_matrix(S, te)); yt = te.cls.values
    Pf = np.zeros((len(te), 9)); Pf[:, mods["out"].classes_] = Pt
    Ptr = D[tr].copy(); Ptr["ob"] = Ptr.over; Ptr["wb"] = Ptr.wkts.clip(0, 6)
    freq = Ptr.groupby(["ob", "wb"]).cls.value_counts(normalize=True).unstack(fill_value=0).reindex(columns=range(9), fill_value=0)
    te2 = te.copy(); te2["ob"] = te2.over; te2["wb"] = te2.wkts.clip(0, 6)
    Pb_ = freq.reindex(pd.MultiIndex.from_frame(te2[["ob", "wb"]])).fillna(1 / 9).values
    Pb_ = np.clip(Pb_, 1e-4, 1); Pb_ = Pb_ / Pb_.sum(1, keepdims=True)
    rows.append(dict(model="outcome", n=len(te), logloss=log_loss(yt, np.clip(Pf, 1e-9, 1), labels=range(9)),
                     base_logloss=log_loss(yt, Pb_, labels=range(9)),
                     const_logloss=log_loss(yt, np.tile(np.bincount(D[tr].cls, minlength=9) / tr.sum(), (len(te), 1)), labels=range(9))))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "validation_models_temporal.csv"), index=False)
    print(pd.DataFrame(rows).to_string())
    print("total", time.time() - t0)
