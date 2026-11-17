"""Step 1: load the supplied files, audit them, and build the pre-delivery match state table.

Every feature in the state table is computed from deliveries BEFORE the delivery in question (no look-ahead).
Outputs: work/deliveries.parquet, work/matches_clean.csv, out/data_integrity.json
"""
import numpy as np, pandas as pd, os, sys
from collections import deque
sys.path.insert(0, os.path.dirname(__file__))
from common import *

b = pd.read_csv(os.path.join(DATA, "ball_by_ball.csv"), low_memory=False)
m = pd.read_csv(os.path.join(DATA, "matches.csv"))
c = pd.read_csv(os.path.join(DATA, "competitions.csv"))
m = m.merge(c, on="comp_id", how="left")
m["match_date"] = pd.to_datetime(m["match_date"])
rep = {"raw_rows": int(len(b)), "raw_matches_bbb": int(b.match_id.nunique()), "matches_file": int(len(m))}

# ---------- scope: IPL and T20I only ----------
m["family"] = np.where(m.match_format.eq("T20I"), "T20I", np.where(m.comp_name.str.startswith("IPL"), "IPL", "OTHER"))
rep["matches_by_family"] = m.family.value_counts().to_dict()
rep["excluded_other_competitions"] = sorted(m[m.family.eq("OTHER")].comp_name.unique().tolist())
m = m[m.family.ne("OTHER")].copy()
rep["no_ballbyball_for_listed_matches"] = int((~m.match_id.isin(b.match_id)).sum())
m = m[m.match_id.isin(b.match_id)].copy()

# ---------- integrity: duplicates, ids, super overs ----------
rep["duplicate_deliveries"] = int(b.duplicated(["match_id", "innings", "innings_ball_num"]).sum())
b = b[b.match_id.isin(m.match_id)].copy()
rep["superover_rows_excluded"] = int((b.innings > 2).sum())
b = b[b.innings <= 2].copy()
b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

# ---------- delivery semantics ----------
et = b.extras_type.fillna("")
b["is_wide"] = et.eq("wide")
b["is_nb"] = et.isin(["Overstep", "other"])
b["is_pen"] = et.eq("Penalty")
b["legal"] = ~(b.is_wide | b.is_nb | b.is_pen)
b["faced"] = ~(b.is_wide | b.is_pen)                       # batter faced the delivery (no-balls included)
b["team_runs"] = b.runs + b.extras1_runs + b.extras_2_runs
b["bat_runs"] = np.where(et.isin(["byes", "legbyes", "Penalty", "wide"]), 0, b.runs)
b["bowler_runs"] = (b.bat_runs
                    + np.where(b.is_wide, b.extras1_runs + b.extras_2_runs, 0)
                    + np.where(b.is_nb, b.extras1_runs, 0))
b["striker_out"] = b.is_wicket & (b.dismissed_player_id == b.striker_id)
b["bowler_wkt"] = b.is_wicket & b.dismissal_type.isin(BOWLER_CREDITED)
b["other_wkt"] = b.is_wicket & ~b.dismissal_type.isin(BOWLER_CREDITED)
rep["wicket_types"] = b[b.is_wicket].dismissal_type.value_counts().to_dict()
rep["dismissed_not_striker"] = int((b.is_wicket & ~b.striker_out).sum())

# innings totals and result reconciliation
tot = b.groupby(["match_id", "innings"]).agg(total=("team_runs", "sum"), wk=("is_wicket", "sum"),
                                             legal=("legal", "sum")).unstack()
tot.columns = [f"{a}{i}" for a, i in tot.columns]
mm = m.set_index("match_id").join(tot)
rep["innings_over_120_legal_balls"] = int(((mm.legal1 > 120) | (mm.legal2 > 120)).sum())
res = mm.result.str.lower()
runs_n = res.str.extract(r"won by (\d+) run")[0].astype(float)
wk_n = res.str.extract(r"won by (\d+) w")[0].astype(float)
chk_runs = mm[runs_n.notna() & mm.revised_target.isna()]
ok_runs = ((chk_runs.total1 - chk_runs.total2) == runs_n[chk_runs.index])
chk_wk = mm[wk_n.notna() & mm.revised_target.isna()]
ok_wk = ((chk_wk.total2 > chk_wk.total1) & ((10 - chk_wk.wk2) >= 0))
rep["reconcile_runs_margin_checked"] = int(len(chk_runs)); rep["reconcile_runs_margin_ok"] = int(ok_runs.sum())
rep["reconcile_wkts_checked"] = int(len(chk_wk)); rep["reconcile_wkts_ok"] = int(ok_wk.sum())
rep["reconcile_wkts_margin_ok"] = int(((10 - chk_wk.wk2) == wk_n[chk_wk.index]).sum())
rep["runs_margin_mismatch_examples"] = chk_runs[~ok_runs].index.tolist()[:20]

# ---------- match filters ----------
rev = m.revised_target.notna() | m.first_inn_revised_overs.notna() | m.second_inn_revised_overs.notna()
m["rain_revised"] = rev
decided = m.win_team_id.notna() | m.result.str.contains("tie", case=False)
m["decided"] = decided
rep["rain_revised_matches_excluded"] = int(rev.sum())
rep["no_result_or_abandoned_in_scope"] = int((~decided).sum())
keep = m[(~rev) & decided].match_id
# both innings must exist
have2 = b.groupby("match_id").innings.nunique()
keep = [x for x in keep if have2.get(x, 0) == 2]
m = m[m.match_id.isin(keep)].copy()
b = b[b.match_id.isin(keep)].copy()
rep["matches_in_analysis"] = int(len(m)); rep["deliveries_in_analysis"] = int(len(b))
# truncated innings (neither all out, nor 20 overs, nor chase complete) -> flag
inn = b.groupby(["match_id", "innings"]).agg(total=("team_runs", "sum"), wk=("is_wicket", "sum"), legal=("legal", "sum")).reset_index()
t1 = inn[inn.innings == 1].set_index("match_id")
inn2 = inn[inn.innings == 2].set_index("match_id")
x = inn2.join(t1[["total"]], rsuffix="_1")
trunc2 = x[(x.legal < 120) & (x.wk < 10) & (x.total <= x.total_1)]
trunc1 = t1[(t1.legal < 120) & (t1.wk < 10)]
rep["truncated_innings_flagged"] = int(len(trunc1) + len(trunc2))
bad = set(trunc1.index) | set(trunc2.index)
m = m[~m.match_id.isin(bad)].copy(); b = b[~b.match_id.isin(bad)].copy()
rep["matches_in_analysis"] = int(len(m)); rep["deliveries_in_analysis"] = int(len(b))

# ---------- pre-delivery state (single pass, strictly causal) ----------
b = b.reset_index(drop=True)
n = len(b)
cols = {k: np.zeros(n, dtype=np.float32) for k in
        ["score", "wkts", "legal_used", "s_order", "s_balls", "s_runs", "ns_order", "ns_balls", "ns_runs",
         "max_order", "pos_in_over", "last12_w", "last12_r", "part_runs", "part_balls", "target"]}
mid = b.match_id.values; inn_a = b.innings.values; stk = b.striker_id.values; nst = b.non_striker_id.values
legal_a = b.legal.values; faced_a = b.faced.values; trun = b.team_runs.values; batr = b.bat_runs.values
wk_a = b.is_wicket.values; dis = b.dismissed_player_id.values
inn1_total = t1.total.to_dict()
i = 0
while i < n:
    j = i
    while j < n and mid[j] == mid[i] and inn_a[j] == inn_a[i]:
        j += 1
    score = 0; wk = 0; lg = 0
    order = {}; pst = {}
    hist = deque(maxlen=12)
    part_r = 0; part_b = 0
    tgt = inn1_total[mid[i]] + 1 if inn_a[i] == 2 else np.nan
    for k in range(i, j):
        for p in (stk[k], nst[k]):
            if p not in order:
                order[p] = len(order) + 1
                pst[p] = [0, 0]
        cols["score"][k] = score; cols["wkts"][k] = wk; cols["legal_used"][k] = lg
        cols["s_order"][k] = order[stk[k]]; cols["s_balls"][k] = pst[stk[k]][0]; cols["s_runs"][k] = pst[stk[k]][1]
        cols["ns_order"][k] = order[nst[k]]; cols["ns_balls"][k] = pst[nst[k]][0]; cols["ns_runs"][k] = pst[nst[k]][1]
        cols["max_order"][k] = len(order); cols["pos_in_over"][k] = lg % 6
        cols["last12_w"][k] = sum(h[0] for h in hist); cols["last12_r"][k] = sum(h[1] for h in hist)
        cols["part_runs"][k] = part_r; cols["part_balls"][k] = part_b
        cols["target"][k] = tgt
        # update
        score += trun[k]
        part_r += trun[k]
        if legal_a[k]:
            lg += 1; part_b += 1
            hist.append((int(wk_a[k]), int(trun[k])))
        if faced_a[k]:
            pst[stk[k]][0] += 1; pst[stk[k]][1] += batr[k]
        if wk_a[k]:
            wk += 1; part_r = 0; part_b = 0
    i = j
for k, v in cols.items():
    b[k] = v
b["balls_left"] = 120 - b.legal_used
b["over"] = (b.legal_used // 6 + 1).astype(int)
b["runs_needed"] = np.where(b.innings == 2, b.target - b.score, np.nan)
b["rrr"] = np.where(b.innings == 2, b.runs_needed / np.maximum(b.balls_left, 1) * 6, np.nan)
b["crr"] = b.score / np.maximum(b.legal_used, 1) * 6
# order sanity check against supplied batting_order for strikers
rep["striker_order_equals_supplied_batting_order_pct"] = float((b.s_order == b.batting_order).mean())

# ---------- match-level pre-match context: Elo, venue par, final result ----------
m = m.sort_values(["match_date", "match_id"]).reset_index(drop=True)
elo = {}; K = 20.0
elo_h = []; elo_a = []
for r in m.itertuples():
    eh = elo.get(r.home_team_id, 1500.0); ea = elo.get(r.away_team_id, 1500.0)
    elo_h.append(eh); elo_a.append(ea)
    if pd.notna(r.win_team_id) and r.win_team_id in (r.home_team_id, r.away_team_id):
        exp_h = 1 / (1 + 10 ** ((ea - eh) / 400))
        sh = 1.0 if r.win_team_id == r.home_team_id else 0.0
        elo[r.home_team_id] = eh + K * (sh - exp_h); elo[r.away_team_id] = ea - K * (sh - exp_h)
m["elo_home"] = elo_h; m["elo_away"] = elo_a
tt = inn.pivot(index="match_id", columns="innings", values="total")
m["inn1_total"] = m.match_id.map(tt[1]); m["inn2_total"] = m.match_id.map(tt[2])
par = []; seen = {}; fam_sum = {}; fam_n = {}
for r in m.itertuples():
    key = (r.family, r.venue_id)
    s, nn = seen.get(key, (0.0, 0))
    fs, fn = fam_sum.get(r.family, 0.0), fam_n.get(r.family, 0)
    fam_mean = (fs / fn) if fn >= 5 else 160.0
    par.append((s + 5 * fam_mean) / (nn + 5))
    seen[key] = (s + r.inn1_total, nn + 1); fam_sum[r.family] = fs + r.inn1_total; fam_n[r.family] = fn + 1
m["venue_par"] = par
mi = m.set_index("match_id")
b["comp"] = b.match_id.map(mi.family).eq("IPL").astype(np.int8)
b["venue_par"] = b.match_id.map(mi.venue_par)
bat_is_home = b.batting_team_id.values == b.match_id.map(mi.home_team_id).values
eh = b.match_id.map(mi.elo_home).values; ea = b.match_id.map(mi.elo_away).values
b["elo_diff"] = np.where(bat_is_home, eh - ea, ea - eh)
b["season_year"] = b.match_id.map(mi.match_date).dt.year
b["family"] = b.match_id.map(mi.family)
b["date"] = b.match_id.map(mi.match_date)
# batting side wins? (ties: 0.5)
win = b.match_id.map(mi.win_team_id)
tie = b.match_id.map(mi.result).str.contains("tie", case=False) & (b.match_id.map(mi.inn1_total) == b.match_id.map(mi.inn2_total))
b["bat_win"] = np.where(tie, 0.5, (win.values == b.batting_team_id.values).astype(float))
rep["tied_matches_super_over"] = int(tie.groupby(b.match_id).first().sum())
rep["seasons"] = sorted(b.season_year.unique().tolist()); rep["date_min"] = str(m.match_date.min().date()); rep["date_max"] = str(m.match_date.max().date())
rep["deliveries_by_family"] = b.family.value_counts().to_dict()
rep["legal_ball_count_check_max_per_innings"] = int(b.groupby(["match_id", "innings"]).legal.sum().max())

b.to_parquet(os.path.join(WORK, "deliveries.parquet"), index=False)
m.to_csv(os.path.join(WORK, "matches_clean.csv"), index=False)
dump_json(rep, os.path.join(OUT, "data_integrity.json"))
print(json.dumps(rep, indent=1, default=str))
