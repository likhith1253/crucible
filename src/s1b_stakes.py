"""Step 1b: match-level stakes, reconstructed ONLY from information available before each match.

stake(team, match) = swing in the probability of that team's tournament objective between winning and losing the match:
  * IPL league match   : P(finish top 4) | win  -  P(finish top 4) | lose, by Monte Carlo of the remaining league fixtures
                         (fixture list = schedule, known in advance; results of future matches are NOT used). Net run rate is
                         not modelled (ties at the cut-off are broken at random) - documented limitation.
  * IPL playoffs       : Qualifier 1 / Eliminator 0.5 each (reach the final), Qualifier 2 1.0, Final 1.0
  * T20I bilateral     : swing in P(win the series) (draw = half) given the series score before the match, remaining games at Elo odds
  * other knockouts    : 1.0 (semi-final, final)
  * multi-team T20I events (Asia Cups): group / Super-4 stages simulated like the IPL league; the number advancing is read from the
                         size of the next stage. The supplied pack contains NO T20 World Cup matches.
Output: work/stakes.csv (match_id, team_id, stake, kind)
"""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *

m = pd.read_csv(os.path.join(WORK, "matches_clean.csv"), parse_dates=["match_date"])
allm = pd.read_csv(os.path.join(DATA, "matches.csv"), parse_dates=["match_date"]).merge(pd.read_csv(os.path.join(DATA, "competitions.csv")), on="comp_id")
rng = np.random.default_rng(SEED)
N = 3000
rows = []

def elo_p(ea, eb):
    return 1 / (1 + 10 ** ((eb - ea) / 400))

# ---------------- IPL league ----------------
ipl = allm[allm.comp_name.str.startswith("IPL")].copy()
mel = m.set_index("match_id")
for comp, g in ipl.groupby("comp_id"):
    lg = g[g.stage.eq("League")].sort_values(["match_date", "match_number", "match_id"]).reset_index(drop=True)
    teams = sorted(set(lg.home_team_id) | set(lg.away_team_id)); ti = {t: i for i, t in enumerate(teams)}; T = len(teams)
    # a fixture whose result is unknown (abandoned) is awarded 1 point each when it is played
    pts = np.zeros(T)
    elo = {t: 1500.0 for t in teams}
    for idx, r in lg.iterrows():
        eh = mel.elo_home.get(r.match_id, 1500.0); ea = mel.elo_away.get(r.match_id, 1500.0)
        rem = lg.iloc[idx + 1:]
        hi = np.array([ti[t] for t in rem.home_team_id], dtype=int); ai = np.array([ti[t] for t in rem.away_team_id], dtype=int)
        # remaining matches at neutral-ish Elo odds using the CURRENT match's team ratings only for these two; others at 0.5 + elo from m if present
        ph = np.array([elo_p(mel.elo_home.get(x, 1500.0), mel.elo_away.get(x, 1500.0)) if x in mel.index else 0.5 for x in rem.match_id])
        # use the ratings known now: the (future) match rows hold ratings valid at THEIR date; to avoid leakage use 0.5 for future games
        ph[:] = 0.5
        u = rng.random((len(rem), N))
        homewin = (u < ph[:, None])
        base = np.repeat(pts[:, None], N, axis=1)
        add = np.zeros((T, N))
        for k in range(len(rem)):
            hw = homewin[k]
            add[hi[k]] += 2 * hw; add[ai[k]] += 2 * (~hw)
        res = {}
        hh, aa = ti[r.home_team_id], ti[r.away_team_id]
        for outcome in ("home", "away"):
            p2 = base + add
            if outcome == "home": p2[hh] += 2
            else: p2[aa] += 2
            jitter = rng.random(p2.shape) * 0.01
            rank = (-(p2 + jitter)).argsort(axis=0).argsort(axis=0)
            q = rank < 4
            res[outcome] = (q[hh].mean(), q[aa].mean())
        swing_h = res["home"][0] - res["away"][0]
        swing_a = res["away"][1] - res["home"][1]
        rows.append((r.match_id, r.home_team_id, max(swing_h, 0.0), "ipl_league_qualification"))
        rows.append((r.match_id, r.away_team_id, max(swing_a, 0.0), "ipl_league_qualification"))
        # update points with the real result after the match
        if pd.notna(r.win_team_id):
            pts[ti[r.win_team_id]] += 2
        else:
            pts[hh] += 1; pts[aa] += 1
    po = g[~g.stage.eq("League")]
    for r in po.itertuples():
        s = {"Qualifier": 0.5, "Eliminator": 0.5, "Final": 1.0}.get(r.stage, 1.0)
        if r.stage == "Qualifier" and r.match_number == po.match_number.max() - 1: s = 1.0  # Qualifier 2 (second qualifier)
        rows.append((r.match_id, r.home_team_id, s, "ipl_playoff")); rows.append((r.match_id, r.away_team_id, s, "ipl_playoff"))

# ---------------- T20I ----------------
t20 = allm[allm.match_format.eq("T20I")].copy()
for comp, g in t20.groupby("comp_id"):
    g = g.sort_values(["match_date", "match_number", "match_id"]).reset_index(drop=True)
    is_series = "Series" in g.comp_name.iloc[0] and g.stage.isin(["Others", "League"]).all() and g.knock_out.sum() == 0
    if is_series and len({*g.home_team_id, *g.away_team_id}) == 2:
        t1, t2 = sorted({*g.home_team_id, *g.away_team_id}); n = len(g)
        w1 = w2 = 0
        for i, r in g.iterrows():
            rem = n - i - 1
            def p_series(a, bb, rem_):  # P(team1 wins series) + 0.5 P(draw), remaining at p=0.5
                from math import comb
                tot = 0.0
                for k in range(rem_ + 1):
                    pr = comb(rem_, k) / 2 ** rem_
                    x1, x2 = a + k, bb + (rem_ - k)
                    tot += pr * (1.0 if x1 > x2 else 0.5 if x1 == x2 else 0.0)
                return tot
            sw1 = p_series(w1 + 1, w2, rem) - p_series(w1, w2 + 1, rem)
            rows.append((r.match_id, t1, abs(sw1), "t20i_series")); rows.append((r.match_id, t2, abs(sw1), "t20i_series"))
            if pd.notna(r.win_team_id):
                if r.win_team_id == t1: w1 += 1
                else: w2 += 1
    else:
        # multi-team event: a round-robin stage is every run of non-knockout matches; the number of teams that advance from it is a
        # FORMAT constant, read as the number of its teams that appear in the next stage (a count, not who). Stake = swing in
        # P(finish in the advancing places), points only, remaining fixtures of the stage at even odds.
        ko = g[g.knock_out == 1]; rr = g[g.knock_out == 0]
        for r in ko.itertuples():
            rows.append((r.match_id, r.home_team_id, 1.0, "t20i_knockout")); rows.append((r.match_id, r.away_team_id, 1.0, "t20i_knockout"))
        for stage, lg in rr.groupby("stage", sort=False):
            lg = lg.reset_index(drop=True); later = g[g.match_date > lg.match_date.max()]
            teams = sorted(set(lg.home_team_id) | set(lg.away_team_id)); nxt = set(later.home_team_id) | set(later.away_team_id)
            n_adv = len([t for t in teams if t in nxt])
            if n_adv == 0 or n_adv >= len(teams) or len(teams) < 3:
                for r in lg.itertuples():
                    rows.append((r.match_id, r.home_team_id, 0.0, "t20i_event_unmodelled")); rows.append((r.match_id, r.away_team_id, 0.0, "t20i_event_unmodelled"))
                continue
            ti = {t: i for i, t in enumerate(teams)}; T = len(teams); pts = np.zeros(T)
            for idx, r in lg.iterrows():
                rem = lg.iloc[idx + 1:]; hi = np.array([ti[t] for t in rem.home_team_id], dtype=int); ai = np.array([ti[t] for t in rem.away_team_id], dtype=int)
                hw = rng.random((len(rem), N)) < 0.5; add = np.zeros((T, N))
                for k in range(len(rem)): add[hi[k]] += 2 * hw[k]; add[ai[k]] += 2 * (~hw[k])
                hh, aa = ti[r.home_team_id], ti[r.away_team_id]; res = {}
                for outcome in ("home", "away"):
                    p2 = np.repeat(pts[:, None], N, axis=1) + add
                    p2[hh if outcome == "home" else aa] += 2
                    q = (-(p2 + rng.random(p2.shape) * 0.01)).argsort(axis=0).argsort(axis=0) < n_adv
                    res[outcome] = (q[hh].mean(), q[aa].mean())
                rows.append((r.match_id, r.home_team_id, max(res["home"][0] - res["away"][0], 0.0), "t20i_event_group"))
                rows.append((r.match_id, r.away_team_id, max(res["away"][1] - res["home"][1], 0.0), "t20i_event_group"))
                if pd.notna(r.win_team_id): pts[ti[r.win_team_id]] += 2
                else: pts[hh] += 1; pts[aa] += 1

st = pd.DataFrame(rows, columns=["match_id", "team_id", "stake", "kind"]).drop_duplicates(["match_id", "team_id"])
st.to_csv(os.path.join(WORK, "stakes.csv"), index=False)
x = st[st.match_id.isin(m.match_id)]
print(x.groupby("kind").stake.describe())
print("share of IPL league team-matches with stake>0.2:", (x[x.kind == "ipl_league_qualification"].stake > 0.2).mean())
