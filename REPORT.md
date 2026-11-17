# CRUCIBLE — The Most Reliable T20 Players Under Pressure

**SupeRR Selector Season 2 · Round 1 · Technical report**

---

## Summary

CRUCIBLE identifies the batters and bowlers who perform most reliably in high-pressure moments of IPL and T20I cricket, measured against their own usual level and against everyone else facing the same situations.

The approach has four parts:

1. **Pressure is measured, not assumed.** Every ball is scored by how much win probability the side stands to lose on it, multiplied by how much the match matters for the season. League games that decide qualification count like knockouts. The top 20% of balls are pressure balls.
2. **Performance is judged against expectation.** Player-blind models give the expected result of every ball from the match state alone. A player is credited only for what he does beyond that.
3. **Opponents are accounted for.** All batters and bowlers are fitted together in one hierarchical model over the 24,054 batter–bowler matchups in the data. Every rating is therefore adjusted for the quality of the opposition actually faced, the conditions of each match and the strength of each competition.
4. **The ranking is checked from several angles.** The model ranking is reviewed against the detailed evidence. The final lists are then tested against four alternative definitions of reliability.

### Final rankings

| # | Batters | Bowlers |
|---|---|---|
| 1 | Tim David | Jasprit Bumrah |
| 2 | Suryakumar Yadav | Josh Hazlewood |
| 3 | Abhishek Sharma | Rashid Khan |
| 4 | Phil Salt | Sunil Narine |
| 5 | Heinrich Klaasen | Bhuvneshwar Kumar |
| 6 | Rinku Singh | Kuldeep Yadav |
| 7 | Shreyas Iyer | Axar Patel |
| 8 | Yashasvi Jaiswal | Varun Chakravarthy |
| 9 | Virat Kohli | Mustafizur Rahman |
| 10 | Tilak Varma | Maheesh Theekshana |

---

## 1. Data

| | |
|---|---|
| Matches analysed | 773 (353 IPL, 420 T20I) |
| Deliveries | 184,930 (177,037 legal balls, 7,893 wides and no-balls) |
| Period | 22 January 2022 – 1 November 2026 |
| Batters / bowlers appearing | 816 / 589 |
| Distinct batter–bowler pairs | 24,054 |

The supplied files contain 200,999 delivery rows from 883 listed matches. The following were removed before analysis:

| Removed | Count | Reason |
|---|---|---|
| Listed matches with no ball-by-ball data | 16 | Nothing to analyse |
| Rain-revised matches | 34 | Revised targets do not match the ball-by-ball state |
| No-results | 24 | No outcome to model |
| Super-over rows | 26 | Different game state |
| Youth and emerging-team events | 2 events | Outside the senior IPL/T20I scope |

Run and wicket totals were reconciled against recorded results: 383 of 385 run margins and 388 of 389 wicket margins agree. The audit is in `results/data_integrity.json`.

---

## 2. Defining pressure

Pressure is a property of the ball, fixed **before** it is bowled. The same ball is a pressure ball whether it goes for six or takes a wicket, so no player can create pressure balls by playing badly.

### 2.1 What the side can lose on this ball

A win-probability model (section 3) gives the batting side's chance of winning from the current state. The outcome model gives the probability of every possible result of the next ball. For each possible result we know the win probability that would follow. The **downside** of the ball for a side is the expected fall in that side's win probability, counting only the results that hurt it.

The downside is computed separately for each side:

- **Batting downside** is driven by wickets in hand, runs needed and balls left.
- **Bowling downside** is driven by the runs being defended and how close the batting side is to taking the game away.

A batter needing 12 off the last over and the bowler defending them are both under pressure, but on different scales.

### 2.2 What the match is worth

Each team in each match has a **stake**: how much its season objective depends on the result. It is reconstructed from information available before the match.

| Match type | Stake |
|---|---|
| IPL league match | Change in the team's probability of finishing in the top four between winning and losing, from 3,000 simulations of the remaining fixtures |
| IPL Qualifier 1 or Eliminator | 0.5 |
| IPL Qualifier 2 or Final | 1.0 |
| T20I bilateral match | Change in the probability of winning the series, given the series score |
| Multi-team T20I event | Group and Super-4 stages simulated in the same way as the IPL league |
| Other knockout matches | 1.0 |

A league match with a stake of 0.3 or more is a **virtual knockout**: the season turns on it. Virtual and real knockouts together form the **high-stakes** matches. 55% of bowling pressure balls fall in high-stakes matches.

### 2.3 Pressure on a ball

```
pressure = downside × (1 + stake)
```

All balls are placed on one percentile scale, separately for batting and bowling pressure. **Balls in the top 20% are pressure balls.**

The definition has two useful properties:

- **Lost causes are not pressure.** At 9 wickets down needing 60, little can still be lost, so the downside is close to zero. The same is true of a chase that is already won.
- **Pressure is not confined to the death overs.** Batting pressure balls split as follows:
  - 25% in the powerplay (a fast start against a big target, or early wickets);
  - 45% in the middle overs;
  - 30% at the death.

Share of second-innings balls that are batting pressure balls:

| Overs | 0–1 wkts | 2–3 wkts | 4–5 wkts | 6–7 wkts | 8+ wkts |
|---|---|---|---|---|---|
| 1–4 | 32% | 22% | 0% | – | – |
| 5–8 | 33% | 24% | 12% | 0% | – |
| 9–12 | 26% | 31% | 18% | 13% | 0% |
| 13–16 | 15% | 40% | 34% | 19% | 1% |
| 17–20 | 5% | 30% | 43% | 38% | 16% |

---

## 3. Measuring performance on a ball

### 3.1 Expected outcome

Four gradient-boosted models describe the situation, not the players. No batter or bowler identity is used.

| Model | Predicts | Inputs |
|---|---|---|
| Win probability, first innings | Batting side's chance of winning | Score, wickets, balls used |
| Win probability, second innings | Batting side's chance of winning | Runs needed, balls left, wickets, required rate |
| Ball outcome | Probability of 0, 1, 2, 3, 4, 6, bowler wicket, run-out, wide/no-ball | Over, wickets, competition, innings, ball in over, runs needed, required rate, both batters' positions and balls faced, wickets in the last two overs, partnership balls, pace or spin |
| Remaining runs | Runs still to come in the innings | Balls used, wickets, competition, batting positions, balls faced |

**Cross-fitting.** Each match's predictions come from models trained without that match (5-fold cross-fitting by match). A ball is never scored by a model that has seen it.

**No own-performance leakage.** The inputs contain no runs already scored by the batters at the crease. If they did, a batter in form would be judged against a bar raised by his own earlier boundaries. That would penalise fast starters.

### 3.2 Value of a ball

The value of a ball has three parts, all expressed in runs:

| Component | Definition | Credited to |
|---|---|---|
| Scoring | Runs off the bat minus expected runs | Batter (+), bowler (−) |
| Dismissal | Expected rest-of-innings runs after a dot ball minus after a wicket, both measured after the ball | Bowler, for bowler-credited dismissals only |
| Extras | Wide and no-ball runs minus expectation | Bowler |

The dismissal value is measured after the ball in both branches, so the ball's own runs are never counted twice. Run-outs are credited to neither player.

---

## 4. The matchup model

Raw "runs above expected" still flatters a batter who faced weak bowling and a bowler who bowled at the tail. CRUCIBLE fits **every batter and every bowler jointly** over the network of matchups. Each player's rating is then estimated relative to the specific opponents he met.

### 4.1 Structure

| Component | Form |
|---|---|
| Scoring (legal balls) | runs = expected + batter effect − bowler effect + match effect, with each player's effect allowed to change as pressure rises |
| Dismissal | Same structure on the log-odds of a bowler-credited wicket |
| Extras | Bowler-only effect on wide and no-ball runs |
| Competition-season levels | Separate baselines for every season of the IPL and of T20Is |
| Match effects | Pitch and conditions on the day |
| Partial pooling | Player effects share a common distribution; players with fewer than 30 balls are pooled |

The competition-season levels handle the IPL–T20I difference in strength. The adjustment is estimated from the data, not set by a fixed weight.

### 4.2 Estimation

- **Method:** approximate restricted maximum likelihood, using EM with a Laplace approximation.
- **Effects:** 4 per batter, 5 per bowler and 2 per match.
- **Convergence:** every fit ran until parameters changed by less than 10⁻⁴, with no early stopping.

### 4.3 From effects to ratings

Each player's effects are converted into runs per ball on a common set of 2,000 real pressure situations, so every player is scored on the same situations. Rankings use the **10th percentile** of each player's posterior rating. A player ranks highly only when the model is confident he is good.

### 4.4 Two fits

| Fit | Data | Strength | Weakness |
|---|---|---|---|
| Full model | All balls, pressure as a slope | Uses all evidence | Can lift a player who is excellent in ordinary play |
| Pressure-only model | Pressure balls only | Pure pressure evidence | Noisier |

The **model ranking** is the average of a player's ranks in the two fits, with ties broken by the full model.

---

## 5. Eligibility

| Requirement | Batters | Bowlers |
|---|---|---|
| Effective pressure evidence (pressure-weighted, match-clustered) | ≥ 15 | ≥ 25 |
| Pressure balls | ≥ 100 | ≥ 100 |
| Seasons with pressure balls | ≥ 2 | ≥ 2 |
| Different opponents | ≥ 5 | ≥ 5 |
| Effective IPL pressure evidence | ≥ 5 | ≥ 8.3 |
| Share of pressure evidence from the IPL | ≥ 1/3 | ≥ 1/3 |
| T20 played in the 12 months to 1 Nov 2026 | Yes | Yes |

68 batters and 58 bowlers meet the evidence requirements. The IPL requirements make sure every ranked player has a substantial IPL pressure record. The full candidate audit, including the reason each excluded player failed, is in `results/eligibility_audit.csv`.

---

## 6. From model ranking to final list

The model ranking was reviewed against each player's detailed evidence:
- record in the IPL and in T20Is separately;
- high-stakes matches;
- survival under pressure;
- pressure workload;
- recent form;
- season-to-season consistency.

The adjustments below are recorded in `src/select_final.py` and in the `basis` column of the final lists.

### Batters

| Adjustment | Model rank | Evidence |
|---|---|---|
| Andre Russell not ranked | 4 | No T20 since July 2025. |
| Nicholas Pooran not ranked | 10 | 17.6 runs per 100 pressure balls below expected in T20Is, over 197 balls. |
| Cameron Green not ranked | 11 | Below expected under pressure in T20Is (−3.5 per 100) and in high-stakes matches (−1.4). |
| Dewald Brevis not ranked | 12 | 106 pressure balls; too little evidence for a reliability ranking. |
| Abhishek Sharma placed 3rd | 5 | +49.8 runs per 100 in high-stakes matches (Phil Salt +11.2) on a larger sample (308 pressure balls to 233); second only to Suryakumar Yadav on match-winning performances. |
| Yashasvi Jaiswal placed 8th | 6 | His T20I figure rests on 37 pressure balls; his IPL record (+10.8 per 100 over 258 balls) is the weakest of the top eight. |
| Virat Kohli ranked 9th | 13 | Hardest batter to dismiss under pressure (one dismissal every 47 pressure balls) over 424 pressure balls; 8th in the pressure-only model; top ten in four of the five ranking methods (section 7). |
| Tilak Varma ranked 10th | 16 | Heaviest pressure workload of any eligible batter: 495 pressure balls, 266 in high-stakes matches, where he is above expected; 12th in the full model. |

### Bowlers

| Adjustment | Model rank | Evidence |
|---|---|---|
| Jasprit Bumrah placed 1st | 3 | The top three are a statistical tie in the model. Bumrah has the best rate of all (1.99 runs saved per over), the best high-stakes record of any eligible bowler (2.43), runs saved in every season, and the best death-over economy of any eligible pace bowler (7.40 over 514 balls). |
| Wanindu Hasaranga not ranked | 10 | Barely above average under pressure in T20Is (+0.06 runs saved per over); no IPL in the latest season. |
| Krunal Pandya not ranked | 11 | No T20I pressure record; death-over economy 10.48. |
| Maheesh Theekshana ranked 10th | 16 | Saved runs under pressure in all 9 season blocks: +1.56 per over in the IPL, +1.37 in T20Is, +1.54 in high-stakes matches over 275 balls; 3rd on the reliability index. |

---

## 7. Robustness: five ways to rank

"Reliable under pressure" can reasonably be defined in more than one way. Every available eligible player was therefore ranked under five methods, each a single fixed rule:

| Method | What it rewards |
|---|---|
| **Model** | Opponent-adjusted pressure performance from the two converged fits |
| **Reliability index** | Equal-weight average of five percentiles: model rating, dependability (batters: balls per dismissal; bowlers: economy), high-stakes value, season-to-season consistency, pressure workload |
| **Win probability added** | Total win probability added above the player-blind expectation on pressure balls |
| **Match-winning** | Stake-weighted count of won matches in which the player moved his side's win probability by 20 points or more |
| **Process + outcome** | Average of the model rank and the match-winning rank |

Ranks of the final ten under each method:

**Batters**

| Final | Player | Model | Reliability | WPA | Match-winning | Process + outcome | Top ten in |
|---|---|---|---|---|---|---|---|
| 1 | Tim David | 1 | 2 | 1 | 11 | 5 | 4 of 5 |
| 2 | Suryakumar Yadav | 2 | 2 | 2 | 1 | 1 | 5 of 5 |
| 3 | Abhishek Sharma | 4 | 8 | 25 | 2 | 2 | 4 of 5 |
| 4 | Phil Salt | 3 | 6 | 18 | 8 | 3 | 4 of 5 |
| 5 | Heinrich Klaasen | 6 | 11 | 30 | 25 | 15 | 1 of 5 |
| 6 | Rinku Singh | 7 | 4 | 3 | 22 | 13 | 3 of 5 |
| 7 | Shreyas Iyer | 8 | 5 | 4 | 3 | 3 | 5 of 5 |
| 8 | Yashasvi Jaiswal | 5 | 1 | 9 | 14 | 8 | 4 of 5 |
| 9 | Virat Kohli | 12 | 7 | 8 | 6 | 6 | 4 of 5 |
| 10 | Tilak Varma | 15 | 9 | 16 | 9 | 11 | 2 of 5 |

**Bowlers**

| Final | Player | Model | Reliability | WPA | Match-winning | Process + outcome | Top ten in |
|---|---|---|---|---|---|---|---|
| 1 | Jasprit Bumrah | 3 | 2 | 10 | 14 | 8 | 4 of 5 |
| 2 | Josh Hazlewood | 1 | 4 | 1 | 2 | 1 | 5 of 5 |
| 3 | Rashid Khan | 2 | 1 | 2 | 1 | 1 | 5 of 5 |
| 4 | Sunil Narine | 4 | 8 | 9 | 27 | 12 | 3 of 5 |
| 5 | Bhuvneshwar Kumar | 5 | 5 | 18 | 8 | 5 | 4 of 5 |
| 6 | Kuldeep Yadav | 6 | 7 | 6 | 5 | 4 | 5 of 5 |
| 7 | Axar Patel | 7 | 6 | 5 | 3 | 3 | 5 of 5 |
| 8 | Varun Chakravarthy | 8 | 9 | 3 | 19 | 11 | 3 of 5 |
| 9 | Mustafizur Rahman | 9 | 10 | 21 | 7 | 6 | 4 of 5 |
| 10 | Maheesh Theekshana | 16 | 3 | 19 | 24 | 16 | 1 of 5 |

What the comparison shows:

- **The core is stable.** Seven batters and seven bowlers on the final lists are top ten under at least four of the five methods.
- **Where methods disagree, the reason is visible.**
  - Narine and Varun rank lower on match-winning counts. Their value comes from steady control rather than single decisive spells.
  - Klaasen's strength is opponent-adjusted scoring rate rather than survival or match-deciding innings.
  - Theekshana is one of only two eligible bowlers, with Bumrah, who saved runs under pressure in every season block. He ranks third on the reliability index, but he rarely takes the decisive wicket, so he ranks lower on match-winning counts.

The full table for every player is in `results/method_comparison_*.csv`.

---

## 8. Evidence for the final lists

**Batters.** "Vs expected" columns are runs above expected per 100 pressure balls. A season block is one season in one competition.

| # | Player | Pressure balls | SR | Balls per dismissal | Vs expected | IPL | T20I | High-stakes | Seasons positive |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Tim David | 327 | 194 | 20.4 | +45.9 | +42.1 | +49.0 | +39.2 | 8 of 10 |
| 2 | Suryakumar Yadav | 506 | 180 | 18.7 | +28.6 | +13.8 | +43.5 | +27.2 | 8 of 10 |
| 3 | Abhishek Sharma | 308 | 169 | 15.4 | +25.4 | −3.2 | +78.3 | +49.8 | 5 of 8 |
| 4 | Phil Salt | 233 | 175 | 19.4 | +32.7 | +54.6 | +18.9 | +11.2 | 8 of 9 |
| 5 | Heinrich Klaasen | 227 | 166 | 16.2 | +18.7 | +18.8 | +18.5 | +18.9 | 5 of 7 |
| 6 | Rinku Singh | 204 | 175 | 29.1 | +19.8 | +10.5 | +46.4 | +58.7 | 7 of 8 |
| 7 | Shreyas Iyer | 315 | 167 | 31.5 | +19.0 | +26.6 | −1.8 | +12.9 | 4 of 7 |
| 8 | Yashasvi Jaiswal | 295 | 161 | 32.8 | +16.5 | +10.8 | +56.5 | +40.4 | 5 of 7 |
| 9 | Virat Kohli | 424 | 150 | 47.1 | −2.8 | −1.4 | −7.3 | −0.2 | 3 of 7 |
| 10 | Tilak Varma | 495 | 141 | 26.1 | −4.9 | −10.8 | +0.7 | +2.4 | 4 of 9 |

Kohli and Tilak are placed on dependability and workload rather than scoring rate: they score close to expectation but rarely get out and carry the heaviest pressure loads in the data.

**Bowlers.** "Saved" columns are runs saved per over against expectation, extras included.

| # | Player | Pressure balls | Economy | Saved | IPL | T20I | High-stakes | Death economy | Dot % | Seasons positive |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Jasprit Bumrah | 329 | 7.93 | 1.99 | 2.05 | 1.83 | 2.43 | 7.40 | 38.6 | 8 of 8 |
| 2 | Josh Hazlewood | 330 | 8.36 | 1.45 | 0.60 | 3.00 | 1.05 | 9.85 | 42.1 | 5 of 7 |
| 3 | Rashid Khan | 488 | 7.30 | 1.49 | 1.26 | 1.86 | 1.39 | 8.41 | 35.0 | 9 of 10 |
| 4 | Sunil Narine | 256 | 7.59 | 1.16 | 1.16 | – | – | 6.12 | 37.1 | 4 of 5 |
| 5 | Bhuvneshwar Kumar | 427 | 9.08 | 1.12 | 1.74 | −0.51 | 0.36 | 10.09 | 37.2 | 6 of 7 |
| 6 | Kuldeep Yadav | 333 | 7.98 | 1.00 | 1.59 | −0.38 | 0.53 | 8.45 | 30.3 | 6 of 9 |
| 7 | Axar Patel | 433 | 8.00 | 0.67 | 0.82 | 0.53 | 0.11 | 9.15 | 31.9 | 7 of 10 |
| 8 | Varun Chakravarthy | 349 | 7.82 | 0.74 | 1.35 | −0.61 | −0.28 | 8.55 | 39.5 | 6 of 8 |
| 9 | Mustafizur Rahman | 355 | 9.26 | 0.75 | 0.54 | 0.85 | 0.81 | 8.91 | 35.8 | 6 of 9 |
| 10 | Maheesh Theekshana | 397 | 7.59 | 1.43 | 1.56 | 1.37 | 1.54 | 8.22 | 37.0 | 9 of 9 |

Narine has no T20I pressure balls in the data. None of his pressure balls came in a match that reached the high-stakes threshold, so those columns are empty for him rather than negative.

---

## 9. Validation and limitations

### Stability over time

Pressure performance was measured on 2022–24 and compared with 2025–26, for players with at least 60 pressure balls in each period (53 batters, 53 bowlers).

| | Rank correlation, 2022–24 vs 2025–26 | Top quarter in 2022–24 still above the median in 2025–26 |
|---|---|---|
| Bowlers | 0.30 | 79% |
| Batters | 0.05 | 71% |

**Bowling performance under pressure repeats; batting performance is noisier from one period to the next.** This is why the method relies on long records, partial pooling, agreement between two fits and a cross-method check, rather than on any single season.

### Uncertainty

The top of each list is clearly separated. Tim David and Suryakumar Yadav are first and second in both fits, and Bumrah, Hazlewood and Rashid are in the top four of both. From roughly fifth place onwards the differences between players are smaller than the uncertainty in their ratings, so neighbouring positions should be read as a band rather than a strict order.

### Limitations

- The data contains no T20 World Cup matches. High-stakes T20I evidence comes from bilateral series and Asia Cups.
- Net run rate is not modelled in the stake simulation. Ties at the qualification cut-off are broken at random.
- Rain-revised matches are excluded rather than modelled.
- Run-outs are credited to neither player.
- Fielding and captaincy are not measured.
- The minimum share of IPL pressure evidence (one third) was set after an initial review of the rankings.

---

## 10. What distinguishes this approach

| | Typical approach | CRUCIBLE |
|---|---|---|
| Pressure | Weighted formula of required rate, wickets and phase | Expected loss of win probability on the ball, from fitted models |
| Match importance | Flat bonus for playoffs | Season simulation; must-win league matches count as virtual knockouts |
| Batting vs bowling pressure | One shared score | Separate measures for each side |
| Opposition | Ignored, or a league-average baseline | Joint model over 24,054 matchups |
| IPL vs T20I | Fixed weights | Competition-season levels estimated from data, plus an IPL evidence requirement |
| Robustness | One formula | Two converged fits, evidence review and five-method comparison |
| Validation | Rarely reported | Out-of-time check reported, including where it is weak |

---

## 11. Reproducing the results

```
pip install -r requirements.txt
python src/s1_data.py        # clean data, pre-ball match states
python src/s1b_stakes.py     # match stakes from season simulation
python src/s2_models.py      # player-blind models, cross-fitted ball values
python src/s5_main.py        # joint matchup model on all balls, eligibility audit
python src/quick_final.py    # full-model ratings              -> results/model_full/
python src/pressure_only.py  # pressure-only fits and ratings  -> results/model_pressure_only/
python src/select_final.py   # model ranking, availability, expert review -> results/final_*.csv
python src/evidence.py       # evidence tables, out-of-time check, dashboard data
python src/robustness.py     # five-method comparison          -> results/method_comparison_*.csv
```

Intermediate files are written to `work/`. The joint-model fit is the slowest step and uses a CUDA GPU through PyTorch when one is available. All random steps use the fixed seed 20260718.
