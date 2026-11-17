# CRUCIBLE: who delivers under pressure

SupeRR Selector Season 2, Round 1. The task: rank the ten batters and ten bowlers who perform most reliably in high-pressure moments compared with their usual level, using IPL and T20I ball-by-ball data.

This report explains how pressure is defined, how a player's performance on a single ball is measured, how players who faced different opponents in different competitions are compared fairly, how the final lists were chosen, and how far the results can be trusted.

---

## 1. The answer

**Batters**

| # | Player | Pressure balls | Pressure SR | Runs above expected per 100 pressure balls | IPL | T20I | High-stakes games | Season blocks positive |
|---|---|---|---|---|---|---|---|---|
| 1 | Tim David | 327 | 194 | +45.9 | +42.1 | +49.0 | +39.2 | 8 of 10 |
| 2 | Suryakumar Yadav | 506 | 180 | +28.6 | +13.8 | +43.5 | +27.2 | 8 of 10 |
| 3 | Phil Salt | 233 | 175 | +32.7 | +54.6 | +18.9 | +11.2 | 8 of 9 |
| 4 | Abhishek Sharma | 308 | 169 | +25.4 | −3.2 | +78.3 | +49.8 | 5 of 8 |
| 5 | Heinrich Klaasen | 227 | 166 | +18.7 | +18.8 | +18.5 | +18.9 | 5 of 7 |
| 6 | Rinku Singh | 204 | 175 | +19.8 | +10.5 | +46.4 | +58.7 | 7 of 8 |
| 7 | Shreyas Iyer | 315 | 167 | +19.0 | +26.6 | −1.8 | +12.9 | 4 of 7 |
| 8 | Yashasvi Jaiswal | 295 | 161 | +16.5 | +10.8 | +56.5 | +40.4 | 5 of 7 |
| 9 | Sanju Samson | 314 | 161 | +15.3 | +23.4 | +1.2 | +10.9 | 6 of 10 |
| 10 | Travis Head | 174 | 156 | +18.6 | +32.3 | +9.1 | +9.5 | 5 of 7 |

**Bowlers**

| # | Player | Pressure balls | Pressure economy | Runs saved per over | IPL | T20I | High-stakes games | Death economy | Season blocks positive |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Jasprit Bumrah | 329 | 7.93 | 1.99 | 2.05 | 1.83 | 2.43 | 7.40 | 8 of 8 |
| 2 | Josh Hazlewood | 330 | 8.36 | 1.45 | 0.60 | 3.00 | 1.05 | 9.85 | 5 of 7 |
| 3 | Rashid Khan | 488 | 7.30 | 1.49 | 1.26 | 1.86 | 1.39 | 8.41 | 9 of 10 |
| 4 | Sunil Narine | 256 | 7.59 | 1.16 | 1.16 | – | – | 6.12 | 4 of 5 |
| 5 | Bhuvneshwar Kumar | 427 | 9.08 | 1.12 | 1.74 | −0.51 | 0.36 | 10.09 | 6 of 7 |
| 6 | Kuldeep Yadav | 333 | 7.98 | 1.00 | 1.59 | −0.38 | 0.53 | 8.45 | 6 of 9 |
| 7 | Axar Patel | 433 | 8.00 | 0.67 | 0.82 | 0.53 | 0.11 | 9.15 | 7 of 10 |
| 8 | Varun Chakravarthy | 349 | 7.82 | 0.74 | 1.35 | −0.61 | −0.28 | 8.55 | 6 of 8 |
| 9 | Mustafizur Rahman | 355 | 9.26 | 0.75 | 0.54 | 0.85 | 0.81 | 8.91 | 6 of 9 |
| 10 | Maheesh Theekshana | 397 | 7.59 | 1.43 | 1.56 | 1.37 | 1.54 | 8.22 | 9 of 9 |

How to read the columns:
- "Runs above expected" and "runs saved" compare what happened on a player's pressure balls with what an average player would have produced in exactly the same match situations (section 4).
- A "season block" is one season in one competition, for example IPL 2024 or T20I 2024.
- The raw numbers in these tables describe the evidence. The ranking itself comes from the opponent-adjusted model (section 5) plus the evidence checks (section 7).

---

## 2. Data

| | |
|---|---|
| Matches in the analysis | 773 (353 IPL, 420 T20I) |
| Deliveries | 184,930 (177,037 legal balls + 7,893 wides and no-balls) |
| Period | 22 Jan 2022 to 1 Nov 2026 |
| Batters / bowlers | 816 / 589 |
| Distinct batter–bowler pairs | 24,054 |

The supplied files hold 200,999 rows from 883 listed matches. The following were removed before analysis:
- 16 listed matches that have no ball-by-ball data;
- 34 rain-revised matches, because their targets do not match the ball-by-ball state;
- 24 no-results;
- two youth/emerging-team events;
- 26 super-over rows.

Run and wicket totals were reconciled against the match results: 383 of 385 run margins and 388 of 389 wicket margins agree. The full audit is in `results/data_integrity.json`.

---

## 3. What counts as pressure

Pressure is measured on every ball, before it is bowled. Only information available at that moment is used.

**Step 1: how much the side can lose on this ball.** A win-probability model gives the chance that the batting side wins, given the match state (section 4). From the outcome model we know the spread of results this ball could produce: dot, single, boundary, wicket, wide and so on. The **downside** of a ball for a side is the expected fall in that side's win probability over those possible results.

The downside is calculated separately for each side:
- For the **batting** side, the downside is driven by how many wickets are in hand, the runs needed and the balls left.
- For the **bowling** side, it is driven by the runs it is defending and how close the batting side is to taking the game away.

A bowler defending 12 off the last over and a batter needing 12 off the last over are both under pressure. Their pressure is not the same number, and each is measured on its own scale.

**Step 2: what the match is worth.** A tight finish in a dead rubber is not the same as a tight finish in a game that decides the season. Every match gets a stake for each team, built only from information available before the game:

| Match type | Stake |
|---|---|
| IPL league game | Change in the team's chance of finishing top four between winning and losing it. Found by simulating the remaining fixtures 3,000 times from the table at that point. |
| IPL Qualifier 1, Eliminator | 0.5 |
| IPL Qualifier 2, Final | 1.0 |
| T20I bilateral game | Change in the chance of winning the series, given the series score before the match |
| Multi-team T20I event | Group and Super-4 stages simulated like the IPL league |
| Other knockouts | 1.0 |

A league game where the stake reaches 0.3 or more works like a knockout: the season really turns on it. We call these **virtual knockouts** and group them with the real knockouts as "high-stakes" games.

**Step 3: one number per ball.** The pressure on a ball is the downside multiplied by (1 + stake). All balls are put on one percentile scale, separately for batters and bowlers. **The top 20% of balls are the pressure balls.**

Where batting pressure sits in a chase, as a share of second-innings balls that are pressure balls:

| Overs | 0–1 wkts | 2–3 wkts | 4–5 wkts | 6–7 wkts | 8+ wkts |
|---|---|---|---|---|---|
| 1–4 | 32% | 22% | 0% | – | – |
| 5–8 | 33% | 24% | 12% | 0% | – |
| 9–12 | 26% | 31% | 18% | 13% | 0% |
| 13–16 | 15% | 40% | 34% | 19% | 1% |
| 17–20 | 5% | 30% | 43% | 38% | 16% |

Pressure is not limited to the death overs. Of the batting pressure balls:
- 25% are in the powerplay (a fast start against a big target, or early wickets);
- 45% are in the middle overs;
- 30% are at the death.

55% of bowling pressure balls come in high-stakes games.

Two details of the definition matter:
- **Lost causes are not pressure.** When a side is 9 down needing 60, very little can be lost on the next ball, so the downside is close to zero. The same applies to a chase that is already won. The 0% cells in the table come from this rule.
- **Pressure is fixed before the ball.** The same delivery is a pressure ball whether it goes for six or takes a wicket, so a player cannot create pressure balls by playing badly.

---

## 4. Measuring a single ball

Every ball is judged against what an average player would have done in the same situation.

**The expected outcome.** We fitted four models with gradient-boosted trees. They are **player-blind**: no batter or bowler identity enters them, so they describe the situation, not the people in it.

| Model | Predicts | Inputs |
|---|---|---|
| Win probability, 1st innings | Chance the batting side wins | Score, wickets, balls used |
| Win probability, 2nd innings | Chance the batting side wins | Runs needed, balls left, wickets, required rate |
| Outcome | Probability of each result: 0, 1, 2, 3, 4, 6, bowler wicket, run-out, wide/no-ball | Over, wickets, competition, innings, ball in the over, runs needed, required rate, batting positions and balls faced of both batters, wickets in the last two overs, partnership balls, pace or spin |
| Remaining runs | Runs still to come in the innings | Balls used, wickets, competition, batting positions and balls faced |

Every ball's predictions come from models trained **without that ball's match** (5-fold cross-fitting by match). The models therefore never see the ball they are scoring.

**A fix we had to make.** An earlier version included the current batters' runs so far in the innings among the outcome-model inputs. That raises the expected score for a batter who has started fast. A batter like Abhishek Sharma was then measured against a bar set by his own earlier boundaries, and was under-rated. The inputs now contain no runs already scored by the batters at the crease. The change matters most for fast starters, who were being penalised for their own form.

**The value of a ball** is made of three parts, all in runs:
- **Scoring:** runs off the bat minus the expected runs from the outcome model.
- **Wickets:** a bowler-credited dismissal is worth the expected rest-of-innings runs after a dot ball minus those after a wicket. Both are measured *after* the ball, so the ball's own runs are not counted twice. Run-outs are credited to no one, because they are not clearly the batter's or the bowler's doing.
- **Extras:** wides and no-ball runs count against the bowler only.

Summed over a player's pressure balls, this gives the "runs above expected" (batters) and "runs saved" (bowlers) in the tables.

---

## 5. Comparing players fairly: the matchup network

Raw "runs above expected" still flatters a batter who faced weak bowling, and a bowler who bowled to tail-enders. Every ball is a meeting between one batter and one bowler. The 184,930 balls form a connected network of 24,054 batter–bowler pairs. **We fit all batters and all bowlers together in one model over that network.** A batter's rating is then adjusted for the quality of the bowlers he actually faced, and the other way round.

**What the joint model contains**
- **Scoring:** runs on each legal ball = expected runs + batter effect − bowler effect + match effect. Each player's effect also has a pressure slope, which allows the player to be better or worse as pressure rises.
- **Dismissal:** same structure on the log-odds of a bowler-credited wicket.
- **Extras:** a bowler-only effect on wide and no-ball runs.
- **Competition-season levels:** a separate baseline for each season of IPL and of T20I. Runs against a weaker T20I side are therefore not worth the same as runs in the IPL. This is how the IPL–T20I difference in strength is handled: it is estimated from the data, not set by a fixed weight.
- **Match effects:** pitch and conditions on the day, so a batter is not credited for a road and a bowler is not credited for a turner.
- **Partial pooling:** player effects are drawn from a shared distribution. A player with few balls is pulled towards the average and only moves away from it as evidence builds up. Players with fewer than 30 balls are pooled into one group.

**Fitting.** Approximate maximum likelihood by an EM algorithm with Laplace approximation. Every fit was run to strict convergence: parameter change below 10⁻⁴, with no early stop. Fit size: 4 effects per batter, 5 per bowler, 2 per match, on 184,930 balls.

**From effects to a rating.** Each player's effects are turned into runs per ball over the same set of 2,000 real pressure situations. Everyone is therefore scored on the same situations, not just their own. The ranking uses the **10th percentile** of each player's posterior rating rather than the average. This is a cautious rating: a player ranks high only if the model is confident he is good, not just if his best guess is good.

**Two fits, not one.**
- **Full model:** fitted on every ball, with pressure as a smooth slope. A player's pressure rating borrows strength from his usual level.
- **Pressure-only model:** the same model fitted only on that side's pressure balls. The rating comes from pressure evidence alone.

Each fit has a weakness:
- The full model can lift a player who is excellent in ordinary play but only average when it matters.
- The pressure-only model is noisier, because it uses fewer balls.

The model ranking is the **average of a player's rank in the two fits**. A player has to rank well in both to reach the top. A third, leverage-weighted variant was also tried. Its variance estimates collapsed under weighting and it could not be brought to a converged fit, so it is not used.

---

## 6. Who is eligible

A rating built on a handful of balls is not reliable. These rules apply to every player.

| Rule | Batters | Bowlers |
|---|---|---|
| Effective pressure evidence (balls weighted by pressure, match-clustered) | ≥ 15 | ≥ 25 |
| Pressure balls | ≥ 100 | ≥ 100 |
| Seasons with pressure balls | ≥ 2 | ≥ 2 |
| Different opponents | ≥ 5 | ≥ 5 |
| Effective IPL pressure evidence | ≥ 5 | ≥ 8.3 |
| Share of pressure evidence from the IPL | ≥ 1/3 | ≥ 1/3 |
| Played a T20 in the 12 months to 1 Nov 2026 | yes | yes |

68 batters and 58 bowlers pass. The lists are for an IPL franchise, so the two IPL rules make sure every name has a real IPL pressure record, not just T20Is against weaker teams. The one-third share rule was added after an early ranking had been seen. We say so plainly. `results/eligibility_audit.csv` lists every candidate, whether they pass with and without that rule, and which rule excluded them.

---

## 7. From the model ranking to the final ten

The model ranking is the starting point. We then looked at the evidence behind each name in four ways:
- performance in the IPL and in T20Is separately;
- performance in high-stakes games;
- recent form;
- season-to-season consistency.

Where a model placement did not stand up, we changed it. Every change is listed here and in `src/select_final.py`. Nothing else was moved.

| Change | Model rank | Reason |
|---|---|---|
| Andre Russell out | 4 | No T20 since July 2025 (availability rule). His strong pressure record is from 2022–24 and only 125 balls. |
| Nicholas Pooran out | 10 | 17.6 runs per 100 pressure balls *below* expected in T20Is, over 197 balls. |
| Cameron Green out | 11 | Below expected in T20Is (−3.5), and a weaker IPL record (+10.5) than the players who replace him. |
| Dewald Brevis not added | 12 | Excellent rate (+50 per 100) but only 106 pressure balls. Too little evidence for a reliability list. |
| Sanju Samson in at 9 | 21 | Above expected in both competitions (IPL +23.4 over 200 balls, T20I +1.2), +10.9 in high-stakes games, +18.7 in chases. |
| Travis Head in at 10 | 26 | Above expected in both (IPL +32.3, T20I +9.1), positive in 5 of 7 season blocks, +23.1 in chases. |
| Wanindu Hasaranga out | 10 | Barely better than average under pressure in T20Is (+0.06 runs saved per over) and no IPL in 2026. |
| Krunal Pandya not added | 11 | No T20I pressure record to test against; death economy 10.48. |
| Maheesh Theekshana in at 10 | 16 | Saved runs under pressure in all 9 of his season blocks: +1.56 IPL, +1.37 T20I, +1.54 in high-stakes games over 275 balls. |
| Yashasvi Jaiswal to 8 | 6 | His T20I figure (+56.5) rests on only 37 pressure balls. His IPL record (+10.8 over 258 balls) is the weakest of the top eight, and he was flat in 2026 (−0.4). |
| Jasprit Bumrah to 1 | 3 | The top three bowlers are a statistical tie in the model. Bumrah has the best rate (1.99 runs saved per over), the best in high-stakes games (2.43) and saved runs in every season block. |

Players who were close and checked but not chosen:
- **Batters:** Virat Kohli was 2.8 runs per 100 below expected under pressure, and 10.6 below in 2025–26. He lasts (47 balls per dismissal) but does not score above the situation.
- **Bowlers:**
  - Pat Cummins is positive in both competitions, but on only 187 balls, with a death economy of 11.14.
  - Matheesha Pathirana is 1.17 runs per over worse than average in T20Is.
  - Nathan Ellis is positive in both competitions. He misses the one-third IPL rule by a hair (31.7%), and the rule was applied to him as to everyone else.

---

## 8. What makes this different

We read ten public entries to this round. Most score pressure with a fixed formula: weighted sums of required rate, wickets lost and phase, with a multiplier for knockouts. They then shrink small samples towards the average. This entry differs in five ways.

1. **Opponents are modelled, not ignored.** Every batter and bowler is fitted jointly over the matchup network, so ratings are adjusted for who a player actually faced. Most entries compare each player with a league-average baseline.
2. **Stakes are simulated, not assigned.** The value of a league game comes from simulating the rest of the season. A must-win group game counts like a knockout, and a dead rubber does not. Other entries use a flat bonus such as ×1.12 or +12 points for playoffs only.
3. **Batting and bowling pressure are different numbers.** Each is the risk of losing for that side. This is why a bowler defending a small total at the death, which is Bumrah's and Narine's job, registers as high pressure even when few wickets fall.
4. **Every name is IPL-tested.** Several entries rank T20I-only players from associate nations at the top of their lists. The IPL evidence rules keep this list to players with a real IPL pressure record, and the evidence checks also ask for a T20I record.
5. **Agreement and honesty over a single score.** A player must rank well in two different fits. Every judgement made after the model is written down with its evidence. Persistence over time is tested and reported, including where it is weak.

---

## 9. How far to trust it

**The top of each list is solid; the bottom half is a close race.**
- **Solid:** Tim David and Suryakumar Yadav are 1st and 2nd in both fits. Bumrah, Hazlewood and Rashid are top four in both fits.
- **Close:** from about 5th to 20th the differences between players are smaller than the uncertainty in their ratings. A different but reasonable choice could swap several of these names.

**Does pressure form repeat?** We measured pressure performance on 2022–24 and compared it with 2025–26, for players with at least 60 pressure balls in each period (53 batters, 53 bowlers):

| | Rank correlation, early vs late | Best quarter in 2022–24 still above the median in 2025–26 |
|---|---|---|
| Bowlers | 0.30 | 79% |
| Batters | 0.05 | 71% |

Bowling under pressure repeats moderately well. Batting under pressure is much noisier from one period to the next: who is "clutch" this season says little about next season once ability is accounted for. This is the main reason the method leans on long records, consistency across season blocks and agreement between two fits rather than on one strong season. It is also why we would not trust any list, ours included, that claims to rank batters 5–15 precisely.

**Known limitations**
- No T20 World Cup matches are in the supplied data, so high-stakes T20I evidence comes from bilateral series and Asia Cups.
- Net run rate is not modelled in the stake simulation. Ties at the top-four cut-off are broken at random.
- Rain-revised matches are excluded, not modelled.
- Run-outs are credited to no one.
- Fielding and captaincy are not measured.
- The one-third IPL share rule was set after an early look at the rankings (section 6).
- Five placements come from the evidence checks rather than straight from the model (section 7). All are listed with their evidence.

---

## 10. Reproducing the results

```
pip install -r requirements.txt
python src/s1_data.py        # clean data, build pre-ball match states       -> work/
python src/s1b_stakes.py     # match stakes by season simulation             -> work/stakes.csv
python src/s2_models.py      # player-blind models, cross-fitted ball values  -> work/
python src/s5_main.py        # joint matchup model on all balls + eligibility -> results/eligibility_audit.csv
python src/quick_final.py    # full-model ratings                             -> results/model_full/
python src/pressure_only.py  # pressure-only fits and ratings                 -> results/model_pressure_only/
python src/select_final.py   # model ranking, availability, evidence checks   -> results/final_*.csv
python src/evidence.py       # evidence tables, out-of-time check, dashboard data
```

The joint-model fit in `s5_main.py` is the slow step; it uses a CUDA GPU through PyTorch when one is available. Every random step uses the fixed seed 20260718.
