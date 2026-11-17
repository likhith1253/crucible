# CRUCIBLE

**T20 performance intelligence.** Pressure reveals performance. Data reveals the truth.

CRUCIBLE ranks the IPL and T20I batters and bowlers who perform most reliably when a match is on the line. It was built for the SupeRR Selector Season 2 challenge (Rajasthan Royals). It does not use a weighted formula. Instead:

- **Pressure** is the win probability a side could lose on a ball, scaled by what the match means for the season. The value of each match comes from simulating the rest of the season.
- **Opponents are accounted for:** every batter and bowler is fitted together over the network of 24,054 batter–bowler matchups.

## Results

| # | Batters | Bowlers |
|---|---|---|
| 1 | Tim David | Jasprit Bumrah |
| 2 | Suryakumar Yadav | Josh Hazlewood |
| 3 | Phil Salt | Rashid Khan |
| 4 | Abhishek Sharma | Sunil Narine |
| 5 | Yashasvi Jaiswal | Bhuvneshwar Kumar |
| 6 | Heinrich Klaasen | Kuldeep Yadav |
| 7 | Rinku Singh | Axar Patel |
| 8 | Shreyas Iyer | Varun Chakravarthy |
| 9 | Sanju Samson | Mustafizur Rahman |
| 10 | Travis Head | Maheesh Theekshana |

The full method, evidence and limitations are in [REPORT.md](REPORT.md).

## Dashboard

Open `dashboard/index.html` in a browser. It is a single static page with no build step and no server. The data it shows comes from `src/evidence.py`.

## Layout

```
data/        supplied ball-by-ball, match and competition files
src/         pipeline (run in the order listed in REPORT.md, section 10)
results/     final lists, model rankings, evidence tables, eligibility audit
dashboard/   static dashboard (index.html + data.js)
```

## Running it

```
pip install -r requirements.txt
python src/s1_data.py && python src/s1b_stakes.py && python src/s2_models.py
python src/s5_main.py && python src/quick_final.py && python src/pressure_only.py
python src/select_final.py && python src/evidence.py
```

Intermediate files go to `work/`, which is not tracked.
