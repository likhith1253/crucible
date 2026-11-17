# CRUCIBLE

**T20 Performance Intelligence** — *Pressure reveals performance. Data reveals the truth.*

CRUCIBLE ranks the IPL and T20I batters and bowlers who perform most reliably when a match is on the line. It was built for Round 1 of the SupeRR Selector Season 2 challenge.

## Rankings

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

## Method in brief

- **Pressure:** the win probability a side stands to lose on a ball, multiplied by what the match is worth for the season. Must-win league matches count as virtual knockouts. The top 20% of balls are pressure balls.
- **Performance:** each ball compared with what player-blind models expect from the match state, cross-fitted by match.
- **Fair comparison:** every batter and bowler is fitted jointly in one hierarchical model over 24,054 matchups, with match conditions and competition-season strength estimated from the data.
- **Ranking:** two converged model fits (all balls, pressure balls only), reviewed against the detailed evidence. The result is checked against four alternative definitions of reliability.

The full method, evidence, validation and limitations are in **[REPORT.md](REPORT.md)**. A formatted PDF version with dashboard screenshots and reproduction steps is in **[CRUCIBLE_Report.pdf](CRUCIBLE_Report.pdf)**.

## Dashboard

Live: **https://likhith1253.github.io/crucible/**

You can also open `dashboard/index.html` locally. It is a single static page and needs no build step.

## Repository

| Path | Contents |
|---|---|
| `data/` | Supplied ball-by-ball, match, competition and squad files |
| `src/` | Pipeline, run in the order below |
| `results/` | Final lists, model rankings, evidence tables, method comparison, eligibility audit |
| `dashboard/` | Interactive dashboard (`index.html`, `data.js`) |
| `REPORT.md` | Technical report |

## Reproducing

```
pip install -r requirements.txt
python src/s1_data.py
python src/s1b_stakes.py
python src/s2_models.py
python src/s5_main.py
python src/quick_final.py
python src/pressure_only.py
python src/select_final.py
python src/evidence.py
python src/robustness.py
```

Intermediate files go to `work/`, which is not tracked. The joint-model fit uses a CUDA GPU when one is available. All random steps use the seed 20260718.
