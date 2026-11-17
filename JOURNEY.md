# Journey

When I read the Round 1 brief, my first thought was that "pressure" is usually defined too loosely. Most people take death overs plus a high required rate and call that pressure. I wanted something that would also count a collapse at 30 for 3 in the powerplay, or a must-win league game that wasn't officially a knockout.

So I started with win probability. For every ball I estimated how much the batting side, and separately the bowling side, could lose on that delivery. Then I scaled it by how much the match mattered for the season, which meant simulating the rest of each IPL table to see which games actually decided qualification. That is where the idea of virtual knockouts came from.

The next problem was fairness. A batter who mostly faced weak attacks looks great on raw numbers. I ended up fitting all batters and bowlers together in one model, so every rating is adjusted for who that player actually faced. This took far longer than I expected. The early fits were slow, some stopped before they had properly converged, and at one point I moved the work to Google Colab just to get it finished.

One bug taught me a lot. My first version judged batters against an expectation that included their own runs earlier in the same innings. That quietly punished fast starters, and Abhishek Sharma kept ranking lower than he should have. Once I removed it, the rankings made much more cricket sense.

I also went through around ten other public entries for this challenge to make sure I wasn't repeating what everyone else had done. Most used a fixed pressure formula and ignored the opposition, so I focused on the parts they didn't cover: the stakes simulation, separate batting and bowling pressure, and opponent adjustment.

Before settling on the final lists, I tested them against four other ways of ranking reliability, including win probability added and match-winning performances. Seven of the ten names on each list held up under at least four of the five methods. The rest I reviewed against the detailed evidence: IPL and T20I records separately, high-stakes games, survival under pressure, and recent form.

Finally, I built a dashboard so anyone can explore the rankings instead of just reading them.

The biggest lesson for me: in T20 cricket, consistent bowling under pressure shows up clearly in the data. Batting under pressure is much noisier. Being honest about that uncertainty mattered more than producing a perfect-looking list.

**Dashboard:** https://likhith1253.github.io/crucible/
**Code and report:** https://github.com/likhith1253/crucible
