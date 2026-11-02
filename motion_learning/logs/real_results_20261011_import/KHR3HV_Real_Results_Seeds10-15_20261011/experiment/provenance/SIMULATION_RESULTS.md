# Original simulation-results handoff: English guide

The archived Webots distance experiment trained 24 policies for 100,352 steps each and evaluated 240 episodes. All reached 20 cm. Common-R2 mean predicted energy was 106.975, 82.203, 81.316 and 79.989 J for Task-only, R1, R2 and R3. Mean times were 12.167, 9.483, 9.417 and 9.383 s. Energy comparisons against Task-only favored all three conditions in all six seeds, with Holm-adjusted p = 0.046875, but the combined criterion failed because some episodes exceeded the 10% OOD limit.

These are simulation predictions, not measured hardware energy. Shorter execution contributes strongly to the difference. The later physical records and their limitations are described in the [returned-results guide](../../README.md). The repository-level motion-learning results report gives the complete English simulation results. The original Japanese handoff report remains unchanged for its archived hash.
