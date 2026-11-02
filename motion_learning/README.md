# Energy-aware motion learning and hardware replay

This directory contains the paper's **20 cm distance task**, not the later foot-lift experiment. The learned gait may include foot sliding; the archive is not a guarantee of lifted-foot walking.

From the repository root:

```sh
python tools/reproduce.py rl
python tools/reproduce.py replay
```

| Read first | Evidence |
|---|---|
| [Experiment plan](docs/DISTANCE_GOAL_EXPERIMENT.md) | Four reward conditions, six training seeds, 100,352 steps, evaluation/gate definitions |
| [Simulation results](docs/distance_goal_20261011_RESULTS.md) | 240 evaluations, common-R2 comparisons, OOD and failed combined criterion |
| [Returned hardware recordings](logs/real_results_20261011_import/KHR3HV_Real_Results_Seeds10-15_20261011/README.md) | The 24 actual single replays and measurement limitations |
| [Original replay/acquisition package](hardware_transfer_distance/README.md) | Command format and historical preparatory procedures |

The initial 12- and 72-trial proposals are retained as history. They are not the sample size of the reported hardware experiment. Raw records and the frozen training source retain their original bytes. English companion documentation and translated labels are indexed in [archive notes](../docs/ARCHIVE_NOTES.md).
