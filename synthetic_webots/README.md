# Supporting synthetic-power experiment

The current manuscript uses **revision_02**, with seven actions and three initial perturbation seeds per action. All 21 tests pass the fixed sustained mechanical criteria. This experiment is separate from physical R0-R3 prediction and PPO motion learning.

- [Revised experiment, source, data and reproduction instructions](runs/revision_02/README.md)
- [Frozen design](runs/revision_02/design.json)
- [All revised-run recordings](runs/revision_02/raw/)
- [Saved analysis](runs/revision_02/analysis/)
- [Sustained mechanical outcomes](runs/revision_02/revision_report/motion_audit.json)
- [Historical original experiment, retained unchanged](runs/full_v2/)

From the repository root:

```sh
python tools/reproduce.py synthetic
# Historical original results only:
python tools/reproduce.py synthetic_original
```

Both commands operate offline. The revised command verifies source/data hashes, model fits, numerical summaries, sustained mechanical outcomes and supplementary time-window results. The power reference remains synthetic; M4 directly represents its dominant terms. It is not a validation of physical electrical power. Known original evaluation seeds were reused, so the revised result is not a new blind test.
