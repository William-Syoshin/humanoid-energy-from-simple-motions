# Numerical paper results

Current physical prediction results are in [repeated_evaluation_20261109/verified_data](../hardware_prediction/repeated_evaluation_20261109/verified_data/) and [analysis](../hardware_prediction/repeated_evaluation_20261109/analysis/), covering nine motions x five recordings.

- [Synthetic results](synthetic_webots/): adopted revision_02 (21/21 sustained mechanical checks), with original results under `historical_full_v2/`; `python tools/reproduce.py synthetic`.
- [Hardware command replay](rl_application/): unchanged 24 policy replays; `python tools/reproduce.py replay`.
- [Reviewer guide](../docs/REVIEWER_GUIDE.md): current expected values and statistical units.

Do not pool physical repetitions, synthetic trials, policy seeds or simulated evaluation episodes.
