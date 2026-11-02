# Supporting synthetic-power experiment

This experiment checks input features against a synthetic power reference defined from Webots observations. It is separate from physical R0-R3 fitting and the PPO application.

Run `python tools/reproduce.py synthetic` **from the repository root**. It checks raw hashes and refits M1-M5 from stored data; it does not launch the simulator.

- [runs/full_v2/design.json](runs/full_v2/design.json): 1,000 training, 90 validation, 21 test, and three separate control trials.
- [runs/full_v2/raw/](runs/full_v2/raw/): 1,114 original NPZ recordings.
- [runs/full_v2/analysis/](runs/full_v2/analysis/): saved model/trajectory summaries and mechanical outcomes.
- [src/protocol.py](src/protocol.py): synthetic reference and feature definitions.
- [analysis/analyze.py](analysis/analyze.py): portable offline fitting and evaluation.
- [archived_source/](archived_source/): original collection-time source; retained for audit, not the recommended analysis entry point.

All mechanical failures remain in the electrical evaluation. M4 directly represents the dominant terms used to construct the reference; its low error is not proof of a calibrated physical power law. Historical Japanese report strings in archived code are explained in [archive notes](../docs/ARCHIVE_NOTES.md). The current offline entry point produces English/structured outputs.
