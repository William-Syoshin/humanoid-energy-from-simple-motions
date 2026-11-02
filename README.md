# Predicting Whole-Body Electrical Energy from Simple Motor Experiences

Code, recorded measurements, trained models, and figures accompanying the humanoid energy-prediction study. Models learn from **116 simple-motion trials**, use **23 validation trials**, and are evaluated on **nine unseen whole-body movements**. A separate application uses the predictors in motion learning and measures the energy of replayed commands on hardware.

**For reviewers:** start with the [reviewer guide](docs/REVIEWER_GUIDE.md), which maps each main result to its data and verification command. No robot, simulator, GPU, or external dataset is required for the offline analyses.

## Start here

| What you want to do | Where to go |
|---|---|
| Check the main claims and limitations | [Reviewer guide](docs/REVIEWER_GUIDE.md) |
| Reproduce the reported numbers | [Reproduction instructions](docs/REPRODUCTION.md) |
| Locate each figure and table | [Figure and table map](docs/FIGURE_MAP.md) |
| Understand splits, raw data, and exclusions | [Data and scope](docs/DATA_AND_SCOPE.md) |
| Read the English experiment reports | [Document index](docs/INDEX.md) |
| Understand archived labels and Japanese source records | [Archive notes and English labels](docs/ARCHIVE_NOTES.md) |
| Prepare the public GitHub release | [Release instructions](docs/RELEASE.md) |

## Quick start

Use Python 3.11. Run these commands **from this repository's root**:

```sh
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-analysis.txt
python tools/check_repository.py
python tools/reproduce.py verify
python tools/reproduce.py hardware
```

The hardware check refits R0-R3 on the original split, reconstructs the nine held-out recordings, and checks the published errors. Expected mean absolute relative energy errors are **10.52%, 5.44%, 3.55%, and 2.59%**, respectively. R2 was selected by validation power RMSE; R3 has the lowest mean test energy error.

For all seven offline analysis/plotting actions:

```sh
python tools/reproduce.py all
```

Allow several minutes and additional disk space. Each action prints its output directory under `generated/` and produces `report.json`. A failed check raises an error and returns a nonzero exit status. The commands do not launch Webots, train PPO again, open serial ports, or overwrite recorded data. Plot typography can vary by platform. See [reproduction scope](docs/REPRODUCTION.md) for what is and is not tested.

## Repository layout

```text
docs/                    Reviewer guide, protocols, results, and release instructions
hardware_prediction/     Physical prediction: data, R0-R3 models, collection/analysis code
synthetic_webots/        Supporting synthetic-power experiment: raw data and M1-M5 code
motion_learning/         Distance-task policies/evaluations and physical command replay
paper_results/           Numerical summaries for the reported comparisons
paper_figures/           Ten manuscript figures (PDF)
tools/                   Portable offline analysis and repository checks
verification/            Recorded verification results and environment versions
provenance/              Source-copy and translation records
licenses/                Existing third-party license material
```

Each experiment directory has its own English README. Archived paths are preserved so that file hashes, model identities, and recorded trials remain traceable. The portable tools resolve paths relative to this repository; historical absolute paths are not installation instructions.

## Results and scope

| Study | Included evidence | Interpretation |
|---|---|---|
| Physical energy prediction | 116 training, 23 validation, nine test recordings; R0-R3 | Mean energy error decreases from 10.52% (R0) to 2.59% (R3); improvements vary by movement and do not guarantee accurate power peaks. |
| Supporting simulation | 1,000 training, 90 validation, 21 test, three separate controls; M1-M5 | The power reference is synthetic, not measured electricity. All failures are retained; nine of 21 test executions meet mechanical endpoint criteria. |
| Energy-aware motion learning | 24 PPO policies, six training seeds per condition, 240 simulated evaluations | Common-R2 predicted energy is lower, but the combined primary criterion was not met because some episodes exceed the OOD limit. |
| Physical command replay | 24 recordings, one per policy | Mean measured energy is about 20% lower for R1-R3, mainly with shorter replay duration. Physical distance was not measured; this is not an equal-distance efficiency test or online hardware PPO deployment. |

Later foot-lift experiments, earlier 20-second models, unrelated pilots, and superseded figures are excluded. Full details are in [data and scope](docs/DATA_AND_SCOPE.md). The paper itself is not bundled; its identifying metadata is in [PAPER_SNAPSHOT.json](docs/PAPER_SNAPSHOT.json).

## Citation, license, and contributions

This is a local release-preparation snapshot, not a claim of publication or an issued DOI. **The authors must complete the author list, repository URL, and reuse-license decision before public release.** [CITATION.cff.example](CITATION.cff.example) is an inactive template; no citation metadata is invented. See [release instructions](docs/RELEASE.md) and [license decision](LICENSE_DECISION.md).

Third-party terms remain in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). A blanket license has not been applied to vendor code, Webots assets, data, and figures. For reporting a reproducibility issue, see [CONTRIBUTING.md](CONTRIBUTING.md).
