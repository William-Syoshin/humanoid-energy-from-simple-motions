# Predicting Whole-Body Electrical Energy from Simple Motor Experiences

Code, recorded measurements, trained models, and figures accompanying the humanoid energy-prediction study. Models learn from **116 simple-motion trials**, use **23 validation trials**, and are evaluated on **nine unseen whole-body movements, each executed five times (45 recordings)**. A separate application uses the predictors in motion learning and measures the energy of replayed commands on hardware.

**For reviewers:** start with the [reviewer guide](docs/REVIEWER_GUIDE.md), which maps each main result to its data and verification command. No robot, simulator, GPU, or external dataset is required for the offline analyses.

## Start here

| What you want to do | Where to go |
|---|---|
| Check the main claims and limitations | [Reviewer guide](docs/REVIEWER_GUIDE.md) |
| Reproduce the reported numbers | [Reproduction instructions](docs/REPRODUCTION.md) |
| Inspect acquisition, preprocessing, and analysis settings | [Detailed methods](docs/MANUSCRIPT_METHODS.md) |
| Inspect the synthetic power equation and simulation settings | [Simulation reference](docs/SIMULATION_REFERENCE.md) |
| Locate each figure and table | [Figure and table map](docs/FIGURE_MAP.md) |
| Understand splits, raw data, and exclusions | [Data and scope](docs/DATA_AND_SCOPE.md) |
| Read the English experiment reports | [Document index](docs/INDEX.md) |
| Understand archived labels and Japanese source records | [Archive notes and English labels](docs/ARCHIVE_NOTES.md) |
| Prepare the public GitHub release | [Release instructions](docs/RELEASE.md) |

## Current physical evaluation (2026-11-09)

The primary test set is [the 45-recording dataset](hardware_prediction/repeated_evaluation_20261109/README.md). Bow replaces Supine get-up in this set; the two sets must not be pooled. Raw files, frozen model parameters and the original acquisition manifest are preserved. [Update details](docs/UPDATE_45_TRIALS.md) distinguish current results from the historical nine-recording analysis.

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

The hardware check verifies the original 116/23 training/validation split and refits the original models only to check their frozen coefficients. It then reconstructs all 45 held-out recordings without adapting the models to the test data. Expected mean absolute relative energy errors are **9.41%, 8.67%, 4.06%, and 3.14%** for R0-R3, respectively. Regularization is chosen using validation power RMSE separately for each model; all four frozen models are evaluated on the test set. R3 has the lowest mean test energy error.

For all offline verification, analysis and plotting actions:

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
paper_figures/           Eleven current manuscript figures (PDF); historical figures separated
tools/                   Portable offline analysis and repository checks
verification/            Recorded verification results and environment versions
provenance/              Source-copy and translation records
licenses/                Existing third-party license material
```

Each experiment directory has its own English README. Archived paths are preserved so that file hashes, model identities, and recorded trials remain traceable. The portable tools resolve paths relative to this repository; historical absolute paths are not installation instructions.

## Results and scope

This repository includes the 45-recording physical evaluation and the adopted [revised supporting simulation](synthetic_webots/runs/revision_02/README.md), with 21/21 sustained mechanical successes. Original simulation data remain archived. The packaged eleven-figure map predates the later Bow/Kick illustration; its two simulation PDFs have been updated to revision_02.

| Study | Included evidence | Interpretation |
|---|---|---|
| Physical energy prediction | 116 training, 23 validation, 45 test recordings (9 motions x 5); R0-R3 | Mean energy error decreases from 9.41% (R0) to 3.14% (R3); improvements vary by movement and do not guarantee accurate power peaks. |
| Supporting simulation | 1,000 training, 90 validation, 21 test, three separate controls; M1-M5 | The power reference is synthetic, not measured electricity. All 21 revised test executions meet fixed sustained mechanical criteria. The original 9/21 and intermediate 20/21 results remain distinguishable; evaluation seeds were reused. |
| Energy-aware motion learning | 24 PPO policies, six training seeds per condition, 240 simulated evaluations | Common-R2 predicted energy is lower, but the combined primary criterion was not met because some episodes exceed the OOD limit. |
| Physical command replay | 24 recordings, one per policy | Mean measured energy is about 20% lower for R1-R3, mainly with shorter replay duration. Physical distance was not measured; this is not an equal-distance efficiency test or online hardware PPO deployment. |

Later foot-lift/CEM experiments, earlier 20-second models and unrelated pilots are excluded. Superseded test data and figures are explicitly marked historical. Full details are in [data and scope](docs/DATA_AND_SCOPE.md). The paper itself is not bundled; its identifying metadata is in [PAPER_SNAPSHOT.json](docs/PAPER_SNAPSHOT.json).

## Citation, license, and contributions

The implementation is public at [GitHub](https://github.com/William-Syoshin/humanoid-energy-from-simple-motions). The repository includes all 45 physical test recordings, the revised 21-trial supporting simulation, their offline analyses, and detailed method documentation. Use the full commit hash of this checkout (`git rev-parse HEAD`) to identify this version. No release tag or DOI has been assigned. **The authors still need to complete citation-author metadata and the reuse-license decision.** [CITATION.cff.example](CITATION.cff.example) is an inactive template; no citation metadata is invented. See [release instructions](docs/RELEASE.md) and [license decision](LICENSE_DECISION.md).

Third-party terms remain in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). A blanket license has not been applied to vendor code, Webots assets, data, and figures. For reporting a reproducibility issue, see [CONTRIBUTING.md](CONTRIBUTING.md).
