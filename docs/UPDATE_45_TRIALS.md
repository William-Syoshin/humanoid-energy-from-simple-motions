# Update: 45 physical test executions

Prepared 2026-11-10 and verified for publication on 2026-11-13. This update adds the 45-recording evaluation to the original implementation at `7b144085ab47673770eac49baea1d5a8d524a46c`. The commit containing this update identifies the published dataset and scripts; retrieve its full hash with `git rev-parse HEAD`.

The first publication of this dataset was commit `4d1d720e943043c1157e20b1d870f74263b2ec57`. A subsequent update on 2026-11-13 adds [revision_02 of the supporting simulation](../synthetic_webots/runs/revision_02/README.md), its 21 test recordings, frozen controllers and offline analysis. The 45-trial physical data and primary models are unchanged. Bow/Kick posture-error illustrations and the latest twelve-figure layout remain outside this figure package.

## Changes

- Added all nine motions x five physical executions, original device streams, frozen model copies, acquisition metadata, trace tables and archive hashes.
- Switched default hardware, window, context/coverage and figure reproduction to these 45 recordings. No primary model is refitted on test data. Original training116/validation23 data and parameter values are retained.
- Updated mean energy errors to R0 9.41%, R1 8.67%, R2 4.06%, R3 3.14%. The old 10.52%/2.59% comparison belongs only to the historical test set.
- Added within-motion sample SD and variance; kept between-motion SD distinct. Bow replaces Supine get-up in the new test set.
- Updated waveform illustrations to Clap repeat3 and Bow repeat3, selected by minimum R3 power RMSE within each motion. All 45 recordings remain in aggregate analyses.
- Updated current artwork and figure numbering (11 figures). Supplied vector PDFs were copied without recompression.
- Kept all original single-execution data, historical figures, source code, and original acquisition records for provenance. The current reading path and default commands point to the repeated dataset.
- At preparation on 2026-11-10, the synthetic benchmark was unchanged; the follow-up publication adds revision_02 separately while retaining the old run. PPO training/evaluations and the 24 physical command replays remain unchanged. New CEM/foot-lift experiments are not part of this paper package.

## Verification

The preparation audit is [verification/repeated_update_20261110.json](../verification/repeated_update_20261110.json). It records executed checks, the current archive identity, and model invariance. Earlier reports under `verification/` describe the original release and remain historical.

The source archive SHA-256 is `c0566b3d6eaa46ab4df564345900d9f7bd54dc846bbde68767de15c854f7b709`. The frozen model SHA-256 is `fb9d381ec3b142353725c83f274fad181b509ef9e5a2916d70954fa43b6c1d5e`.

## Publishing this update

Use the existing repository and remote; do not repeat `git init` or create another remote. Review `git diff` and `git status`, run the checks in the reviewer guide, commit the update, then push the intended branch. A manuscript should cite the verified public commit hash after pushing. Do not invent a release tag or DOI.

Original-language raw acquisition documents remain byte-exact; English reviewer documentation is provided separately. Author/citation and reuse-license metadata that were pending in the original release still require the authors' decision.

## Manuscript consistency review (2026-11-10)

Figure 7 now explicitly labels the 45-recording evaluation set. Its numerical curves are unchanged from the verified repeated-data analysis. Training-subset IQR is distinguished from physical repeat variability. The manuscript clarifies that exploratory models are fitted only to training data, the current test set contains Bow, and the remaining single-replay limitation concerns the learned-command application. Abstract and primary result values already described 45 executions and remain unchanged. At that preparation date, the update was unpushed; the original audit retains this historical status. A fresh hardware reconstruction was performed for publication on 2026-11-13.
