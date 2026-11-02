# Archive notes and English reading path

The [document index](INDEX.md), [reviewer guide](REVIEWER_GUIDE.md), and all new experiment guides are in English. Original data, frozen source and supplier records are not silently translated in place: their exact bytes are part of the evidence checked by manifests. English companions remove the need to read those originals during review.

## English companions

- [Motion labels and selection decisions](english_metadata/README.md): English CSV views; numerical cells and row selection are unchanged.
- [Returned hardware guide](../motion_learning/logs/real_results_20261011_import/KHR3HV_Real_Results_Seeds10-15_20261011/README.md) and [English results table](../motion_learning/logs/real_results_20261011_import/KHR3HV_Real_Results_Seeds10-15_20261011/RESULTS_EN.html): the actual 24 measurements and their interpretation.
- [Returned transfer guide](../motion_learning/logs/real_results_20261011_import/KHR3HV_Real_Results_Seeds10-15_20261011/experiment/README.md): English explanation beside the archived original Japanese handoff.
- [Simulation-result handoff](../motion_learning/logs/real_results_20261011_import/KHR3HV_Real_Results_Seeds10-15_20261011/experiment/provenance/SIMULATION_RESULTS.md): English explanation beside the archived original report.
- [Source-code guide](SOURCE_CODE_GUIDE.md): where to find each model/collection/analysis component.
- [Vendor-library guide](../hardware_prediction/external/Rcb4Lib/README_EN.md): role of the original serial library, whose notices and comments remain intact.

[documentation_translations.json](../provenance/documentation_translations.json) maps original documents to English files and records hashes. [preserved_language_records.json](../provenance/preserved_language_records.json) lists remaining Japanese-bearing originals and why they are preserved. Original unprotected Japanese instructions replaced during preparation are backed up outside the upload directory.

## Historical statements that need context

1. The returned Japanese README calls the simulator MuJoCo. This is a historical naming error. The frozen source, saved Webots evaluations and command-identity audit establish Webots provenance. The English guide corrects the description without altering the original record.
2. The transfer package originally proposed 12 or 72 recordings. The manuscript uses **24 actual replays**, one per condition and training seed, in fixed order.
3. Initial `VALIDATION.json` files describe software/export checks before acquisition. They do not replace the returned per-trial configuration, provisional mapping or commissioning status.
4. Support labels are assigned context, not continuous contact measurements. In particular, Supine get-up retains the `ground` label throughout.
5. Original selection metadata includes exclusions decided after inspection. The translated views retain these statements; they are not recast as preregistered criteria.
6. `Baseline` in the returned replay archive means Task-only. `Work` is the paper's display name for the original forward-walking entry. Internal IDs remain file identifiers, not additional experimental conditions.
7. Absolute paths in original logs describe the acquisition machines. Portable offline tools use repository-relative paths. These historical strings are not credentials or required local directories.

## Original operator confirmations

Japanese operator-confirmation fields are source records, not independent sensor measurements. The returned archive reports the confirmation of forward motion for R1, seed 10. Other success claims are not added. The six leg-pitch direction checks and initial-pose checks do not establish an exact mapping of all axes. Read the [hardware guide](../motion_learning/logs/real_results_20261011_import/KHR3HV_Real_Results_Seeds10-15_20261011/README.md) for the distinction between completed logging and physical task success.

## Source and license preservation

Do not normalize line endings or replace frozen source with translated copies. `.gitattributes` disables automatic text conversion. The release checksum covers the new English documentation; experiment-level manifests continue to verify original records. Supplier terms are retained and listed in [third-party notices](../THIRD_PARTY_NOTICES.md). The author-owned license and final citation metadata are still to be completed before public release.
