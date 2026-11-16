# Repeated evaluation: nine unseen motions, five executions each

This is the current physical test set for the manuscript dated 2026-11-10. It contains **45 recordings**, scored by the **original, unchanged R0-R3 models** trained on 116 simple-motion trials and selected using 23 validation trials. No test execution is used for fitting or selecting model parameters.

| Manuscript label | Archive console number | Repetitions |
|---|---:|---:|
| Work (source label: Walk) | 1 | 5 |
| Clap | 2 | 5 |
| Bow | 10 | 5 |
| Cheer | 4 | 5 |
| Kick | 5 | 5 |
| Crawl | 6 | 5 |
| Crawl turn left | 7 | 5 |
| Crawl turn right | 8 | 5 |
| Backward crawl | 9 | 5 |

Bow is the third displayed motion but retains console number 10. Supine get-up is not part of these 45 recordings. The historical nine-recording set is a different test set, not an additional repetition.

## Files

- [source/export_manifest.json](source/export_manifest.json): all 45 capture identifiers and relative raw-data locations.
- [source/recordings/](source/recordings/): current, voltage, joint states, accelerometer data and original acquisition metadata.
- [source/original_models/](source/original_models/): frozen coefficients and model metadata, identical to the original training results.
- [source/analysis/traces/](source/analysis/traces/): 45 aligned 10 Hz measured/predicted power traces.
- [verified_data/per_trial_verified.csv](verified_data/per_trial_verified.csv): 180 rows (45 executions x 4 models), reconstructed independently from raw streams.
- [verified_data/mean_sd_variance.csv](verified_data/mean_sd_variance.csv): 36 motion/model summaries, including sample SD and variance across five repetitions.
- [analysis/summary.json](analysis/summary.json): overall results, waveform selection, integration-window results, and exploratory comparisons.
- [scripts/](scripts/): portable, offline verification, statistical analysis and plotting. Run these via the root reproduction entry point.

The `source/` directory is a byte-preserved snapshot of the supplied archive, including original-language reports and acquisition notes. It is provenance, not the English reading path. The embedded `analysis_code_snapshot` has original workspace assumptions; use the separate portable scripts here. No acquisition metadata is rewritten or discarded.

## Reproduce

From the repository root after installing `requirements-analysis.txt`:

```sh
python tools/reproduce.py hardware
python tools/reproduce.py windows
python tools/reproduce.py coverage
python tools/reproduce.py figures
```

These commands create new `generated/` outputs and never open a serial port, start Webots, retrain the primary models on new data, or overwrite measurements. The hardware action refits on the original split solely to verify equality of the archived coefficients. Coverage/context analyses fit exploratory models on the original training data and retain the original validation selection and training subsets.

## Statistics and interpretation

For each execution, energy error is `100 * abs(predicted_J - measured_J) / measured_J`. Errors are first averaged within each motion (five executions), then equally across the nine motions. Overall R0-R3 errors are **9.41%, 8.67%, 4.06%, 3.14%**.

Figure 4 uses sample SD within each motion/model (`ddof=1`, n=5); Figure 5 uses sample SD across nine motion means (`ddof=1`, n=9). Neither is a confidence interval. Figure 7 bands show the interquartile range across 20 training subsets, a different source of variation.

Figure 6 shows Clap repetition 3 and Bow repetition 3, each the lowest R3 power-RMSE execution within that motion's five recordings. These are favorable illustrative examples, not random or median trials; all 45 recordings remain in the aggregates. R3 RMSEs are 0.5271 W and 0.3290 W, respectively. No extra waveform smoothing, time shifting or peak clipping is used for display.

For 2/5/10-second integration, complete nonoverlapping windows are primary, incomplete tails are excluded, and overlapping windows with a 0.1 s stride are secondary. Window errors are averaged within executions, then within motions, then across motions. Primary window counts are 628, 240 and 110. Overlapping windows are not independent repetitions.

The 21 synthetic test trials and 24 physical learned-command replays belong to separate studies. This dataset does not add repetitions to those studies or establish equal-distance hardware walking efficiency.
