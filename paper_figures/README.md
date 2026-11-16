# Current manuscript figures

The eleven PDFs `fig01_...` through `fig11_...` are byte-exact copies of the manuscript dated 2026-11-10. Their image quality and vector content have not been reduced. See the [figure map](../docs/FIGURE_MAP.md) for numbering and data sources.

`python tools/reproduce.py figures` rebuilds current Figures 4-7 (45 physical executions, Clap/Bow examples, and the updated coverage analysis) into a new `generated/` directory. The supplied manuscript files are not overwritten. Helvetica is used when available, otherwise DejaVu Sans; numerical content is unchanged.

`historical_20261102/` holds the superseded figures for traceability. `archived_build_scripts/` preserves earlier builders with historical paths; it is not the portable entry point.
