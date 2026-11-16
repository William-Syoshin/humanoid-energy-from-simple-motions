# Reporting problems and proposing changes

Use the repository's Issues page after publication to report a reproduction problem. Include the release/tag or commit, operating system, Python version, command, and the error or generated `report.json`. Do not include credentials or unrelated private files.

For an analysis change, describe the affected paper result and run the corresponding `tools/reproduce.py` action. Keep raw measurements, frozen model files, original source manifests, and their line endings unchanged. Put new outputs in `generated/` and do not commit them as replacements for recorded results.

Documentation corrections can be proposed separately from numerical or protocol changes. Changes to archived scientific artifacts require a new, explicitly documented version. Run `python tools/check_repository.py` before proposing a change. Regenerate `checksums.json` only after an intentional reviewed update with `python tools/create_checksums.py`.

Authorship, citation metadata, and reuse-license decisions belong to the research authors. The preparation template does not set those decisions on their behalf.
