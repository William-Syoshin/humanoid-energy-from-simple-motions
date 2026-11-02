# Preparing and publishing the research repository

## Upload scope

Upload the **contents of `humanoid-energy-from-simple-motions/`**, with `README.md` at the GitHub repository root. Do not upload its parent delivery directory. The parent contains private copy-provenance records, local verification outputs, and a distribution ZIP; these are not repository content. The original research archive is separate.

## Before publication

1. Confirm the complete author/rights-holder list and the terms for author-owned code, measurements, and figures. Follow [LICENSE_DECISION.md](../LICENSE_DECISION.md) and preserve [third-party notices](../THIRD_PARTY_NOTICES.md). This package does not assign a blanket license.
2. Complete [CITATION.cff.example](../CITATION.cff.example) using the actual authors, repository URL, version, and release date; save it as `CITATION.cff`. Add a DOI only after one exists. The `.example` file is not an active GitHub citation file.
3. Inspect the [reviewer guide](REVIEWER_GUIDE.md), then run:

```sh
python tools/create_checksums.py
python tools/check_repository.py --release
python tools/reproduce.py verify
```

The ordinary repository check verifies structure, links, English reading documents, and file integrity. `--release` additionally requires completed citation metadata and a license declaration; it currently reports the unresolved author decisions rather than filling them with invented values.

## Upload using Git or GitHub Desktop

GitHub Desktop or command-line Git is more practical than browser upload for the thousands of recorded data files. Create an empty repository on GitHub. With Git, run the following **inside the inner project directory**, substituting the real URL:

```sh
git init -b main
git add .
git status --short
git commit -m "Add paper code, recorded data, models, and reproduction guide"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

The URL above is a placeholder command, not an existing project address. No upload is performed by this package. Do not commit the distribution ZIP or `generated/`. `.gitignore` excludes local outputs and environments. `.gitattributes` disables automatic line-ending conversion because original files are checked byte for byte.

Individual packaged files are below GitHub's normal-Git size limit. The outer distribution ZIP is much larger than the single-file limit and belongs outside the repository (or as a release asset), not in a commit. See the [current size audit](../verification/repository_review.json).

## Freeze the version corresponding to the paper

Create a release/tag, such as `v1.0.0`, after verification. Retain that version when making later changes. Optionally archive the release through Zenodo and use its version DOI. A release identifies a fixed artifact more precisely than a changing default branch.

Once the public links exist, an availability statement can read:

> The code, recorded data, trained models, and analysis scripts supporting this study are available at [repository URL]. The version corresponding to the reported results is archived as [release version] at [version DOI].

Replace the bracketed fields with actual values. If no DOI exists, use the release URL and omit the DOI claim. Include only materials that are actually accessible. Preparing reviewer-accessible code and data before review helps readers check the reported evidence; a GitHub URL alone is not a guarantee of reproducibility.

## Official references

- [GitHub repository practices](https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories)
- [GitHub file-size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [GitHub citation files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)
- [Archiving and citing repository content](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)
- [Frontiers publication and data policies](https://www.frontiersin.org/guidelines/policies-and-publication-ethics)

GitHub guidance checked on 2026-11-02. Journal submission policies should be checked again for the actual submission.
