# Before uploading to GitHub

The directory is prepared locally; nothing has been published or pushed.

1. Confirm repository ownership, contributors, and the software license. No license has been assigned automatically. Check third-party and dataset terms separately.
2. Upload the contents of this directory as the repository root, not the full research workspace. Do not upload raw images, checkpoints, credentials, local environments, or training logs.
3. Install the package and run the README quick start. `VERIFICATION.json` records the checks performed locally; full GPU training has not been repeated for this upload candidate.
4. Decide how to provide processed data and trusted checkpoints, including download URLs and checksums where redistribution is permitted. Until those artifacts are available, do not describe the repository as a complete end-to-end reproduction package.
5. Update author/contact information and repository links after choosing the public address. Add that verified address to the manuscript's code-availability statement.
6. Create a release/tag corresponding to the submitted paper. Keep the current manuscript tables distinct from the historical exports.

`CHECKSUMS.json` covers the packaged files except itself. Regenerate it after modifying release contents. No repository URL, author identity, DOI, or license is fabricated in this package.
