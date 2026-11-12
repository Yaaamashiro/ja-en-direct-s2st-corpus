# Input manifest I/O report

This report records the implementation on `feature/kokoro-tts`, based on commit `8bf0ca8`.

## Changes

- `input_index.py` adds a local SQLite index shared within each process. It stores the ordinal, pair ID, shard, and full JSON record, and hashes the original bytes as it reads the source JSONL.
- `external.py` caches export validation and run identity. `run_input_sha256` reuses the index digest. Smoke-test records are checked against the index.
- `runner.py` uses the index to select shards, resume completed shards, recheck, and plan for external profiles. The legacy Qwen/JESC/KFTT path uses its existing API.
- `cli.py` initializes caches on startup. Status queries pair IDs with `SELECT pair_id ... WHERE shard=?` to avoid rescanning the source manifest for each completed shard.
- `release.py` copies the validated input index through local SQL and joins the QC results while retaining ordinal order and all integrity checks.
- The notebook sets the index root to `/content/s2st-input-index` and retains the explicit production approval.

Initial validation reads the source JSONL once. Record checks run against the local index, and all shards in the same CLI process share the result. Each shard checks the manifest and summary for changes using size, mtime, ctime, and inode. The process stops if a source changes.

After verifying QC results and staging the release files, `consolidate` rechecks the SHA-256 digests of the source manifest and summary before publication. Smoke runs also check the selected subset. Standalone `run-shard` and `run-next-shard` perform the same final verification. Each new CLI process builds its own index.

A typical `run-remaining-shards` reads the entire source manifest twice, once for indexing and once for final verification. This removes the repeated reads equivalent to 512 full passes in a 128-shard run. Memory holds roughly one shard and the index stays on local disk. The CLI removes the temporary index it owns at exit. If a forced disconnect leaves an index behind, subsequent processes build a fresh one.

## Local validation

Local pytest reported **97 passing tests** and two existing tar-extraction deprecation warnings.

The 12 added cases covered the following checks.

- Fetching 128 shards, checking identity, and planning opens the source manifest once.
- Status with 100 completed shards opens the source manifest once.
- Changes to either the manifest or summary during a run are rejected.
- Final SHA-256 verification catches simulated changes that file attributes miss.
- The source manifest is read twice, including final verification.
- Smoke-test shard reassignment and record comparison reuse the local index.
- A cache reset equivalent to a CLI restart triggers validation and indexing again.
- Shard index bounds and maximum pair counts are enforced.
- Raw SHA-256 hashing accounts for CRLF line endings and blank lines.
- A source change during indexing is rejected and the failed temporary index is cleaned up.
- Resetting only the index still rejects a mismatch with an existing validation digest.

Existing tests also passed for canonical order, split releases, the publication marker, WAV format and duration, SHA-256, provenance, resuming, stratified benchmarking, and Qwen. The implementation preserved input text, train/dev splits, voices, model revisions, and QC thresholds.

## Validation pending at the time

This change was tested before validating the real JParaCrawl export, running the 10-, 100-, and 1,000-pair tests on Colab T4, or producing 300,000 pairs. Quality, RTF, and compute-unit usage for real data remained to be measured.
