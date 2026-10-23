# V3 reproducibility and experiment-contract preflight

Install this package's runner, contract and evidence/ directory in the current v3 project root. It adds files only. Run from that root, retaining configs/config.yaml and the shared processed-folder helper. Dependencies: Python >=3.10, PyYAML, joblib. Torch is not imported or executed; installed versions are recorded through package metadata.

```python
INPUT_DIR = f"{PROJECT_ROOT_DIR}/data/processed/CU/input_states/cu_input_state_v1/train/20261021T122002Z_4ce2146c"
COMPARISON_DIR = f"{PROJECT_ROOT_DIR}/data/processed/CU/v2_validation/transition_optimization/20261023T075812Z_07d5ca16"
RESIDUAL_DIR = f"{PROJECT_ROOT_DIR}/data/processed/CU/v2_validation/frozen_residual_transition/20261023T084132Z_23cb92e4"

!python -m run_v3_reproducibility_preflight \
  --project_root_dir "$PROJECT_ROOT_DIR" \
  --input_dir "$INPUT_DIR" \
  --comparison_dir "$COMPARISON_DIR" \
  --residual_dir "$RESIDUAL_DIR"
```

Use the actual saved run directories if relocated. Pass directories containing comparison_report.json and residual_report.json directly, not an enclosing extraction directory. No new checkpoint is required: the runner verifies the original initial_model.pt and all output files against the saved experiment reports. It does not deserialize checkpoints.

Outputs: a new unique directory below data/processed/CU/v2_validation/v3_reproducibility_preflight/. No inputs are overwritten. Rough work bound: one streamed hash pass over the 25 prepared shards, all files declared by the two small bounded experiments, pinned source and evidence files, followed by an unchanged-file recheck. No raw CSV scan, tensor forward, backward, or optimizer step. SHA256 checks are integrity checks, not authenticity signatures.

## Meaning of the result

- BLOCKED (exit 2): a required file, hash, source link, relevant configuration, or split check failed, or an exception occurred.
- PASS_WITH_LIMITATIONS (exit 0): saved-artifact and source contract checks passed. This is not permission to start optimization and does not certify numerical replay, complete raw-to-experiment reproducibility, learned representation quality, behavior discrimination, or Level 2 success.
- Runtime equality is reported separately. A mismatch requires a fresh bounded numerical compatibility gate before optimization. Even equality of version strings does not prove numerical reproducibility.

This is an explicitly frozen reference contract, not a general validator for arbitrary new experiments. Do not regenerate its reference hashes to make changed artifacts pass. Future methodological changes need a new named contract and baseline comparison.

## Explicit boundaries

Text source accepts only exact or CRLF-to-LF normalized hashes, with both reported. All scientific artifacts require exact bytes. Source-run links connect the core modules to executed reports; the configuration is compared structurally after excluding relocated project root and two unused generic settings: models and hdeg.e2e_training. Their differences are retained in the report. Fixed diagnostic budgets are anchored by the original reports, protocol files and split, not those generic epoch settings.

The provided v3 still omits the upstream preparation entry points. Their absence is recorded; the older scalar trainer and generic loader are unsupported here. This package does not restore or modify them. Prepared-data provenance does not settle source freshness/backfill concerns. The current experiment probes have been inspected repeatedly and remain development data.

The included analytical-baseline result and evidence note are frozen records. Their hashes are checked, but this runner does not recompute their statistics. Previous scalar-input Level 1 evidence remains historical.

## Research direction agreed on 23 October 2026

The original HDEG methodology may change when justified by evidence. The scientific goal is useful representations capable of distinguishing user behaviors. Forecasting loss alone is not sufficient evidence of this property. Before a future methodological experiment, define what counts as behavior (rather than identity, chronology or device availability), its independent ground truth, intended invariances, legitimate evaluation split, baselines and success criterion. Do not use the already-inspected probes as untouched validation. Neither supervised identity learning nor an architecture replacement is selected by this preflight.

## Local verification

The supplied v3 snapshot and saved completed experiment outputs produced PASS_WITH_LIMITATIONS. Runtime differences and incomplete raw-to-experiment packaging were explicitly retained. Additional negative checks confirm that a changed source file and missing checkpoint block the report. No training was run.
