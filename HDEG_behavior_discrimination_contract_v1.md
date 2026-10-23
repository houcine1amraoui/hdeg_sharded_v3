# HDEG behavior-discrimination evaluation contract v1

23 October 2026. Status: scientific scope and decision rules specified; execution blocked pending annotation alignment and coverage audit. No methodology change, model fitting, new label assignment, or final-test evaluation is authorized by this document alone.

## 1. Goal and claims

The scientific goal is to learn useful representations of observable smart-home behavior. The original HDEG architecture and forecasting objective may change when justified by controlled evidence. Forecast loss is an auxiliary diagnostic, not the definition of representation quality.

Primary development claim: a frozen learned representation makes independently logged activity categories distinguishable across separated days better than prespecified simple input baselines and the same untrained hierarchy.

Secondary claim: the representation retains this activity information across acquisition periods and, where independently matched labels and observations support it, across actors.

Separate later claim: a model fitted on resident-normal behavior distinguishes departures from that behavior while controlling false alarms in later normal periods. Activity recognition, actor classification, and anomaly detection are different tasks. Success on one does not establish the others.

Actor 2 is a different actor in a different period; this is not an unconfounded identity or anomaly ground truth for every window. A normal activity by Actor 2 must not automatically be called intrinsically anomalous. Preserve the dataset's actor-based anomaly convention explicitly when reporting that separate benchmark.

## 2. Available evidence and its limits

Inspected sources: supplied dataset documentation, Section 5.1–5.2; Misc/ActivityLogR1.csv; Misc/ActivityLogR2.csv; Misc/ActivityLabels.json; current v3 config and actor-timeline utility; current manuscript Experimental Benchmark/Dataset Preparation/Validation Hierarchy sections.

- R1: 586 records, columns StartTime, EndTime, ActID; starts as early as 2022-10-20 12:52:00 and ends as late as 2022-11-16 15:48:05. There are 29 distinct literal ActID strings. All timestamps parse and all intervals have positive duration in this initial inspection.
- R2: 69 records, columns StartTime, EndTime, ActDetails; coverage extrema 2022-11-08 10:32:56 to 2022-11-10 14:36:00. All timestamps parse and intervals have positive duration in this initial inspection.
- R1 IDs -1.0, 0.0, 1.4 and 1.5 are absent as literal keys from the supplied ActivityLabels.json. Their semantics are unresolved; do not infer meanings from their numerical prefixes.
- R2 is free text. No reviewed R2-to-R1 label mapping has been established.
- The documentation states that actors logged start/end times and activity details. This gives an annotation source independent of the learned representation; annotation completeness and accuracy remain to be assessed.
- Simultaneous labels occur in the R1 log. They cannot be resolved by silently choosing the first CSV row.
- The activity dictionary references multiple data modalities and device IDs. A label's presence does not imply that the selected 23 CU channels observe its defining action.

Record hashes, original row numbers and literal text. Never parse ActID as a floating-point class code. Verify timezone convention, timestamp alignment, duplicates, overlaps, activity-to-channel mappings and temporal coverage before producing labels.

## 3. Label and window policy

The primary task is activity-category discrimination, not identification of a person. Use original activity IDs whose meanings can be established independently. Any coarse grouping must be justified from annotation definitions and selected-channel observability, frozen before representation inspection, and versioned. Do not group classes because it improves model scores.

Treat intervals as half-open [start,end). This is an evaluation convention, not a claim about annotation precision. Preserve original timestamps. Do not shift them to maximize model agreement. Any clock correction requires independent documentation and a recorded transform.

For the first bounded development analysis, use current 30-row, one-second observation windows. A labeled window must be wholly contained in one unambiguous activity episode and wholly contained in one temporal partition. Unlabeled time is unknown, not inactivity or a negative class. Windows touching unresolved overlaps or conflicting labels are excluded from the primary single-label analysis and counted separately. Genuine simultaneous activities remain candidates for a later multilabel task.

Duplicate annotations do not create extra episodes. Group overlapping annotation intervals into connected components before splitting or counting support. Do not split an episode across fit, selection or evaluation roles. Exclude partition-crossing episodes. No source row may be shared across roles. Account for Y-window rows too if future model fitting uses paired windows.

Bound episode sampling to at most eight nonoverlapping 30-row windows, spaced deterministically across the eligible interior of each episode without looking at sensor values or predictions. Use the same windows for every method. Short or ambiguous episodes with no eligible window are excluded and counted. This deliberately limits the claim to sustained, unambiguous episodes; brief actions require a separately specified event-centered protocol.

Labels generated directly from device states, availability masks, BSE membership or transition strata are proxy diagnostics only. They cannot serve as independent evidence that those representations capture activity semantics.

## 4. Data roles and exposure

Do not reuse the frozen 256 transition pairs as an independent activity test. They remain development diagnostics with recorded prior inspection.

For a future controlled development comparison, start from the existing prepared training-period support only. Derive chronological cutoffs at 60% and 80% of its complete calendar days, rounding down and recording the exact dates before model fitting. The first portion is encoder/probe fit, the second probe/model selection, and the last a development assessment. Require whole episodes, no row overlap and a full one-day gap before each later portion. The coverage audit must report whether this leaves usable data. This is a proposed bounded development split, not untouched final validation: the broader period has already contributed to earlier experiments and design decisions.

A candidate encoder evaluated under this contract may fit only the first portion. Scaling and any learned feature transform must also fit only that portion. Existing full-training-period min/max parameters cannot silently be reused as fit-only parameters. Reusing them would instead require explicit transductive-development wording and would not pass the strict protocol. The next audit is metadata-only and does not refit or regenerate inputs.

Future comparison budgets, seeds and numerical gates must be fixed in the experiment specification. Historical trained checkpoints that used later portions are excluded from the strict comparison; they may appear as explicitly labeled historical diagnostics. An untrained hierarchy is allowed when its initialization is frozen independently of labels and evaluation scores.

Reserve later Actor 1 and Actor 2 periods for subsequent assessment only after checking their exposure history. Earlier analysis may already have inspected them. Call them historical holdouts if used in design; do not promise pristine final-test status without an exposure ledger. Within-actor temporal generalization precedes cross-actor claims. Actor 2 mapping must be reviewed without model outputs, but class inclusion and all model choices must be frozen before its scores are inspected.

## 5. Eligibility gate

Before any model comparison, report each class's usable episodes, overlap components, days, durations and retained-channel observability by role. The initial bounded protocol requires at least three eligible classes, each with at least ten fit episodes over three days, five selection episodes over two days, and five assessment episodes over two days. These are feasibility floors, not a power guarantee.

If support fails, stop the confirmatory claim. Do not lower thresholds after inspecting model scores. A revised label grouping, timescale or task may be proposed using annotations and observability evidence only, with a new contract version. Do not classify absence from logs as absence of activity.

## 6. Frozen representation probes and baselines

Use g as the primary hierarchy level to avoid selecting the best of four after seeing results. Report Z, S and S_tilde as secondary, flattened in fixed device/state order. No new projection layer is trained inside the encoder during probing. Report dimensionality and all four results, including failures.

Primary probe: multinomial logistic regression with L2 regularization, standardized using fit data only. Choose C from {0.01,0.1,1,10} on selection macro-F1; ties choose smaller C. Fit-only class balancing and episode-balanced sample weights are fixed for every method. The encoder remains frozen. Average window probabilities within each episode and produce one episode-level prediction. No episode may dominate simply because it is longer.

Required representation baselines on identical windows and episodes:
1. Original six-feature inputs flattened in fixed order, using the same probe.
2. Simple per-channel summaries: valid-value mean, standard deviation, minimum, maximum and last valid value; valid-value transition count; five status fractions and status-transition count. Undefined value summaries are zero with explicit no-valid-value indicators. Fit-only scaling applies.
3. The same randomly initialized hierarchy, with matching seed/architecture, before learning.
4. Class-prior and majority-class trivial baselines.

Required nuisance diagnostics: availability/status-only summaries and time-of-day/day-of-week features with the same probe. Calendar date, actor ID and absolute recording index are never features of the intended behavioral model. Report performance by availability burden and activity class. Strong nuisance performance is evidence of confounding, not automatically proof of leakage. Do not delete inconvenient samples post hoc.

Learned g must beat both raw-input and summary baselines as well as untrained g to support this contract's added-value claim. A successful higher-dimensional secondary level does not rescue a failed prespecified g claim. Such a finding can motivate a later explicitly revised hypothesis.

## 7. Metrics and decision rule

Primary endpoint: episode-level macro-F1 across the frozen eligible class set. Also report balanced accuracy, per-class recall, confusion matrix, class support, and all quiet/availability-related exclusions. Never use overlapping windows as independent observations for confidence intervals.

For the future candidate experiment, propose three matched initialization seeds; retain all seeds and fix final update/checkpoint selection before assessment. The candidate must improve macro-F1 by at least 0.05 absolute over each of the three substantive baselines in the mean across seeds, with nonnegative difference in every seed. This is an engineering relevance threshold, not a universal scientific constant. Freeze it before fitting.

Where at least ten assessment days are available, use 1,000 paired day-block bootstrap replicates, resampling the same days for all methods and retaining all their episodes. Report paired intervals, missing-class replicate counts and the number of distinct days. Require a positive lower 95% bound against each substantive baseline for the stronger evidence claim; disclose that separate intervals are not a familywise simultaneous guarantee. With fewer days, report descriptive differences and per-day sensitivity only; do not claim statistically established generalization. These data may be insufficient for that stronger claim even if feasibility floors pass.

A result driven by availability or temporal partition, or unsupported by enough days, stays qualified. Forecasting performance, representation variance and collapse/drift measures remain secondary diagnostics, not substitutes for this endpoint. Activity discrimination alone cannot validate abnormal-user detection or an anomaly threshold.

## 8. Gates and next action

Current state: experiment-contract integrity PASS_WITH_LIMITATIONS; behavior-evaluation readiness PENDING. Four missing R1 label definitions, R2 semantic alignment, observable class support, clock alignment and split exposure must be resolved or explicitly excluded.

Next focused implementation: one standalone, read-only annotation-and-coverage auditor. Inputs: the two original logs, activity dictionary, selected-channel registry, timestamp metadata and this contract. Outputs: source hashes; timestamp/duplicate/overlap diagnostics; unresolved IDs; episode/day coverage; a proposed split manifest; observability evidence and unresolved mappings; exclusion counts; READY/BLOCKED per task. No raw full-data scan, representation calculation, classifier fitting, model optimization or automatic R2 text classification.

Only after this audit should we select a methodology change. Preserve the original HDEG baseline. Candidate objectives must be judged by the specified task, with normal-only self-supervised training retained unless a separately declared supervised research question is chosen. Labels may train the evaluation probe; they do not silently become encoder-training supervision.
