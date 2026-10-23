# HDEG — Current Level 2 forecasting evidence boundary

Consolidated 23 October 2026. Scope: the current six-feature CU input pipeline and the completed bounded, training-period forecasting diagnostics. This note records the present evidence; it neither declares the full manuscript Level 2 protocol complete nor changes the manuscript, implementation, or previous results. It does not replace an exact claim-by-claim audit of a later manuscript revision.

## Current decision

The implementation and optimization diagnostics are operationally verified. Learned input-conditioned forecasting superiority is not established. Standard absolute-output HBF reduces loss but fails persistence; joint optimization further reduces raw loss while moving and contracting its target space; a persistence-initialized residual variant removes the initial prediction offset and achieves selective gains, largely reproduced by constant corrections or the update-fitted mean true increment.

Preserve the completed experiments as separate evidence. No further training experiment, budget extension, model promotion, or progression to anomaly-detection claims is selected by this consolidation.

## Data and measurement scope

- Prepared CU training inputs: 23 channels, six features per channel, 1-second rows, 30-row X windows and one-row-shifted Y windows. The six features represent value and five status indicators.
- Training manifest SHA256: `a33175f859178a61292dd49ce3d62f70fbfac604a3c3c292c5a41962bd35c296`.
- Census: 1,632,960 prepared rows and 1,632,930 eligible paired windows. Quiet pairs account for 76.84%, older-only changes 22.03%, and newly arriving changes 1.13%.
- Frozen diagnostic selection: 256 source-span-disjoint pairs, eight chronological portions, 64 quiet and 32 from each of six other strata. Nonoverlap does not establish independence.
- Optimization split: 128 update pairs and 128 probes, assigned by the frozen input-only hash rule within portion/stratum. Each role has 32 quiet and 16 per other stratum. Each update pair appears eight times across 128 updates. No probe pair enters gradient updates.
- Probes come from the training period and were inspected in initialization and successive diagnostic analyses. They are development probes, not untouched validation or a generalization test.
- Selection is deliberately enriched. Unweighted errors are sample diagnostics, not natural-population performance. The update-fitted means likewise describe this enriched update subset.
- Quiet denotes unchanged prepared values/statuses, not necessarily physical inactivity or fresh valid observations. In the census, 64.27% of quiet pairs contain at least one unavailable channel. Status transitions must remain distinct from valid-value changes.
- A new_mixed pair has both transition types somewhere in its full 31-row span; its newest transition need not itself be mixed.

All level-specific errors average coordinates within a representation before averaging pairs. Nodes are not treated as independent samples. Zero denominators remain undefined/null. No significance or confidence claim is made from this bounded analysis.

## Separate experimental records

| Record | Encoder/target space | Forecasting condition | What the result establishes |
|---|---|---|---|
| Initialization diagnostic | Fixed untrained six-feature hierarchy | Original randomly initialized absolute-output HBF | Exact quiet-pair persistence and nonzero increments for all 192 nonquiet pairs; original HBF loses to persistence on all 256 pairs at all four levels |
| Arm A: absolute-output HBF | Original encoder frozen | 128 HBF-only Adam updates | Large loss reductions in a genuinely fixed representation space, but no pairwise persistence wins at any recorded evaluation |
| Arm B: joint optimization | Live shared encoder, gradients through X and Y | 128 updates to hierarchy and HBF | Lower raw losses accompanied by target drift and reduced representation variation; no pairwise persistence wins at recorded evaluations |
| Output-residual variant | Original untrained encoder frozen | Prediction=current+correction; output-head weights/biases zero at start; 128 HBF-only updates | Exact persistence initialization; selective final mean gains and quiet/value-only costs; no encoder drift |
| Constant target-mean baseline | Corresponding saved representation state | Coordinate-wise mean Y fitted on 128 updates | Standard HBF fails a simple constant-output baseline on every probe pair at every level in both arms |
| Constant learned-correction diagnostic | Fixed residual experiment representation space | Mean learned correction fitted on 128 updates | Most residual-model gains persist when its correction is replaced by a constant |
| Mean true increment baseline | Fixed residual experiment representation space | Mean(Y-X) fitted on 128 updates, prediction=X+mean increment | Lower probe mean MSE than learned residual HBF in 23/28 stratum-level comparisons |

The residual variant changes output parameterization and head initialization together. Its advantage over Arm A cannot isolate their separate effects. It remains an experimental variant; the standard HBF module has not been replaced.

## Absolute-output and joint-optimization evidence

Both arms start from identical original model tensors, use fresh Adam state, the same update pairs and fixed batch schedule, lr=1e-4, and 128 updates. Evaluation steps are 0,16,32,64,128; step 128 is the predetermined final result.

Arm A's probe forecasting MSE falls by 82.64–83.08% at Z, 82.28–82.64% at S, 82.38–82.96% at S_tilde, and 98.15–98.54% at g across strata. Its encoder parameters, buffers and representations remain unchanged. Nevertheless, neither Arm A nor Arm B beats persistence on an individual update or probe pair at any recorded evaluation step.

Arm B retains 67.6–70.7% of initial probe target variation at Z, 56.4–65.2% at S, 43.9–62.7% at S_tilde, and 39.2–63.7% at g across strata. Target drift is nonzero. Thus lower raw loss is not an equivalent forecasting-quality comparison with Arm A. These measurements establish contraction and movement, not complete collapse or a unique cause of loss reduction.

Read-only decomposition shows that the mean residual accounts for most final standard-HBF MSE: 94–99% across Z/S/S_tilde strata and roughly 76–94% at g. This residual offset is not identical to a network bias parameter. Removing learned head biases makes final errors worse. Both arms' original HBF predictions lose to the update-fitted constant target mean on every probe pair at every level.

## Residual-variant evidence at final step 128

The variant loads the saved original checkpoint, freezes the hierarchy, and zero-initializes all four correction-head weights and biases. It starts at exact persistence, with zero backbone gradient at step one and nonzero backbone gradient from step two. Frozen encoder tensor equality and representation equality hold throughout.

Probe skill is 1 - sum(forecast squared error)/sum(persistence squared error), reported separately per stratum and level:

| Stratum (16 probes each except quiet) | Z | S | S_tilde | g |
|---|---:|---:|---:|---:|
| older_value_only | -13.84% | -20.49% | -18.98% | -12.60% |
| older_status_only | +0.44% | +0.96% | +0.36% | -1.65% |
| older_mixed | +0.09% | +7.23% | +6.36% | +3.46% |
| new_value_only | -2.05% | -6.34% | -5.48% | -3.21% |
| new_status_only | +0.55% | +4.26% | +3.03% | -0.48% |
| new_mixed | +3.61% | +6.56% | +5.42% | +1.99% |

All 32 quiet probes have zero persistence error. Learned corrections add mean MSE approximately 5.83e-8, 9.91e-8, 9.98e-8 and 8.24e-9 at Z/S/S_tilde/g. Quiet skill is undefined.

The residual variant beats standard Arm A on every probe pair at every level, but this includes the structural advantage of starting from persistence. It does not establish broadly reliable learned dynamics.

## Attribution and analytical baselines

For effective correction c=prediction-X and true increment d=Y-X, the verified identity is:

    persistence_MSE - forecast_MSE = 2 mean(c*d) - mean(c*c).

Positive alignment must outweigh correction energy. Quiet pairs have no alignment benefit; negatively aligned increments incur additional harm.

The descriptive probe mean accounts for 99.816%, 99.773%, 99.727%, and 97.555% of correction energy at Z/S/S_tilde/g. Replacing corrections with their mean fitted exclusively on update pairs produces nearly the same new_mixed skills. This establishes predominantly offset-like behavior at this budget, not exactly zero input dependence.

Gains are concentrated: index 767603 is the sole improving older_status_only probe at every level; index 136131 contributes approximately 81–90% of gross older_mixed gains. The leading new_mixed gains include indices 1010806, 1183876 and 571709, all with status-only newest transitions. Index 917346 has the largest Z/S/S_tilde harm, with strongly negative correction/increment alignment. These are representation-space observations; no causal device-event explanation is established.

Fit a single coordinate-wise mean true increment on all 128 update pairs, without using probe targets. It outperforms learned HBF in probe mean MSE in 23 of 28 stratum-level comparisons. HBF's five lower-error cases are new_mixed at all four levels and older_value_only at g; both predictors still lose to persistence in the latter case.

| New_mixed probes | Mean true increment skill | Learned residual HBF skill |
|---|---:|---:|
| Z | +3.406% | +3.610% |
| S | +6.251% | +6.556% |
| S_tilde | +5.065% | +5.422% |
| g | +0.785% | +1.992% |

Both methods introduce quiet errors and worsen both value-only strata. Mean true increment is the least-squares optimal constant correction on the update sample; its probe performance is measured rather than guaranteed by that property. Cosines between the learned update-mean correction and update-mean true increment are 0.979, 0.985, 0.985 and 0.790.

## Claims that are supported and claims that remain open

| Claim | Current status |
|---|---|
| The current pipeline produces finite hierarchical representations and executes the bounded optimization protocols | Supported within the verified executions |
| Current-window persistence is a strong baseline for these overlapping paired windows | Supported within the audited training-period census/sample |
| HBF-only optimization can reduce absolute-output error with a fixed encoder | Supported |
| Joint optimization changes and contracts the representation space under this objective/budget | Supported |
| Output-residual initialization guarantees persistence at step zero | Supported structurally and verified empirically |
| The residual variant achieves selective final mean gains over persistence | Supported for named strata/levels, with costs and heterogeneous pair outcomes |
| Gains demonstrate useful input-conditioned prediction beyond simple constant corrections | Not established |
| Joint optimization is superior to frozen-encoder forecasting | Not established by raw-loss comparisons |
| Full representation collapse or a coding bug explains these results | Not established |
| The model has converged, or cannot learn with another justified protocol | Not established |
| Forecasting generalizes to untouched validation, other periods, homes or seeds | Not established |
| Forecasting results validate anomaly detection, thresholds or higher validation levels | Not established |

## Suggested manuscript wording

In bounded training-period diagnostics, the absolute-output forecaster reduced error under both frozen-encoder and joint optimization, but did not outperform persistence on the evaluated pairs. Joint optimization also changed the target representations and reduced their between-window variation, limiting interpretation of raw loss reductions as forecasting improvement. An experimental persistence-initialized residual forecaster produced modest mean gains in selected transition strata while adding error to quiet periods and worsening value-only strata. Pair-level analysis showed predominantly shared-offset corrections, and a mean-increment baseline fitted exclusively on update pairs achieved lower probe mean error in 23 of 28 stratum-level comparisons. These results establish operational correctness and characterize the current forecasting limitations; they do not establish generalizable input-conditioned forecasting superiority.

This wording is a proposed evidence summary, not an assertion that the existing manuscript's formal Level 2 criteria have been met. The bounded diagnostics use an originally untrained hierarchy in the frozen conditions and must be described accordingly.

## Conditions before another experiment is chosen

First identify one unresolved question and a comparison that would answer it. A further experiment should explicitly distinguish optimization of the current representation problem from useful conditional predictability of increments. It must retain persistence and update-fitted mean increment as baselines, report transition strata and quiet costs, and identify the model/target space used.

State what new evidence would change the conclusion, freeze the budget and decision rule, and document the already-inspected status of the current probes. Do not present continued tuning on these probes as independent validation. Any later generalization claim requires a separately specified evaluation on data not used for those development decisions, subject to the unresolved preprocessing/provenance boundaries.

Do not automatically extend updates, tune a learning rate, replace the standard HBF, or move to another validation level on the basis of this consolidation. The immediate deliverable is this evidence boundary and its source ledger.

## Provenance and historical limits

The two-arm source run is `20261023T075812Z_07d5ca16`; the residual source run is `20261023T084132Z_23cb92e4`. The source ledger records archive/file SHA256 values. The preceding diagnostic/selection artifacts retain their own hash-linked provenance. Original model tensor hash is `6f8b846f2a8137133578cc1965cbb7f8255828eeb41c66dfc122093149c74b81`; split-manifest hash is `2a5b91bf255d8b8f3a77918357c009c609eef0f8d0bb5507ed850bd5452eabd5`.

Completed Colab runs reproduce the original reference outputs. Separate local-runtime reproduction failures were blocked before scientific updates and are not mixed with completed evidence. Older scalar-input, earlier checkpoint, or earlier residual-pilot results remain historical; they cannot be transferred automatically to the six-feature experiments. These experiments also do not resolve upstream source freshness or backward-fill provenance concerns recorded earlier in the project, nor retroactively validate earlier Level 1 references against the current representation state.
