# Independent scientific review of GNN-CB

Review date: 2026-10-09. Scope: the two image/graph archives in `Graphs_Analysis_Corrugated_Boards`, their tracking workbooks and load curves, and the related `Graph-CB` source. This is a review of the proposed experiment, not certification of training results.

## Defensible objective

The strongest immediately supportable task is a **supervised, progress-conditioned surrogate for tracked two-dimensional node displacements**, using the initial graph and an explicitly supplied compression-progress coordinate. This can establish whether a message-passing implementation runs, learns the published tracking labels, and transfers descriptively between two recorded specimens. It does not establish a universal constitutive law, a three-dimensional mechanics solver, or reliable prediction of new materials and geometries.

The underlying paper selected two specimens for clear cross-sections and videos and presented graph extraction and node tracking as a feasibility study. It also called for more experiments and geometries. Thus there are two independent experimental units, even though there are hundreds of frames. The initial graph is an engineered representation extracted from images; training on it is not an end-to-end neural image-segmentation model. Source: Belfekih et al., *Materials* 17(24), 6083 (2024), DOI `10.3390/ma17246083`; publisher: https://www.mdpi.com/1996-1944/17/24/6083 ; accessible indexed full-text mirror: https://pmc.ncbi.nlm.nih.gov/articles/PMC11727766/ .

## Facts checked directly in the delivered data

| Quantity | Probe1_22 | Probe2_7 |
|---|---:|---:|
| Raw image frames in archive | 205 | 205 |
| Tracking sheets with observations | 40 | 44 |
| Empty tracking sheets | 20 | 16 |
| Author compression start/end, inclusive | 40–164 | 33–158 |
| Possible tracked node/frame pairs in compression window | 5,000 | 5,544 |
| Observed pairs in that window, before any quality filtering | 4,902 | 5,542 |
| Coverage before quality filtering | 98.04% | 99.964% |
| Minimum observed nodes in a compression frame | 20 | 42 |
| Repeated current coordinates within a compression frame | 0 | 0 |
| Missing internal frame numbers over all nonempty tracks | 265 | 292 |
| Largest recorded tracking distance over entire archive, px | 9.2195 | 7.2801 |

All nonempty sheets begin with frame 1; its `pre_row, pre_col` is a usable frame-0 reference. Subsequent rows must be indexed by `frame_number`, never by row position: sheets contain internal gaps. Coordinates are `(row, column)` in files; physical plotting convention should state whether this is converted to `(x=column,y=row)` with image y downward or y upward. The archive metadata is `img_analysis_*`, not a directly supplied hardware time synchronization.

Exact matching of every available `(Crush (mm), Force (N))` pair in each `modeling_data_*` file to the raw curve proves the following mapping, to floating-point precision: **Probe1_22 → Ex2 (header Specimen 1.10); Probe2_7 → Ex1 (header Specimen 1.14)**. Maximum paired discrepancies were below `6e-14`. This verifies the specimen-to-curve association; it does not independently verify every frame-to-force alignment.

## Leakage and evaluation

1. **Split whole specimens.** Training on one specimen and evaluating on all eligible frames of the other is the appropriate transfer check available here. Randomly splitting neighboring frames and advertising the result as generalization would primarily measure interpolation of nearly repeated observations. Any within-specimen holdout must be separately labeled as temporal interpolation or future-block extrapolation.
2. **Specify the progress information.** If `p=(frame-start)/(end-start)`, author-provided end frames use knowledge of the completed test. That is acceptable for a retrospectively aligned, protocol-conditioned surrogate, but cannot support an autonomous forecast at an unknown test duration. An operational replacement would be known commanded platen displacement or a predeclared loading protocol, measured independently of the future response. `p=1` also need not represent identical physical strain in the two specimens.
3. **Prevent target-derived inputs.** Use initial geometry, initial topology, initial boundary labels, and supplied progress only. Future positions, final collapsed thickness, measured future force, late-frame cluster assignments, track duration, or the current observation mask must not enter a supposed initial-state forecast. The target mask can be used to select loss/evaluation entries.
4. **Normalize causally.** Per-specimen normalization from its initial geometry is deployable. Any target standardization, regression coefficients, or statistical normalizers must be fit on the training specimen only. Preserve aspect ratio if coordinates have separate x/y scales, and report which scale defines displacement and edge geometry.
5. **Do not tune to the two transfer scores.** Use fixed prespecified hyperparameters, or a training-only temporal validation subset with its limitations stated. Selecting epochs, seeds, architecture, or hyperparameters based on the held-out specimen makes that specimen a validation set, leaving no untouched test specimen.
6. **Use an informative comparator.** A fitted affine field as a function of initial position and progress controls for mostly homogeneous compression. A shared node MLP controls for whether messages add value. If the MLP still receives degree/neighbor-derived features, call it a message-free MLP; it is not entirely graph-free. Same inputs, training data, loss mask, and evaluation mask are needed for fair comparison.
7. **Report the result even when the GNN loses.** Show both fold scores separately and the direction of transfer. Pixel endpoint error and displacement-component RMSE are interpretable; displacement R² can be secondary. R² of absolute positions is misleading because the large initial x span dominates. Avoid confidence intervals based on resampling individual frames/nodes as independent experiments. Seed spread is optimization variability, not population uncertainty.

## Missing observations and label quality

Empty sheets and absent frame IDs mean unknown tracking labels. Filling them with zero displacement introduces fictitious stationary nodes; carrying the last position creates fictitious measurements; interpolation uses a modeling assumption and, if it reaches across splits, future information. Use explicit masks for both loss and metrics, and compute residuals only at observed entries so masked NaNs cannot poison gradients.

The late Probe1_22 frames lose many nodes: there are only 20 observations at frame 164. Errors over surviving nodes alone can understate error in the most severely collapsed regions. Include frame-wise coverage beside error curves and avoid presenting the last-frame prediction as fully validated.

The nearest-neighbor approach in `Graph-CB/data/Combination_code_v5.py` does not enforce one-to-one assignment and retains old graph coordinates when matching fails. The actual archive's tracking workbooks did not show duplicate current coordinates during compression, which is reassuring but does not prove tracking identity correctness. Tracking labels are image-processing pseudo-labels with quantization, filtering, localization, and association error; they are not independently measured ground-truth material-point displacements. Raw images at early/mid/late compression should be visually overlaid with tracked and predicted nodes.

## Force, calibration, and mechanics

The source script maps machine displacement to frame number using a linear coefficient from the inferred compression start and end and the last retained machine crush value. It notes nonuniform sampling of load measurements and therefore does not align by measurement-row index. It also trims the final 25 curve rows. This is an approximate synchronization model, not a provided common timestamp stream.

In `Graph-CB/data/Combination_code.py`, `pixel_mm` is estimated as the machine's initial-to-final crush divided by the image/graph's initial-to-final collapse. Consequently, millimetre conversion is calibrated using the complete deformation trajectory. It should not be described as an independent camera calibration or silently used to normalize a held-out initial-state prediction. Native pixels or a scale derived exclusively from initial geometry are the clean primary benchmark; any physical-unit output must expose the source and uncertainty of its calibration.

Force prediction requires separately validated force labels, specimen dimensions, material properties, and loading information. A small force decoder could fit these two curves, but this would not establish transferable mechanics. Initial image geometry and normalized progress do not uniquely identify paper stiffness, plasticity, moisture dependence, contact/friction, or force. Likewise, a fixed initial graph does not represent newly formed contact edges, and displacement accuracy alone does not verify force balance, strain energy, stress, failure, or constitutive consistency. Do not call this model physics-informed unless explicit physics constraints are actually implemented and verified.

## High-value verification tests

- Mask invariance: alter all masked labels to arbitrary large values or NaNs; the loss and finite observed gradients remain unchanged.
- Explicit frame indexing: a track with observations at frames 1 and 3 remains missing at frame 2.
- Permutation equivariance: permute node features and remap edges; predictions permute identically.
- Zero-progress condition: if the architecture imposes it, predicted displacement is exactly zero at progress 0.
- Split integrity: no frame of the held-out specimen appears in fitting or parameter selection.
- Normalization provenance: persisted fit statistics list only the training specimen; test-only targets cannot change them.
- Metric sanity: perfect predictions give zero pixel errors; a known one-pixel displacement perturbation produces the analytically expected endpoint error.
- CPU reproducibility and artifact replay: load the saved checkpoint and configuration and reproduce recorded predictions within a declared numerical tolerance.

## Suggested report wording

“We implemented and trained a message-passing displacement surrogate on image-derived corrugated-board graphs. The evaluation contains two whole-specimen transfer directions with masks for unobserved tracking labels. Compression progress is retrospectively normalized from the author-supplied test windows. Results establish software functionality and a small-sample feasibility baseline; they do not establish generalization across materials, geometries, or loading protocols. Observed displacements are tracking-derived labels, and further independently calibrated experiments are needed before mechanical or industrial deployment claims.”

This wording must be supplemented with the actual fold metrics, baseline comparison, final architecture, and any deviations from the recommended evaluation protocol.
