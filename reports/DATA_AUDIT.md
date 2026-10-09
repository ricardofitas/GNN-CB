# Corrugated-board data audit

Sources inspected at frozen commits:

- Graph-CB: `765baa23e364538b05ec293cea2d8ade8b4c4a03`.
- Graphs_Analysis_Corrugated_Boards: `16eed74ff025e0e0f40f4f5d81c2b71af58b93b5`.

The first contains newer tracking exports for five unique experiments, not a 122-specimen dataset. `data/data_st/Ex1` and `Ex2` repeat the Probe2_7 and Probe1_22 CSVs byte-for-byte. Probe2_7 rate3 and rate10 are different processing/subsampling of the same specimen. Never treat these as independent experiments.

## Coherent original archives used

| Dataset | Workbook sheet | Nodes | Undirected edges | Camera frames | Compression frames | Observed node/frame labels in compression | Coverage |
|---|---|---:|---:|---:|---|---:|---:|
| Probe2_7 | Ex1 | 44 | 64 | 205 | 33–158 inclusive | 5542 / 5544 | 99.964% |
| Probe1_22 | Ex2 | 40 | 58 | 205 | 40–164 inclusive | 4902 / 5000 | 98.040% |

Each `Graphs_Analysis_Corrugated_Boards/data_{specimen}.zip` contains a coherent set:

- `Frames_{specimen}_rate10/frame_0000.jpg` ... `frame_0204.jpg`: 205 camera images, already cropped to 1024 × 256 px.
- `{specimen}_ske_rate10/graphs_{specimen}_ske_rate10.pkl`: 205 NetworkX graphs with node `o=(row,column)` coordinates and actual skeleton edges.
- `{specimen}_ske_rate10/Track_data_{specimen}_ske_rate10.xlsx`: a sheet per node; fields `frame_number, pre_row, pre_col, new_row, new_col, min_distance`.
- `{specimen}_ske_rate10/img_analysis_{specimen}_ske_rate10.xlsx`: original author annotations of compression start/end and inferred image dimensions.
- `{specimen}_ske_rate10/graphs_figs_{specimen}_ske_rate10/Graph_Frame_0000.jpg`: original graph overlay plot, separately resized/cropped; its pixel coordinates must not be confused with raw camera coordinates.

Every nonempty track begins at frame 1. For all 84 nodes, first `pre_row/pre_col` exactly equals the graph0 coordinate. Original graph node IDs therefore safely index the original tracks. Empty sheets above IDs 43/39 are ignored. Source PKLs were disassembled using pickletools before use; importer restricts NetworkX/NumPy classes and verifies source archive SHA-256 before deserialization. Prepared data are ordinary NumPy arrays readable with `allow_pickle=False`.

## Tracking quality

Internal gaps: 59 track-gap occurrences in Probe2_7, 54 in Probe1_22. No duplicate observed coordinates were found in either archive across all 205 frames. Median frame-to-track displacement is 0 px; 95th percentile 1.414 px in both specimens; maxima 7.280 and 9.220 px. This is quantized, automatically matched image tracking, not independent DIC or FEM truth. Missing rows are strictly masked, never interpolated, smoothed, or filled. All initial node observations are preserved at frame0. Labels in the compressed NPZ are displacements relative to those initial observations.

Pixel scale is retained. In the source code, millimetres per pixel are estimated retrospectively from total initial-final experimental deformation and image height change. That is not an independently supplied camera calibration; it is unsuitable as an unqualified predictor input or prospective physical scale. Initial graph heights used for nondimensionalization are 41 px (Probe2_7) and 39 px (Probe1_22).

## Recommended prediction task and evaluation

Initial skeleton graph plus prescribed offline compression phase -> per-node displacement. This is a phase-conditioned deformation surrogate. Phase equals `(frame-start)/(end-start)`, using the author's annotated compression endpoints. It is not mechanical strain, a measured timestamp, autonomous next-frame forecasting, or a learned constitutive material law.

Hold out complete specimens in two directions. Correlated frames or nodes cannot be randomly mixed across training and test. Learn any regression coefficients, affine baseline or neural weights only on the training specimen. Use fixed architecture/training settings if no third independent validation specimen exists. Do not select architecture/epochs by the held-out specimen. Report pixel MAE/RMSE and initial-height-normalized errors, both per specimen and pooled with explicit weighting. Show vertical and horizontal errors separately and compare with a simple affine compression model plus a node MLP without graph messaging.

Two independent specimens support a functional proof of concept, not a claim of generalization to new flute architectures, materials or arbitrary loading. Both are from the same corrugated-board study and appear geometrically similar.

## Other exports and why excluded

- Graph-CB Probe2_7_v1 has 44 nodes and Probe1_22_v1 has 43 (versus 40 originally). Node labels cannot be transferred between versions without geometric matching.
- Exp3 has frame0 coordinate CSV with 46 nodes but displacement CSV for 36 nodes and late births: only 24 begin at frame1, remaining tracks begin at frames2–134. Its CSV references each node's first tracking position, not a universal frame0 reference. It lacks matching original image/topology archive.
- Exp4 has 48 tracked nodes over frames1–184, but no committed initial-coordinate CSV or matching images/topology.
- Exp5 has 42 tracked nodes over frames1–71, with five late-start tracks, but no committed initial-coordinate CSV or matching images/topology.
- Peak-load XLSXs for Exp3/4/5 are sparse curve extrema, not full force records. Interpolating them as dense force ground truth would invent targets.

## Force alignment evidence

`Load_Deformation_Curves.xlsx` contains only sheets Ex1 and Ex2. Top human-readable workbook labels are Specimen1.14 and Specimen1.10. The separate reviewer verified the original archive modeling-data force/crush pairs match Ex1 for Probe2_7 and Ex2 for Probe1_22 to numerical precision. Thus specimen association is supported. The source author maps force to frames using constant compression speed and start/end annotations, with `frame≈start+(end-start)*crush/crush_last`; this is inferred synchronization rather than camera/Zwick shared timestamps. The present importer exports no force labels.

## Implementation delivered

- `GNN-CB/gnn_cb/data.py`: frozen source manifests, checksum check, restricted graph import, exact-mask preprocessing, image export, NPZ loading helpers.
- `GNN-CB/scripts/prepare_data.py`: local-source or pinned-download CLI.
- `GNN-CB/data/processed/*.npz`: x0, y, mask, edge_index, progress, frame, node_ids, initial_pos_px, displacement_px, origin_px, scale_px, compression_window, specimen_id and experiment.
- `GNN-CB/data/processed/manifest.json` and per-specimen JSON: complete provenance, coverage, assumptions and exclusions.
- `GNN-CB/data/processed/images/`: source images at frame0/start/middle/end and original source initial graph overlay for both specimens.

Reproduction: `python scripts/prepare_data.py --source-dir ../Graphs_Analysis_Corrugated_Boards --output-dir data/processed`, or `--download` with a normal public-GitHub-capable connection. Importer ran successfully on both local source archives.

Citation: Belfekih, T.; Fitas, R.; Schaffrath, H.-J.; Schabel, S. *Graph-Based Analysis for the Characterization of Corrugated Board Compression.* Materials 2024, 17(24), 6083. DOI: 10.3390/ma17246083. Original repository license is MIT; include attribution when redistributing extracted data or source figures.
