# Audit of the expanded image-derived graph dataset

The original two-specimen baseline is preserved byte-for-byte in `data/processed`.
The five-specimen version in `data/expanded` adds the original, coherent graph and
tracking bundles for three more physical specimens found in `uncertain_cb`.

| Physical specimen | Archived image frames | Initial nodes | Undirected edges | Compression frames | Valid compression node observations |
|---|---:|---:|---:|---:|---:|
| Probe2_7 | 205 | 44 | 64 | 126 | 5,542 |
| Probe1_22 | 205 | 40 | 58 | 125 | 4,902 |
| Exp3 / 2024-08-05 D12 | 174 | 46 | 67 | 95 | 2,566 |
| Exp4 / 2024-08-05 D14 | 185 | 48 | 71 | 126 | 5,403 |
| Exp5 / 2024-12-03 ID03 | 72 | 42 | 61 | 40 | 839 |
| **Total** | **841** | **220** | **321** | **512** | **19,252** |

The total is five physical experiments, not 841 independent experiments. All
frames and all processing variants of one physical specimen belong to one fold.

## Why the three additional specimens are now admissible

The earlier import could not reliably join the later CSV exports to initial
geometry. The private source contains the original NetworkX graph list, original
tracking workbook, compression annotations and original image crops from each
same processing run. Their Git blob hashes were verified before parsing.

For each included node, its first workbook `pre_col,pre_row` equals graph0;
each subsequent pre-coordinate equals the last observed position; and every
`new_col,new_row` equals the same node in the corresponding archived graph.
All three checks passed for every observed record in all three new datasets.
Graph frame indices exactly match the image filenames. All **431** additional
image files were downloaded and their source blob hashes verified.

Exp3 has nodes whose first successful tracking match occurs much later than
frame0. Those nodes already exist in graph0, and their first pre-coordinate
agrees with graph0. The importer therefore keeps their true initial reference
and masks the earlier missing observations. It never treats the first successful
match as a new zero-displacement reference.

## Curation and exclusions

- A stored graph position without an explicit tracking-workbook observation is
  excluded. The source tracker can carry a last coordinate forward after a
  failed match; this is not a measurement.
- If multiple nodes share an identical observed coordinate in a frame, all
  involved observations are masked. This removed **20** observations from Exp3
  and **427** from Exp4 across all phases. Of these, **227** Exp4 observations
  were in the compression window. Exp5 had no such collisions.
- No coordinates, missing trajectories, mechanical forces or edge labels are
  interpolated or fabricated. Missing targets have `mask=False` and zero
  placeholders. Source observation and collision masks are retained for audit.
- Exp5 rate10 and rate3 are two processing versions of the same experiment.
  Only rate3 is admitted. Repository aliases under `Taieb/Sample_videos` and the
  campaign experiment folders are not extra specimens.
- The original two NPZ datasets remain unchanged. The extra collision filter
  applies to the three newly imported specimens.

Initial graph overlays on the raw camera crops are shown in
`figures/expanded_source_graphs.png`. The graph coordinates register with the
board. Some original edges bridge fibre defects or simplify curved flute paths;
these source limitations remain and are examined separately by re-extraction.

## What the labels mean

The targets remain automated image-tracked nodal displacement relative to
frame0, normalized by initial graph height. New camera crops have different
resolution from the original pair, so normalized error is the primary basis for
comparison across experiments. Pixel errors remain useful within each camera
view. Label coverage within compression is about **58.7%** for Exp3, **89.3%**
for Exp4 and **49.9%** for Exp5 after curation.

The phase input is the author's offline compression-window annotation, not a
measured mechanical strain. All three new windows start at frame20; this has not
been independently synchronized to force or acquisition timestamps. Exact
graph/workbook agreement establishes file consistency, but does not prove that
nearest-skeleton-node matching preserves physical identity after buckling.
These are noisy image-derived labels, not independently validated DIC or FEM
ground truth.

## Reproduction

Restore `ricardofitas/uncertain_cb` at commit
`1df856a4f2c71378744d4a73e3cd958f6f6d7a71`. From this repository root:

```bash
PYTHONPATH=. python scripts/prepare_expanded.py --source-root /path/to/uncertain_cb
```

The tracked `data/expanded/source_tree.json` records the exact source paths and
blob hashes. `expanded_data_audit.json` and each specimen JSON record additional
SHA-256 fingerprints and every exclusion count. The restricted pickle loader
admits only the reviewed NetworkX and NumPy data classes, including the NumPy2
module spelling.

Four further candidate recordings were restored from 17 multi-volume RAR files
in `ML_Corrugated_Boards`, yielding **8,192 decoded frames**. They are documented
separately in `video_source_audit.json`; their fresh extraction and any quality
admission are separate from the five-specimen archive benchmark above.
