# Additional experimental image audit

Source: `ricardofitas/uncertain_cb`, commit `1df856a4f2c71378744d4a73e3cd958f6f6d7a71`.

| Experiment | Raw PNG sequence | Raw frames | Existing sampled frames recovered | Sampling | Initial image size |
|---|---|---:|---:|---:|---|
| Exp3 | experiments_2024_08_05_D12 | 1,732 | 174 | Every 10th frame | 2752 × 300 |
| Exp4 | experiments_2024_08_05_D14 | 1,846 | 185 | Every 10th frame | 2752 × 300 |
| Exp5 | experiments_2024_12_03_ID_03 | 214 | 72 | Every 3rd frame | 2752 × 400 |
| Total | Three experiments | 3,792 | 431 | — | — |

All 431 sampled camera JPGs were downloaded and checked against the Git blob SHA and byte count. All 431 are distinct image blobs. These frames are the existing image-to-graph pipeline inputs, not plots. Complete graph pickles, node-tracking workbooks, and image-analysis workbooks are recovered for each sequence. The importer separately audits graph/track consistency and usable observation masks; inventory counts are not final accepted training counts.

## Deduplication and labels

`Taieb/Sample_videos/Exp3`, `Exp4`, and `Exp5` have exactly the same Git tree SHAs as the three root experimental folders listed above. They are duplicate mirrors, not six experiments. Exp5 rate10 is a subsampling of the same experiment as rate3 and is not counted again. `data_Probe1_22_old`, `data_Probe1_22_v1`, `data_Probe2_7_old`, `data_Probe2_7_v1`, and denser rate2/rate3 variants retain the two previous experimental identities.

The mechanical workbook contains these exact mappings: `Load_Deformation_Curves.xlsx/Ex3` = `Extra_exp.xlsx/E04` (header Specimen 1.7), `Ex4` = `E03` (Specimen 1.6), and `Ex5` = `E05` (Specimen 1.5). Ex3 must not be matched to E03 by name. Frame-force synchronization remains unvalidated; none of these curves alone establishes force or stress supervision.

## Other repositories inspected

- `ricardofitas/ML_Corrugated_Boards`: 17 multipart RAR objects containing four named video candidates, Probe1_2, Probe1_8, Probe1_9, Probe1_10. Exact paths, SHAs, sizes are in its saved tree. They are processed separately from these three recovered graph sequences. Mechanical workbook `Data/druckversuche_neuereinsatz 231205_2.xlsx` contains correspondingly named sheets. Its Probe 1.10 curve is different from the older GNN Ex2/Specimen 1.10 curve (750 versus 976 spreadsheet rows; different observations); specimen numbering must be namespaced by campaign rather than treated as a global identifier.
- `rfitas-lab/corrugated-sheet-fem-pinn`: mirrors the two original Probe1_22/Probe2_7 archives and selected annotated frames; numerous other PNGs are simulation or result figures.
- `rfitas-lab/geometry-based-large-deformation-model`: two-probe diagnostic figures and solver coordinate/plot outputs; no additional raw camera sequence identified in its 1,224-entry tree.
- `rfitas-lab/transferability-ai`: 4,316-entry tree consists of unrelated validation research, numerical arrays, external datasets and paper figures; no additional corrugated camera sequence identified.
- `ricardofitas/projecthub`: `public/graph-cb` contains three data files; `public/opt-cb` contains 14,993 entries dominated by optimization iteration renders; branding and solver renders are not additional experimental observations.

`private_image_inventory.json` is the compact machine-readable provenance with every recovered camera file path, Git SHA, byte count, aliases, source commit and paired label paths. It contains no signed download URLs or credentials.
