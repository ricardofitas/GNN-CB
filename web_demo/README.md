# Interactive GNN prediction

This is genuine inference from the original five-specimen LOSO GNN checkpoints,
seed 11. Each displayed specimen was excluded from that model's fitting and
validation. It is a demonstration of one seed, not a replacement benchmark.

Run from the repository root:

```sh
python -m http.server 8000
# Open http://localhost:8000/web_demo/
```

Select a specimen and move the phase slider to compare at an exact archived
frame. Enter any continuous phase in [0,1] to evaluate the network between
recorded frames; observed markers and RMSE are hidden where no exact observation
exists. Selecting one of the three image snapshots also displays its pixel-aligned
prediction and observed nodes. Download exports the actual current predictions.

The trained model uses only initial normalized coordinates, initial edges and
protocol phase. The camera images, masks and observed displacements are not
inputs. Missing observations remain missing; no target interpolation is used.
Phase is retrospective video-protocol information, not measured strain or a
future load forecast. Coordinates are pixels; normalized error is divided by
the initial graph height. No millimetre or force calibration is implied.

`engine.mjs` implements the original three message blocks, degree-mean
aggregation, SiLU, LayerNorm, residual decoder and training-fitted affine base.
Exported tensors retain exact float32 bytes in little-endian base64. The JavaScript
calculation uses doubles and is compared with PyTorch on every archived phase
and three additional non-archived phases per specimen. There is no prediction
lookup table in the UI. `verification.json` records executed parity/UI checks.

Re-export and verify:

```sh
PYTHONPATH=. python scripts/export_web_demo.py
node web_demo/verify.mjs
# Optional real-browser QA; specify installed Playwright and Chromium paths:
PLAYWRIGHT_PATH=/path/to/playwright CHROMIUM_PATH=/path/to/chromium node web_demo/verify.mjs
```

Browser screenshots and the PyTorch numerical reference are reproducible local
test artifacts under `results/diagnosis/cache/web_demo/`, not inference inputs.
The source images load on demand from the exact public GNN-CB commit
`50f888bcddffe3284dae77cfba0769cbd54a62fb`. No bulk video storage or private service
is needed. The original image hashes and dimensions accompany every case.

For ProjectHub, copy the static runtime files (`index.html`, `style.css`,
`engine.mjs`, `app.mjs`, `data/*.json`) to `public/gnn-cb/` and embed
`/gnn-cb/index.html` within the existing Graph-CB card. Keep existing source image
galleries and FEM viewers distinct. No authentication, provider, DNS or storage
changes are required by this component.
