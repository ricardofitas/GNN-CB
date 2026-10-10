# Generated inputs, not additional physical specimens

`native/` preserves four original PNG outputs of the native image generation
tool. Both provenance JSON files contain the exact prompts and limitations.
No physical specimen images, measured targets or test masks conditioned generation.
The internal generator version/seed is not exposed; the preserved PNG hashes,
rather than bit-exact re-generation from text, fix downstream reproducibility.

The first pair supplies roughly regular/asymmetric controls. The second pair
was requested specifically to expose geometric defects. Local crush and lean
are visible in `ai_crushed.png`; the requested open contact is not unequivocally
realized in `ai_contact_requested.png`. Prompt text is not a certified defect label.

`scripts/synthetic_image_geometry.py` also draws new analytical raster images,
including fixed pitch variation, lean, local 28% height reduction and a 12-pixel
open contact. It renders 168 frames and runs both fixed/Otsu extraction (336
calls). Selected PNGs are reproducible local outputs under `rendered/`.
The renderer defines geometric motion, **not FEM or physical constitutive truth**.

`results/synthetic_images/geometry_bank.json` fixes all 32 normalized graph
variants, in four families of eight. Real validation/test observations remain
unchanged. Fitting uses one synthetic graph per update, total loss mass 0.1,
and either an original training-only affine prior or frozen GNN pseudo-labels.
There are **zero additional independent mechanical measurements**.
