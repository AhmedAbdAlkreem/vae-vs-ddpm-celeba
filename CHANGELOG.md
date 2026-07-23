# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/).

## v1.2

**Added**
- Cosine beta schedule (Nichol & Dhariwal, "Improved DDPM", 2021) as the new
  default, alongside the original linear schedule (`configs/diffusion.py`:
  `beta_schedule`). Verified the cosine schedule degrades signal more
  gradually than linear (85% signal retained at t=250 vs. 52% for linear).
- Self-attention at an additional U-Net resolution (16x16), not just the 8x8
  bottleneck (`configs/model.py`: `unet_attn_resolutions`). Verified
  attention modules land at exactly the intended resolution level, both
  down and up paths, via a direct architecture inspection test.
- Gradient clipping for DDPM training (`configs/training.py`:
  `ddpm_grad_clip_norm`, default max norm 1.0). Correctly unscales AMP
  gradients before clipping.
- Cosine learning-rate decay over DDPM training
  (`configs/training.py`: `ddpm_use_lr_scheduler`).

**Fixed**
- **LR scheduler + resume interaction.** Resuming training with a different
  total epoch count than the original run (a normal part of this project's
  workflow) corrupted the cosine LR curve, because `CosineAnnealingLR`
  computes each step recursively from the *current* optimizer LR rather than
  a fixed base value — and the resumed optimizer's LR was already decayed
  near zero from the previous run, so fast-forwarding multiplied zero by
  zero forever. Fixed by resetting to the true base LR before fast-forwarding
  the scheduler to the correct point on the new curve. Verified the resulting
  LR sequence matches the exact theoretical cosine formula.
- **U-Net checkpoint loading crash.** `evaluate.py`, `sample.py`, and
  `src/evaluation/evaluator.py` manually reconstructed the U-Net for
  inference without passing `img_size`/`attn_resolutions`, so the rebuilt
  model didn't match the trained checkpoint's architecture the moment
  attention-resolution support was added (`RuntimeError: Unexpected key(s)
  in state_dict`). Fixed all three call sites to use `UNet.from_config(cfg)`
  consistently instead of repeating constructor arguments by hand.

**Results**

| Metric | v1.1 | v1.2 | Change |
|---|---|---|---|
| VAE FID | 97.94 | 97.06 | -0.9% (flat, expected — no VAE changes this version) |
| VAE Inception Score | 2.00 +/- 0.08 | 2.00 +/- 0.06 | flat |
| VAE reconstruction PSNR | 22.48 dB | 22.47 dB | flat |
| DDPM FID | 28.37 | 26.42 | -6.9% |
| DDPM Inception Score | 2.59 +/- 0.13 | 2.61 +/- 0.17 | ~flat |

Smaller gain than v1.0->v1.1, as expected: v1.1's improvement included fixing
an outright sampling bug (large, one-time win), while v1.2 is architectural
refinement on an already-working baseline (smaller, incremental win).

**Artifacts:** `outputs/v1.2/results.json`,
`outputs/comparison/version_comparison.png` (cross-version FID/IS chart,
regenerate with `python scripts/generate_comparison_chart.py`)

---

## v1.1

**Fixed**
- **DDPM ancestral sampler numerical drift.** `src/sampling/ddpm_sampler.py`
  never clipped the intermediate predicted-x0 estimate back to the valid
  `[-1, 1]` pixel range at each reverse step. Over 1000 sequential steps,
  small per-step numerical error compounded — verified directly by comparing
  sampler output with clipping disabled: pixel values reached **-6.06 to
  6.72** (should be within [-1, 1]). This produced solid black/white panels
  in place of faces in some v1.0 samples.
  - Fix: predict x0 from the model's noise estimate at each step, clip it to
    `[-1, 1]`, then compute the posterior mean from the clipped x0 (standard
    approach from the official DDPM / Improved-DDPM implementations).
  - Added `posterior_mean_coef1` / `posterior_mean_coef2` to
    `src/models/diffusion.py`'s `GaussianDiffusion` to support this.
  - This fix applies to sampling only — it required no retraining, only
    re-running generation with the corrected sampler against existing
    checkpoints.
- **matplotlib backend leakage in interactive notebooks.** Several
  visualization modules (`sampling_grid.py`, `training_curves.py`,
  `reconstruction.py`, `interpolation.py`, `latent_space.py`,
  `denoising.py`) called `matplotlib.use("Agg")` at import time. In a Kaggle
  notebook, importing any of these silently switched the *entire session's*
  matplotlib backend to Agg, which made all subsequent `plt.show()` calls
  (including the user's own plotting cells) render nothing with no error.
  Fix: removed the forced backend — `plt.savefig()` works correctly on
  whichever backend is already active, so forcing one wasn't necessary.
- Checkpoint resume was missing `history` in the saved state dict, which
  would have raised `KeyError: 'history'` on resume. Caught by an explicit
  interrupt-and-resume test, not just an import/syntax check. Fixed in
  `src/training/train_vae.py` and `src/training/train_ddpm.py`.

**Changed**
- Dataset subset size: 30,000 -> 40,000 images
- VAE training: 30 -> 40 epochs
- DDPM training: 40 -> 50 epochs

**Results**

| Metric | v1.0 | v1.1 | Change |
|---|---|---|---|
| VAE FID | 120.69 | 97.94 | -18.9% |
| VAE Inception Score | 2.04 +/- 0.06 | 2.00 +/- 0.08 | ~flat |
| VAE reconstruction PSNR | 22.27 dB | 22.48 dB | ~flat |
| DDPM FID | 43.77 | 28.37 | -35.2% |
| DDPM Inception Score | 2.79 +/- 0.12 | 2.59 +/- 0.13 | slightly lower |

Note: this release bundles the sampler fix with a larger training budget in
a single run — the FID improvement reflects both, not the fix in isolation.

**Artifacts:** `outputs/v1.1/` (results.json, sample grids, comparison grid,
reconstruction grid, latent-space PCA, interpolation grid, denoising GIF)

---

## v1.0

Initial release: from-scratch VAE and DDPM implementation, professional
modular repo structure (`configs/`, `src/{datasets,models,losses,training,
sampling,evaluation,visualization,utils,engine}`, `tests/`).

- Dataset: CelebA, 30,000-image subset
- VAE: 30 epochs, latent_dim=128, conv encoder/decoder
- DDPM: 40 epochs, U-Net (base_ch=64, attention at 8x8 bottleneck only),
  linear beta schedule (1e-4 -> 0.02), 1000 timesteps, EMA (decay=0.999)
- Ancestral DDPM sampling only (no DDIM yet at this point in development)
- Resumable checkpointing (model, optimizer, AMP scaler, EMA state) every 2
  epochs

**Results**

| Metric | Value |
|---|---|
| VAE FID | 120.69 |
| VAE Inception Score | 2.04 +/- 0.06 |
| VAE reconstruction PSNR | 22.27 dB |
| DDPM FID | 43.77 |
| DDPM Inception Score | 2.79 +/- 0.12 |

**Known issue at this version:** some DDPM samples rendered as solid
black/white panels due to the sampler numerical drift described above (fixed
in v1.1).

**Artifacts:** `outputs/v1.0/` (results.json, sample grids, comparison grid)

---

## Planned (v1.3, not yet implemented)

Organized by which part of the codebase each change touches.

**VAE (`src/models/vae.py`, `src/losses/vae_loss.py`) — currently unchanged
since v1.0, all v1.1/v1.2 work targeted DDPM:**
- Enable the already-implemented perceptual loss
  (`configs/training.py`: `use_perceptual_loss`, currently off by default) —
  directly targets the VAE's main weakness (blur), since it compares
  reconstructions in a pretrained feature space instead of raw pixels.
- Consider a learned (rather than fixed) decoder variance, or a larger
  latent dimension — both documented ways to reduce reconstruction blur
  further.

**DDPM sampling (`src/sampling/`):**
- Benchmark the DDIM sampler (already implemented,
  `cfg.diffusion.sampler = "ddim"`) — no `results.json` has been generated
  with it yet. Worth reporting as a speed/quality trade-off data point
  (~50 steps vs. 1000) alongside the full-DDPM numbers.

**DDPM architecture (`src/models/unet.py`):**
- Try attention at 32×32 as well, or a wider U-Net (`unet_base_ch=128`) —
  the v1.1→v1.2 attention-resolution change gave a real but modest gain;
  worth checking whether more capacity continues that trend or plateaus.
- v-prediction parameterization (predicting a velocity term instead of raw
  noise) — used in several post-DDPM papers, reported to improve stability
  at high timesteps.

**Data / training budget (`configs/dataset.py`, `configs/training.py`):**
- Full ~200k CelebA images (currently using a 40k subset) and more epochs,
  now that the architecture and training loop are stable and bug-free.
- Larger image resolution (e.g. 128×128) — would require re-tuning the
  U-Net's channel multipliers and attention resolutions.

**Evaluation (`src/evaluation/`):**
- Add LPIPS or SSIM alongside FID/IS/PSNR for a more complete reconstruction-
  quality picture.
- Compute metrics separately per CelebA attribute (e.g. by gender, glasses,
  smiling) to check whether either model has attribute-specific weaknesses
  the aggregate FID hides.

**Engineering (repo-wide):**
- GitHub Actions workflow to run `pytest` automatically on every push —
  currently the test suite is only run manually.
- Replace print-based logging with TensorBoard or Weights & Biases for
  richer training-curve tracking across versions.
