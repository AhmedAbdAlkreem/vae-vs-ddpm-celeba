# Gen_AI — VAE vs DDPM from Scratch

**Task code:** GenCV003 · **Position:** CV Engineer

A from-scratch PyTorch implementation of a **Variational Autoencoder (VAE)**
and a **Denoising Diffusion Probabilistic Model (DDPM)**, trained on CelebA
and benchmarked quantitatively (FID, Inception Score, reconstruction PSNR)
and qualitatively (sample grids, latent-space visualization, interpolation,
denoising-process animation). Both models are built from base `torch.nn`
primitives only — no pretrained VAE, no `diffusers` library.

---

## Table of contents

- [Dataset](#dataset)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Setup](#setup)
- [How to run](#how-to-run)
- [Version comparison: v1.0 -> v1.1 -> v1.2](#version-comparison-v10---v11---v12)
- [Results](#results)
- [Report notes](#report-notes)
- [Known limitations / next steps](#known-limitations--next-steps)

---

## Dataset

[CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
— 64×64 aligned/cropped face crops, loaded via Kaggle's dataset picker.

CelebA is a good fit for this specific comparison: large enough to be a
meaningful benchmark, simple enough (aligned, centered faces) for both models
to converge within a single Kaggle GPU session, and the VAE-vs-DDPM
blur/sharpness trade-off is very visible on faces — making the qualitative
comparison easy to see, not just a number in a table.

## Architecture

| | VAE | DDPM |
|---|---|---|
| **Core idea** | Encode to a low-dimensional Gaussian latent, decode back | Learn to reverse a fixed Markov noising process |
| **Objective** | Reconstruction (MSE) + KL divergence to N(0,I) | Noise-prediction MSE (`L_simple`, Ho et al. 2020) |
| **Generation** | 1 decoder forward pass | `T` (1000) sequential U-Net forward passes |
| **Building blocks** | Conv encoder/decoder (`src/models/encoder.py`, `decoder.py`) | U-Net with residual blocks + self-attention + sinusoidal time embedding (`src/models/unet.py`, `attention.py`) |
| **Sampling options** | Latent prior sampling (`src/sampling/latent_sampler.py`) | Full DDPM (`ddpm_sampler.py`) or fast DDIM, ~10-20x fewer steps (`ddim_sampler.py`) |

## Project structure

```
Gen_AI/
├── README.md
├── CHANGELOG.md              # full version history, see below
├── requirements.txt
├── pytest.ini
├── train.py                  # Train models from the command line
├── sample.py                 # Generate new samples
├── evaluate.py                # Evaluate trained models
├── inference.py                # Run inference on custom images
│
├── configs/
│   ├── dataset.py             # Dataset configuration
│   ├── model.py                # Model hyperparameters
│   ├── diffusion.py             # Diffusion settings
│   ├── training.py               # Training configuration
│   └── config.py                  # Combines all configurations
│
├── src/
│   ├── datasets/
│   │   ├── preprocessing.py    # Prepare dataset paths and metadata
│   │   ├── transforms.py        # Image preprocessing
│   │   ├── augmentations.py      # Data augmentation
│   │   ├── dataset.py             # PyTorch dataset
│   │   └── dataloader.py           # DataLoader creation
│   │
│   ├── models/
│   │   ├── encoder.py           # VAE encoder
│   │   ├── decoder.py            # VAE decoder
│   │   ├── vae.py                 # Variational Autoencoder
│   │   ├── attention.py            # Self-attention block
│   │   ├── unet.py                  # U-Net backbone
│   │   └── diffusion.py              # Diffusion process (noise schedule)
│   │
│   ├── losses/
│   │   ├── vae_loss.py           # VAE loss
│   │   ├── diffusion_loss.py      # Diffusion loss
│   │   └── perceptual_loss.py      # Optional perceptual loss (VGG-based)
│   │
│   ├── training/
│   │   ├── ema.py                # EMA updates
│   │   ├── optimizer.py           # Optimizer setup
│   │   ├── scheduler.py            # Learning-rate scheduler
│   │   ├── callbacks.py             # Training callbacks (checkpoint, sample preview)
│   │   ├── trainer.py                # Base trainer (resume/checkpoint logic)
│   │   ├── train_vae.py               # VAE training
│   │   └── train_ddpm.py               # DDPM training
│   │
│   ├── sampling/
│   │   ├── latent_sampler.py      # Sample latent vectors (VAE)
│   │   ├── ddpm_sampler.py         # Full ancestral DDPM sampling
│   │   ├── ddim_sampler.py          # Fast DDIM sampling
│   │   └── image_sampler.py          # Unified sampling interface
│   │
│   ├── evaluation/
│   │   ├── fid.py                 # FID metric
│   │   ├── quality.py              # Inception Score
│   │   ├── reconstruction.py        # Reconstruction evaluation (VAE only)
│   │   └── evaluator.py              # Evaluation pipeline orchestrator
│   │
│   ├── visualization/
│   │   ├── sampling_grid.py       # Sample grids
│   │   ├── training_curves.py      # Training plots
│   │   ├── reconstruction.py        # Reconstruction results
│   │   ├── interpolation.py          # Latent interpolation
│   │   ├── latent_space.py            # Latent-space PCA visualization
│   │   ├── denoising.py                # Denoising process strip
│   │   └── gif.py                       # Animation generation
│   │
│   ├── utils/
│   │   ├── seed.py                # Random seed utilities
│   │   ├── device.py               # Device selection
│   │   ├── helpers.py               # Common helpers
│   │   ├── image_utils.py            # Image tensor conversion utilities
│   │   ├── checkpoint.py              # Checkpoint save/load
│   │   ├── paths.py                    # Project paths
│   │   └── logger.py                    # Logging utilities
│   │
│   └── engine/
│       ├── trainer.py             # Training orchestrator (thin — no duplicate logic)
│       ├── evaluator.py            # Evaluation orchestrator
│       └── inferencer.py            # Inference orchestrator
│
├── scripts/
│   └── generate_comparison_chart.py  # Regenerates the cross-version FID/IS chart
│
├── notebooks/
│   └── kaggle_run.ipynb          # Kaggle entry point
│
├── tests/                         # Unit tests (pytest)
│
└── outputs/                       # Generated outputs (mostly gitignored —
                                       see below for the exception)
```

**Design note on `engine/`:** it does not duplicate `training/` or
`evaluation/` — it only orchestrates them (builds the dataloader once, calls
`train_vae` then `train_ddpm`, loads checkpoints for evaluation/inference).
All actual logic lives in exactly one place.

**Note on `outputs/`:** checkpoints, logs, and samples are gitignored (too
large for git), but `outputs/v1.0/`, `v1.1/`, `v1.2/`, and
`outputs/comparison/` are explicitly kept — they hold each version's
`results.json` and comparison charts referenced throughout this README.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.x, `torchmetrics` + `torch-fidelity` (FID/IS),
`scikit-learn` (latent-space PCA), `imageio` (denoising GIF).

## How to run

### On Kaggle

1. New Notebook → Settings → Accelerator → GPU (T4 x2 or P100).
2. "Add Input" → search **CelebFaces Attributes (CelebA) Dataset** → Add.
3. Open `notebooks/kaggle_run.ipynb`, paste its cells in order. They write
   this exact file tree via `%%writefile`, then run:

   ```bash
   python train.py
   python evaluate.py
   ```

4. If your dataset mounts at a non-standard path, check
   `src/datasets/preprocessing.py`'s `CELEBA_DIR_CANDIDATES` — add your exact
   path if `train.py` raises `FileNotFoundError`.

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

Trains the VAE first, then the DDPM. Training automatically **resumes** from
the latest checkpoint if one exists (model, optimizer, AMP scaler, and EMA
state are all restored) — safe to re-run after an interrupted session.

### Evaluation

```bash
python evaluate.py
```

Computes FID, Inception Score, and reconstruction metrics, and generates all
visualizations (sample grids, latent-space PCA, interpolation, denoising GIF).

### Sampling

```bash
python sample.py --model ddpm --n 16
```

Generates 16 new images using the selected sampler
(`--model vae`, `--model ddpm`, or `--model both`).

### Inference

Reconstruct an input image:

```bash
python inference.py --reconstruct path/to/image.jpg
```

Generate a new image:

```bash
python inference.py --generate ddpm
```

### Tests

```bash
pytest
```

Runs the project's test suite (model shapes, loss values, sampler output,
dataset pipeline).

---

## Version comparison: v1.0 -> v1.1 -> v1.2

Three full training runs were completed. Each version bundles more than one
change, and the tables below are transparent about that rather than
attributing improvements to a single factor in isolation.

![FID and Inception Score across versions](outputs/comparison/version_comparison.png)

| | **v1.0** | **v1.1** | **v1.2** |
|---|---|---|---|
| Dataset subset | 30,000 images | 40,000 images | 40,000 images |
| VAE epochs | 30 | 40 | 40 |
| DDPM epochs | 40 | 50 | 50 |
| Beta schedule | linear | linear | **cosine** |
| DDPM sampler | ancestral, **no x0 clipping** | ancestral, **x0 clipped to [-1,1] each step** | ancestral, x0 clipped |
| U-Net attention | bottleneck (8x8) only | bottleneck (8x8) only | bottleneck (8x8) **+ 16x16** |
| Gradient clipping | none | none | **max norm 1.0** |
| LR schedule | constant | constant | **cosine decay** |

### What changed at each step

**v1.0 -> v1.1 — bug fix + more training.** The v1.0 ancestral DDPM sampler
never clipped its intermediate predicted-x0 estimate back to the valid pixel
range. Over 1000 sequential steps, small per-step numerical error
compounded — verified directly: an unclipped run produced pixel values
ranging from **-6.06 to 6.72** (valid range is [-1, 1]), which is what caused
some v1.0 DDPM samples to render as solid black/white panels instead of
faces. v1.1 fixes this (`src/sampling/ddpm_sampler.py`) and also increases
the training budget, so the FID improvement reflects both changes combined.

**v1.1 -> v1.2 — architectural refinements, same bug-free baseline.** With
the sampling bug already fixed and the training budget unchanged, v1.2 tests
four documented DDPM quality improvements together: a cosine beta schedule
(Nichol & Dhariwal, "Improved DDPM"), self-attention at an additional
resolution (16x16, not just the 8x8 bottleneck), gradient clipping, and
cosine LR decay. Because v1.1 was already a working baseline, this step
tests refinement rather than bug-fixing — correspondingly, the gain is
smaller than v1.0->v1.1's, which is the expected pattern (large one-time
wins from fixing bugs, smaller incremental wins from architecture tuning).
None of the v1.2 changes touch the VAE, so its numbers are flat by design.

### Quantitative results

| Metric | v1.0 | v1.1 | v1.2 | v1.0->v1.2 |
|---|---|---|---|---|
| VAE FID (lower is better) | 120.69 | 97.94 | 97.06 | -19.6% |
| VAE Inception Score | 2.04 +/- 0.06 | 2.00 +/- 0.08 | 2.00 +/- 0.06 | ~flat |
| VAE reconstruction PSNR (higher is better) | 22.27 dB | 22.48 dB | 22.47 dB | ~flat |
| **DDPM FID (lower is better)** | 43.77 | 28.37 | **26.42** | **-39.6%** |
| DDPM Inception Score | 2.79 +/- 0.12 | 2.59 +/- 0.13 | 2.61 +/- 0.17 | ~flat* |

\* Inception Score is a weaker signal for a single-class domain like faces
(the underlying classifier is trained on 1000 ImageNet classes, none of
which are "face") — FID is the more meaningful metric across all three runs.

Each version's artifacts (`results.json`, sample grids where available) are
kept in `outputs/v1.0/`, `outputs/v1.1/`, `outputs/v1.2/` for direct
comparison, and `outputs/comparison/version_comparison.png` (regenerate with
`python scripts/generate_comparison_chart.py` after adding a new version) —
see [CHANGELOG.md](CHANGELOG.md) for the complete version history.

---

## Results

Final reported numbers (v1.2, current best):

| Model | FID (lower better) | Inception Score (higher better) | Reconstruction PSNR (higher better) | Sampling cost |
|-------|---------------------|-----------------------------------|----------------------------------------|----------------|
| VAE   | 97.06               | 2.00 +/- 0.06                       | 22.47 dB                                | 1 forward pass |
| DDPM  | 26.42               | 2.61 +/- 0.17                       | n/a                                      | 1000 forward passes (full DDPM) |

## Qualitative Analysis

The assignment asks for a subjective assessment covering diversity, realism,
and thematic consistency — based on directly viewing generated sample grids,
not just the FID/IS numbers above.

> **Version note:** this analysis is based on the **v1.0** sample grids
> (`outputs/v1.0/vae_grid.png`, `ddpm_grid.png`). v1.1 fixed the DDPM
> sampling bug described below, so v1.1/v1.2 samples should show
> meaningfully better thematic consistency than what's described here — this
> section should be re-examined against `outputs/v1.2/` grids once available
> and updated accordingly.

**Diversity.** The VAE's 16 samples show genuine variation in age, hair
color, and expression, with no visible mode collapse. DDPM's successful
generations are similarly varied, but 2 of 16 samples in the v1.0 grid
failed to produce a face at all (rendering as a near-blank white or near-solid
black panel), which reduces *effective* diversity — a failed generation
contributes nothing to the usable output distribution.

**Realism.** This is where the architectural difference is most visible.
Every VAE sample shares the same soft, out-of-focus quality — hair blends
into the background, eye and mouth edges are indistinct, colors look
slightly washed out. Several DDPM samples are noticeably sharper — visible
individual hair strands, photographic-looking skin texture, defined eye
catchlights — clearly exceeding any VAE sample's fidelity. DDPM v1.0 also
has realism failures beyond blur: one sample shows an unnatural, matted
texture across the whole face, qualitatively different from a blurry-but-
coherent VAE failure mode.

**Thematic consistency** (does the model reliably stay "on-topic," i.e.
produce a face at all). VAE: 16/16 samples are recognizably human faces,
even the blurriest ones. DDPM v1.0: roughly 14/16 — the blank/corrupted
panels are genuine thematic failures. This is the direct visual signature of
the sampler numerical-drift bug described in
[CHANGELOG.md](CHANGELOG.md#v11) (verified pixel values reaching -6.06 to
6.72, well outside the valid [-1,1] range) — it is a sampling-implementation
issue fixed in v1.1, not an inherent limitation of diffusion models. Worth
stating explicitly in the report: DDPM's v1.0 thematic-consistency score
understates the architecture's real capability once sampled correctly.

**Summary:** VAE prioritizes reliability (always produces *a* face) at the
cost of sharpness. DDPM v1.0 produces higher peak realism but at the cost of
occasional total failures — a trade-off that the v1.1 sampling fix
substantially closes, which the FID numbers (43.77 -> 28.37 -> 26.42) are
consistent with even though this qualitative section hasn't yet been
re-verified against the newer sample grids.



- **Architecture difference:** the VAE learns an explicit, low-dimensional
  latent distribution via an encoder/decoder trained with a
  reconstruction + KL objective; the DDPM has no explicit compressed latent
  — it learns to reverse a fixed Markov noising process, trained with a
  simple noise-prediction MSE loss.
- **Sample quality:** DDPM FID is ~3.7x better than VAE FID, consistent
  across all three training runs — matches the theoretical expectation that
  diffusion models produce sharper, more realistic samples, since each
  denoising step only has to make a small, local correction rather than
  reconstruct the whole image in one pass.
- **Speed trade-off:** VAE sampling is a single forward pass; DDPM (full)
  requires 1000 sequential passes. The DDIM sampler (`src/sampling/ddim_sampler.py`)
  narrows this gap to ~50 steps at some quality cost — set
  `cfg.diffusion.sampler = "ddim"` to use it.
- **Latent space sanity check:** `outputs/v1.1/latent_space/vae_latent_pca.png`
  and `outputs/v1.1/interpolation/vae_interpolation.png` show the VAE learned
  a smooth, meaningful latent space (no posterior collapse).
- **Denoising process:** `outputs/v1.1/gifs/ddpm_denoising.gif` makes DDPM's
  iterative refinement process visually concrete.

## Known limitations / next steps

- **Training budget is still modest** relative to published DDPM results on
  CelebA (which typically use the full ~200k images and hundreds of epochs);
  FID of 26.42 is a credible result at this budget, not a ceiling.
- **DDIM sampling hasn't been benchmarked yet** — the repo supports it
  (`cfg.diffusion.sampler = "ddim"`), but no results.json has been generated
  with it. Worth adding as a speed/quality trade-off data point.
- **VAE architecture is unchanged since v1.0.** All improvements so far
  targeted DDPM; a candidate v1.3 could apply a matching set of upgrades to
  the VAE (e.g. perceptual loss, already implemented but off by default in
  `configs/training.py`).

## Deliverables mapping

- **(a) Report:** the Results table, Version comparison section, and Report
  notes above as an outline; embed the visualization PNGs/GIF from
  `outputs/v1.2/` (or `v1.1/` where v1.2-specific images weren't generated)
  for qualitative evidence.
- **(b) GitHub repo:** this repository, structured so `git clone` +
  `pip install -r requirements.txt` + `python train.py` reproduces everything.
