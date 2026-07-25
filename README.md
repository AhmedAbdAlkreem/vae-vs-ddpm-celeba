# Gen_AI — VAE vs DDPM from Scratch

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)


A from-scratch PyTorch implementation and head-to-head comparison of two
generative model families — a **Variational Autoencoder (VAE)** and a
**Denoising Diffusion Probabilistic Model (DDPM)** — trained on CelebA faces.
Both models are built from base `torch.nn` primitives only: no pretrained
VAE, no `diffusers` library. Benchmarked with quantitative metrics (FID,
Inception Score, reconstruction PSNR) and a qualitative visual analysis
(diversity, realism, thematic consistency), across three iterated training
versions.

---

## Table of contents

- [Dataset](#dataset)
- [Architecture at a glance](#architecture-at-a-glance)
- [Project structure](#project-structure)
- [Installation and usage](#installation-and-usage)
  - [Run on your own device](#run-on-your-own-device)
  - [Run on Kaggle](#run-on-kaggle)
- [Quantitative comparison](#quantitative-comparison)
- [Qualitative comparison](#qualitative-comparison)
- [Results summary](#results-summary)
- [Known limitations / roadmap](#known-limitations--roadmap)

---

## Dataset

[CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
— 64×64 aligned/cropped face crops.

CelebA is a good fit for this specific comparison: large enough to be a
meaningful benchmark, simple enough (aligned, centered faces) for both models
to converge within a modest GPU budget, and the VAE-vs-DDPM
blur/sharpness trade-off is very visible on faces — making the qualitative
comparison easy to see, not just a number in a table.

## Architecture at a glance

| | VAE | DDPM |
|---|---|---|
| **Core idea** | Encode to a low-dimensional Gaussian latent, decode back | Learn to reverse a fixed Markov noising process |
| **Objective** | Reconstruction (MSE) + KL divergence to N(0,I) | Noise-prediction MSE (`L_simple`, Ho et al. 2020) |
| **Generation cost** | 1 decoder forward pass | 1000 sequential U-Net forward passes (full DDPM) |
| **Building blocks** | Conv encoder/decoder (`src/models/encoder.py`, `decoder.py`) | U-Net: residual blocks + self-attention (8x8 and 16x16) + sinusoidal time embedding |
| **Sampling options** | Latent prior sampling | Full DDPM or fast DDIM (~50 steps, some quality trade-off) |

## Project structure

```
Gen_AI/
├── README.md
├── CHANGELOG.md              # full version history
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
│   ├── datasets/       # dataset, dataloader, transforms, preprocessing, augmentations
│   ├── models/          # encoder, decoder, vae, unet, attention, diffusion (noise schedule)
│   ├── losses/            # vae_loss, diffusion_loss, perceptual_loss (optional VGG)
│   ├── training/            # trainer (base), train_vae, train_ddpm, optimizer, scheduler, callbacks, ema
│   ├── sampling/               # ddpm_sampler (full), ddim_sampler (fast), latent_sampler, image_sampler
│   ├── evaluation/                # evaluator (orchestrates), fid, quality (IS), reconstruction
│   ├── visualization/                # sampling_grid, training_curves, reconstruction, interpolation,
│   │                                   latent_space, denoising (trajectory), gif
│   ├── utils/                          # seed, device, logger, checkpoint, paths, image_utils, helpers
│   └── engine/                          # thin orchestration layer (Trainer/Evaluator/Inferencer) —
│                                           no duplicate logic, wires together training/evaluation/sampling
│
├── scripts/
│   └── generate_comparison_chart.py    # regenerates the cross-version FID/IS chart
│
├── notebooks/
│   └── kaggle_run.ipynb          # Kaggle entry point (writes this file tree, then runs it)
│
├── tests/                         # pytest sanity tests
│
└── outputs/                       # results (mostly gitignored — v1.0/, v1.1/, v1.2/, and
                                       comparison/ are kept, see below)
```

**Design note on `engine/`:** it does not duplicate `training/` or
`evaluation/` — it only orchestrates them. All actual logic lives in exactly
one place.

---

## Installation and usage

### Run on your own device

**1. Clone and install dependencies**

```bash
git clone https://github.com/AhmedAbdAlkreem/vae-vs-ddpm-celeba
cd Gen_AI
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.x. A GPU is strongly recommended — the
DDPM sampling step (1000 sequential U-Net passes per generated image) is
very slow on CPU.

**2. Download the CelebA dataset**

Unlike Kaggle, there's no auto-mounted dataset on your own machine — get it
one of two ways:

```bash
# Option A: Kaggle API (needs a Kaggle account + API token in ~/.kaggle/kaggle.json)
pip install kaggle
kaggle datasets download -d jessicali9530/celeba-dataset
unzip celeba-dataset.zip -d celeba-dataset
```

Option B: download manually from the
[dataset page](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
and unzip it locally.

Either way, note the full path to the folder containing the `.jpg` files
(e.g. `.../celeba-dataset/img_align_celeba/img_align_celeba`).

**3. Point the config at your local dataset path**

```python
from configs.config import Config

cfg = Config()
cfg.dataset.data_dir = "/full/path/to/celeba-dataset/img_align_celeba/img_align_celeba"
cfg.output_dir = "outputs"
```

**4. Sanity-check the install before training**

```bash
pytest
```

Validates every model, loss, and sampler in the codebase in well under a
minute — run this first so a broken local environment is caught immediately,
not after an hour of training.

**5. Train, evaluate, generate**

```bash
python train.py        # trains VAE then DDPM; resumes automatically if interrupted
python evaluate.py      # FID / Inception Score / reconstruction metrics + all visualizations
python sample.py --model ddpm --n 16
python inference.py --reconstruct path/to/your_photo.jpg
python inference.py --generate ddpm
```

### Run on Kaggle

1. New Notebook -> Settings -> Accelerator -> GPU (T4 x2 or P100).
2. "Add Input" -> search **CelebFaces Attributes (CelebA) Dataset** -> Add.
3. Open `notebooks/kaggle_run.ipynb`, paste its cells in order. They write
   this exact file tree via `%%writefile`, then run `train.py` and
   `evaluate.py`.
4. If your dataset mounts at a non-standard path, check
   `src/datasets/preprocessing.py`'s `CELEBA_DIR_CANDIDATES` — add your exact
   path if `train.py` raises `FileNotFoundError`.

---

## Quantitative comparison

Three full training versions were completed, each documented in
[CHANGELOG.md](CHANGELOG.md). The table and chart below compare all three
directly rather than showing only the final result — the iteration itself is
part of the story.

![FID and Inception Score across versions](outputs/comparison/version_comparison.png)

| | **v1.0** | **v1.1** | **v1.2** |
|---|---|---|---|
| Dataset subset | 30,000 images | 40,000 images | 40,000 images |
| VAE / DDPM epochs | 30 / 40 | 40 / 50 | 40 / 50 |
| Beta schedule | linear | linear | **cosine** |
| DDPM sampler | ancestral, **no x0 clipping** | ancestral, **x0 clipped** | ancestral, x0 clipped |
| U-Net attention | 8x8 only | 8x8 only | 8x8 **+ 16x16** |
| Gradient clipping / LR decay | none | none | **yes / yes** |
| VAE FID (lower better) | 120.69 | 97.94 | 97.06 |
| VAE Inception Score | 2.04 +/- 0.06 | 2.00 +/- 0.08 | 2.00 +/- 0.06 |
| VAE reconstruction PSNR (higher better) | 22.27 dB | 22.48 dB | 22.47 dB |
| **DDPM FID (lower better)** | 43.77 | 28.37 | **26.42** |
| DDPM Inception Score | 2.79 +/- 0.12 | 2.59 +/- 0.13 | 2.61 +/- 0.17 |

**Overall change, v1.0 -> v1.2: VAE FID -19.6%, DDPM FID -39.6%.**

- **v1.0 -> v1.1** (largest single jump): fixed a real sampling bug — the
  ancestral DDPM sampler never clipped its intermediate predicted-x0 estimate
  back to the valid `[-1, 1]` pixel range. Over 1000 steps, numerical error
  compounded; verified directly, an unclipped run produced pixel values from
  **-6.06 to 6.72**. This produced solid black/white panels in place of
  faces in some v1.0 samples. Combined with a larger training budget in the
  same run, so the FID gain reflects both factors together.
- **v1.1 -> v1.2** (smaller, incremental gain): with the bug already fixed,
  this step tested architectural refinements on an already-working baseline
  — cosine beta schedule, attention at an additional resolution, gradient
  clipping, cosine LR decay. Smaller gains here are the expected pattern:
  large one-time wins from bug fixes, smaller incremental wins from tuning.
  None of the v1.2 changes touched the VAE, hence its flat numbers.

Inception Score is a secondary signal throughout — it's derived from a
1000-class ImageNet classifier, and faces aren't one of those classes, so IS
is less informative here than FID.

---

## Qualitative comparison

The numbers in the previous section tell you *that* DDPM improved. This
section shows *what that improvement actually looked like* — comparing the
broken v1.0 output against the current v1.2 model, then digging into what
each model's internal representation reveals.

### The story in one comparison

**v1.0 — first full training run, sampler bug present:**

![DDPM samples v1.0, showing corrupted panels](outputs/v1.0/ddpm_grid.png)

Two of these sixteen DDPM samples aren't faces at all — a near-blank white
panel and a near-solid black panel. This is the direct visual fingerprint of
the numerical drift bug described in the Quantitative comparison above
(verified pixel values reaching -6.06 to 6.72, outside the valid [-1, 1]
range).

**v1.2 — current best, sampler fixed, architecture refined:**

![Real vs VAE vs DDPM comparison, v1.2](outputs/v1.2/comparison_grid.png)

No blank or corrupted panels. Every DDPM sample in this v1.2 grid is a
coherent face — several are hard to distinguish from the real reference
images at a glance (the man in the dark cap, the woman with wavy brown
hair). This single side-by-side is the clearest evidence that the v1.1 bug
fix and v1.2 refinements produced a real, visible improvement — not just a
better number on a chart.

### Diversity

Both models produce varied output across age, hair color, gender
presentation, and expression — no visible mode collapse in either. The VAE's
diversity is easy to see despite the blur; DDPM's v1.2 diversity is now
fully *usable* diversity, since (unlike v1.0) none of the sampled faces fail
outright.

### Realism

The architectural difference remains clearly visible even after DDPM's
improvements. Every VAE face in the comparison grid shares the same soft,
out-of-focus quality — hair blends into the background, eyes and mouths lack
sharp edges, colors read as slightly washed out. This is architecture-bound,
not a training deficiency: it comes directly from the pixel-wise
reconstruction loss VAEs optimize. The v1.2 DDPM faces, by contrast, show
photographic-level sharpness — individual hair strands, defined eye
catchlights, natural skin texture — with none of the v1.0 "matted texture"
or corrupted-panel artifacts.

### Thematic consistency

VAE: every sample across every version has been a recognizable face — this
has never been in question. DDPM: v1.0 was roughly 14/16 (2 outright
failures); the v1.2 comparison grid above shows **0 visible failures** in
the 9 DDPM samples shown. This is the qualitative counterpart to the FID
improvement, and arguably the more convincing evidence, since it's a direct
"does this look broken" check rather than a statistical distance metric.

### Beyond sample grids: what's inside each model

Sample grids show final output. These four visualizations (all from v1.2)
show *how* each model represents and generates a face internally —
important supporting evidence for claims made in the report notes above.

**VAE reconstruction fidelity** — encode a real photo, decode it back, and
compare side by side (top row: real; bottom row: reconstructed):

![VAE reconstructions](outputs/v1.2/vae_reconstructions.png)

The VAE clearly preserves identity-relevant structure — hair color, pose,
approximate facial geometry all carry through — while losing fine detail,
which is the direct visual meaning behind the 22.47 dB reconstruction PSNR
reported above.

**VAE latent space (PCA projection)** — a scatter of 2000 real images'
encoded latent vectors, projected to 2D:

![VAE latent space PCA](outputs/v1.2/vae_latent_pca.png)

A single dense, roughly Gaussian-shaped cloud centered near the origin, with
no isolated clusters and no collapse to a single point. This is the
qualitative check for **posterior collapse** — a common VAE failure mode
where the model learns to ignore the latent code entirely. This plot shows
that didn't happen: the encoder is genuinely using the latent space.

**VAE latent interpolation** — decoding a straight line between two random
latent vectors, ten steps apart:

![VAE latent interpolation](outputs/v1.2/vae_interpolation.png)

The transition between the two endpoint identities is smooth and gradual,
with no abrupt jumps between frames — evidence that the learned latent space
is continuous and semantically meaningful, not a lookup table of
memorized points.

**DDPM denoising trajectory** — intermediate snapshots of a single image
being generated, from pure noise to final output:

![DDPM denoising strip](outputs/v1.2/ddpm_denoising_strip.png)

The first several frames are visually indistinguishable from random static;
a face silhouette becomes visible only in the last third of the sequence,
sharpening rapidly in the final few steps. This is the most direct visual
demonstration of what "1000 sequential forward passes" (from the
Architecture table) actually means in practice — the model spends most of
its steps making imperceptibly small corrections, with the recognizable
image only emerging very late in the process.

### Summary

The v1.0 → v1.2 journey is visible qualitatively, not just numerically: a
model that produced outright broken output roughly 1-in-8 times now produces
consistently coherent, often sharp results, while the VAE's characteristic
blur — an architectural trait, not a bug — remains present but stable across
all three versions. The supporting visualizations above back up every claim
made in the Report notes with actual evidence rather than assertion.

---

## Results summary

Final reported numbers (v1.2, current best):

| Model | FID (lower better) | Inception Score (higher better) | Reconstruction PSNR (higher better) | Sampling cost |
|-------|---------------------|-----------------------------------|----------------------------------------|----------------|
| VAE   | 97.06 | 2.00 +/- 0.06 | 22.47 dB | 1 forward pass |
| DDPM  | 26.42 | 2.61 +/- 0.17 | n/a | 1000 forward passes (full DDPM) |

## Known limitations / roadmap

- **VAE unchanged since v1.0** — enabling the already-implemented perceptual
  loss (`configs/training.py`: `use_perceptual_loss`) is the highest-leverage
  next step for reducing VAE blur.
- **DDIM sampler is implemented but not yet benchmarked** — no results.json
  generated with `cfg.diffusion.sampler = "ddim"` yet.
- **Training budget is modest** relative to published DDPM results on
  CelebA (full ~200k images, hundreds of epochs); current numbers are
  credible at this budget, not a ceiling.

See [CHANGELOG.md](CHANGELOG.md) for the complete, version-by-version
technical history, including every bug found and fixed along the way.
