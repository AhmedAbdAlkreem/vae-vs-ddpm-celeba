# Gen_AI — VAE vs DDPM from Scratch

**Task code:** GenCV003 · **Position:** CV Engineer

A production-structured implementation of a **Variational Autoencoder (VAE)**
and a **Denoising Diffusion Probabilistic Model (DDPM)**, both built entirely
from scratch (only basic `torch.nn` building blocks), trained on CelebA, and
benchmarked quantitatively (FID, Inception Score, reconstruction PSNR) and
qualitatively (sample grids, latent-space visualization, interpolation,
denoising-process animation).

## Dataset

[CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
(64x64 aligned/cropped face crops). Good fit for this comparison: large
enough to be a meaningful benchmark, simple enough (aligned, centered faces)
for both models to converge within a Kaggle GPU quota, and the VAE-vs-DDPM
blur/sharpness trade-off is very visible on faces.

## Project structure

```
Gen_AI/
├── train.py / sample.py / evaluate.py / inference.py   # CLI entry points
├── configs/            # dataset.py, model.py, diffusion.py, training.py -> composed in config.py
├── src/
│   ├── datasets/        # dataset, dataloader, transforms, preprocessing, augmentations
│   ├── models/           # encoder, decoder, vae, unet, attention, diffusion (noise schedule)
│   ├── losses/           # vae_loss, diffusion_loss, perceptual_loss (optional VGG)
│   ├── training/          # trainer (base), train_vae, train_ddpm, optimizer, scheduler, callbacks, ema
│   ├── sampling/           # ddpm_sampler (full T-step), ddim_sampler (fast, ~10-20x fewer steps),
│   │                        latent_sampler, image_sampler (high-level API)
│   ├── evaluation/          # evaluator (orchestrates), fid, quality (IS), reconstruction
│   ├── visualization/        # sampling_grid, training_curves, reconstruction, interpolation,
│   │                          latent_space, denoising (trajectory), gif
│   ├── utils/                 # seed, device, logger, checkpoint, paths, image_utils, helpers
│   └── engine/                 # thin orchestration layer (Trainer/Evaluator/Inferencer) —
│                                  wires together training/evaluation/sampling, contains no
│                                  duplicate logic of its own
├── notebooks/kaggle_run.ipynb   # Kaggle entry point (writes this file tree, then runs it)
├── tests/                       # pytest sanity tests (shapes, finite losses, dataset pipeline)
└── outputs/                     # checkpoints/, logs/, samples/, reconstructions/, interpolation/,
                                    latent_space/, denoising/, training_curves/, gifs/  (all gitignored)
```

**Design note on `engine/`:** it does not duplicate `training/` or
`evaluation/` — it only orchestrates them (builds the dataloader once, calls
`train_vae` then `train_ddpm`, loads checkpoints for evaluation/inference).
All actual logic lives in exactly one place.

## How to run on Kaggle

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

## How to run locally / any GPU box

```bash
pip install -r requirements.txt
python train.py       # trains VAE then DDPM, with resumable checkpointing
python evaluate.py    # FID / IS / reconstruction metrics + all visualizations
python sample.py --model ddpm --n 16     # generate new images
python inference.py --reconstruct my_face.jpg
python inference.py --generate ddpm
pytest                # run the test suite
```

## Resumability

Both trainers checkpoint every `training.checkpoint_every` epochs (default:
2) to `outputs/checkpoints/{vae,ddpm}_checkpoint.pt`, including optimizer,
AMP scaler, and EMA state. Re-running `train.py` after an interrupted session
(e.g. a Kaggle timeout) automatically resumes from the last checkpoint
instead of restarting.

## DDPM sampling speed: DDPM vs DDIM

Vanilla DDPM sampling requires `T` (default 1000) sequential U-Net forward
passes per image — the main practical bottleneck of diffusion models. Set
`cfg.diffusion.sampler = "ddim"` (and tune `ddim_steps`, default 50) to use
the DDIM sampler instead: same trained model, ~10-20x fewer steps, small
quality trade-off. Useful for fast iteration; use `"ddpm"` for final reported
numbers to match the training-time process exactly.

## Environment

Python 3.10+, PyTorch 2.x, torchvision, `torchmetrics` + `torch-fidelity`
(FID/IS), `scikit-learn` (latent-space PCA), `imageio` (denoising GIF).

## Results

Filled in after training (also written to `outputs/results.json` and the
various `outputs/*` visualization folders):

| Model | FID ↓ | Inception Score ↑ | Reconstruction PSNR ↑ | Sampling cost |
|-------|-------|--------------------|-------------------------|----------------|
| VAE   |       |                    |                          | 1 forward pass |
| DDPM  |       |                    |         n/a              | T (or ddim_steps) forward passes |

## Report notes (deliverable a)

- **Architecture difference:** the VAE learns an explicit, low-dimensional
  latent distribution via an encoder/decoder trained with a
  reconstruction + KL objective; the DDPM has no explicit compressed latent
  — it learns to reverse a fixed Markov noising process, trained with a
  simple noise-prediction MSE loss.
- **Sample quality trade-off:** VAEs are typically blurrier because the
  pixel-wise reconstruction loss and Gaussian decoder assumption average over
  plausible outputs; DDPM samples are usually sharper/more realistic because
  each denoising step only has to make a small, local correction. Use
  `outputs/comparison_grid.png` as direct visual evidence.
- **Speed trade-off:** VAE sampling is a single forward pass; DDPM (full)
  requires `T` sequential passes — DDIM narrows this gap at a quality cost.
- **Latent space sanity check:** `outputs/latent_space/vae_latent_pca.png`
  and `outputs/interpolation/vae_interpolation.png` show whether the VAE
  learned a smooth, meaningful latent space (vs. posterior collapse).
- **Denoising process:** `outputs/gifs/ddpm_denoising.gif` makes the
  iterative refinement process visually concrete for the report.
- Use the FID/IS/PSNR numbers in `outputs/results.json` as the quantitative
  evidence for all of the above claims.

## Deliverables mapping

- **(a) Report:** the Results table + "Report notes" section above as an
  outline; embed the visualization PNGs/GIF for qualitative evidence.
- **(b) GitHub repo:** this repository, structured so `git clone` +
  `pip install -r requirements.txt` + `python train.py` reproduces everything.
