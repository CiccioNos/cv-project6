# Frequency-Domain Diffusion Models for Deepfake Analysis

## Idea

Generative models leave detectable artifacts in the frequency spectrum of their outputs, even when images
look realistic in the spatial domain; this is what state-of-the-art deepfake detectors exploit (Frank et
al., 2020). Instead of training a standard DDPM on RGB pixels, we train a DDPM
**directly on the Fourier spectrum** of face images, and compare it against an identical RGB-DDPM baseline
(same architecture, data, and budget) to see whether modeling the true frequency distribution reduces the
detectable spectral fingerprint of generated images.

## Data and normalization

Dataset: FFHQ, resized to 128×128, scaled to `[-1, 1]`. A held-out split is never used in training and 
serves as the real-image reference for evaluation.

For the frequency model, each image is transformed with a centered 2D FFT and split into real + imaginary
channels (3 RGB channels to 6). We use real/imaginary rather than magnitude/phase because diffusion adds
unbounded Gaussian noise at every step: real/imaginary are well-behaved under this, while magnitude is
non-negative and phase wraps at ±π, both incompatible with plain Gaussian noise and an L2 loss. Since raw
FFT coefficients span an extreme dynamic range (low frequencies dominate), the spectrum is compressed with
a symmetric log transform and standardized per channel (statistics computed once on the training set) before
diffusion — without this, high frequencies are effectively ignored during training.

## Model and training

Both models share the same UNet (residual blocks, sinusoidal timestep embedding, self-attention at the
bottleneck), differing only in input/output channels (3 vs. 6).
The UNet is charachterized by 4 iterative ResBlock with a downsapling convolution, from a resolution of 128x128
to a 8x8, where the ResBlock is concatenated with a SelfAttention block that implements a 64x64 matrix. The same
pattern of the downsampling part is replicated for the upsampling, using a ConTranspose2D for the resizing.
Timesteps are included in each block with a SinusoidalPosEmb.
This thinner version of the standard DDPM UNet description was chosen for the training constrains that we had.
Training follows the standard DDPM recipe: a linear noise schedule over `T=1000` steps, network trained to
predict the added Gaussian noise (L2 loss) on RGB pixels for the baseline, on the normalized spectrum for
the frequency model.

## Sampling and evaluation

Sampling uses DDIM (Song et al. 2021) with 50 steps and `η = 0` (deterministic, best quality in
this low-step regime, and reproducible for a fair comparison), reusing the same trained network — no extra
training required.

Metrics:
- **FID / KID**, computed on DDIM samples vs. the held-out real set, same sample count for
  both models.
- **Radial power spectrum** compared across real images, RGB-DDPM samples, and frequency-DDPM samples; 
  the core metric of the project, showing whether the frequency-domain model
  reduces the grid-like spectral artifacts typical of spatial generative models.
- Qualitative visual inspection of generated samples.

## Repository contents

- `freq_diffusion_ddpm_starter.ipynb` — full pipeline (Imports, Globals, Utils, Data, Network, Train,
  Evaluation); `DOMAIN` toggles between the RGB baseline and the frequency model.
- Dataset link: FFHQ 128×128.
- Project presentation (slides).
- requirements.txt: packages required for running.

## How to run

1. Use a 3.12 version of python and install necesary packages described in the requirements.txt file.
2. Download the correct version of torch, torchvision for your hardware (if needed, e.g. cuda support).
3. Set `DATA_DIR` to the FFHQ folder or download it via the designated box.
4. Set `DOMAIN` (`"rgb"` or `"fft"`) and run Imports → Train to train that model (both need a separate run).
5. Run Evaluation to sample via DDIM and compute FID/IS and the radial spectrum comparison.

## Results

Both models were trained for 100 epochs under an identical protocol (same UNet, 11.4M parameters; batch
size 32; AdamW, lr 2e-4; linear β schedule, T=1000; seed 42; mixed-precision on an RTX 5070 Ti — the domain
is the only variable that changes between the two runs). Evaluation uses deterministic DDIM sampling (50
steps, η=0) from the epoch-100 checkpoints. FID/IS are computed with `torchmetrics` on 5,000 generated
images against the 7,000-image held-out validation split.

| | RGB baseline | Frequency model |
|---|---|---|
| FID ↓ | 84.41 | 174.02 |
| KID ↓ | 0.0543 ± 0.0020 | 0.1590 ± 0.0024 |

The radial power spectrum (log scale, 64 radial bins, 256 images per condition) tells the same story: the
RGB baseline tracks the real spectrum closely across the informative range, with no periodic grid
artifacts, while the frequency model sits roughly one order of magnitude below the real curve at every
radius and falls further behind in the high-frequency tail — it under-generates spectral energy rather than
matching it.

**Under this training budget, the hypothesis is not confirmed**: diffusing directly on the Fourier spectrum
did not improve spectral realism, and performed worse than the RGB baseline on both perceptual quality
(FID/IS) and spectral fidelity. Two likely causes: (1) *Hermitian redundancy* — the real/imaginary
parameterization carries about twice the necessary information, and the inverse transform discards the
residual imaginary component, so part of the network's capacity is spent on an output that is thrown away;
(2) errors in the spectral domain are *global* (every coefficient contributes to every output pixel), while
RGB errors stay local — with only 100 epochs and 50 DDIM steps neither model is fully converged, but this
asymmetry hits the frequency model much harder, producing distorted-but-recognizable faces in RGB versus
drifting colors/shapes in frequency.

**Future work:** a half-spectrum (`rfft2`) or magnitude/phase parameterization to remove the redundancy;
EMA weights; a larger training budget before drawing conclusions about the domain itself.

## References

1. Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
2. Song, Meng & Ermon, *Denoising Diffusion Implicit Models*, ICLR 2021.
3. Ronneberger, Fischer & Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
4. Frank et al., *Leveraging Frequency Analysis for Deep Fake Image Recognition*, ICML 2020.
5. Corvi et al., *On the Detection of Synthetic Images Generated by Diffusion Models*, ICASSP 2023.
6. Karras, Laine & Aila, *A Style-Based Generator Architecture for GANs*, CVPR 2019 (FFHQ dataset).
