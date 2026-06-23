# RCIR: Enhanced Single-View Image Inverse Rendering via Pre-trained Diffusion and Relighting Consistency

RCIR is a single-view inverse rendering project built on DPI, IntrinsicAnything, and StableNormal. The current codebase extends the original DPI pipeline with the main algorithmic ideas from the RCIR paper:

- environment map sampling with a pre-trained illumination DDPM and differentiable path tracing;
- albedo and normal prior initialization from pre-trained models;
- joint optimization of material and optional normal-map parameters after the high-noise denoising stage;
- relighting consistency regularization with a low-rank intrinsic-property constraint.

The repository currently contains the source code and configuration templates. Datasets, scene assets, and model checkpoints are not bundled.

## Project Structure

```text
RCIR-main/
├── README.md
└── DPI/
    ├── sample_condition.py                  # main inverse-rendering / sampling entry
    ├── material_optimization.py             # optional post-refinement script
    ├── configs/
    │   ├── model_config_outdoor.yaml
    │   ├── model_config_indoor.yaml
    │   ├── diffusion_config.yaml
    │   ├── raytracing_config_outdoor.yaml
    │   └── raytracing_config_indoor.yaml
    ├── guided_diffusion/
    │   ├── condition_methods.py             # RT posterior sampling and timestep scheduling
    │   ├── gaussian_diffusion.py
    │   ├── measurements.py                  # Mitsuba operator, priors, relighting consistency
    │   └── unet.py
    ├── IntrinsicAnything-master/            # albedo prior project snapshot
    ├── StableNormal-main/                   # normal prior project snapshot
    ├── camera/
    └── util/
```

## Environment

The original implementation targets PyTorch 1.11, CUDA 11.3, and Mitsuba 3. A CUDA GPU is strongly recommended because the renderer uses Mitsuba's CUDA AD variant.

```bash
cd DPI
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mitsuba
```

If you want to generate priors inside this repository, also install the dependency files under:

```bash
DPI/IntrinsicAnything-master/requirements.txt
DPI/StableNormal-main/requirements.txt
```

## Required Assets

Prepare the following assets before running inverse rendering.

1. Illumination DDPM checkpoints:

```text
DPI/models/outdoor.pt
DPI/models/indoor.pt
```

The checkpoint paths are configured in `DPI/configs/model_config_outdoor.yaml` and `DPI/configs/model_config_indoor.yaml`.

2. Scene data for Mitsuba:

```text
DPI/data/<scene_name>/
├── images/
│   ├── 0.exr or 0.png
│   ├── 1.exr or 1.png
│   └── ...
├── scene.xml
└── camera.xml
```

The ray-tracing configs expect a Mitsuba scene with material parameters named like:

```yaml
param_keys:
  basecolor: OBJMesh.bsdf.base_color.data
  roughness: OBJMesh.bsdf.roughness.data
  metallic: OBJMesh.bsdf.metallic.data
  normal:
```

If your XML uses different parameter names, update `measurement.operator.param_keys` in the task config.

3. Optional albedo and normal priors:

Use IntrinsicAnything to estimate an albedo map and StableNormal to estimate a normal map from the input view, then write their paths into the task config:

```yaml
prior:
  albedo_path: /path/to/albedo.png
  normal_path: /path/to/normal.png
  normal_space: rgb01
```

If these fields are left empty, RCIR falls back to constant initialization:

- albedo: `0.01`
- roughness: `0.1`
- metallic: `0.1`
- normal: `[0.5, 0.5, 1.0]`

## Run Inverse Rendering

Outdoor example:

```bash
cd DPI
python3 sample_condition.py \
  --model_config=configs/model_config_outdoor.yaml \
  --diffusion_config=configs/diffusion_config.yaml \
  --task_config=configs/raytracing_config_outdoor.yaml \
  --gpu=0 \
  --save_dir=./results
```

Indoor example:

```bash
cd DPI
python3 sample_condition.py \
  --model_config=configs/model_config_indoor.yaml \
  --diffusion_config=configs/diffusion_config.yaml \
  --task_config=configs/raytracing_config_indoor.yaml \
  --gpu=0 \
  --save_dir=./results
```

Main outputs are written under:

```text
DPI/results/<operator_name>/
├── input/
├── label/
├── recon/                 # sampled environment maps
├── recon_measurement/     # rendered reconstruction previews
├── progress/
└── material/
    ├── *.pt
    ├── basecolor/
    ├── roughness/
    ├── metallic/
    └── normal/
```

## Optional Material Refinement

After sampling, you can run the original DPI-style material refinement:

```bash
cd DPI
python3 material_optimization.py \
  --task_config=configs/raytracing_config_outdoor.yaml \
  --gpu=0 \
  --save_dir=./results
```

This script reads sampled environment maps and material outputs from `results/<operator_name>/`.

## Important Configuration Fields

`conditioning.params` controls the posterior sampling update:

```yaml
scale_method: paper
rho_min: 0.1
rho_max: 1.0
rho_transition: 0.6
joint_optimization_start: 0.8
grad_clip: 0.1
spp: 16
```

With `scale_method: paper`, RCIR keeps the rendering-gradient coefficient small in early denoising and increases it when the timestep drops below `rho_transition`. Joint material and normal optimization starts when normalized timestep is lower than `joint_optimization_start`.

`measurement.operator.relighting` controls relighting consistency:

```yaml
relighting:
  enabled: true
  lambda_initial: 0.2
  lambda_final: 0.5
  rank_weight: 1.0
  image_weight: 1.0
  env_shift: 64
  spp: 8
```

If no optimizable normal texture parameter is found in the Mitsuba scene, normal consistency is disabled automatically and material optimization still runs.

## Notes

- The code assumes 256 x 256 environment maps and texture maps by default.
- For LDR inputs set `measurement.operator.ldr: true`; for EXR/HDR inputs set it to `false`.
- `illumi_gamma`, `illumi_scale`, and `illumi_normalize` are scene-dependent and usually need tuning.
- The bundled `IntrinsicAnything-master` and `StableNormal-main` directories are upstream snapshots used to generate initialization priors; their checkpoints are not included.

## Acknowledgments

This project is mainly based on:

- DPI: https://github.com/LinjieLyu/DPI
- IntrinsicAnything: https://github.com/zju3dv/IntrinsicAnything
- StableNormal: https://github.com/Stable-X/StableNormal

We thank the authors of these projects for their excellent work.
