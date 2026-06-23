# DPI Subdirectory

This directory contains the executable RCIR implementation built on top of the original DPI codebase.

Please read the project-level documentation at [../README.md](../README.md) for:

- environment setup;
- required scene assets and DDPM checkpoints;
- albedo and normal prior configuration;
- inverse-rendering commands;
- relighting consistency options;
- output directory structure.

Main entry points:

```bash
python3 sample_condition.py \
  --model_config=configs/model_config_outdoor.yaml \
  --diffusion_config=configs/diffusion_config.yaml \
  --task_config=configs/raytracing_config_outdoor.yaml \
  --gpu=0 \
  --save_dir=./results
```

Optional post-refinement:

```bash
python3 material_optimization.py \
  --task_config=configs/raytracing_config_outdoor.yaml \
  --gpu=0 \
  --save_dir=./results
```
