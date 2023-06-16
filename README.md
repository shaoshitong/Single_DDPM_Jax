## Denoise Diffusion Probability Model

All code in this repository are copied from Yang Song's SDE repository, but I separate DDPM's code and only present it.


### Install

```bash
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade jaxlib
pip install --upgrade git+https://github.com/google/flax.git
pip insrall ml-collections tensorflow-gan tensorflow_io tensorflow_datasets tensorflow tensorflow-addons tensorboard absl-py
```
Note that for Jax, you should additionally install cudnn and cuda, you can follow [jax](https://github.com/google/jax#pip-installation-gpu-cuda-installed-locally-harder)'s guidance.

### Start Training

```bash
CUDA_VISIBLE_DEVICES=0,1  python main.py --config ./configs/ddpm/ddpm_uncontinuous.py --eval_folder ./tmp/ddpm/eval --mode train --workdir ./tmp/ddpm
```

### Evaluating


```bash
CUDA_VISIBLE_DEVICES=0,1  python main.py --config ./configs/ddpm/ddpm_uncontinuous.py --eval_folder ./tmp/ddpm/eval --mode eval --workdir ./tmp/ddpm
```