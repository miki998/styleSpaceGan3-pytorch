## Description
We re-apply style channel searching techniques on stylegan3. (inspired from https://github.com/betterze/StyleSpace)

## Requirements

Our requirements are very similar to the requirements from StyleGAN3 repo.

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory.
* 64-bit Python 3.8 and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.1 or later.
* GCC 7 or later (Linux) or Visual Studio (Windows) compilers. Recommended GCC version depends on CUDA version, see for
  example [CUDA 11.4 system requirements](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-installation-guide-linux/index.html#system-requirements)
  .
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies. You can use the following
  commands with Miniconda3 to create and activate your LELSD Python environment:
    - `conda env create -f environment.yml`
    - `conda activate stylegan3`
* Docker users:
    - Ensure you have correctly installed
      the [NVIDIA container runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu).
    - Use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies.

## Getting started

Open following notebooks 

```
jupyter notebook demoPipeline.ipynb
jupyter notebook generateInfos.ipynb
```

## Acknowledgement

We borrow a lot from these repositories:

**IVRL — LELSD (we branch off from this repository)

```
https://github.com/IVRL/LELSD
```

**StyleGAN2-ADA — Official PyTorch implementation**

```
https://github.com/NVlabs/stylegan2-ada-pytorch
```
## Citation
```
@article{pajouheshgar2021optimizing,
  title={Optimizing Latent Space Directions For GAN-based Local Image Editing},
  author={Pajouheshgar, Ehsan and Zhang, Tong and S{\"u}sstrunk, Sabine},
  journal={arXiv preprint arXiv:2111.12583},
  year={2021}
}
@misc{https://doi.org/10.48550/arxiv.2011.12799,
  author = {Wu, Zongze and Lischinski, Dani and Shechtman, Eli},
  title = {StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation},
  publisher = {arXiv},
  year = {2020}
}
```
