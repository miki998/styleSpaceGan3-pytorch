## Description
Our code is largely notebook based and is applying stable scripts from previously known repositories. 
Main points are as follows:
- We re-apply style channel searching techniques on stylegan3-R and stylegan3-T. (inspired from https://github.com/betterze/StyleSpace)
- The found channels are used for editing on both generated images and real-inverted images
- Inversion was done by projection (through respective stylegan3 model)
- Some experiments were done to qualitatively and quantitatively compare edit quality across models

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

Open jupyter notebooks inside the `notebooks` folder

```
jupyter notebook "NOTEBOOK-NAME"
```

## NOTICE
1/ Due to losing the large part of our code base written during semester, there might be some differences of generated images from the report because of us forgetting which seed was used for the generating and inverting.
2/ This repository is largely inspired by `IVRL — LELSD` as mentioned in acknowledgement.
Even though our project is largely notebook-based, we modified partially code from the original repository the following way (to adapt for stylegan3):
- lelsd.py 
- apps/sequential_editing.py 
- utils/stylegan3_utils.py 

3/ Outputed values
ALSO stylespace channels found within Stylegan2 and Stylegan3 are saved in both `logs` and `out` folders for stylespace method and LELSD methods respectively. (there is an explained readme in the `logs` folder explaining what the values are)

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

**StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation
```
https://github.com/betterze/StyleSpace
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
