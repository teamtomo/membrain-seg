# membrain-seg

[![License](https://img.shields.io/pypi/l/membrain-seg.svg?color=green)](https://github.com/teamtomo/membrain-seg/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/membrain-seg.svg?color=green)](https://pypi.org/project/membrain-seg)
[![Python Version](https://img.shields.io/pypi/pyversions/membrain-seg.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/membrain-seg/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/membrain-seg/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/membrain-seg/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/membrain-seg)

Membrane segmentation in 3D for cryo-ET.

<p align="center" width="100%">
    <img width="100%" src="https://user-images.githubusercontent.com/34575029/248259282-ee622267-77fa-4c88-ad38-ad0cfd76b810.png">
</p>

Membrain-Seg is currently under early development, so we may make breaking changes between releases.

Our best model is changing quickly, so if you would like to give MemBrain-seg a try, please reach out to us (e.g. Lorenz.Lamm@helmholtz-munich.de) and we are happy to provide you with the latest version and advice on how to best use this repository.

Preliminary [documentation](https://github.com/teamtomo/membrain-seg/blob/main/docs/index.md) is available, but far from perfect. Please let us know if you encounter any issues, and we are more than happy to help (and get feedback what does not work yet).

Our current best model is available for download [here](https://drive.google.com/file/d/15ZL5Ao7EnPwMHa8yq5CIkanuNyENrDeK/view?usp=sharing). Please let us know how it works for you.

The U-Net architecture and training parameters are largely inspired by nnUNet<sup>1</sup>.
```
[1] Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method 
for deep learning-based biomedical image segmentation. Nature Methods, 1-9.
```