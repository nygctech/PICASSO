# napari-PICASSO

[![License](https://img.shields.io/pypi/l/napari-curtain.svg?color=green)](https://github.com/nygctech/PICASSO/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-PICASSO.svg?color=green)](https://pypi.org/project/napari-PICASSO)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-PICASSO.svg?color=green)](https://python.org)
[![tests](https://github.com/nygctech/PICASSO/actions/workflows/test_and_deploy.yml/badge.svg?event=push)](https://github.com/nygctech/PICASSO/actions/workflows/test_and_deploy.yml)
[![codecov](https://codecov.io/gh/nygctech/PICASSO/branch/main/graph/badge.svg?token=4xOocXnFHt)](https://codecov.io/gh/nygctech/PICASSO)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-PICASSO)](https://napari-hub.org/plugins/napari-PICASSO)

Unmix spectral spillover

![](https://user-images.githubusercontent.com/72306584/176486552-50e1bca9-65fd-4466-8c92-a114e48d2278.gif)

## Automatic Usage

You can find the `PICASSO` plugin in the menu `Plugins > napari-PICASSO: PICASSO`. Select sink images that have spectral spillover from corresponding source images, then click run to optimise the mixing parameters with PICASSO. 

## Manual Usage

![](https://user-images.githubusercontent.com/72306584/176505151-572bd762-abe6-47b1-9821-4f3aaa4704c9.gif)

Select the manual button in the options pop up window. Then select sink images that have spectral spillover from corresponding source images. In the source images window, sliders for each $source_i$ control the mixing spillover, 
$\alpha_i$ 
(top), and background 
$\beta_i$ 
(bottom, optional).

## Mixing model

$$ unmixed sink = sink - \sum_{i} \alpha_i(source_i - \beta_i) $$

## Installation

You can install `napari-PICASSO` via [pip]:

    pip install napari-PICASSO

## Details

napari-PICASSO is a napari widget to blindly unmix fluorescence images of known members using PICASSO<sup>1</sup>. 

For example, if 2 fluorophores with overlapping spectra are imaged, spillover fluorescesce from a channel into an adjacent channel could be removed if you know which channel is the source of the spillover fluorescence and which channel is the sink of the spillover fluorescence. 

PICASSO is an algorithm to remove spillover fluorescence by minimizing the mutual information between sink and source images. The original algorithm described by Seo et al, minimized the mutual information between pairs of sink and source images using a Nelson-Mead simplex algorithm and computing the mutual information outright with custom written MATLAB code<sup>1</sup>. The napari plugin uses a neural net to estimate and minimize the mutual information (MINE<sup>2</sup>) between pairs of sink and source images using stochastic gradient descent with GPU acceleration.

## References

1. Seo, J. et al. PICASSO allows ultra-multiplexed fluorescence imaging of spatially overlapping proteins without reference spectra measurements. Nat Commun 13, 2475 (2022).
2. Belghazi, M. I. et al. MINE: Mutual Information Neural Estimation. arXiv:1801.04062 [cs, stat] (2018).


[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
