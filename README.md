napari-PICASSO is a napari widget to blindly unmix fluorescence images of known members using PICASSO<sup>1</sup>. 

For example, if 2 fluorophores with overlapping spectra are imaged, spillover fluorescesce from a channel into an adjacent channel could be removed if you know which channel is the source of the spillover fluorescence and which channel is the sink of the spillover fluorescence. 

PICASSO is an algorithm to remove spillover fluorescence by minimizing the mutual information between sink and source images. The original algorithm described by Seo et al, minimized the mutual information between pairs of sink and source images using a Nelson-Mead simplex algorithm and computing the mutual information outright with custom written MATLAB code. The napari plugin uses a neural net to estimate and minimize the mutual information (MINE<sup>2</sup>) between pairs of sink and source images using stochastic gradient descent with GPU acceleration. 

1. Seo, J. et al. PICASSO allows ultra-multiplexed fluorescence imaging of spatially overlapping proteins without reference spectra measurements. Nat Commun 13, 2475 (2022).
2. Belghazi, M. I. et al. MINE: Mutual Information Neural Estimation. arXiv:1801.04062 [cs, stat] (2018).

