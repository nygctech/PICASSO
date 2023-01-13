from math import log, ceil, floor

import pkg_resources
import imageio
import pooch
import numpy as np
import picasso

# # Get the version
version = picasso.__version__

# Create a new friend to manage your sample data storage
sample_data = pooch.create(
    path=pooch.os_cache("picasso"),
    base_url="https://github.com/nygctech/PICASSO/raw/{version}/sample_data/",
    version=version,
    version_dev="main",
    env="PICASSO_DATA_DIR",
    registry={"GFAP_sink.tiff": "ccc29ee5a9ac6cfe917d9c8c85448c0edda7822e61e7ce8f67ed6313700987f1",
              "LMNB1_source.tiff": "4747ba8c69376f99bb3d84d24da0aeec78447b72d0c11b1e4724c1314815d847"}
)

def adj_to_mixing(adj_matrix):
    '''Convert adjacency matrix to mixing matrix.'''

    try:
        adj_matrix = np.array(adj_matrix)
    except TypeError:
        print('Expected array')

    assert adj_matrix.ndim == 2, f'Expected 2D array, got {adj_matrix.ndim}D'
    sinks, sources = adj_matrix.shape
    assert sinks == sources, f'Expected square matrix'
    assert (adj_matrix.diagonal() == 0).all(), f'Expected diagonals to be zero'
    assert (adj_matrix >= 0).all(), f'Expected all values to be >= 0'

    n_sinks = (adj_matrix.sum(axis=1) > 0).sum()

    mm = np.zeros((sources, n_sinks))

    sink_ind = 0
    for i in sinks:
        if adj_mat[i,:].sum() > 0:                                              # Check if row has sink
            mm[i, sink_ind] = 1                                                 # Mark sink in mm with 1
            for ii in sources:                                                  # Mark sources in mm with -(mixing parameter)
                if adj_mat[i,ii] > 0:
                    mm[ii, sink_ind] = -adj_mat[i,ii]
            sink_ind += 1

    return mm

def exp_ceil(x, base=2):

    return base**ceil(log(x, base))

def exp_floor(x, base=2):

    return base**floor(log(x, base))

def sample_images():
    """
    Load some sample images to unmix.

    Channel is last dimension, sink = 0, source = 1
    """

    sink = imageio.imread(sample_data.fetch("GFAP_sink.tiff"))
    source = imageio.imread(sample_data.fetch("LMNB1_source.tiff"))

    return np.stack([sink, source],-1)
