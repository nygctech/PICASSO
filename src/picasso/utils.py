from math import log, ceil, floor

import pkg_resources
import imageio
import pooch

# # Get the version
# from _version import version as __version__

# Create a new friend to manage your sample data storage
sample_data = pooch.create(
    # Folder where the data will be stored. For a sensible default, use the
    # default cache folder for your OS.
    path=pooch.os_cache("picasso"),
    # Base URL of the remote data store.
    #base_url="https://github.com/nygctech/PICASSO/tree/main/sample_data/",
    base_url="https://github.com/nygctech/PICASSO/tree/fix_unmix/sample_data/",
    # # Base URL of the remote data store. Will call .format on this string
    # # to insert the version (see below).
    # base_url="https://github.com/myproject/mypackage/raw/{version}/data/",
    # # Pooches are versioned so that you can use multiple versions of a
    # # package simultaneously. Use PEP440 compliant version number. The
    # # version will be appended to the path.
    # version=version,
    # # If a version as a "+XX.XXXXX" suffix, we'll assume that this is a dev
    # # version and replace the version with this string.
    # version_dev="main",
    # An environment variable that overwrites the path.
    env="PICASSO_DATA_DIR",
    # The cache file registry. A dictionary with all files managed by this
    # pooch. Keys are the file names (relative to *base_url*) and values
    # are their respective SHA256 hashes. Files will be downloaded
    # automatically when needed (see fetch_gravity_data).
    registry={"GFAP_sink.tiff": "f036be044e0f957a74766b7ed8373fc90bf0d95594db912cedddd40cd752d3ee",
              "LMNB1_source": "4747ba8c69376f99bb3d84d24da0aeec78447b72d0c11b1e4724c1314815d847"}
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
    """

    # Fetch the path to a file in the local storage. If it's not there,
    # we'll download it.
    sink = imageio.imread(sample_data.fetch("GFAP_sink.tiff"))
    source = imageio.imread(sample_data.fetch("LMNB1_source.tiff"))

    return sink, source
