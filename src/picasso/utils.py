from math import log, ceil, floor


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
