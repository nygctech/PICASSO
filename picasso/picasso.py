import dask.array as da
import numpy as np
from math import floor, ceil


def mutual_information(X,Y):
    """Compute mutual information between X and Y dask arrays."""

    if not isinstance(X, da.Array):
        X = da.from_array(X)
    if not isinstance(Y, da.Array):
        Y = da.from_array(Y)

    # compute limits
    #X = X.compute()
    xmin = floor(da.min(X).compute())
    xmax = ceil(da.max(X).compute())

    #Y = Y.compute()
    ymin = floor(da.min(Y).compute())
    ymax = ceil(da.max(Y).compute())

    # reshape for histogram
    XY = da.rechunk(da.stack([X.flatten(),Y.flatten()],axis=1),chunks=(1e6,-1))

    # joint probability distribution
    xbins = range(xmin, xmax+2)
    ybins = range(ymin, ymax+2)
    H, edges = da.histogramdd(XY, bins=[xbins, ybins], density = True)
    H = H.compute()

    # x marginal probability distribution
    p_x = H.sum(axis=1)
    # y marginal probability distribution
    p_y = H.sum(axis=0)

    # print(H)
    # print(p_x)
    # print(p_y)

    # compute indices
    xi = da.digitize(X,xbins).compute()-1
    yi = da.digitize(Y,ybins).compute()-1

    # print(xi)
    # print(yi)

    H_ = da.from_array(H[xi,yi])
    x_ = da.from_array(p_x[xi])
    y_ = da.from_array(p_y[yi])

    # print(H_.compute())
    # print(x_.compute())
    # print(y_.compute())

    # Mutual information I(X,Y)
    try:
        #MI=da.sum(H_*da.log(H_/(x_*y_)))
        MI = da.mean(da.log((H_/(x_*y_))))
        MI=MI.compute()
    except:
        print('H', H.shape)
        print('px', p_x.shape)
        print('py', p_y.shape)
        print('MI failed',xbins, ybins)
        MI = None

    return MI


def get_joint_samples(correlation, size, mean=[0,0]):
    """Return 2 arrays with specified correlation."""

    cov = [[1, correlation],[correlation,1]]
    joint_samples = np.random.multivariate_normal(mean, cov, size)

    return joint_samples[:,0], joint_samples[:,1]
