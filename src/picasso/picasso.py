import dask.array as da
import numpy as np
from math import floor, ceil
import torch
from numpy.random import multivariate_normal


def mutual_information_np(X,Y):
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

_range = range



def histogramdd(sample,bins=None,range=None,weights=None,remove_overflow=True):

    edges=None
    device=None
    custom_edges = False
    D,N = sample.shape
    if device == None:
        device = sample.device
    if bins == None:
        if edges == None:
            bins = 10
            custom_edges = False
        else:
            try:
                bins = edges.size(1)-1
            except AttributeError:
                bins = torch.empty(D)
                for i in _range(len(edges)):
                    bins[i] = edges[i].size(0)-1
                bins = bins.to(device)
            custom_edges = True
    try:
        M = bins.size(0)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except AttributeError:
        # bins is either an integer or a list
        if type(bins) == int:
            bins = torch.full([D],bins,dtype=torch.long,device=device)
        elif torch.is_tensor(bins[0]):
            custom_edges = True
            edges = bins
            bins = torch.empty(D,dtype=torch.long)
            for i in _range(len(edges)):
                bins[i] = edges[i].size(0)-1
            bins = bins.to(device)
        else:
            bins = torch.as_tensor(bins)
    if bins.dim() == 2:
        custom_edges = True
        edges = bins
        bins = torch.full([D],bins.size(1)-1,dtype=torch.long,device=device)
    if custom_edges:
        use_old_edges = False
        if not torch.is_tensor(edges):
            use_old_edges = True
            edges_old = edges
            m = max(i.size(0) for i in edges)
            tmp = torch.empty([D,m],device=edges[0].device)
            for i in _range(D):
                s = edges[i].size(0)
                tmp[i,:]=edges[i][-1]
                tmp[i,:s]=edges[i][:]
            edges = tmp.to(device)
        k = torch.searchsorted(edges,sample)
        k = torch.min(k,(bins+1).reshape(-1,1))
        if use_old_edges:
            edges = edges_old
        else:
            edges = torch.unbind(edges)
    else:
            if range == None: #range is not defined
                range = torch.empty(2,D,device=device)
                if N == 0: #Empty histogram
                    range[0,:] = 0
                    range[1,:] = 1
                else:
                    range[0,:]=torch.min(sample,1)[0]
                    range[1,:]=torch.max(sample,1)[0]
            elif not torch.is_tensor(range): #range is a tuple
                r = torch.empty(2,D)
                for i in _range(D):
                    if range[i] is not None:
                        r[:,i] = torch.as_tensor(range[i])
                    else:
                        if N == 0: #Edge case: empty histogram
                            r[0,i] = 0
                            r[1,i] = 1
                        r[0,i]=torch.min(sample[:,i])[0]
                        r[1,i]=torch.max(sample[:,i])[0]
                range = r.to(device=device,dtype=sample.dtype)
            singular_range = torch.eq(range[0],range[1]) #If the range consists of only one point, pad it up
            range[0,singular_range] -= .5
            range[1,singular_range] += .5
            edges = [torch.linspace(range[0,i],range[1,i],bins[i]+1) for i in _range(len(bins))]
            tranges = torch.empty_like(range)
            tranges[1,:] = bins/(range[1,:]-range[0,:])
            tranges[0,:] = 1-range[0,:]*tranges[1,:]
            k = torch.addcmul(tranges[0,:].reshape(-1,1),sample,tranges[1,:].reshape(-1,1)).long() #Get the right index
            k = torch.max(k,torch.zeros([],device=device,dtype=torch.long)) #Underflow bin
            k = torch.min(k,(bins+1).reshape(-1,1))


    multiindex = torch.ones_like(bins)
    multiindex[1:] = torch.cumprod(torch.flip(bins[1:],[0])+2,-1).long()
    multiindex = torch.flip(multiindex,[0])
    l = torch.sum(k*multiindex.reshape(-1,1),0)
    hist = torch.bincount(l,minlength=(multiindex[0]*(bins[0]+2)).item(),weights=weights)
    hist = hist.reshape(tuple(bins+2))
    if remove_overflow:
        core = D * (slice(1, -1),)
        hist = hist[core]
    return hist,edges



def joint_distribution(X, Y, xmin = None, xmax = None, ymin = None, ymax = None, device = None):

    X = X.detach(); Y = Y.detach()

    if device is None:
        assert X.device == Y.device
        device = X.device

    # compute limits
    if xmin is None:
        xmin = torch.floor(X.min()).long()
    if xmax is None:
        xmax = torch.ceil(X.max())
    if ymin is None:
        ymin = torch.floor(Y.min()).long();
    if ymax is None:
        ymax = torch.ceil(Y.max())

    # digitize
    X = X.long()-xmin
    Y = Y.long()-ymin

    # make bins
    xmax = xmax-xmin+2
    ymax = ymax-ymin+2
    xbins  = torch.arange(0, xmax.short(), device=device).detach()
    ybins  = torch.arange(0, ymax.short(), device=device).detach()
    bins = [xbins, ybins]

    # Joint Probability distribution
    H, edges = histogramdd(torch.stack([X, Y], dim=0), bins, remove_overflow=False)
    H = H.detach()/X.size()[0]

    return H, edges

def mutual_information(X, Y, Pxy = None, device = None):

    if Pxy is None:
        Pxy, edges = joint_distribution(X, Y, device)

    if device is None:
        assert X.device == Y.device
        device = X.device

    X = X.long()
    Y = Y.long()

    # marginal probability distributions
    p_x = Pxy.sum(1)
    p_y = Pxy.sum(0)

    # Compute mutual information
    H_ = Pxy[X,Y]
    x_ = p_x[X]
    y_ = p_y[Y]

    try:
        MI = torch.mean(H_*torch.log(H_/(x_*y_)))
    except:
        print('H', H.shape)
        print('px', p_x.shape)
        print('py', p_y.shape)
        print('MI failed', xbins, ybins)
        MI = None

    return MI


def get_joint_samples(correlation, size, mean = [0,0], variance = [1,1], scale=4095, dtype='int16'):
    """Return 2 arrays with specified correlation."""

    cov = [[1+variance[0], correlation], [correlation, 1+variance[1]]]

    mvn = multivariate_normal(mean, cov, size = size)

    x = mvn[:,0]; y = mvn[:,1]
    xmin = x.min(); xmax = x.max()
    ymin = y.min(); ymax = y.max()
    x = (x-xmin)*scale/(xmax-xmin)
    y = (y-ymin)*scale/(ymax-ymin)

    x = x.astype(dtype)
    y = y.astype(dtype)

    return x, y
