import numpy as np
import xarray as xr
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.stats import mode
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
import time
import dask.array as da
from dask import delayed
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, wait
from dask_image.ndfilters import gaussian_filter
from dask_image.ndmeasure import label
from dask_image.ndmorph import binary_closing
from skimage.registration import phase_cross_correlation
from skimage.util import apply_parallel
from skimage.morphology import local_maxima, disk
from skimage.segmentation import watershed
from math import floor
from os.path import exists, join
from os import makedirs, getcwd
from pyseq import image_analysis as ia


xr.set_options(keep_attrs=True)


#dask.config.set({'temporary-directory': '/scratch'})
def get_cluster(queue_name = 'pi3', log_dir=None):
    """ Make dask cluster w/ workers = 2 cores, 32 G mem, and 1 hr wall time."""

    if log_dir is None:
        log_dir = join(getcwd(),'dask_logs')
        makedirs(log_dir, exist_ok=True)

    cluster = SLURMCluster(
                queue = queue_name,
                cores = 2,
                memory = '32G',
                walltime='1:00:00',
                log_directory=log_dir,
                extra=["--lifetime", "55m", "--lifetime-stagger", "4m"])
    client = Client(cluster)

    return cluster, client

def q_bin(X, downscale=2):
    """Bin and quantize X (2D XArray), int16 quantization limit.

        Parameters:
        downscale (int): Factor to downscale both dimensions of X by.

        Returns:
        (dask array): Quantized and binned dask array, dtype=int16

    """

    X = da.coarsen(np.mean,X.data,{0:downscale,1:downscale},trim_excess=True,dtype='int16')

    return X

def mutual_information(X, Y):
    """Compute mutual information between X and Y dask arrays."""


    # compute limits
    X = X.compute()
    xmin = da.min(X).compute()
    xmax = da.max(X).compute()

    Y = Y.compute()
    ymin = da.min(Y).compute()
    ymax = da.max(Y).compute()

    # compute indices
    xi = X-xmin
    yi = Y-ymin

    # reshape for histogram
    XY = da.rechunk(da.stack([da.ravel(X),da.ravel(Y)],axis=1),chunks=(1e6,-1), balance=True)

    # joint probability distribution
    xbins = range(xmin, xmax+2)
    ybins = range(ymin, ymax+2)
    H, edges = da.histogramdd(XY, bins=[xbins, ybins], density = True)
    H = H.compute()

    # x marginal probability distribution
    p_x = H.sum(axis=1)
    # y marginal probability distribution
    p_y = H.sum(axis=0)


    H_ = da.from_array(H[xi,yi])
    x_ = da.from_array(p_x[xi])
    y_ = da.from_array(p_y[yi])

    # Mutual information I(X,Y)
    try:
        MI=da.sum(H_*da.log(H_/(x_*y_)))
        MI=MI.compute()
    except:
        print('H', H.shape)
        print('px', p_x.shape)
        print('py', p_y.shape)
        print('MI failed',xbins, ybins)
        MI = None

    return MI

def picasso(im_chi, im_chj, alpha_min=0, alpha_max=2, n_alpha=20, alpha_path=None, downscale=2,
            fit_alpha=True, **kwargs):
    """Estimate mixing parameter to minimize mutual information between images.

    Parameters:
    im_chi (Xarray): Image to remove crosstalk from.
    im_chj (Xarray): Image that is source of crosstalk.
    alpha_min (int/float): minimum mixing parameter to consider
    alpha_max (int/float): maximum mixing parameter to consider
    n_alpha (int): Number of mixing paramters between alpha_min and alpha_max
    alpha_path (path): File to save computed mutual information for each mixing parameter
    downscale (int): Factor to downscale images by mutual information calculation

    Returns:
    (float): Estimate of optimal mixing parameter

    """
    # im_chi = image from channel i
    # im_chj = image from channel j

    assert len(im_chi.shape) == 2
    assert len(im_chj.shape) == 2

    Y = q_bin(im_chj, downscale)

    print('Computing mutual information across range of alphas')

    if alpha_path is not None:
        if exists(alpha_path):
            alpha = np.loadtxt(alpha_path)
        else:
            alpha_path = None

    if alpha_path is None:
        alpha_path = make_alpha_path(im_chi,im_chj)
        alpha = np.linspace(alpha_min,alpha_max,n_alpha)
        alpha = np.vstack([alpha,np.zeros_like(alpha)]).T
        np.savetxt(alpha_path, alpha)

    for i, (a, mi) in enumerate(alpha):
        if mi == 0:
            X = q_bin(im_chi-a*im_chj, downscale)
            mi = mutual_information(X,Y)
            alpha[i,1] = mi
            np.savetxt(alpha_path, alpha)
            print('PICASSO::'+alpha_path[0:-4]+'::',a,mi)

    if fit_alpha:
        fit = interp1d(alpha[:,0], alpha[:,1], kind='cubic')
        fine_alpha = np.linspace(alpha_min,alpha_max,n_alpha*10)
        opt_alpha = fine_alpha[np.argmin(fit(fine_alpha))]
    else:
        opt_alpha = -1

    return opt_alpha

def gaussian(x, *args):
    """Gaussian function for curve fitting that can do multiple peaks.

        Parameters:
        x (array): array of x values
        args (list): model parameters with length 3*n_peaks
              [amp0, amp1, ...ampN, cen0, cen1, ...cennN, sig0, sig1, ...sigN]

        Returns:
        (array): result of model with parameters computed with x
    """

    if len(args) == 1:
      args = args[0]

    n_peaks = int(len(args)/3)


    if len(args) - n_peaks*3 != 0:
      print('Unequal number of parameters')
    else:
      for i in range(n_peaks):
        amp = args[0:n_peaks]
        cen = args[n_peaks:n_peaks*2]
        sigma = args[n_peaks*2:n_peaks*3]

      g_sum = 0
      for i in range(len(amp)):
          g_sum += amp[i]*(1/(sigma[i]*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen[i])/sigma[i])**2)))

      return g_sum

def res_gaussian(args, xfun, yfun):
    """Gaussian residual function for curve fitting."""

    g_sum = gaussian(xfun, args)

    return yfun-g_sum

def fit_mixed_gaussian(x, y, max_peaks=8, amplb=-np.inf, ampub=np.inf, tolerance=0.98, **kwargs):
    """Fit x vs y to a mixed gaussian model.

        Parameters:
        x (array): dependent variable
        y (array): independt variable
        max_peaks (int): maximum number of peaks to fit
        amp_lb: lower bound of amplitude
        amp_ub: upper bound of amplitude
        tolerance (float): Add peaks to model until R^2 is above this threshold

        Returns:
        (list): Model parameters with length 3*n_peaks
                [amp0, amp1, ...ampN, cen0, cen1, ...cennN, sig0, sig1, ...sigN]

    """


    # initialize values
    peaks = None; R2 = None
    # Initialize variables
    amp = []; amp_lb = []; amp_ub = [];
    cen = []; cen_lb = []; cen_ub = []; cen_guess = x[np.argmax(y)]
    cenlb = np.min(x); cenub=np.max(x)
    sigma = []; sigma_lb = []; sigma_ub = []; sigma_guess = np.sum(y**2)**0.5
    SST = np.sum((y-np.mean(y))**2)
    R2 = 0

    # Add peaks until fit reaches threshold
    while len(amp) <= max_peaks and R2 < tolerance:

        # calculate initial guesses
        if len(amp) == 0:
            amp_guess = np.max(y)
            cen_guess = x[np.argmax(y)]
            sigma_guess = np.sum(y**2)**0.5
        else:
            ind = np.argmax(y-results.fun)
            amp_guess = y[ind]
            cen_guess = x[ind]
            sigma_guess = np.sum(results.fun**2)**0.5

        # set initial guesses
        amp.append(amp_guess)
        cen.append(cen_guess)
        sigma.append(sigma_guess)
        p0 = np.array([amp, cen, sigma])
        p0 = p0.flatten()

        # set bounds
        amp_lb.append(amplb); amp_ub.append(ampub)
        cen_lb.append(cenlb); cen_ub.append(cenub)
        sigma_lb.append(0); sigma_ub.append(np.inf)
        lo_bounds = np.array([amp_lb, cen_lb, sigma_lb])
        up_bounds = np.array([amp_ub, cen_ub, sigma_ub])
        lo_bounds = lo_bounds.flatten()
        up_bounds = up_bounds.flatten()

        # Optimize parameters
        results = least_squares(res_gaussian, p0, bounds=(lo_bounds,up_bounds),
                                args=(x,y))


        if not results.success:
            print(results.message)
        else:
            R2 = 1 - np.sum(results.fun**2)/SST
            print('R2=',R2,'with',len(amp),'peaks')


        if results.success and R2 > tolerance:
            peaks = np.reshape(results.x,(-1,len(amp))).T
            peaks = peaks[np.argsort(peaks[:,1])[::-1],:]
            for i, row in enumerate(peaks):
                print('peak',i,':: amplitude =', row[0],'center=',row[1],'sigma=',row[2])


        else:
            if len(amp) == max_peaks:
                print('Bad fit')
                break

    return peaks, R2

def make_alpha_path(im_chi, im_chj):
    """Return filename to save mixing parameter vs mutual information."""

    alpha_path = im_chi.name
    chi = im_chi.channel.values
    chj = im_chj.channel.values
    cy = im_chi.cycle.values
    alpha_path+= '_i'+str(chi)+'j'+str(chj)+'c'+str(cy)+'.txt'

    return alpha_path


def get_mixing_matrix(im, channel_pairs, Mpath=None, **kwargs):
    """Return mixing matrix to unmix channels.

        Uses PICASSO algorithm to calculate mixing parameters between images as
        defined in channel pairs.
        chanels_pairs={cycle number:[(channel1, channel2), (channel3, channel4)]}

        Parameters:
        im (HiSeqImage): Images from a HiSeq
        channel_pairs: dict of channels to unmix, keys are cycle numbers, values
                       are channel pairs, see above
        Mpath(path): Filename to save mixing matrix
        downscale(int): Factor to downscale images during computatin of mixing matrix

        Returns:
        (array): ncycles x nchannels x nchannels mixing matrix

    """

    im = one_z_plane(im)

    cycles = list(im.cycle.values)
    channels = list(im.channel.values)
    ncy = len(cycles)
    nch = len(channels)

    if Mpath is None:
        Mpath = im.name+'_MixingMatrix.txt'

    if exists(Mpath):
        M = np.loadtxt(Mpath)
        M = M.reshape((ncy,nch,nch))
    else:
        M = np.zeros((ncy,nch,nch))
        for i in range(ncy):
            M[i,:,:] = np.identity(nch)
        np.savetxt(Mpath,M.flatten())

    for cyi, cy in enumerate(cycles):
        # Get Channel Pairs
        if type(channel_pairs) == dict:
            ch_pairs = channel_pairs[cy]
        else:
            ch_pairs = channel_pairs

        for chi, chj in ch_pairs:

            i = channels.index(chi)
            j = channels.index(chj)

            if M[cyi,i,j] == 0:

                # Open images
                im_chi = im.sel(channel = chi, cycle = cy)
                im_chj = im.sel(channel = chj, cycle = cy)
                alpha_path = make_alpha_path(im_chi, im_chj)

                # remove background
                mini = im_chi.min(dim=['row','col']).astype('int16')
                minj = im_chj.min(dim=['row','col']).astype('int16')
                im_chi = im_chi - mini
                im_chj = im_chj - minj

                # Find alpha that minimizes mutual information
                start = time.time()
                try:
                    alpha = picasso(im_chi, im_chj, alpha_path=alpha_path, **kwargs)
                    stop = time.time()
                    print('PICASSO::',im.name,'::',chi,chj,'::',alpha,(stop-start)/60)
                    M[cyi,i,j] = alpha
                    np.savetxt(Mpath,M.flatten())
                except:
                    print('PICASSO::',im.name,'::',chi,chj,'::FAILED')

    return M


def unmix(im, MM, update_size =1, DOT = True, client = None):
    """Unmix images using mixing matrix

        Parameters:
            im (Xarray): Images to unmix
            MM (array): mixing matrix
            update_size: Not in use potentially for PICASSO
            DOT (bool): True to use matrix dot product, or
                        False to compute channels seperately (for large images)
            client (dask scheduler): Used with DOT = False

        Returns:
            (Xarray): unmixed image

    """

    im = im.sortby('channel')

    ncycles, n_chi, n_chj = MM.shape
    cycles = im.cycle.values
    if 'obj_step' in im.dims:
        zplanes = im.obj_step.values
    else:
        im = im.expand_dims('obj_step')
        zplanes = im.obj_step.values
    nch = len(im.channel.values)

    if ncycles != len(cycles):
        print('Image cycles and Mixing Matrix cycles do not match')
        raise RuntimeError

    if n_chi != nch or n_chj != nch:
        print('Image channels and Mixing Matrix channels do not match')
        raise RuntimeError

    cycle_ims = []
    for i, cy in enumerate(cycles):
        print('Unmixing cycle', cy)
        # make unmixing matrix
        M = MM[i,:,:]
        mask = ~np.eye(n_chi, dtype=bool)
        M_unmix = np.identity(n_chi)
        M_unmix[mask] = M[mask]*(-update_size)
        dims = ['channeli', 'channel']
        ch_coords = im.coords['channel'].values
        coords={'channeli':ch_coords, 'channel':ch_coords}
        M_unmix = xr.DataArray(M_unmix, dims=dims, coords=coords)

        z_ims = []
        for z in zplanes:
            X = im.sel(cycle=cy, obj_step = z)

            if DOT:
                # Unmix X and and update dimension names
                X_update = M_unmix.dot(X, dims='channel').swap_dims({'channeli':'channel'})
                X_update = X_update.astype('int16')
                X_update = X_update.where(X_update > 0, 0) #remove pixels less than 0
                X_update = X_update.rename({'channeli':'channel'})
                X_unmix = X_update.assign_coords({'channel':ch_coords})
            else:
                X_unmix = unmix_split_channel(M_unmix, X, client)

            #Save unmixed image
            z_ims.append(X_unmix)

        # Stack z planes
        zstack = xr.concat(z_ims,dim='obj_step')
        #Save unmixed z stack
        cycle_ims.append(zstack)

    # Stack Cycles
    unmix_im = xr.concat(cycle_ims,dim='cycle')
    unmix_im.attrs = im.attrs
    unmix_im.name = im.name

    return unmix_im

def unmix_split_channel(M_unmix, X, client=None):
    """Unmix channels seperately.

        Parameters:
        M_unmix (array): Unmixing matrix
        X (3-d image): Channels x row x col images
        client (dask scheduler): Schedular for cluster

        Returns:
        (xarray DataArray): Unmixed Channels x row x col images

    """

    channels = X.channel.values

    ch_list = []
    for chi in channels:
        alphas = M_unmix.sel(channeli = chi)
        alphas = alphas.reset_coords(names='channeli',drop=True)

        if np.sum(alphas) == 1:
            ch_list.append(X.sel(channel = chi))
        else:
            im = None
            for a, chj in zip(alphas, channels):
                if a != 0 and im is None:
                    im = a*X.sel(channel = chj)
                elif a !=0 and im is not None:
                    im = im+a*(X.sel(channel = chj))
            im = im.assign_coords({'channel':chi})
            im = im.astype('int16')
            im = im.where(im > 0, 0) #remove pixels less than 0
            if client is not None:
                im = client.persist(im, retries=10)
                wait(im)


            ch_list.append(im)

    # Stack channels planes
    chstack = xr.concat(ch_list,dim='channel')

    return chstack


def get_cycle_shift(im, ref_channel=610, ref_cycle=1):
    """Calculate shift to align images across cycles.

        Uses phase cross correlation

        Parameters:
            im (Xarray): Images to align
            ref_channel: Reference channel to align images to
            ref_cycle: Refernce cycle to align images to

        Returns:
            (array): shape = ncycles x 2

    """

    rows = len(im.row)
    cols = len(im.col)
    cycles = im.cycle.values

    im = one_z_plane(im)

    # Crop if image is large
    crop_size = 2048*3
    if rows*cols > 2*crop_size**2:
        center_row = int(rows/2); center_col = int(cols/2)
        upper_row = int(center_row + crop_size/2); lower_row = int(center_row - crop_size/2)
        left_col  = int(center_col - crop_size/2); right_col = int(center_col + crop_size/2)
        bound_box = [lower_row, left_col,
                     upper_row, left_col,
                     upper_row, right_col]
        bound_box = np.reshape(np.array(bound_box), (3,2))
        im = ia.HiSeqImages(im = im)
        im.crop_section(bound_box)
        im = im.im


    im_ref = im.sel(cycle=ref_cycle, channel=ref_channel)
    cycles_ = cycles[cycles != ref_cycle]
    n_cy = len(cycles)

    shift = [[ref_cycle, ref_cycle, ref_cycle]]
    for cy in cycles_:
        im_cy = im.sel(cycle=cy, channel=ref_channel)
        start = time.time()
        detected_shift = phase_cross_correlation(im_ref,im_cy)[0]
        shift.append([cy]+list(detected_shift))
        stop = time.time()
        print(stop-start,cy, detected_shift)

    shift = np.array(shift, dtype='int8')
    np.savetxt(im.name+'_shift.txt',shift)

    return shift

def register_across_cycles(im, shift_path=None):
    """Register images across cycles according to shift input.

    """

    rows = len(im.row)
    cols = len(im.col)
    tiles = int(cols/im.chunks[-1][0])

    if shift_path is None:
        shift_path = im.name+'_shift.txt'
    assert exists(shift_path)
    shift = np.loadtxt(shift_path,dtype='int8')

    ref_cycle = shift[0,0]
    cycles_ = shift[1:,0]
    shift = shift[1:,1:]

    # adjust for global pixel shifts
    max_row = int(np.max(shift[shift[:,0]>=0,0], initial=0))
    min_row = int(abs(np.min(shift[shift[:,0]<=0,0], initial=0)))
    max_col = int(np.max(shift[shift[:,1]>=0,1], initial=0))
    min_col = int(abs(np.min(shift[shift[:,1]<=0,1], initial=0)))
    row_del = int(max_row+min_row)
    col_del = int(max_col+min_col)

    # adjust reference image
    row_slice = slice(max_row, rows-min_row)
    col_slice = slice(max_col, cols-min_col)
    im_list = [im.sel(cycle=ref_cycle,row=row_slice, col=col_slice)]

    # adjust offset cycle images
    for i, cy in enumerate(cycles_):
        if shift[i,0] >= 0:
            top_row = max_row-shift[i,0]
            bot_row = rows-(row_del-top_row)
        else:
            bot_row = min_row+shift[i,0]
            top_row = row_del-bot_row
            bot_row = rows-bot_row
        row_slice = slice(top_row,bot_row)

        if shift[i,1] >= 0:
            l_col = max_col - shift[i,1]
            r_col = cols-(col_del - l_col)
        else:
            r_col = min_col+shift[i,1]
            l_col = col_del-r_col
            r_col = cols-r_col
        col_slice = slice(l_col, r_col)

        im_list.append(im.sel(cycle=cy, row=row_slice, col=col_slice))

    # Concatenate shifted images together
    shifted = xr.concat(im_list, dim='cycle')

    # Rechunk so chunks are regular
    dims = {'channel':1,'cycle':1,'obj_step':1,'row':rows,'col':int(cols/tiles)}
    chunk_shape = {}
    for d in shifted.dims:
        chunk_shape[d] = dims[d]
    shifted = shifted.chunk(chunk_shape)

    return shifted

def reshape_af(im, af_ch=610, af_cy=0):
    """Append autofluorescence image to each cycles of im."""

    # Reorganize images so autofluorescence image is in each cycle
    channels = im.channel.values
    cycles = im.cycle.values
    af_im = im.sel(channel=af_ch, cycle=af_cy)

    af_stack = []
    for cyi in cycles:
        if cyi != af_cy:
            af_im = af_im.assign_coords({'channel':-af_ch,'cycle':cyi})
            af_stack.append(af_im)

    af_stack = xr.concat(af_stack,dim='cycle')
    im_stack = im.sel(cycle=cycles[cycles != af_cy])
    reshaped = xr.concat([af_stack, im_stack], dim='channel').transpose('channel',...)
    reshaped = reshaped.rename('af_'+im.name)

    return reshaped

def one_z_plane(im):
    """Return only middle plane of image."""

    if 'obj_step' in im.dims:
        # Pick middle obj_step to get unmixing matrix
        if len(im.obj_step) % 2 == 0:
            mid_obj_step = np.median(im.obj_step[:-1]).astype('int32')
        else:
            mid_obj_step = np.median(im.obj_step).astype('int32')
        im = im.sel(obj_step=mid_obj_step)

    return im


def segment(image, client=None, min_radius = 5, sigma=1, bg_dev =3):
    """Segment signal in image using watershed.

        Regions labeled 1 is background, labels > 1 are objects with signal.

        Parameters:
            image (dask array): Image to segment
            client (dask client): Scheduler for workers
            min_radius (int): Radius used for binary closing
            sigma (int/float): Sigma used for gaussian filter
            bg_dev (int/float): Background deviation for pixel signal threshold

        Returns:
            (dask array, int): Watersheded image and the number of labels

    """


    chunksize = image.chunksize

    bg_px = da.mean(image, axis=None)
    bg_sigma = da.std(image, axis=None)
    px_cutoff = bg_px+bg_sigma*bg_dev
    px_cutoff = px_cutoff.compute()
    print('pixel cutoff value:', px_cutoff)

    # Filter image
    print('Gaussian filter with sigma =', sigma)
    filtered_image = gaussian_filter(image, sigma=sigma).astype('int16')
    if client is not None:
        filtered_image = client.persist(filtered_image)
        wait(filtered_image)

    # Get local maxima of entire image
    print('Finding local maxima of entire image')
#     marker_mask = apply_parallel(local_maxima, filtered_image, chunks=chunksize, compute=False,
#                                  extra_arguments=(), extra_keywords={'connectivity':2}, dtype='bool')
    marker_mask = da.map_blocks(local_maxima, filtered_image, dtype='bool', meta='np.ndarray', connectivity=2)
    markers = marker_mask.astype('int8')


    # Filter local maxima that are only above pixel threshold
    print('Filtering local maxima: threshold =', px_cutoff)
    cell_mask = filtered_image >= px_cutoff

    # Euclidean distance transform
    print('Euclidean distance transform')
    #edt_image = distance_transform_edt(cell_mask).astype('int8')
    #edt_image = da.from_array(edt_image,chunks=chunksize)
    edt_image = da.map_overlap(distance_transform_edt, cell_mask, depth = [0,100], meta='np.ndarray', dtype='uint8')
    edt_image = gaussian_filter(edt_image, sigma=sigma).astype('uint8')

    # Find seed points for cells
    print('Find cell markers')
#     cell_markers = apply_parallel(local_maxima, edt_image, chunks=chunksize, compute=False,
#                                   extra_arguments=(), extra_keywords={'connectivity':2}, dtype='bool')
    cell_markers = da.map_blocks(local_maxima, edt_image, dtype='int8', meta='np.ndarray', connectivity=2)
    cell_markers = binary_closing(cell_markers, structure=disk(min_radius))

    # Label seed points
    print('Label cells')
    cell_labels, n_cells = label(cell_mask)
    n_cells = n_cells.compute()
    print('Found', n_cells, 'cell markers')
    cell_labels = cell_labels+1


    # Markers for watershed, 1 is background, > 1 is cells
    print('Set markers for watershed')
    markers = da.where(cell_mask == True, cell_labels, markers)

    watershed_image = dask_watershed(-filtered_image, markers, client)

    return watershed_image, n_cells

def dask_watershed(image, markers, client, **kwargs):
    """Dask wrapper around watershed

        There will be edge effects around chunks

        Parameters:
            image (dask array): Image to segment
            markers (dask array: Seeds for watershedding
            client (dask client): Scheduler for workers

        Returns:
            (dask array): Watersheded image and the number of labels

    """
    assert len(image.shape) == 2
    nrows, ncols = image.shape
    nchunks = len(image.chunks[-1])
    chunksize = image.chunksize[1]

    chunk_list = []
    for i in range(nchunks):
        print('watershedding chunk',i+1,'of',nchunks)
        if i != nchunks-1:
            cols =  slice(i*chunksize,(i+1)*chunksize)
        else:
            cols =  slice(i*chunksize,ncols)
        im = image[:,cols]
        #client.persist(im)
        marks = markers[:,cols]
        #client.persist(marks)
        im = im.compute()
        marks = marks.compute()
        ws_im = client.submit(watershed, im, markers = marks, connectivity=1)
        chunk_list.append(da.from_delayed(ws_im, shape=im.shape, meta=np.array((),dtype=np.int32)))

    print('Waiting for chunks to finish computing')
    wait(chunk_list)
    watershed_im = da.concatenate(chunk_list,axis=1)
    client.persist(watershed_im)

    return watershed_im
