
import dask.array as da
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from dask.utils import format_bytes
import psutil
from math import ceil, floor, log
from typing import Union

from mine.mine import MINE

from picasso.utils import exp_ceil, exp_floor


DA_TYPE = type(da.zeros(0))
NP_TYPE = type(np.zeros(0))
TT_TYPE = type(torch.tensor(0))

try:
    import xarray as xr
    XR_TYPE = type(xr.DataArray([0]))
except:
    XR_TYPE = None

def get_device(device = None):
    assert device in [None, 'cpu', 'cuda']

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return device

def collate(chunk_, dtype = torch.float32):
    '''Return tensor to dataloader.

       Take chunk and transfer to cuda asynchronously if chunk is tensor

       If chunk is a dask array, compute chunk and convert into tensor

    '''

    device = get_device()
    iscuda = device == 'cuda'

    _type = type(chunk_)

    assert _type in [DA_TYPE, TT_TYPE, list], f'Expected dask array or list, got {_type}'
    if _type is list:
        _type = type(chunk_[0])
        assert _type in [DA_TYPE, TT_TYPE]
        chunk_ = chunk_[0]

    if _type is TT_TYPE:
        if chunk_.device.type != device:
            chunk_ = chunk_.to(device, non_blocking = iscuda)
        return chunk_
    else:
        return torch.tensor(chunk_.compute(), dtype=dtype, device=device)



def get_memory(device = None):
    '''Get size of memory on system.'''

    device = get_device(device)

    if device == 'cuda':
        torch.cuda.empty_cache()
        gpu_prop = torch.cuda.get_device_properties('cuda')
        memory_size = gpu_prop.total_memory - torch.cuda.memory_reserved()
    else:
        sys_prop = psutil.virtual_memory()
        memory_size = sys_prop.available

    return memory_size



class PICASSOnn(nn.Module):
    def __init__(self, mixing_matrix,
                 transform: Union[None, nn.Module] = None,
                 background: bool = True,
                 px_bit_depth: int = 12,
                 mi_weight: float  = 0.9,
                 **kwargs):

        super().__init__()


        self.device = get_device(kwargs.get('device', None))
        print('Using', self.device)

        self.mixing_matrix = mixing_matrix
        self.pairs = self.mixing_matrix                                         # Pairs property take mixing matrix as input to set

        self.bit_depth = px_bit_depth

        if transform is None:
            self.transform = MixModel(self.n_images, self.n_sinks,
                                      self.mixing_matrix, background = background,
                                      device=self.device)
        else:
            self.transform = transform
        self.transform.to(self.device)


        self.mine_ = []                                                         # initialize list of mutual information neural estimators (MINE)
        self.mine_params = []                                                   # initialize list of parameters for each MINE
        for i in range(self.n_pairs):
            self.mine_.append(MINE(neurons = px_bit_depth))
            for param in self.mine_[-1].parameters():
                self.mine_params.append(param)
        self.mi_loss = torch.zeros((self.n_pairs,), requires_grad = False, device = self.device)

        assert 0 < mi_weight < 1, f'Expected 0 < mi_weight < 1, got {mi_weight}'
        self.mi_weight = mi_weight
        self.contrast_weight = 1-mi_weight

        self.train_info = None



    def contrast_loss(self, x, y, k2 = 0.03):

        c2 = (1*k2)**2
        varx = torch.var(x, dim = 0)
        vary = torch.var(y, dim = 0)
        score = (2*varx**0.5*vary**0.5+c2)/(varx+vary+c2)

        score = 1-score                                                         #keep contrast similiar

        return score


    def forward(self, images):
        ''' images NxC: N = number of px, C = number of channels'''

        # Remove spillover spectras in sink images
        no_spill = self.transform.forward(images)

        # Keep contrast of sink image the same
        self.c_loss = self.contrast_loss(images[:,self.sink_mask], no_spill)*self.contrast_weight
        c_loss = torch.sum(self.c_loss)

        # Minimize mutual information between cleaned sink image and source images
        mi_loss = torch.tensor(0)
        for i, (snk, src) in enumerate(self.pairs):
            xy = torch.stack([no_spill[:,snk], images[:,src]], 1)
            mi = (1-self.mine_[i].forward(xy))*self.mi_weight

            # Need to scale mutual information loss
            # MINE loss is an lower bound estimate of mutual information
            # Minimizing a lower bound leads to exploding loss / loss gradients
            # For now scaling as mutual information goes to 0
            # TODO: scale loss by contrast loss gradient norm
            scale = 1
            if 0 <= mi < 1:
                scale = mi
            elif mi < 0:
                scale = 0
            mi = mi * scale

            self.mi_loss[i] = mi
            mi_loss = mi_loss + mi

        total_loss = mi_loss + c_loss

        return total_loss, self.mi_loss, self.c_loss

    def train_loop(self, images, max_iter=100, batch_size=-1, lr=1e-3, opt=None, **kwargs):

        mix_params = [self.transform.alpha]
        bg_params = [self.transform.background]

        if opt is None:
            opt = torch.optim.Adam([{'params':self.mine_params, 'lr':lr},
                                    {'params':mix_params, 'lr': lr/3},
                                    {'params':bg_params, 'lr': lr/10}
                                   ])

        mix_params_ = []
        mi_loss_ = []
        contrast_loss_ = []

        dataset = self.get_dataset(images, batch_size = batch_size)
        self.max_px = dataset.max_px
        num_workers = kwargs.get('num_workers', 0)

        for i in range(1, max_iter + 1):

            #dataloader = DataLoader(dataset, shuffle=True, collate_fn = collate, num_workers = num_workers)
            dataloader = DataLoader(dataset, collate_fn = collate, num_workers = num_workers,
                                    sampler=RandomSampler(dataset, num_samples=dataset.subset_chunks, replacement=True))
            batch_loss = 0; batch_mi_loss = 0; batch_contrast_loss = 0
            for batch, ims in enumerate(dataloader):

                opt.zero_grad(set_to_none=True)                                 # per documentation saves memory
                total_loss, mi_loss, contrast_loss  = self.forward(ims)
                total_loss.backward()

                # Save mixing parameters over iterations
                if kwargs.get('SAVE_MIX_PARAMETERS', False):
                    mix_params_.append(self.transform.get_parameters())

                opt.step()
                self.transform.constrain()

                batch_loss += total_loss.item()
                b_mi_loss = torch.sum(mi_loss).item()
                batch_mi_loss += b_mi_loss
                batch_contrast_loss += torch.sum(contrast_loss).item()

                # Save losses over iterations
                mi_loss_.append(mi_loss.tolist())
                contrast_loss_.append(contrast_loss.tolist())


            loss_ = np.array([batch_loss, batch_mi_loss, batch_contrast_loss])
            loss_ /= (batch+1)
            if i % 10 == 0:
                print(f"It {i} - total loss: {loss_[0]}, total MI loss: {loss_[1]}, total contrast loss: {loss_[2]}")

            if loss_[1] == 0:
                yield max_iter
                break

            yield i-1

        train_info = {'mutual information loss': mi_loss_,
                      'contrast loss': contrast_loss_}
        if kwargs.get('SAVE_MIX_PARAMETERS', False):
            train_info['mixing parameters'] = mix_params_

        self.train_info = train_info

    def get_dataset(self, images, batch_size=-1, ch_dim=0):
        'Format images so that are flattened and stacked in columns by channel'

        im_type = type(images)

        im_stack = []
        if im_type in [list, tuple]:
        # Handle iterable of images
            assert images[0].ndim == 2, f'Expected iterable of 2D images'
            n_im = len(images)
            assert n_im == self.n_images, f'Expected {self.n_images}, got {n_im}'
            rows, cols = images[0].shape
            dtype = images[0].dtype

            for i in images:
                im_type = type(i)

                if im_type is XR_TYPE:
                    i = i.data
                    im_type = type(i)

                if im_type is not DA_TYPE:
                    im_stack.append(da.from_array(i).flatten())
                else:
                    im_stack.append(i.flatten())

        else:
        # Hanndle array of images with channel dimension at the first or last axis
            dim_shape = images.shape
            if ch_dim == 0:
            # assume if channel first, row and cols are the last 2 axis
                n_im = dim_shape[0]; rows = dim_shape[-2]; cols = dim_shape[-1]
            elif ch_dim == -1:
            # assume if channel last, row and cols are the first 2 axis
                n_im = dim_shape[-1]; rows = dim_shape[0]; cols = dim_shape[1]
            else:
                raise ValueError(f'expected ch_dim to be 0 or -1, got {ch_dim}')

            assert n_im == self.n_images, f'Expected {self.n_images}, got {n_im}'
            dtype = images.dtype

            # Handle xarray wrapped arrays
            if im_type is XR_TYPE:
                images = images.data
                im_type = type(images)

            for i in range(n_im):
                if ch_dim == 0:
                    if im_type is not DA_TYPE:
                        im_stack.append(da.from_array(images[i,...]).flatten())
                    elif im_type is DA_TYPE:
                        im_stack.append(images[i,...].flatten())
                else:
                    if im_type is not DA_TYPE:
                        im_stack.append(da.from_array(images[...,i]).flatten())
                    elif im_type is DA_TYPE:
                        im_stack.append(images[...,i].flatten())

        dataset = da.stack(im_stack, axis=1)

        return PICASSO_Dataset(dataset, rows, cols, dtype, px_per_chunk=batch_size, device = self.device)



    def unmix_images(self, images, batch_size: int = 1):

        n_ims = len(images)
        dataset = self.get_dataset(images)

        #dataloader = DataLoader(dataset, collate_fn = self.dask_collate)
        dataloader = DataLoader(dataset, collate_fn = collate)

        batches = []
        for im in dataloader:

            # if len(im) == 1:
            #     im = im[0]
            # else:
            #     im = torch.cat(im, dim=0)
            batches.append(da.from_array(self.transform(im).cpu().detach().numpy()))
        unmixed = da.concatenate(batches, axis=0)

        unmixed_ = []
        px, n_ims = unmixed.shape
        for i in range(n_ims):
            im = unmixed[:,i].reshape(dataset.rows,dataset.cols)
            unmixed_.append((im*dataset.max_px).astype(dataset.dtype))

        return da.stack(unmixed_, axis = 0)

    #
    # def dask_collate(self, chunk_, dtype = torch.float32):
    #
    #     _type = type(chunk_)
    #
    #     assert _type in [DA_TYPE, list], f'Expected dask array or list, got {_type}'
    #     if _type is list:
    #         assert isinstance(chunk_[0], DA_TYPE)
    #         chunk_ = chunk_[0]
    #
    #     chunk = torch.tensor(chunk_.compute(), dtype=dtype, device = self.device)
    #
    #     return chunk


    @property
    def pairs(self):
        return self._pairs

    @pairs.setter
    def pairs(self, mixing_matrix):
        '''Set list of pairs (sink index, source index).'''

        images, sinks = self._mixing_matrix.shape

        self._pairs = []
        for i in range(images):
            for ii in range(sinks):
                if self.mixing_matrix[i,ii] == -1:
                    self._pairs.append((ii, i))

        self.n_pairs = len(self._pairs)

    @property
    def mixing_matrix(self):
        return self._mixing_matrix

    @mixing_matrix.setter
    def mixing_matrix(self, mm: NP_TYPE):
        '''Check and set the mixing matrix.

           rows = all images
           cols = unmixed images

           Mark a sink image with a 1, only 1 sink image per column.
           Mark a source image with a -1
           All other images should be 0.

        '''

        assert mm.ndim in [1,2], f'Expected 2D mixing matrix, got {mm.ndim}'
        if mm.ndim == 1:
            mm = np.expand_dims(mm, -1)
        n_src, n_snk = mm.shape
        assert n_src >= n_snk, f'Number of sources {n_src} must be >= number of sinks {n_snk}'
        # if n_src == n_snk:
        #     assert (mm.diagonal() == 1).all(), f'Diagonal of mixing matrix should be 1s'
        #     off_diag = mm[~torch.eye(mm.shape[0],dtype=bool)]
        #     assert ((off_diag==0) | (off_diag==-1)).all(), f'Off diagonal of mixing matrix should be -1s or 0s'
        # else:
        #     assert ((mm==0) | (mm==-1) | (mm==1)).all(), f'Mixing matrix should only include -1s, 0s, or 1s'

        mm[mm < 0] = -1                                                         #Mark source images as -1
        assert ((mm==1).sum(axis=0) == 1).all(), f'1 image must be marked as a sink per column in the mixing matrix'
        assert ((mm==1).sum(axis=1) <= 1).all(), f'Image marked as sink in multiple columns in the mixing matrix'
        assert ((mm==-1).sum(axis=0) >= 1).all(), f'At least 1 image must be marked as a source per column in the mixing matrix'

        # Remove unused sources
        #mm = mm[~((mm==0).sum(axis=1) == 0)]
        # Remove unused sinks
        #mm = mm[~((mm==1).sum(axis=0) == 1)]

        #self.source_ind = ~(mm.sum(axis=1) == 1)
        self.sink_mask = (mm==1).sum(axis=1) == 1
        #self.sink_mask = (mm == 1).sum(axis=0).T == 1
        #print(self.sink_mask)

        # Get images x sink mixing matrix (columns have 1 sink and at least one source image)
        #self._mixing_matrix = mm[:, self.sink_ind]
        #self._mixing_matrix = mm[:, self.sink_ind]

        self._mixing_matrix = torch.tensor(mm, device = self.device, dtype=torch.float32)
        self.n_images, self.n_sinks = self._mixing_matrix.shape


    @property
    def mixing_parameters(self):

        alpha = (self.mixing_matrix * self.transform.alpha).detach().cpu()
        if self.transform.bg:
            bg = (self.transform.background * self.max_px).detach().cpu()
        else:
            bg = np.zeros(alpha.shape)

        self._mixing_parameters = np.stack([alpha, bg], axis = 0)

        return self._mixing_parameters


class MixModel(nn.Module):
    def __init__(self, images:int, sinks:int, mixing_matrix, device='cpu', background: bool = True,
                 min_alpha:float = 0.01, max_alpha:float = 2.0, max_background:float = 0.2):

        super().__init__()

        assert (images, sinks) == mixing_matrix.shape, f'Mixing matrix rows should = images, and cols should = sinks'

        self.device = device
        self.mixing_matrix = mixing_matrix
        self.bg = background
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.max_background = max_background
        self.images = images
        self.sinks = sinks

        self.Hardtanh = nn.Hardtanh(min_val=0.0, max_val=1.0)

        self.alpha = nn.Parameter(torch.ones((images, sinks), dtype=torch.float32)/1)

        if background:
            self.background = nn.Parameter(torch.ones((images, sinks), dtype=torch.float32)/100)

        self.constrain()

    def forward(self, x):
        assert x.ndim == 2, f'Got {x.ndim}D matrix, expected 2D matrix of flattened images, rows = px, cols = images'
        px, images = x.shape

        assert images == self.images, f'Expected {self.images} images, got {images}'

        y = torch.zeros((px, self.sinks), dtype = torch.float32, device = self.device)

        if self.bg:
            for i in range(self.sinks):
                y[:,i] = self.Hardtanh(x-self.background[:,i].T) @ (self.alpha[:,i]*self.mixing_matrix[:,i])
        else:
            y = x @ (self.alpha*self.mixing_matrix)

        return self.Hardtanh(y)

    def constrain(self):

        alpha = self.alpha.data
        alpha = alpha.clamp(self.min_alpha, self.max_alpha)
        alpha[self.mixing_matrix == 1] = 1
        alpha[self.mixing_matrix == 0] = 0
        self.alpha.data = alpha

        if self.bg:
            background = self.background.data
            background = background.clamp(0.0, self.max_background)
            background[(self.mixing_matrix == 1) | (self.mixing_matrix == 0)] = 0.0
            self.background.data = background






class PICASSO_Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, rows, cols, dtype, device = 'cpu', px_per_chunk=-1, subset=True, **kwargs):

        super().__init__()



        # Get info about dataset
        self.rows = rows; self.cols = cols; self.dtype = dtype                  # height, width, and dtype of images
        self.total_px = rows*cols
        self.max_px = dataset.max().compute()                                   # max pixel value in image
        self.min_px = dataset.min().compute()                                   # min pixel value in image
        min_n_px = ceil(self.min_samples(**kwargs))
        print('min number pixels', min_n_px)
        dataset = dataset.astype('float32')/self.max_px                         # pixels normalized to 1
        #self.dataset = torch.tensor(dataset.compute(), dtype=torch.float32, )

        self.blocks_per_image, self.n_images = dataset.blocks.shape             # chunk/blocks in each image, number of images
        self.px_byte = 4                                                        # bytes of each pixel
        #size of chunk in bytes
        self.chunk_byte = 1
        chunksize = dataset.chunksize
        assert len(chunksize) == 2 and chunksize[1] == 1, f'Dataset should be flattened images arranged by column'
        for s in chunksize:
            self.chunk_byte *= s
        self.chunk_byte *= self.px_byte*self.n_images
        self.chunk_px = chunksize[0]                                            # number of pixels in each chunk

        self.memory_size = get_memory(device)
        f_mem = format_bytes(self.memory_size)

        # See if chunk will fit into memory
        self.max_px_in_mem = exp_floor(self.memory_size/8000)
        self.fit_chunks = self.chunk_px < self.max_px_in_mem
        if not self.fit_chunks:
            self.n_breaks = ceil(self.chunk_px/self.max_px_in_mem)
            self.px_per_chunk = 2**floor(log(self.chunk_px/self.n_breaks,2))

        # If dataset can fit into GPU, moving to tensor on GPU
        # If dataset is larger the GPU memory but can fit into CPU memory, put dataset into tensor  in pinned CPU memory
        # If dataset is larger than CPU memory, keep as dask arrray, and let collate function move chunks to GPU during training (slow)
        dataset_size = self.chunk_byte * self.blocks_per_image
        if dataset_size < self.max_px_in_mem:
            # entire dataset can fit into gpu
            # assume if it can fit into gpu it can also fit into cpu
            self.dataset = torch.tensor(dataset.compute(), dtype=torch.float32, device=device)
        else:
            if dataset_size < get_memory('cpu') :
                # entire dataset can fit on cpu memory
                self.dataset = torch.tensor(dataset.compute(), dtype=torch.float32, device='cpu')
                if device == 'cuda':
                    self.dataset = self.dataset.pin_memory()
            else:
                # dataset is big and can't fit into cpu memory, leave as dask array
                # training will be slow
                self.dataset = dataset





        # if not self.fit_chunks:
        #     figure out how to break chunks
        #     chunk_px/max_px_in_mem

        # print('available memory', f_mem)
        # print('number of images',self.n_images)
        # self.chunks_in_memory = self.memory_size/self.chunk_byte
        # print(self.chunks_in_memory)
        # self.px_in_mem = format_bytes(torch.cuda.get_device_properties('cuda').total_memory/262144)
        # if self.chunks_in_memory >= 2:
        #     self.fit_chunks = True

        # overide max_px_in_mem with px_per_chunk if it can fit into memory
        if px_per_chunk > 0:
            assert px_per_chunk <= self.max_px_in_mem, f'Can only fit {self.max_px_in_mem} px in {f_mem} memory, not {px_per_chunk}'
            self.fit_chunks = False
            self.max_px_in_mem = px_per_chunk


        self.chunks = self.dataset

        if subset:
            print('pixels in chunk', self.max_px_in_mem)
            if min_n_px < self.total_px:
                self.subset_chunks = ceil(min_n_px/self.max_px_in_mem)
            else:
                self.subset_chunks = 1
            print('Chunks per iteration:', self.subset_chunks)
            percent_px_used = (self.subset_chunks*self.max_px_in_mem)/(rows*cols)*100
            if percent_px_used > 100:
                percent_px_used = 100
            print(f'{percent_px_used} % of pixels used')
        else:
            self.subset_chunks = self.length


    def break_chunk(self, chunk):
        '''Break chunk into smaller chunks.'''

        small_px = self.px_per_chunk
        big_px = chunk.shape[0]

        chunks_ = []
        for i in range(ceil(big_px//small_px)):
            start_ind = i*small_px
            stop_ind = min(start_ind + small_px, big_px)
            test = chunk[start_ind:stop_ind,:]
            chunks_.append(chunk[start_ind:stop_ind,:])

        return chunks_

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.chunks[index]

    def __iter__(self):

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
            # single-process data loading, return the full iterator
                return iter(self.chunks)
            else:
            # in a worker process
            # split workload
                #print(worker_info.id, 'in iter')
                n_chunks = len(self.chunks)
                per_worker = int(ceil(n_chunks / float(worker_info.num_workers)))
                worker_id = worker_info.id
                #print(worker_id, n_chunks, per_worker)
                start_ind = worker_id * per_worker
                stop_ind = min(start_ind + per_worker, n_chunks+1)
                #print(worker_id, start_ind, stop_ind)
                return iter(self.chunks[start_ind:stop_ind])



    @property
    def chunks(self):
        return self._chunks

    @chunks.setter
    def chunks(self, dataset):

        cp = self.chunk_px
        samples = []
        indices = np.arange(ceil(self.total_px/cp))

        for i in indices:
            stop_ind = min((i+1)*cp, self.total_px)
            chunks = dataset[i*cp:stop_ind,:]
            if not self.fit_chunks:
                samples = samples + self.break_chunk(chunks)
            else:
                samples.append(chunks)

        self.length = len(samples)

        self._chunks = samples

    def min_samples(self, accuracy = 0.1, confidence = 0.9, **kwargs):
        '''See equation 15 in theorem 3 from:
         Belghazi, M. I. et al. MINE: Mutual Information Neural Estimation.
         arXiv:1801.04062 [cs, stat] (2018).
        '''

        M = 1                                                                   # Max MINE model value
        d = self.max_px - self.min_px                                           # dimension of parameter space
        K = 1                                                                   # Max MINE parameter value
        L = 10                                                                  # Lipshitz constant, no idea just overestimate
        e = accuracy                                                            # accuracy
        c = confidence                                                          # confidence

        return (2*M**2*(d*log(16*K*L*d**(0.5)/e) + 2*d*M + log(2/c)))/(e**2)
