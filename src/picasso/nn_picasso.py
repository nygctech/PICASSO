
import dask.array as da
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dask.utils import format_bytes
import psutil
from math import ceil
from typing import Union

from ../mine import mine



DA_TYPE = type(da.zeros(0))
NP_TYPE = type(np.zeros(0))
try:
    import xarray as xr
    XR_TYPE = type(xr.DataArray([0]))
except:
    XR_TYPE = None

class PICASSOnn(nn.Module):
    def __init__(self, mixing_matrix,
                 transform: Union[None, nn.Module] = None,
                 background: bool = True,
                 px_bit_depth: int = 12,
                 mi_weight: float  = 0.9):

        super().__init__()

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print('Using', self.device)
        self.mixing_matrix = torch.tensor(mixing_matrix, device = self.device)
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
            self.mine_.append(mine.MINE(neurons = px_bit_depth))
            for param in self.mine_[-1].parameters():
                self.mine_params.append(param)
        self.mi_loss = torch.zeros((self.n_pairs,), requires_grad = False, device = self.device)

        assert 0 < mi_weight < 1, f'Expected 0 < mi_weight < 1, got {mi_weight}'
        self.mi_weight = mi_weight
        self.contrast_weight = 1-mi_weight



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
        self.c_loss = self.contrast_loss(images[:,self.sink_ind], no_spill)*self.contrast_weight
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



    def train_loop(self, images, max_iter=40, batch_size=500, lr=1e-3, opt=None, **kwargs):

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

        for i in range(1, max_iter + 1):

            dataloader = DataLoader(dataset, shuffle=True, collate_fn = self.dask_collate)
            batch_loss = 0; batch_mi_loss = 0; batch_contrast_loss = 0
            for batch, im_list in enumerate(dataloader):

                if len(im_list) == 1:
                    im = im_list[0]
                else:
                    im = torch.stack(im_list, dim=0)

                opt.zero_grad()
                total_loss, mi_loss, contrast_loss  = self.forward(im)
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
            loss_ /= batch
            if i % (max_iter // 10) == 0:
                print(f"It {i} - total loss: {loss_[0]}, total MI loss: {loss_[1]}, total contrast loss: {loss_[2]}")

            if loss_[1] == 0:
                break

        train_info = {'mutual information loss': mi_loss_,
                      'contrast loss': contrast_loss_}
        if kwargs.get('SAVE_MIX_PARAMETERS', False):
            train_info['mixing parameters'] = mix_params_

        return train_info

    def get_dataset(self, images, batch_size=-1):


        im_type = type(images)

        im_stack = []
        if im_type in [list, tuple]:
        # Handle iterable of images
            assert images[0].ndim == 2, f'Expected iterable of 2D images'
            n_im = len(images)
            rows, cols = images[0].shape
            dtype = images[0].dtype

            for i in images:
                im_type_ = type(i)
                #TODO check if generic array type instead of numpy array
                if im_type_ is NP_TYPE:
                    im_stack.append(da.from_array(i).flatten())
                elif im_type_ is DA_TYPE:
                    im_stack.append(i.flatten())

        elif im_type in [DA_TYPE, XR_TYPE, NP_TYPE]:
        #Handle 3D array of images
            #TODO handle n dim
            assert images.ndim == 3, f'3D stack of images, with dim 0 as the stack dimension'
            n_im, rows, cols = images.shape
            dtype = images.dtype

            if im_type is XR_TYPE:
                images = images.data
                im_type = type(images)

            for i in range(n_im):
                if im_type is NP_TYPE:
                    im_stack.append(da.from_array(images[i,:,:]).flatten())
                elif im_type is DA_TYPE:
                    im_stack.append(images[i,:,:].flatten())
        else:
            raise TypeError('Did not recognize images')

        dataset = da.stack(im_stack, axis=1)

        return PICASSO_Dataset(dataset, rows, cols, dtype, px_per_chunk=batch_size)



    def unmix_images(self, images, batch_size: int = 1):

        n_ims = len(images)
        dataset = self.get_dataset(images)
        dataloader = DataLoader(dataset, collate_fn = self.dask_collate)

        batches = []
        for im in dataloader:
            if len(im) == 1:
                im = im[0]
            else:
                im = torch.cat(im, dim=0)
            batches.append(da.from_array(self.transform(im).cpu().detach().numpy()))
        unmixed = da.concatenate(batches, axis=0)

        unmixed_ = []
        px, n_ims = unmixed.shape
        for i in range(n_ims):
            im = unmixed[:,i].reshape(dataset.rows,dataset.cols)
            unmixed_.append((im*dataset.max_px).astype(dataset.dtype))

        return da.stack(unmixed_, axis = 0)


    def dask_collate(self, chunks_, dtype = torch.float32):

        _type = type(chunks_)

        assert _type in [DA_TYPE, list], f'Expected dask array or list, got {_type}'

        if _type is DA_TYPE:
            chunks = torch.tensor(chunks_.compute(), dtype=dtype, device = self.device)
        elif _type is list:
            chunks = [torch.tensor(c.compute(), dtype=dtype, device = self.device) for c in chunks_ ]

        return chunks


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
    def mixing_matrix(self, mm):
        '''Check and set the mixing matrix.

           rows = all images
           cols = unmixed images

           Mark a sink image with a 1, only 1 sink image per column.
           Mark a source image with a -1
           All other images should be 0.

        '''

        n_src, n_snk = mm.shape
        assert n_snk >= n_src, f'Number of sinks {n_snk} must be >= number of sources {n_src}'
        if n_src == n_snk:
            assert (mm.diagonal() == 1).all(), f'Diagonal of mixing matrix should be 1s'
            off_diag = mm[~torch.eye(mm.shape[0],dtype=bool)]
            assert ((off_diag==0) | (off_diag==-1)).all(), f'Off diagonal of mixing matrix should be -1s or 0s'
        else:
            assert ((mm==0) | (mm==-1) | (mm==1)).all(), f'Mixing matrix should only include -1s, 0s, or 1s'

        assert (mm==1).sum(axis=1).all() <= 1, f'Only 1 image can be marked as a sink per column in the mixing matrix'

        # Remove unused sources
        mm = mm[~((mm==0).sum(dim=0) == n_snk)]
        # Remove unused sinks
        mm = mm[~((mm==0).sum(dim=1) == n_src)]

        self.source_ind = ~(mm.sum(dim=1) == 1)
        self.sink_ind = (mm==-1).sum(dim=0) >= 1

        # Get images x sink mixing matrix (columns have 1 sink and at least one source image)
        self._mixing_matrix = mm[:, self.sink_ind]
        self.n_images, self.n_sinks = self._mixing_matrix.shape



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

    def get_parameters(self):

        params = []
        for i in range(self.images):
            for ii in range(self.sinks):
                if self.mixing_matrix[i,ii] == -1:
                    params.append([self.alpha[i,ii], self.background[i,ii]])

        return torch.tensor(params)





class PICASSO_Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, rows, cols, dtype, device = None, px_per_chunk=-1):

        super().__init__()

        # Get info about dataset
        self.rows = rows; self.cols = cols; self.dtype = dtype                  # height, width, and dtype of images
        self.max_px = dataset.max().compute()                                   # max pixel value in images
        self.dataset = dataset.astype('float32')/self.max_px                    # pixels normalized to 1
        self.blocks_per_image, self.n_images = dataset.blocks.shape             # chunk/blocks in each image, number of images
        self.px_byte = np.dtype(dataset.dtype).itemsize                         # bytes of each pixel
        #size of chunk in bytes
        self.chunk_byte = 1
        chunksize = dataset.chunksize
        assert len(chunksize) == 2 and chunksize[1] == 1, f'Dataset should be flattened images arranged by column'
        for s in chunksize:
            self.chunk_byte *= s
        self.chunk_byte *= self.px_byte*self.n_images
        self.chunk_px = chunksize[0]                                            # number of pixels in each chunk

        # Get info about system
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.gpu_prop = torch.cuda.get_device_properties('cuda')
                self.memory_size = self.gpu_prop.total_memory
            else:
                self.device = 'cpu'
                self.sys_prop = psutil.virtual_memory()
                self.memory_size = self.sys_prop.available
        f_mem = format_bytes(self.memory_size)

        # See if chunk will fit into memory
        self.fit_chunks = False
        self.mem_per_chunk = self.memory_size/(self.chunk_byte*self.n_images)
        self.px_in_mem = self.mem_per_chunk/2*self.chunk_px
        if self.mem_per_chunk > 0.5:
            self.fit_chunks = True

        # set pixels per chunk
        if px_per_chunk > 0:
            assert px_per_chunk <= self.px_in_mem, f'Can only fit {self.px_in_mem} px in {f_mem} memory, not {px_per_chunk}'
            self.fit_chunks = False
            self.px_in_mem = px_per_chunk

        self.chunks = self.dataset

    def break_chunk(self, chunks):
        pim = self.px_in_mem
        n_breaks = ceil(self.chunk_px/pim)
        chunks_ = []
        for i in range(n_breaks):
            if i < n_breaks-1:
                chunks_.append(chunks[i*pim:(i+1)*pim,:])
            else:
                chunks_.append(chunks[i*pim:-1,:])

        return chunks_

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        return self.chunks[index]

    def __iter__(self):
        return iter(self.chunks)


    @property
    def chunks(self):
        return self._chunks

    @chunks.setter
    def chunks(self, dataset):

        cp = self.chunk_px
        samples = []
        indices = np.arange(self.blocks_per_image)

        for i in indices:
            chunks = dataset[i*cp:(i+1)*cp,:]
            if not self.fit_chunks:
                samples = samples + self.break_chunk(chunks)
            else:
                samples.append(chunks)

        self._chunks = samples
