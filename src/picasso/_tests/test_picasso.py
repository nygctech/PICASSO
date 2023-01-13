import pytest
from picasso.nn_picasso import PICASSOnn, collate
from torch.utils.data import DataLoader
#from skimage.data import astronaut
import numpy as np
import dask.array as da
import xarray as xr
from picasso.utils import sample_images

def image_formats():

    np_image = sample_images()
    rows, cols, chs = np_image.shape
    chs = range(chs)

    np_list = []
    for ch in chs:
        np_list.append(np_image[:,:,ch])

    da_image = da.from_array(np_image)
    da_list = []
    for ch in chs:
        da_list.append(da.from_array(np_list[ch]))

    xr_np_image = xr.DataArray(np_image)
    xr_da_image = xr.DataArray(da_image)
    xr_np_list = []
    xr_da_list = []
    for ch in chs:
        xr_np_list.append(xr.DataArray(np_list[ch]))
        xr_da_list.append(xr.DataArray(da_list[ch]))

    return [np_image, np_list, da_image, da_list, xr_np_image, xr_np_list, xr_da_image, xr_da_list]

@pytest.mark.parametrize('image', image_formats())
def test_dataloader_gpu(image):
    mm = np.array([[1],[-1]])
    model = PICASSOnn(mm)
    assert model.device in ['cuda', 'cpu']

    dataset = model.get_dataset(image, ch_dim = -1)
    dataloader = DataLoader(dataset, shuffle=True, collate_fn = collate)

    for batch, im_list in enumerate(dataloader):
        assert batch == 0

def test_unmix():

    ims = sample_images()
    ims_ = [ims[:,:,0], ims[:,:,1]]
    mixing_matrix = np.array([[1],[-1]])
    model = PICASSOnn(mixing_matrix)
    for i in model.train_loop(ims_):
        pass
    unmixed = model.unmix_images(ims_)

    R = np.corrcoef(ims[:,:,1].flatten(), unmixed.compute().flatten())

    assert abs(R[0,1]) <= 0.1
