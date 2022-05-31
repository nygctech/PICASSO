import napari
from pyseq import image_analysis as ia

image_path = 'Z:\\gpfs\\commons\\groups\\nygcfaculty\\PySeq\\20210323_4i4color\\zarrs\\m4.zarr'
o = 25933
im = ia.get_HiSeqImages(image_path)
cropped_im = im.im.sel(row = slice(3800,4000), col = slice(4100, 4300), cycle=5, obj_step=o)

viewer = napari.Viewer()
viewer.add_image(cropped_im.sel(channel=558).compute(), name='Laminin1b', blending = 'additive', contrast_limits = (0,4095))
viewer.add_image(cropped_im.sel(channel=610).compute(), name='GFAP', blending = 'additive',  contrast_limits = (0,4095))
