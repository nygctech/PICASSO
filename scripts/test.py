
from napari_picasso.picasso import nn_picasso
import matplotlib.pyplot as plt
import imageio
import numpy as np

#TODO use pooch to fetch sample data from repository

mixed = imageio.mimread('/Users/kpandit/PICASSO/sample_data/mouse_spinalcord_small.tiff')
mixed = np.stack(mixed)

mixing_matrix = [[1, -1, 0, 0],# sink 558, no spillover
                 [0, 1, 0, 0],# sink 610, spillover from 558
                 [0, 0, 1, -1],# sink 687, no spillover
                 [0, 0, 0, 1]]# sink 740, spillover from 687


picasso = nn_picasso.PICASSOnn(mixing_matrix, mi_weight = 0.9)
train_info = picasso.train_loop(mixed, batch_size = 10000, max_iter=100)
unmixed = picasso.unmix_images(mixed)
print(picasso.transform.get_parameters())

unmixed_  = []
c, h, w = unmixed.shape
for i in range(c):
    unmixed_.append(unmixed[i,:,:].compute())
imageio.mimwrite('unmixed.tiff', unmixed)

# Plot Loss
c_loss  = np.array(train_info['contrast loss'])
mi_loss = np.array(train_info['mutual information loss'])
total_loss = c_loss.sum(axis=1) + mi_loss.sum(axis=1)

def format_axs(ax, title=None, xlabel=None, ylabel=None):

    if title is not None:
        ax.set_title(str(title))

    if xlabel is not None:
        ax.set_xlabel(str(xlabel))

    if ylabel is not None:
        ax.set_ylabel(str(ylabel))

fig, axs = plt.subplots(1,5, figsize=(20,4))
axs[0].plot(total_loss)
axs[1].plot(mi_loss[:,0])
axs[2].plot(mi_loss[:,1])
axs[3].plot(c_loss[:,0])
axs[4].plot(c_loss[:,1])

titles = ['Total Loss', '610:558 MI Loss', '740:687 MI Loss', '610 Contrast Loss', '740 Contrast Loss']

for i in range(5):
     format_axs(axs[i], title=titles[i], xlabel = 'iteration', ylabel = 'loss')

axs[3].set_ylim((0, 0.1))
axs[4].set_ylim((0, 0.1))

fig.tight_layout()
plt.savefig('loss.pdf', dpi=300)
