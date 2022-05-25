from magicgui.widgets import PushButton, FloatSlider, Container, ComboBox
from napari.types import ImageData
import numpy as np
import dask.array as da
from napari_picasso.utils import get_layer_info, get_image_layers
from napari_picasso.sink_widget import SinkWidget
from napari_picasso.options_widget import Options
from napari.qt.threading import create_worker, thread_worker
from napari.utils import progress

DA_TYPE = type(da.zeros(0))
NP_TYPE = type(np.zeros(0))

try:
    import xarray as xr
    XR_TYPE = type(xr.DataArray([0]))
except:
    XR_TYPE = None


class PicassoWidget(Container):
    '''Main picasso widget.'''

    def __init__(self, viewer: 'napari.viewer.Viewer', **kwargs):

        super().__init__(**kwargs)

        # Header buttons
        add_sink_btn = PushButton(text="+sink", name="add_snk")
        open_options_btn = PushButton(text="options", name="options")
        run_btn = PushButton(text="run", name="run")
        #self.run_btn = run_btn

        # Connect header buttons to header functions
        add_sink_btn.changed.connect(self.add_sink_widget)
        open_options_btn.changed.connect(self.open_options)
        run_btn.changed.connect(self.make_model)

        # Size header buttons
        widgets = [add_sink_btn, open_options_btn, run_btn]
        for w in widgets:
            w.max_width = 70

        header = Container(widgets = widgets, layout='horizontal', labels=False)
        self.append(header)

        self.mixing_params = {}                                                 # Current mixing paramers {sink : {source:alpha}}
        self._viewer = viewer                                                   # napari viewer
        self._progress = None
        self._worker = None
        self._options_widget = None
        self._options = {}

        self.picasso_params = None
        self.sliders_visible = False


    def make_model(self, *args, **kwargs):
        '''Make picasso neural network model and run asynchronously.'''
        from picasso.nn_picasso import PICASSOnn
        mm = self.mixing_matrix
        model = PICASSOnn(mm[0,:,:])
        self._progress = progress(range(kwargs.get('max_iter', 100)))
        self._progress.set_description("Optimizing mixing parameters")
        self._worker = create_worker(self.train_model, model,
                                _connect={
                                    'returned': self.unmix_images,
                                    'yielded': self.update_progress,
                                    # 'started': self.start_train,
                                    # 'finished': self.finished_train,
                                    # 'errored': self.errored_train
                                    },
                                    **kwargs)
    # widget._make_model = make_model                                         #handle for testing
    def start_train(self, *args):
        print('Start training')
        print(*args)

    def finished_train(self, *args):
        print('Finished training')
        print(*args)


    def errored_train(self, *args):
        print('Error in training')
        print(*args)

    def train_model(self, model, **kwargs):
        # print('model sinks', model.n_sinks)
        # print('model images', model.n_images)
        # print('n images', len(self.images))
        for i in model.train_loop(self.images, **kwargs):
            yield i

        # print(model.mixing_parameters)

        return model.mixing_parameters

    def update_progress(self, iter):

        if self._progress is None:
            # print some warning message
            pass
        elif iter < self._progress.total-1:
            self._progress.update(1)
        elif iter >= self._progress.total - 1:
            self._progress.display(['Unmixing images', iter])
            self._progress.close()
            self._progress = None

    def add_sink_widget(self: Container) -> None:

        i = 0
        while i in self.sinks:
            i += 1

        if len(self.sinks) >= len(get_image_layers(self._viewer)):
            print('Number of sinks equal to number of images')
        elif len(self.image_choices) > 0:
            self.sinks = i
            ALPHA = self._options.get('manual', False)
            BG = self._options.get('background', True)
            self.insert(i+1, SinkWidget(self._viewer, i, ALPHA=ALPHA, BG=BG))
        else:
            print('No more images to add as sink')



    def open_options(self: Container):
        if self._options_widget is None:
            self._options_widget = Options()
            self._options_widget.changed.connect(self.update_options)
        self._options_widget.show()

    def update_options(self: Container):
        self._options = self._options_widget()
        self.toggle_parameter_sliders()

    def toggle_parameter_sliders(self: Container):

        visible = self._options.get('manual', False)
        if self.sliders_visible != visible:
            for s in self.sinks:
                self[f'sink{s}'].src_opts.toggle_sliders(visible)
            self.sliders_visible = visible

    def unmix_images(self, mixing_matrix = None):
        '''Unmix images and add unmixed images to new image layer.'''

        changed = False
        if mixing_matrix is not None:
            self.mixing_matrix = mixing_matrix
            changed = True

        mm = self.mixing_matrix
        images = self.images
        image_names = self.image_names
        alpha = mm[0,:,:]
        bg = mm[1,:,:]

        nimages, nsinks = alpha.shape
        assert len(images) == nimages, f'Expected {nimages} images, got {len(images)} images'

        fimages_ = []
        for im in images:
            if type(im) is XR_TYPE:
                fimages_.append(im.data.flatten())
            else:
                fimages_.append(im.flatten())
        fimages = da.stack(fimages_, axis=1)
        row, col = im.shape

        for i in range(nsinks):
            sink_ind = np.where(mm[0,:,i] == 1)[0][0]
            sink_name = image_names[sink_ind]
            worker = create_worker(self.unmix, fimages, bg[:,i].T, alpha[:,i], row, col, sink_name,
                                    _connect = {'returned':self.add_image})
        if changed:
            self.picasso_params = mixing_matrix

    def unmix(self, fimages, bg, alpha, row, col, sink_name):

        layer_info = get_layer_info(self._viewer, sink_name)
        layer_info['name'] = 'unmixed_' + sink_name

        if self._options.get('BG', True):
            unmixed = (fimages - bg) @ alpha
        else:
            unmixed = fimages @ alpha
        unmixed = da.clip(unmixed, 0, None)
        unmixed = unmixed.reshape((row, col))

        return unmixed.compute(), layer_info

    def add_image(self, *args):
        image = args[0][0]
        layer_info = args[0][1]
        self._viewer.add_image(image, **layer_info)

    @property
    def sinks(self: Container) -> [int]:
        '''List of sink image indices'''
        self._sinks = [int(s.name[4:]) for s in self._list if 'sink' in s.name and s.enabled]
        return self._sinks

    @sinks.setter
    def sinks(self, index: int) -> None:
        self._sinks.append(index)


    @property
    def mixing_dict(self):
        '''Dictionary of mixing parameters. source : sink : alpha'''

        mp = {}
        for s in self.sinks:
            sink = self[f'sink{s}']
            mp[sink.sink_list.current_choice] = sink.mixing_params

        self._mixing_dict = mp

        return mp

    @mixing_dict.setter
    def mixing_dict(self, mix_dict):
        '''Dictionary of mixing parameters. source : sink : alpha'''

        mp = {}
        for s in self.sinks:
            sink = self[f'sink{s}']
            sink_name = sink[f'sinklist{s}'].current_choice
            sink.src_opts.set_mixing_parameters(mix_dict[sink_name])

        self._mixing_dict = mp

        return mp


    @property
    def mixing_matrix(self):
        '''Get unsorted mixing matrix from mixing dictionary.

            columns of mixing matrix are unmixed sinks, rows are images
            1 sink image per column labeled with 1
            source images in column labeled with value < 0

        '''

        mix_dict = self.mixing_dict                                              # dictionary = {sink image :{source image: mixing param}}
        images = self.image_names                                               # list of image names selected as sink or source

        mm = np.zeros((2, len(images), len(mix_dict)))
        for i, sink in enumerate(mix_dict.keys()):
            for ii, source in enumerate(images):
                mix_params  = mix_dict[sink].get(source, {'alpha':0})
                if mix_params['alpha'] > 0 and sink != source:
                    mm[0,ii,i] = -mix_params['alpha']
                    mm[1,ii,i] = mix_params.get('background', 0)
                elif mix_params['alpha'] == 0 and sink == source:
                    mm[0,ii,i] = 1

        self._mixing_matrix = mm

        return mm


    @mixing_matrix.setter
    def mixing_matrix(self, mm):

        mix_dict = self.mixing_dict
        images = self.image_names                                               # list of image names

        BG = self._options.get('BG', True)
        if BG:
            assert mm.ndim == 3
        else:
            assert mm.ndim == 2
            mm = np.expand_dims(mm,axis=0)

        dum, nimages, nsinks = mm.shape
        if len(mix_dict) != nsinks or len(images) != nimages:
            raise ValueError(f'Mismatch between number of selected sink and source images and mixing matrix, try rerunning PICASSO')

        for i, sink in enumerate(mix_dict.keys()):
            assert len(mix_dict[sink]) == (mm[0,:,i] < 0).sum(), f'Mismatch between number of sources for sink {sink}'
            for ii, source in enumerate(images):
                if mm[0,ii,i] < 0 and sink != source:
                    mix_dict[sink][source]['alpha'] = -mm[0,ii,i]
                    if BG:
                        mix_dict[sink][source]['background'] = mm[1,ii,i]

        self.mixing_dict = mix_dict
        self._mixing_matrix = mm


    @property
    def image_names(self):
        'List of sink/source image names'

        mix_dict = self.mixing_dict
        images = dict()

        # Unique images that are marked as sinks
        for sink in mix_dict.keys():
            images.update({sink:None})
        # Add unique images that are marked as sources
        for sinksource in mix_dict.values():
            for source in sinksource.keys():
                images.update({source:None})
        images = list(images)

        # Reorder to match viewer layer List
        images_ = []
        for l in self._viewer.layers:
            if l.name in images:
                images_.append(l.name)
        self._image_names = images_

        return self._image_names

    @property
    def images(self):
        'List of sink/source images'

        image_names = self.image_names
        self._images = []
        layers =  self._viewer.layers

        for l in layers:
            if l.name in image_names:
                self._images.append(l.data)

        return self._images

    @property
    def image_choices(self):
        '''List of images choices to select as sink.'''

        images = get_image_layers(self._viewer)
        image_names = [i.name for i in images]

        # Filter out images that have already been used as sinks
        for s in self.sinks:
            sink_name = self[f'sink{s}'].sink_list.current_choice
            try:
                ind = image_names.index(sink_name)
                image_names.pop(ind)
                images.pop(ind)
            except ValueError:
                print(f'{sink_name} selected as sink image more than once')

        self._image_choices = images

        return images
