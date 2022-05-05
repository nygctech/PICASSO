from magicgui.widgets import PushButton, FloatSlider, Container, ComboBox, FunctionGui
from napari.types import ImageData


class PicassoWidget(Container):
    '''Main picasso widget.'''

    def __init__(self, viewer: 'napari.viewer.Viewer'):

        super().__init__()

        # Header buttons
        add_sink_btn = PushButton(text="+sink", name="add_snk")
        open_options_btn = PushButton(text="options", name="options")
        run_btn = PushButton(text="run", name="run")

        # Connect header buttons to header functions
        add_sink_btn.changed.connect(self.add_sink_widget)
        open_options_btn.changed.connect(self.open_options)
        run_btn.changed.connect(self.run_picasso)

        # Size header buttons
        widgets = [add_sink_btn, open_options_btn, run_btn]
        for w in widgets:
            w.max_width = 70

        header = Container(widgets = widgets, layout='horizontal', labels=False)
        self.append(header)

        self.mixing_params = {}                                                 # Current mixing paramers {sink : {source:alpha}}
        self._viewer = viewer                                                   # napari viewer

        #self.changed.connect(self.get_mixing_params)

        self.picasso_params = None
        self.BG = True


    def add_sink_widget(self: Container) -> None:

        i = 0
        while i in self.sinks:
            i += 1

        self.sinks = i
        self.append(SinkWidget(i, self._viewer.layers, BG = self.BG))


    def open_options(self: Container):
        print('Open options not implemented yet')

    def run_picasso(self: Container):
        from picasso import PICASSOnn

        mm = self.mixing_matrix
        model = PICASSOnn(mm)
        model.train_loop(self.images)
        self.mixing_matrix = model.mixing_parameters                            #get mixing parameters
        self.unmix_images()                                                     #unmix images and add new layer

    def unmix_images(self):

        mm = self.mixing_matrix
        images = self.images

        if self.BG:
            assert mm.ndim == 3
            alpha = mm[0,:,:]
            bg = mm[1,:,:]
        else:
            assert mm.ndim == 2
            alpha = mm

        nsinks, nimages = alpha.shape
        assert len(images) == nimages, f'Expected {nimages}, got {len(images)}'

        fimages_ = []
        for im in images:
            fimages_.append(im.data.flatten())
        fimages = da.stack(fimages_, axis=1)

        for i in range(nsinks):
            sink_ind = np.where(mm[:,i] == 1)[0][0]
            layer_info = images[sink_ind].as_layer_data_tuple()[1]
            layer_info['name'] = 'unmixed_' + layer_info['name']
            if self.BG:
                unmixed = (fimages - bg[:,i].T) @ alpha[:,i]
            else:
                unmixed = fimages @ alpha[:,i]
            self.viewer.add_image(unmixed, **layer_info)

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

        sources = {}
        for s in self.sinks:
            sink = self[f'sink{s}']
            mp[sink.sink_list.value] = sink.mixing_params

        self._mixing_dict = mp


    @property
    def mixing_matrix(self):
        '''Get unsorted mixing matrix from mixing dictionary.

            columns of mixing matrix are unmixed sinks, rows are images
            1 sink image per column labeled with 1
            source images in column labeled with value < 0

        '''

        mixdict = self.mixing_dict                                              # dictionary = {sink image :{source image: mixing param}}
        images = self.images                                                    # list of images selected as sink or source

        mm = np.zeros((2, len(images), len(mixdict)))
        for i, sink in enumerate(mixdict.keys()):
            for ii, source in enumerate(images):
                mix_params  = mix_dict[sink].get(source, {'alpha':0})
                if mix_params['alpha'] > 0 and sink is not source:
                    mm[0,ii,i] = -mix_param['alpha']
                    mm[1,ii,i] = mix_param.get('background', 0)
                elif mix_params == 0 and sink is source:
                    mm[ii,i] = 1

        if mm[1,:,:].sum() == 0:                                                # Remove background if not used
            mm = mm[0,:,:]

        self._mixing_matrix = mm

        return mm


    @mixing_matrix.setter
    def mixing_matrix(self, mm):

        mixdict = self.mixing_dict
        images = self.images

        if self.BG:
            assert mm.ndim == 3
        else:
            assert mm.ndim == 2
            mm = np.expand_dims(mm,axis=0)

        nsources, nsinks = mm.shape
        if len(mix_dict) != nsinks or len(sources) != nsources:
            raise ValueError(f'Mismatch between number of selected sink and source images and mixing matrix, try rerunning PICASSO')

        for i, sink in enumerate(mixdict.keys()):
            for ii, source in enumerate(images):
                if mix_param < 0 and sink is not source:
                    mix_dict[sink][source]['alpha'] = -mm[0,ii,i]
                    if self.BG:
                        mix_dict[sink][source]['background'] = mm[1,ii,i]

        self._mixing_matrix = mm


    @property
    def images(self):
        'List of source image names'

        mixdict = self.mixing_dict
        images = dict()

        for sink in mixdict.keys():
            images.update({sink:None})

        for sinksource in mix_dict.values():
            for source in sinksource.keys():
                images.update({source:None})

        self._images = list(images)

        return self._images





class SinkWidget(Container):
    '''Container to select sink image and spectra spillover source images.'''

    def __init__(self, index, images, BG: bool = False):

        sink_list = ComboBox(choices = images, name = f'sinklist{index}', label=f'sink{index}')
        sink_list.max_width = 175
        self.sink_list = sink_list

        sink_options_btn = PushButton(name=f'sinkopts{index}', label = 'sources')
        sink_options_btn.max_width = 70
        sink_options_btn.changed.connect(self.show_sources)

        sink_del_btn = PushButton(name=f'del{index}', label = '-')
        sink_del_btn.max_width = 30
        sink_del_btn.changed.connect(self.del_sink)



        super().__init__(widgets = (sink_list, sink_options_btn, sink_del_btn),
                         layout='horizontal',
                         name=f'sink{index}',
                         label = ' '
                        )

        self.mixing_params = {}                                                 # {source image: alpha}
        self.index = index                                                      # index of sink: int
        self.current_src_opts = None                                            # source options widget
        self.n_sources = 0                                                      # Number of sources that spillover
        self._images = images                                                   # List of images in napari
        self.BG = BG                                                            # Flag to show background parameter


    def show_sources(self) -> None:
        '''Show widget to select source images.'''

        if self.current_src_opts is None:
            self.current_src_opts = SourceOptions(self.source_images(), self.mixing_params, BG = self.BG)
            self.current_src_opts.changed.connect(self.update_mixing_params)

        self.current_src_opts.show()


    def update_mixing_params(self) -> None:
        '''Update mixing parameters, saved as {source image: alpha}.'''

        self.mixing_params = self.current_src_opts()


    def del_sink(self) -> None:
        '''Delete sink widget.'''

        self.enabled = False
        self.close()

    def source_images(self) -> [ImageData]:
        '''Source image options (sink image removed).'''

        src_imgs = self._images.copy()
        sink_img = self.sink_list.value
        if sink_img in src_imgs:
            src_imgs.remove(sink_img)

        return src_imgs




class _SourceList(Container):
    '''Source image ComboBox and delete button.'''

    def __init__(self, images: [ImageData], index: int = 0, **kwargs):

        delete_btn = PushButton(name=f'del{index}', label = '-')
        delete_btn.max_width = 30
        self.delete_btn = delete_btn


        source_list = ComboBox(choices= images, name=f'source{index}', **kwargs)
        source_list.max_width = 175
        self.source_list = source_list


        super().__init__(widgets = (delete_btn, source_list),
                         layout = 'horizontal',
                         name=f'source_list{index}',
                         labels = False
                        )




class SourceWidget(Container):
    '''Select single source image and alpha mixing parameter pair.'''

    def __init__(self, images: [ImageData], index: int = 0, alpha: float = 0.0,
                       background: float = 0.0, BG: bool = False, **kwargs):

        self.index = index                                                      # Source index, int

        source_list = _SourceList(images, index, **kwargs)
        source_list.delete_btn.changed.connect(self.delete_source)
        self.source_list = source_list

        alpha_slider = FloatSlider(max=2.0, step=0.01, name=f'alpha{index}', value = alpha, tracking = False)
        alpha_slider.max_width = 175
        self.alpha = alpha_slider


        bg_slider = FloatSlider(name=f'background{index}', value = background,
                                tracking = False, visible = BG)
        bg_slider.max_width = 175
        self.background = bg_slider
        self.update_bg_slider()

        super().__init__(
                         name=f'source{index}',
                         labels = False
                        )

        self.append(source_list)
        self.append(alpha_slider)
        self.append(bg_slider)



    def delete_source(self) -> None:
        '''Remove source widget.'''

        self.enabled = False
        self.alpha.value = 0.0
        self.close()


    @property
    def mixing_param(self) -> (ImageData, float):
        '''Tuple of source image and alpha mixing parameter.'''

        img = self.source_list.source_list.value
        alpha = self.alpha.value
        background = self.background.value

        self._mixing_param = (img, alpha, background)

        return img, alpha, background

    def update_bg_slider(self):
        '''Update max, min, and step values of background slider.'''

        try:
            self.min_px, self.max_px = self.source_list.source_list.value.contrast_limits
        except AttributeError:
            self.min_px = 0; self.max_px = 1000
        self.step = 1 if self.max_px > 1 else 0.01

        if self.background.max != self.max_px:
            self.background.max = self.max_px
        if self.background.min != self.min_px:
            self.background.min = self.min_px
        if self.background.step != self.step:
            self.background.step = self.step



class SourceOptions(Container):
    '''Define multiple source image and alpha pairs that spill over into the sink image.'''

    def __init__(self, images: [ImageData], index: int = 0,  mixing_params: dict = {}, BG: bool = False):

        self._images = images                                                   # List of possible source images
        self.BG = BG

        add_source_btn = PushButton(name=f'addsrc', label = '+')
        add_source_btn.max_width = 30
        add_source_btn.changed.connect(self.add_source)


        super().__init__(widgets = [add_source_btn],
                         name=f'sink{index}_sources',
                         labels = False
                        )

        self.add_source()
        # if bool(mixing_params):
        #     self.add_source()
        # else:
        #     for sink_im, alpha in mixing_params.items():
        #         self.add_source(alpha, background, value = sink_im)


    def add_source(self, alpha: float = 0.0, background: float = 0.0, **kwargs) -> None:
        '''Add new source widget.'''

        i = 0
        while i in self.sources:
            i += 1

        self.sources = i
        src = SourceWidget(self._images, i, alpha, background, BG = self.BG, **kwargs)

        src[f'source_list{i}'][f'source{i}'].changed.connect(self.__call__)
        src[f'alpha{i}'].changed.connect(self.__call__)
        src[f'background{i}'].changed.connect(self.__call__)
        self.insert(i+1,src)

        self.__call__()



    @property
    def sources(self) -> [int]:
        '''List of sink indices.'''
        self._sources = [int(s.name[6:]) for s in self._list if 'source' in s.name and s.enabled]
        return self._sources

    @sources.setter
    def sources(self, index: int):
        self._sources.append(index)


    def __call__(self) -> {ImageData:float}:
        '''Return dictionary of mixing paremeters {sink image : alpha}.'''

        mp = {}
        for s in self.sources:
            self[f'source{s}'].update_bg_slider()
            img, alpha, background = self[f'source{s}'].mixing_param
            mp[img] = {'alpha':alpha, 'background':background}

        return mp
