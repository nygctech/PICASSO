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

        self.changed.connect(self.get_mixing_params)


    def add_sink_widget(self: Container) -> None:

        i = 0
        while i in self.sinks:
            i += 1

        self.sinks = i
        self.append(SinkWidget(i, self._viewer.layers))


    def open_options(self: Container):
        print('Open options not implemented yet')

    def run_picasso(self: Container):
        print('Run picasso not implemented yet')

    @property
    def sinks(self: Container) -> [int]:
        '''List of sink image indices'''
        self._sinks = [int(s.name[4:]) for s in self._list if 'sink' in s.name and s.enabled]
        return self._sinks

    @sinks.setter
    def sinks(self, index: int) -> None:
        self._sinks.append(index)


    def get_mixing_params(self) -> {ImageData:{ImageData:float}}:
        '''Dictionary of mixing parameters. source : sink : alpha'''

        mp = {}
        for s in self.sinks:
            sink = self[f'sink{s}']
            mp[sink.sink_list.value] = sink.mixing_params

        self.mixing_params = mp




class SinkWidget(Container):
    '''Container to select sink image and spectra spillover source images.'''

    def __init__(self, index, images):

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


    def show_sources(self) -> None:
        '''Show widget to select source images.'''

        if self.current_src_opts is None:
            self.current_src_opts = SourceOptions(self.source_images(), self.mixing_params)
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

        return mp




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

    def __init__(self, images: [ImageData], index: int = 0, alpha: float = 0.0, **kwargs):

        self.index = index                                                      # Source index, int

        source_list = _SourceList(images, index, **kwargs)
        source_list.delete_btn.changed.connect(self.delete_source)
        self.source_list = source_list

        alpha_slider = FloatSlider(max=2.0, step=0.01, name=f'alpha{index}', value = alpha, tracking = False)
        alpha_slider.max_width = 175
        self.alpha = alpha_slider

        super().__init__(
                         name=f'source{index}',
                         labels = False
                        )

        self.append(source_list)
        self.append(alpha_slider)


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

        return img, alpha



class SourceOptions(Container):
    '''Define multiple source image and alpha pairs that spill over into the sink image.'''

    def __init__(self, images: [ImageData], index: int = 0,  mixing_params: dict = {}):


        add_source_btn = PushButton(name=f'addsrc', label = '+')
        add_source_btn.max_width = 30
        add_source_btn.changed.connect(self.add_source)


        super().__init__(widgets = [add_source_btn],
                         name=f'sink{index}_sources',
                         labels = False
                        )

        if bool(mixing_params):
            self.add_source()
        else:
            for sink_im, alpha in mixing_params.items():
                add_source(alpha, value = sink_im)

        self._images = images                                                   # List of possible source images



    def add_source(self, alpha: float = 0.0, **kwargs) -> None:
        '''Add new source widget.'''

        i = 0
        while i in self.sources:
            i += 1

        self.sources = i
        src = SourceWidget(self._images, i, alpha, **kwargs)

        src[f'source_list{i}'][f'source{i}'].changed.connect(self.__call__)
        src[f'alpha{i}'].changed.connect(self.__call__)

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
            img, alpha = self[f'source{s}'].mixing_param
            mp[img] = alpha

        return mp
