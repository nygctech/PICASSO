from magicgui.widgets import PushButton, FloatSlider, Container, ComboBox
from napari.types import ImageData
import numpy as np
from napari_picasso.utils import get_layer_info, get_image_layers
from napari.utils.notifications import show_info

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

    def __init__(self, viewer, images: [ImageData], index: int = 0, alpha: float = 0.01,
                       background: float = 0.0, BG: bool = False, ALPHA: bool = False, **kwargs):

        self._viewer = viewer
        self.index = index                                                      # Source index, int

        source_list = _SourceList(images, index, **kwargs)
        source_list.delete_btn.changed.connect(self.delete_source)
        self.source_list = source_list

        alpha_slider = FloatSlider(max=2.0, step=0.01, name=f'alpha{index}',
                                    value = alpha, tracking = False, visible = ALPHA)
        alpha_slider.max_width = 175
        self.alpha = alpha_slider


        bg_slider = FloatSlider(name=f'background{index}', value = background,
                                tracking = False, visible = BG)
        bg_slider.max_width = 175
        self.background = bg_slider
        self.update_bg_slider()

        super().__init__(
                         name=f'source{index}',
                         labels = False, **kwargs
                        )

        self.append(source_list)
        self.append(alpha_slider)
        self.append(bg_slider)



    def delete_source(self) -> None:
        '''Remove source widget.'''

        self.enabled = False
        self.alpha.bind(0.0)
        self.hide()


    @property
    def mixing_param(self):
        '''Tuple of source image and alpha mixing parameter.'''

        img = self.source_list.source_list.current_choice
        alpha = self.alpha.get_value()
        background = self.background.get_value()

        self._mixing_param = (img, alpha, background)

        return img, alpha, background

    @mixing_param.setter
    def mixing_param(self, mix_dict: dict):
        '''Set parameter widgets with values from mixing dictionary'''

        img = self.source_list.source_list.current_choice
        alpha = mix_dict[img]['alpha']
        background = mix_dict[img]['background']

        self.alpha.bind(alpha)
        self.background.bind(background)

        self._mixing_param = (img, alpha, background)

    def update_bg_slider(self):
        '''Update max, min, and step values of background slider.'''

        try:
            source_name = self.source_list.source_list.current_choice
            self.min_px, self.max_px = get_layer_info(self._viewer, source_name)['contrast_limits']
            #self.min_px, self.max_px = self.source_list.source_list.value.contrast_limits
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

    def __init__(self, viewer, sink, index: int = 0,  mixing_params: dict = {},
                    ALPHA: bool = False, BG: bool = True, **kwargs):

        self._viewer = viewer                                                   # List of possible source images
        self.BG = BG                                                            # Flag to show background slider
        self.ALPHA = ALPHA                                                      # Flag to show alpha slider
        self._sink = sink

        add_source_btn = PushButton(name=f'addsrc', label = '+')
        add_source_btn.max_width = 30
        add_source_btn.changed.connect(self.add_source)


        super().__init__(widgets = [add_source_btn],
                         name=f'sink{index}_sources',
                         labels = False, **kwargs
                        )

        self.add_source()
        # if bool(mixing_params):
        #     self.add_source()
        # else:
        #     for sink_im, alpha in mixing_params.items():
        #         self.add_source(alpha, background, value = sink_im)


    def add_source(self, alpha: float = 0.01, background: float = 0.0, **kwargs) -> None:
        '''Add new source widget.'''

        i = 0
        while i in self.sources:
            i += 1

        if len(self.sources) >= len(self.image_choices):
            show_info('Max number of sources added')
        else:
            BG = False if not self.ALPHA else self.BG
            self.sources = i
            src = SourceWidget(self._viewer, self.image_choices, i, alpha, background,
                                        BG = BG, ALPHA = self.ALPHA, **kwargs)
            src.source_list.source_list.changed.connect(self.__call__)
            src.alpha.changed.connect(self.__call__)
            src.background.changed.connect(self.__call__)
            self.append(src)

        self.__call__()

    def toggle_sliders(self, ALPHA):
        '''if ALPHA is True show sliders else hide sliders.'''

        for s in self.sources:
            if ALPHA:
                self[f'source{s}'].alpha.show()
                self[f'source{s}'].background.show()
            else:
                self[f'source{s}'].alpha.hide()
                self[f'source{s}'].background.hide()
        self.ALPHA = ALPHA

    @property
    def sources(self) -> [int]:
        '''List of source indices.'''

        self._sources = []

        for s in self._list:
            if 'source' in s.name and s.enabled:
                self._sources.append(int(s.name[6:]))
            elif 'source' in s.name and not s.enabled:
                self._list.remove(s)

        return self._sources

    @sources.setter
    def sources(self, index: int):
        self._sources.append(index)

    def reset_sources(self):
        for s in self.sources:
            self[f'source{s}'].delete_source()
        self._sources = []
        self.add_source()

    # @property
    # def images(self) -> [str]:
    #
    #     # Possible image choices from viewer with sink image removed
    #     image_layers = self.image_choices
    #     image_names = [l.name for l in image_layers]
    #
    #     # Remove images already used as sources
    #     for s in self.sources:
    #         source_name = self[f'source{s}'].source_list.source_list.current_choice
    #         try:
    #             ind = image_names.index(source_name)
    #             image_names.pop(ind)
    #             image_layers.pop(ind)
    #         except ValueError:
    #             print(f'{source_name} selected as source image more than once')
    #
    #     self._images = image_layers
    #
    #     return self._images

    @property
    def image_choices(self):
        '''Potential source images.'''

        # get update list of images
        image_layers = get_image_layers(self._viewer)
        image_names = [l.name for l in image_layers]

        # Remove sink image
        ind = image_names.index(self.sink)
        image_names.pop(ind)
        image_layers.pop(ind)

        self._image_choices = image_layers

        return image_layers



    def __call__(self) -> {ImageData:float}:
        '''Return dictionary of mixing parameters {sink image : {'alpha':a, 'background':b}}.'''

        mp = {}
        for s in self.sources:
            self[f'source{s}'].update_bg_slider()
            img, alpha, background = self[f'source{s}'].mixing_param
            mp[img] = {'alpha':alpha, 'background':background}

        return mp

    def set_mixing_parameters(self, mix_dict: dict):
        for s in self.sources:
            source = self[f'source{s}'].mixing_param = mix_dict



    @property
    def sink(self) -> str:
        '''Corresponding sink name for sources.'''
        return self._sink

    @sink.setter
    def sink(self, sink_name:str):
        '''Update the sink name and delete sources.'''

        if self._sink != sink_name:
            self._sink = sink_name
            self.reset_sources()
