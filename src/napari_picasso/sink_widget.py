from magicgui.widgets import PushButton, Container, ComboBox
from napari.types import ImageData
from napari_picasso.utils import get_layer_info, get_image_layers
from napari_picasso.source_widget import SourceOptions

class SinkWidget(Container):
    '''Container to select sink image and spectra spillover source images.'''

    def __init__(self, viewer, index, ALPHA: bool = False, BG: bool = True):

        images = get_image_layers(viewer)

        sink_list = ComboBox(choices = images, name = f'sinklist{index}', label=f'sink{index}')
        sink_list.max_width = 175
        sink_list.changed.connect(self.set_sink)
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

        self.mixing_params = {}                                                 # {source image: {'alpha':a,'background':b}
        self.index = index                                                      # index of sink: int
        self.src_opts = None                                                    # source options widget
        self.n_sources = 0                                                      # Number of sources that spillover
        self.BG = BG                                                            # Flag to show background parameter
        self.ALPHA = ALPHA                                                      # Flag to show alpha parameter
        self._viewer = viewer

    def show_sources(self) -> None:
        '''Show widget to select source images.'''

        if self.src_opts is None:
            initial_sink = self.sink_list.current_choice
            self.src_opts = SourceOptions(self._viewer, initial_sink, self.index, self.mixing_params,
                                            BG = self.BG, ALPHA = self.ALPHA)
            self.src_opts.changed.connect(self.update_mixing_params)

        self.src_opts.show()
        self.update_mixing_params()


    def update_mixing_params(self) -> None:
        '''Update mixing parameters, saved as {source image: alpha}.'''

        if self.src_opts is not None:
            self.mixing_params = self.src_opts()


    def del_sink(self) -> None:
        '''Delete sink widget.'''

        self.enabled = False
        self.close()

    def set_sink(self) -> None:
        '''Assign sink name and clear source options.'''

        sink_name = self.sink_list.current_choice
        if self.src_opts is not None:
            self.src_opts.sink = sink_name
        self.update_mixing_params()
        # self.mixing_params = self.src_opts()


        # if self.src_opts is not None:
        #     self.src_opts.sink = sink_name
        #     self.mixing_params = None

        #update sources

    # def source_images(self) -> [ImageData]:
    #     '''Source image options (sink image removed).'''
    #
    #     src_imgs = self._images.copy()
    #     src_names = [s.name for s in src_imgs]
    #
    #     sink_img = self.sink_list.current_choice
    #
    #     if sink_img in src_names:
    #         src_imgs.remove(sink_img)
    #
    #     return src_imgs
