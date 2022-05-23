from magicgui.widgets import RadioButtons, Container

class Options(Container):
    '''Source image ComboBox and delete button.'''

    def __init__(self):

        mode = RadioButtons(name='mode', label = 'mode',
                            choices=['picasso', 'manual'], value='picasso')
        self.mode = mode

        super().__init__(widgets = (mode, ),
                         layout = 'horizontal',
                         name='options',
                        )

    def __call__(self):
        opts = {'manual': self.mode.value == 'manual'}

        return opts
