from magicgui.widgets import RadioButtons, Container, CheckBox, SpinBox

class Options(Container):
    '''Source image ComboBox and delete button.'''

    def __init__(self):

        mode = RadioButtons(name='mode', label = 'mode',
                            choices=['picasso', 'manual'], value='picasso')
        self.mode = mode

        picasso_params = CheckBox(name='picasso_params', label='picasso parameters',
                                    value=False, enabled = False)
        self.picasso_params = picasso_params

        background = CheckBox(name='background', label='background',
                                    value=True)
        self.background = background

        max_iter = SpinBox(name='max_iterations', label='max iterations', value=100)
        self.max_iter = max_iter

        super().__init__(widgets = (mode, picasso_params, background, max_iter),
                         layout = 'vertical',
                         name='options',
                        )


    def __call__(self):
        opts = {'manual': self.mode.value == 'manual',
                'picasso_params': self.mode.value,
                'background': self.background.value,
                'max_iter': self.max_iter.value}

        return opts
