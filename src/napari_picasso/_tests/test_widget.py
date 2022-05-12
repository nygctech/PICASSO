from napari_picasso.main_widget import PicassoWidget
from napari_picasso.utils import get_image_layers
import numpy as np
import napari
import pytest

@pytest.fixture
def loaded_viewer(sink_source):
    viewer = napari.Viewer(show = False)
    viewer.add_image(sink_source[0], name='sink')
    viewer.add_image(sink_source[1], name='source')

    return viewer

@pytest.fixture
def loaded_widget(loaded_viewer, sink_source):

    widget = PicassoWidget(loaded_viewer, visible=False)
    widget.add_sink_widget()

    sink = widget['sink0']
    sink.sink_list.bind(sink_source[0])
    sink.show_sources()
    sink.update_mixing_params()

    for s in widget.sinks:
        sink = widget[f'sink{s}']

    return widget

@pytest.fixture
def sink_source():
    s = 512; background = 100; signal = 500; alpha = 0.5

    sink = np.random.rand(s,s)*10+background
    sink[:100, :100] += np.random.rand(100,100)*2000+signal

    source = np.zeros((s,s))
    source[-100:,-100:] += np.random.rand(100,100)*2000+signal

    sink += alpha*source

    source += np.random.rand(s,s)*10+background

    return sink, source


def test_mine_import():
    from mine.mine import MINE
    model =  MINE()
    print(f'Using {model.device}')

    assert model.device in ['cuda', 'cpu']

def test_picassoNN_import():
    mm = np.array([1, -1])
    from picasso.nn_picasso import PICASSOnn
    model = PICASSOnn(mm)
    print(f'Using {model.device}')

    assert model.device in ['cuda', 'cpu']

def test_get_napari_image_names(loaded_widget):
    im_names = loaded_widget.image_names

    assert len(im_names) == 2
    for i in im_names:
        assert i in ['sink', 'source']


def test_mixing_dict_to_matrix(loaded_widget):
    mm = loaded_widget.mixing_matrix

    assert mm.ndim == 3
    assert mm[0,0,0] == 1
    assert mm[0,1,0] == -0.01
    assert int(mm[1,1,0]) == 100

def test_set_mixdict(loaded_widget):
    alpha = (2000*0.5)/(2000-100)
    mm = np.array([[[1], [-alpha]],[[0],[100]]])
    loaded_widget.mixing_matrix = mm

    assert loaded_widget.mixing_dict['sink']['source']['alpha'] == alpha
    assert  loaded_widget.mixing_dict['sink']['source']['background'] == 100

def test_unmix_images(loaded_widget):

    alpha = (2000*0.5)/(2000-100)
    mm = np.array([[[1], [-alpha]],[[0],[100]]])
    loaded_widget.mixing_matrix = mm

    loaded_widget.unmix_images()

    layer_names = [l.name for l in loaded_widget._viewer.layers]

    assert 'unmixed_sink' in layer_names

#
#
# def test_mixing_dict_to_matrix(loaded_widget):
#     mm = loaded_widget.mixing_dict
#
#     assert mm[0]



# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
#def test_picasso_widget(make_napari_viewer, capsys):
def test_picasso_widget(loaded_widget):
    loaded_widget.run_picasso(max_iter = 10)
    image_layers = get_image_layers(loaded_widget._viewer)
    image_names = [l.name for l in image_layers]

    assert 'unmixed_sink' in image_names

    #loaded_widget.run_picasso()
    # # make viewer and add an image layer using our fixture
    #viewer = make_napari_viewer()

    #viewer = napari.Viewer()
    # viewer.add_image(sink_source[0], name='sink')
    # viewer.add_image(sink_source[1], name='source')
    #
    # # create our widget, passing in the viewer
    # my_widget = PicassoWidget(loaded_viewer)
    # # for im in in
    # my_widget.mixing_dict = {}

    #

    # call our widget method
    # my_widget._on_click()

    # read captured output and check that it's as we expected
    # print('Test capsys')
    # captured = capsys.readouterr()
    # print(captured.out)
    # assert captured.out == "Test capsys\n"

# def test_example_magic_widget(make_napari_viewer, capsys):
#     viewer = make_napari_viewer()
#     layer = viewer.add_image(np.random.random((100, 100)))
#
#     # this time, our widget will be a MagicFactory or FunctionGui instance
#     my_widget = example_magic_widget()
#
#     # if we "call" this object, it'll execute our function
#     my_widget(viewer.layers[0])
#
#     # read captured output and check that it's as we expected
#     captured = capsys.readouterr()
#     assert captured.out == f"you have selected {layer}\n"
