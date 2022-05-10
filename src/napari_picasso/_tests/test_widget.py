#from napari_picasso.widgets import PicassoWidget
import numpy as np
import napari
import pytest

@pytest.fixture
def viewer():
    return napari.Viewer(show = False)

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

def test_widget_import(viewer, sink_source):
    from napari_picasso.widgets import PicassoWidget
    # viewer.add_image(sink_source[0], name='sink')
    # viewer.add_image(sink_source[0], name='source')

    widget = PicassoWidget(viewer)

    assert widget.BG

def test_mine_import():
    from mine.mine import MINE
    model =  MINE()
    print(f'Using {model.device}')

    assert model.device in ['cuda', 'cpu']

def test_picassoNN_import():
    mm = np.zeros([1 -1]).T
    from picasso.nn_picasso import PICASSOnn
    model = PICASSOnn(mm)
    print(f'Using {model.device}')

    assert model.device in ['cuda', 'cpu']


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
#def test_picasso_widget(make_napari_viewer, capsys):
def test_picasso_widget(capsys):
    # # make viewer and add an image layer using our fixture
    #viewer = make_napari_viewer()

    #viewer = napari.Viewer()
    #viewer.add_image(astronaut(), channel_axis=2)

    # create our widget, passing in the viewer
    #my_widget = PicassoWidget(viewer)

    # call our widget method
    # my_widget._on_click()

    # read captured output and check that it's as we expected
    print('Test capsys')
    captured = capsys.readouterr()
    print(captured.out)
    assert captured.out == "Test capsys\n"

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
