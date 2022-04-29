from napari_plugin_engine import napari_hook_implementation
from magicgui import magic_factory
from typing import List, Dict, Any

# @napari_hook_implementation(specname="napari_get_writer")
# def picasso_plugin(path: str, layer_data: List[Tuple[Any, Dict, str]])-> List[str]:
#
#     for layer in layer_data:
#         print(layer)[2]
#         if layer[2] == 'image':
#             layer_name = layer[2]['name']
#             data = layer[0]
#             layer[0] = data * 0.5
#
#     return path



@magic_factory(
    auto_call=True,
    run_button={"widget_type": "PushButton", 'label':'run'},
    new_pair_button={"widget_type": "PushButton", 'label':'new pair'},
    alpha={"widget_type": "FloatSlider", 'max': 2, 'min':0},
)
def picasso_plugin(
    source_image: 'napari.types.ImageData',
    sink_image: 'napari.types.ImageData',
    run_button,
    new_pair_button,
    slider_float=1.0):

    pass

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return picasso_plugin
