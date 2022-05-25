def get_layer_info(viewer, layer_name):
    layer_names = [l.name for l in viewer.layers]

    if layer_name in layer_names:
        ind = layer_names.index(layer_name)
        return viewer.layers[ind].as_layer_data_tuple()[1]
    else:
        return None
        # raise ValueError(f'{layer_name} not in viewer')


def get_image_layers(viewer):
    images = []
    for l in viewer.layers:
        if l.as_layer_data_tuple()[2] == 'image':                               #TODO: Find better way to filter image layers
            images.append(l)

    return images
