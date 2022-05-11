def get_layer_info(viewer, layer_name):
    layer_names = [l.name for l in viewer.layers]

    if layer_name in layer_names:
        ind = layer_names.index(layer_name)
        return viewer.layers[ind].as_layer_data_tuple()[1]
    else:
        raise ValueError(f'{layer_name} not in viewer')
