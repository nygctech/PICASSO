from napari_picasso.main_widget import PicassoWidget
from napari.qt.threading import thread_worker

def return_widget(viewer: 'napari.viewer.Viewer', **kwargs):
    widget = PicassoWidget(viewer, **kwargs)

    def make_model(**kwargs):
        from picasso.nn_picasso import PICASSOnn
        mm = widget.mixing_matrix
        model = PICASSOnn(mm[0,:,:])
        worker = train_model(model, widget.images, **kwargs)

    widget._make_model = make_model

    @thread_worker(connect={"returned": widget.unmix_images})
    def train_model(model, images, **kwargs):
        model.train_loop(images, **kwargs)
        return model.mixing_parameters

    widget.run_btn.changed.connect(make_model)

    return widget
