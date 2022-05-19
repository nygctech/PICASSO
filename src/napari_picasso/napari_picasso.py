# from napari_picasso.main_widget import PicassoWidget
# from napari.qt.threading import thread_worker
# from napari.utils import progress
#
# def return_widget(viewer: 'napari.viewer.Viewer', **kwargs):
#     widget = PicassoWidget(viewer, **kwargs)
#
#     def make_model(*args, **kwargs):
#         from picasso.nn_picasso import PICASSOnn
#         mm = widget.mixing_matrix
#         model = PICASSOnn(mm[0,:,:])
#         widget._progress = progress(range(kwargs.get('max_iter', 100)))
#         widget._progress.set_description("Optimizing mixing parameters")
#         worker = train_model(model, widget.images, **kwargs)
#     widget._make_model = make_model                                             #handle for testing
# 
#     @thread_worker(connect={'returned': widget.unmix_images,
#                             'yielded': widget.update_progress})
#     def train_model(model, images, **kwargs):
#
#         # max_iter = kwargs.get('max_iter', 100)
#         # pbr = progress(range(max_iter))
#         for i in model.train_loop(images, **kwargs):
#             yield i
#             pass
#         #     pbr.update(i)
#         # pbr.close()
#
#         return model.mixing_parameters
#
#     widget.run_btn.changed.connect(make_model)
#
#     return widget
