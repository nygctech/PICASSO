
try:
    from _version import version as __version__
    from _version import version_tuple
except ImportError:
    __version__ = 0.3.0
    version_tuple = (0, 3, 0)






# from ._widget import ExampleQWidget, example_magic_widget
