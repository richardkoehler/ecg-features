from .features import (
    FeatureSettings,
    FFTArgs,
    MorphologicalArgs,
    NonlinearArgs,
    StatisticalArgs,
    WelchArgs,
    get_fft_features,
    get_morphological_features,
    get_nonlinear_features,
    get_statistical_features,
    get_welch_features,
)
from .pipelines import Settings, get_features
from .preprocessing import (
    BandpassArgs,
    NormalizeArgs,
    NotchArgs,
    PreprocessingSettings,
    ResampleArgs,
    preprocess,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "get_fft_features",
    "get_morphological_features",
    "get_nonlinear_features",
    "get_statistical_features",
    "get_welch_features",
    "get_features",
    "preprocess",
    "Settings",
    "PreprocessingSettings",
    "ResampleArgs",
    "BandpassArgs",
    "NotchArgs",
    "NormalizeArgs",
    "FeatureSettings",
    "FFTArgs",
    "WelchArgs",
    "StatisticalArgs",
    "MorphologicalArgs",
    "NonlinearArgs",
]


def __dir__():
    return __all__
