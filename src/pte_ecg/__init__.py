"""PTE-ECG: A Python package for ECG signal processing and feature extraction.

This package provides tools for preprocessing ECG signals and extracting various types of features
including statistical, morphological, and nonlinear features. It is designed to work with
multi-channel ECG data and supports parallel processing for efficient computation.
"""

from ._logging import logger, set_log_file, set_log_level
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
    "logger",
    "set_log_level",
    "set_log_file",
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
