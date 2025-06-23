import numpy as np
import pandas as pd
import pydantic

from .features import (
    FeatureArgs,
    get_fft_features,
    get_morphological_features,
    get_nonlinear_features,
    get_statistical_features,
    get_welch_features,
)
from .preprocessing import (
    PreprocessingArgs,
    preprocess,
)


class Settings(pydantic.BaseModel):
    preprocessing: PreprocessingArgs = PreprocessingArgs()
    features: FeatureArgs = FeatureArgs()


def get_features(
    ecg: np.ndarray,
    sfreq: float,
    settings: Settings = Settings(),
) -> pd.DataFrame:
    if settings.preprocessing.enabled:
        ecg, sfreq = preprocess(ecg, sfreq=sfreq, preprocessing=settings.preprocessing)
    feature_list = []
    if settings.features.fft.enabled:
        feature_list.append(get_fft_features(ecg, sfreq))
    if settings.features.morphological.enabled:
        feature_list.append(get_morphological_features(ecg, sfreq, n_jobs=-1))
    if settings.features.nonlinear.enabled:
        feature_list.append(get_nonlinear_features(ecg, sfreq))
    if settings.features.statistical.enabled:
        feature_list.append(get_statistical_features(ecg, sfreq, n_jobs=-1))
    if settings.features.welch.enabled:
        feature_list.append(get_welch_features(ecg, sfreq))
    return pd.concat(feature_list, axis=1)
