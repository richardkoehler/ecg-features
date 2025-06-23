from typing import Literal, Self

import numpy as np
import pandas as pd
import pydantic
from pydantic import Field

from .features import (
    FeatureSettings,
    get_fft_features,
    get_morphological_features,
    get_nonlinear_features,
    get_statistical_features,
    get_welch_features,
)
from .preprocessing import (
    PreprocessingSettings,
    preprocess,
)


class Settings(pydantic.BaseModel):
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)

    @pydantic.model_validator(mode="after")
    def check_any_features(self) -> Self:
        if not any(feature.enabled for _, feature in self.features):
            raise ValueError("No features enabled. Please enable at least one feature.")
        return self


def get_features(
    ecg: np.ndarray,
    sfreq: float,
    settings: Settings | Literal["default"] = "default",
) -> pd.DataFrame:
    if settings == "default":
        settings = Settings()
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
