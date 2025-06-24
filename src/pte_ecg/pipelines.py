"""Pipeline for end-to-end ECG feature extraction.

This module provides a high-level interface for preprocessing ECG data and extracting
various types of features in a configurable pipeline. It combines preprocessing steps
with feature extraction into a single, easy-to-use function.
"""

from typing import Literal, Self

import numpy as np
import pandas as pd
import pydantic
from pydantic import Field

from ._logging import logger
from .features import (
    FeatureSettings,
    get_fft_features,
    get_morphological_features,
    get_nonlinear_features,
    get_statistical_features,
    get_welch_features,
)
from .preprocessing import PreprocessingSettings, preprocess


class Settings(pydantic.BaseModel):
    """Configuration settings for the ECG processing pipeline.

    This class holds all configuration parameters for both preprocessing and
    feature extraction steps of the pipeline.

    Attributes:
        preprocessing: Settings for the preprocessing steps.
        features: Settings for the feature extraction steps.
    """

    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)

    @pydantic.model_validator(mode="after")
    def check_any_features(self) -> Self:
        """Validate that at least one feature type is enabled.

        Raises:
            ValueError: If no feature types are enabled.

        Returns:
            The validated settings instance.
        """
        if not any(feature.enabled for _, feature in self.features):
            raise ValueError(
                "No features enabled. Please enable at least one feature type."
            )
        return self


def get_features(
    ecg: np.ndarray,
    sfreq: float,
    settings: Settings | Literal["default"] = "default",
) -> pd.DataFrame:
    """Extract features from ECG data using the specified settings.

    This is the main entry point for the ECG feature extraction pipeline. It handles:
    1. Input validation
    2. Preprocessing (if enabled)
    3. Feature extraction based on enabled feature types
    4. Concatenation of all extracted features

    Args:
        ecg: Input ECG data with shape (n_samples, n_channels, n_timepoints).
        sfreq: Sampling frequency of the ECG data in Hz.
        settings: Either a Settings object or "default" to use default settings.

    Returns:
        DataFrame containing all extracted features with appropriate column names.

    Raises:
        ValueError: If input data is invalid or no features are enabled.
        TypeError: If settings has an invalid type.
    """
    if not isinstance(settings, Settings | str):
        raise TypeError(
            f"settings must be a Settings instance or 'default', got {type(settings).__name__}"
        )
    if isinstance(settings, str) and settings != "default":
        raise ValueError(
            f"settings must be a Settings instance or 'default', got {settings}"
        )

    if settings == "default":
        logger.info(
            "No settings passed. Using default settings for ECG feature extraction."
        )
        settings = Settings()

    if not isinstance(ecg, np.ndarray) or ecg.ndim != 3:
        raise ValueError(
            "ECG data must be a 3D numpy array with shape (n_samples, n_channels, n_timepoints)"
        )

    logger.info("Starting ECG feature extraction pipeline...")

    # Apply preprocessing if enabled
    if settings.preprocessing.enabled:
        ecg, sfreq = preprocess(ecg, sfreq=sfreq, preprocessing=settings.preprocessing)

    feature_list = []

    if settings.features.fft.enabled:
        feature_list.append(get_fft_features(ecg, sfreq))

    if settings.features.morphological.enabled:
        feature_list.append(
            get_morphological_features(
                ecg, sfreq, n_jobs=settings.features.morphological.n_jobs
            )
        )

    if settings.features.nonlinear.enabled:
        feature_list.append(get_nonlinear_features(ecg, sfreq))

    if settings.features.statistical.enabled:
        feature_list.append(
            get_statistical_features(
                ecg, sfreq, n_jobs=settings.features.statistical.n_jobs
            )
        )

    if settings.features.welch.enabled:
        feature_list.append(get_welch_features(ecg, sfreq))

    if not feature_list:
        raise ValueError(
            "No features were extracted. Please enable at least one feature type."
        )

    features = pd.concat(feature_list, axis=1)
    logger.info(
        "Feature extraction complete. Extracted %s features from %s samples.",
        features.shape[1],
        features.shape[0],
    )

    return features
