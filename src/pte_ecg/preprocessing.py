"""Preprocessing module for ECG signal processing.

This module provides functionality for preprocessing ECG signals, including:
- Resampling
- Bandpass filtering
- Notch filtering
- Normalization

The module is designed to work with multi-channel ECG data and provides a
configurable preprocessing pipeline.
"""

from typing import Literal

import mne
import mne.baseline
import numpy as np
import pydantic
from pydantic import Field

from ._logging import logger


class ResampleArgs(pydantic.BaseModel):
    """Settings for resampling the ECG signal.

    Attributes:
        enabled: Whether to apply resampling.
        sfreq_new: New sampling frequency in Hz. If None, no resampling is performed.
    """

    enabled: bool = False
    sfreq_new: float | None = None


class BandpassArgs(pydantic.BaseModel):
    """Settings for bandpass filtering the ECG signal.

    Attributes:
        enabled: Whether to apply bandpass filtering.
        l_freq: Lower cutoff frequency in Hz. If None, no high-pass filtering is applied.
        h_freq: Higher cutoff frequency in Hz. If None, no low-pass filtering is applied.
    """

    enabled: bool = False
    l_freq: float | None = 0.5
    h_freq: float | None = None


class NotchArgs(pydantic.BaseModel):
    """Settings for notch filtering the ECG signal.

    Attributes:
        enabled: Whether to apply notch filtering.
        freq: Frequency to notch filter in Hz. If None, no notch filtering is applied.
    """

    enabled: bool = False
    freq: float | None = None


class NormalizeArgs(pydantic.BaseModel):
    """Settings for normalizing the ECG signal.

    Attributes:
        enabled: Whether to apply normalization.
        mode: Normalization method to use. One of:
            - 'mean': Subtract mean of each channel
            - 'ratio': Divide by mean of each channel
            - 'logratio': Log of ratio
            - 'percent': Scale to percentage of mean
            - 'zscore': Standard score (z-score) normalization
            - 'zlogratio': Z-score of log ratio
    """

    enabled: bool = False
    mode: Literal["mean", "ratio", "logratio", "percent", "zscore", "zlogratio"] = (
        "zscore"
    )


class PreprocessingSettings(pydantic.BaseModel):
    """Container for all preprocessing settings.

    Attributes:
        enabled: Whether to apply any preprocessing.
        resample: Settings for resampling.
        bandpass: Settings for bandpass filtering.
        notch: Settings for notch filtering.
        normalize: Settings for normalization.
    """

    enabled: bool = True
    resample: ResampleArgs = Field(default_factory=ResampleArgs)
    bandpass: BandpassArgs = Field(default_factory=BandpassArgs)
    notch: NotchArgs = Field(default_factory=NotchArgs)
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)


def preprocess(
    ecg_data: np.ndarray, sfreq: float, preprocessing: PreprocessingSettings
) -> tuple[np.ndarray, float]:
    """Apply preprocessing steps to ECG data.

    This function applies the following preprocessing steps in order:
    1. Resampling (if enabled)
    2. Bandpass filtering (if enabled)
    3. Notch filtering (if enabled)
    4. Normalization (if enabled)

    Args:
        ecg_data: Input ECG data with shape (n_samples, n_channels, n_timepoints).
        sfreq: Sampling frequency of the input data in Hz.
        preprocessing: Preprocessing settings.

    Returns:
            Tuple containing:
            - Processed ECG data with the same shape as input.
            - Updated sampling frequency (may change if resampling is applied).

    Raises:
        ValueError: If input data has invalid dimensions or preprocessing settings are invalid.
        RuntimeError: If preprocessing fails.
    """
    if not preprocessing.enabled:
        logger.debug("Preprocessing is disabled, returning original data.")
        return ecg_data, sfreq

    if not isinstance(ecg_data, np.ndarray) or ecg_data.ndim != 3:
        raise ValueError(
            "ECG data must be a 3D numpy array with shape (n_samples, n_channels, n_timepoints)"
        )
    import warnings

    if ecg_data.shape[-1] < ecg_data.shape[-2]:
        warnings.warn(
            "ECG data must be a 3D numpy array with shape (n_samples, n_channels, n_timepoints)."
            f"ECG data may have more channels than timepoints. Got shape: {ecg_data.shape}. Reshaping data."
        )
        ecg_data = ecg_data.transpose(0, 2, 1)

    if sfreq <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {sfreq}")

    n_samples, n_channels, n_timepoints = ecg_data.shape
    logger.info(
        f"Starting preprocessing of {n_samples} samples with {n_channels} channels "
        f"and {n_timepoints} timepoints at {sfreq} Hz"
    )
    sfreq_new = sfreq
    if preprocessing.resample.enabled and sfreq_new is not None:
        sfreq_new = preprocessing.resample.sfreq_new
        logger.info(f"Resampling from {sfreq} Hz to {sfreq_new} Hz")
        ecg_data = mne.filter.resample(
            ecg_data,
            up=1.0,
            down=sfreq / sfreq_new,
            axis=-1,
            n_jobs=1,
            verbose="WARNING",
        )
        n_timepoints = ecg_data.shape[-1]

    if preprocessing.bandpass.enabled:
        ecg_data = mne.filter.filter_data(
            ecg_data,
            sfreq_new,
            l_freq=preprocessing.bandpass.l_freq,
            h_freq=preprocessing.bandpass.h_freq,
            n_jobs=1,
            copy=True,
            verbose="WARNING",
        )
    if preprocessing.notch.enabled:
        ecg_data = mne.filter.notch_filter(
            ecg_data,
            sfreq_new,
            freqs=preprocessing.notch.freq,
            n_jobs=1,
            copy=True,
            verbose="WARNING",
        )
    ecg_data = ecg_data.reshape(n_samples, -1)
    if preprocessing.normalize.enabled:
        ecg_data = mne.baseline.rescale(
            ecg_data,
            times=np.arange(ecg_data.shape[-1]),
            baseline=(None, None),
            mode=preprocessing.normalize.mode,
            verbose="WARNING",
        )
    ecg_data = ecg_data.reshape(n_samples, n_channels, n_timepoints)
    return ecg_data, sfreq_new
