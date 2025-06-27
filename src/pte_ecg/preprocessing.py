"""Preprocessing module for ECG signal processing.

This module provides functionality for preprocessing ECG signals, including:
- Resampling
- Bandpass filtering
- Notch filtering
- Normalization

The module is designed to work with multi-channel ECG data and provides a
configurable preprocessing pipeline.
"""

import warnings
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
    sfreq_new: int | float | None = None


class BandpassArgs(pydantic.BaseModel):
    """Settings for bandpass filtering the ECG signal.

    Attributes:
        enabled: Whether to apply bandpass filtering.
        l_freq: Lower cutoff frequency in Hz. If None, no high-pass filtering is applied.
        h_freq: Higher cutoff frequency in Hz. If None, no low-pass filtering is applied.
    """

    enabled: bool = False
    l_freq: int | float | None = 0.5
    h_freq: int | float | None = None


class NotchArgs(pydantic.BaseModel):
    """Settings for notch filtering the ECG signal.

    Attributes:
        enabled: Whether to apply notch filtering.
        freq: Frequency to notch filter in Hz. If None, no notch filtering is applied.
    """

    enabled: bool = False
    freq: int | float | None = None


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
    ecg: np.ndarray, sfreq: float, preprocessing: PreprocessingSettings
) -> tuple[np.ndarray, float]:
    """Apply preprocessing steps to ECG data.

    This function applies the following preprocessing steps in order:
    1. Resampling (if enabled)
    2. Bandpass filtering (if enabled)
    3. Notch filtering (if enabled)
    4. Normalization (if enabled)

    Args:
        ecg: Input ECG data with shape (n_samples, n_channels, n_timepoints).
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

    if sfreq <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {sfreq}")

    if not isinstance(ecg, np.ndarray) or ecg.ndim != 3:
        raise ValueError(
            "ECG data must be a 3D numpy array with shape (n_samples, n_channels, n_timepoints)"
        )

    if ecg.shape[-1] < ecg.shape[-2]:
        warnings.warn(
            "ECG data must be a 3D numpy array with shape (n_samples, n_channels, n_timepoints)."
            f"ECG data may have more channels than timepoints. Got shape: {ecg.shape}. Reshaping data."
        )
        ecg = ecg.transpose(0, 2, 1)

    if not preprocessing.enabled:
        logger.debug("Preprocessing is disabled, returning original data.")
        return ecg, sfreq

    ecg = _check_flats(ecg, drop_flat_recs=True)

    n_samples, n_channels, n_timepoints = ecg.shape
    logger.info(
        f"Starting preprocessing of {n_samples} samples with {n_channels} channels "
        f"and {n_timepoints} timepoints at {sfreq} Hz"
    )
    sfreq_new = sfreq
    if preprocessing.resample.enabled and sfreq_new is not None:
        sfreq_new = preprocessing.resample.sfreq_new
        logger.info(f"Resampling from {sfreq} Hz to {sfreq_new} Hz")
        ecg = mne.filter.resample(
            ecg,
            up=1.0,
            down=sfreq / sfreq_new,
            axis=-1,
            n_jobs=1,
            verbose="WARNING",
        )
        n_timepoints = ecg.shape[-1]

    if preprocessing.bandpass.enabled:
        ecg = mne.filter.filter_data(
            ecg,
            sfreq_new,
            l_freq=preprocessing.bandpass.l_freq,
            h_freq=preprocessing.bandpass.h_freq,
            n_jobs=1,
            copy=True,
            verbose="WARNING",
        )
    if preprocessing.notch.enabled:
        ecg = mne.filter.notch_filter(
            ecg,
            sfreq_new,
            freqs=preprocessing.notch.freq,
            n_jobs=1,
            copy=True,
            verbose="WARNING",
        )
    ecg = ecg.reshape(n_samples, -1)
    if preprocessing.normalize.enabled:
        ecg = mne.baseline.rescale(
            ecg,
            times=np.arange(ecg.shape[-1]),
            baseline=(None, None),
            mode=preprocessing.normalize.mode,
            verbose="WARNING",
        )
    ecg = ecg.reshape(n_samples, n_channels, n_timepoints)
    return ecg, sfreq_new


def _check_flats(ecg: np.ndarray, drop_flats_recs: bool) -> np.ndarray:
    are_flat_chs = np.all(np.isclose(ecg, ecg[..., 0:1]), axis=-1)
    n_flats = np.sum(are_flat_chs)
    if n_flats == ecg.shape[0] * ecg.shape[1]:
        raise ValueError(
            f"All channels of all recordings are flat lines ({n_flats}). Check your data"
        )
    if n_flats > 0:
        logger.warning(
            f"{n_flats} channels of {ecg.shape[0]} recordings are flat lines."
        )
    are_empty_recordings = np.all(are_flat_chs, axis=-1)
    n_empty_recordings = np.sum(are_empty_recordings)
    empty_recordings = np.where(are_empty_recordings)[0]
    if n_empty_recordings > 0 and drop_flats_recs:
        logger.warning(
            f"Discarding {n_empty_recordings} recordings with flat lines in all channels."
            f" Recording indices: {empty_recordings}."
        )
        ecg = ecg[~are_empty_recordings]
    return ecg
