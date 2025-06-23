""""""

from typing import Literal

import mne
import mne.baseline
import numpy as np
import pydantic
from pydantic import Field

from .logging import logger


class ResampleArgs(pydantic.BaseModel):
    enabled: bool = False
    sfreq_new: float | None = None


class BandpassArgs(pydantic.BaseModel):
    enabled: bool = False
    l_freq: float | None = 0.5
    h_freq: float | None = None


class NotchArgs(pydantic.BaseModel):
    enabled: bool = False
    freq: float | None = None


class NormalizeArgs(pydantic.BaseModel):
    enabled: bool = False
    mode: Literal["mean", "ratio", "logratio", "percent", "zscore", "zlogratio"] = (
        "zscore"
    )


class PreprocessingSettings(pydantic.BaseModel):
    enabled: bool = True
    resample: ResampleArgs = Field(default_factory=ResampleArgs)
    bandpass: BandpassArgs = Field(default_factory=BandpassArgs)
    notch: NotchArgs = Field(default_factory=NotchArgs)
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)


def preprocess(
    ecg_data: np.ndarray, sfreq: float, preprocessing: PreprocessingSettings
) -> tuple[np.ndarray, float]:
    """
    Präprozessiert EKG-Daten durch Normalisierung und Transposition.

    Diese Funktion normalisiert jede EKG-Sequenz auf den Bereich [0, 1] und
    transponiert die Dimensionen, um die für die Feature-Extraktion erforderliche
    Form zu erhalten.

    Parameters:
    -----------
    ecg_data : numpy.ndarray
        Rohe EKG-Daten mit Form (n_patients, n_times, n_leads)
    sfreq : float
        Abtastrate der EKG-Daten in Hz
    preprocessing : Preprocessing
        Preprocessing Parameters

    Returns:
    --------
    numpy.ndarray
        Normalisierte und transponierte EKG-Daten mit Form (n_patients, n_leads, n_times)
    """
    logger.info("Preprocessing ECG data")
    assert ecg_data.ndim == 3
    n_patients, n_times, n_leads = ecg_data.shape
    assert n_leads < n_times  # Sanity check
    ecg_data = ecg_data.transpose(0, 2, 1)

    sfreq_new = sfreq
    if preprocessing.resample.enabled:
        sfreq_new = preprocessing.resample.sfreq_new
        ecg_data = mne.filter.resample(
            ecg_data,
            up=1.0,
            down=sfreq / sfreq_new,
            axis=-1,
            n_jobs=1,
            verbose=None,
        )
    n_times = ecg_data.shape[-1]
    if preprocessing.bandpass.enabled:
        ecg_data = mne.filter.filter_data(
            ecg_data,
            sfreq_new,
            l_freq=preprocessing.bandpass.l_freq,
            h_freq=preprocessing.bandpass.h_freq,
            n_jobs=1,
            copy=True,
        )
    if preprocessing.notch.enabled:
        ecg_data = mne.filter.notch_filter(
            ecg_data,
            sfreq_new,
            freqs=preprocessing.notch.freq,
            n_jobs=1,
            copy=True,
        )
    ecg_data = ecg_data.reshape(n_patients, -1)
    if preprocessing.normalize.enabled:
        ecg_data = mne.baseline.rescale(
            ecg_data,
            times=np.arange(ecg_data.shape[-1]),
            baseline=(None, None),
            mode=preprocessing.normalize.mode,
            verbose=None,
        )
    ecg_data = ecg_data.reshape(n_patients, n_leads, n_times)
    assert ecg_data.shape == (n_patients, n_leads, n_times)
    return ecg_data, sfreq_new
