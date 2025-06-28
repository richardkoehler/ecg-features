"""Feature extraction module for ECG signal analysis.

This module provides functions to extract various types of features from ECG signals,
including statistical, morphological, and nonlinear features. It supports parallel
processing for efficient computation on multi-channel ECG data.
"""

import multiprocessing
import os
import sys
import time
import warnings
from typing import Literal

import neurokit2 as nk

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="nolds")
    import nolds
import numpy as np
import pandas as pd
import pydantic
import scipy.fft
import scipy.signal
import scipy.stats
from pydantic import Field

from ._logging import logger

EPS = 1e-10  # Small constant for numerical stability

_METHODS_FINDPEAKS = [
    "neurokit",
    "pantompkins",
    "nabian",
    "slopesumfunction",
    "zong",
    "hamilton",
    "christov",
    "engzeemod",
    "elgendi",
    "kalidas",
    "rodrigues",
    "vg",
    "emrich2023",
    "promac",
    # "gamboa", # Does not currently work reliably
    # "manikandan", # Does not currently work reliably
    # "martinez",# Does not currently work reliably
]


class BaseFeature(pydantic.BaseModel):
    """Base class for feature extraction settings.

    Attributes:
        enabled: Whether this feature type should be extracted.
    """

    enabled: bool = True


class FFTArgs(BaseFeature):
    """Settings for Fast Fourier Transform (FFT) feature extraction.

    Attributes:
        enabled: Whether to compute FFT features.
    """


class WelchArgs(BaseFeature):
    """Settings for Welch's method power spectral density feature extraction.

    Attributes:
        enabled: Whether to compute Welch spectral features.
    """


class StatisticalArgs(BaseFeature):
    """Settings for statistical feature extraction.

    Attributes:
        enabled: Whether to compute statistical features.
        n_jobs: Number of parallel jobs to run. -1 means using all processors.
    """

    n_jobs: int = -1


class MorphologicalArgs(BaseFeature):
    """Settings for morphological feature extraction.

    Attributes:
        enabled: Whether to compute morphological features.
        n_jobs: Number of parallel jobs to run. -1 means using all processors.
    """

    n_jobs: int = -1


class NonlinearArgs(BaseFeature):
    """Settings for nonlinear feature extraction.

    Attributes:
        enabled: Whether to compute nonlinear features.
    """

    enabled: bool = False
    n_jobs: int = -1


class FeatureSettings(pydantic.BaseModel):
    """Container for all feature extraction settings.

    Attributes:
        fft: Settings for FFT feature extraction.
        welch: Settings for Welch's method feature extraction.
        statistical: Settings for statistical feature extraction.
        morphological: Settings for morphological feature extraction.
        nonlinear: Settings for nonlinear feature extraction.
    """

    fft: FFTArgs = Field(default_factory=FFTArgs)
    welch: WelchArgs = Field(default_factory=WelchArgs)
    statistical: StatisticalArgs = Field(default_factory=StatisticalArgs)
    morphological: MorphologicalArgs = Field(default_factory=MorphologicalArgs)
    nonlinear: NonlinearArgs = Field(default_factory=NonlinearArgs)


class ECGDelineationError(Exception):
    """Raised when all ECG delineation methods fail to detect peaks properly."""

    pass


def assert_3_dims(ecg_data: np.ndarray) -> None:
    """Ensure the input array has 3 dimensions.

    Args:
        ecg_data: Input array to check.

    Raises:
        ValueError: If input array doesn't have exactly 3 dimensions.
    """
    if ecg_data.ndim != 3:
        raise ValueError("ECG data must be 3D (n_samples, n_channels, n_timepoints)")


def get_fft_features(ecg_data: np.ndarray, sfreq: float) -> pd.DataFrame:
    """Extract FFT features from ECG data for each sample and channel.

    This function calculates various FFT-based features, including:
    - Sum of frequencies
    - Mean of frequencies
    - Variance of frequencies
    - Dominant frequency
    - Bandwidth (95% cumulative energy)
    - Spectral entropy
    - Spectral flatness
    - Frequency band masks (e.g., HF, LF, VLF)

    Args:
        ecg_data: ECG data with shape (n_samples, n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz

    Returns:
        DataFrame containing the extracted FFT features

    Raises:
        ValueError: If input data has incorrect dimensions
    """
    assert_3_dims(ecg_data)
    start = _log_start("FFT", ecg_data.shape[0])

    n_samples, n_channels, n_timepoints = ecg_data.shape
    xf = np.fft.rfftfreq(n_timepoints, 1 / sfreq)  # (freqs,)
    yf = np.abs(np.fft.rfft(ecg_data, axis=-1))  # (samples, channels, freqs)

    sum_freq = np.sum(yf, axis=-1)
    mean_freq = np.mean(yf, axis=-1)
    var_freq = np.var(yf, axis=-1)

    # Dominant frequency
    dominant_freq_idx = np.argmax(yf, axis=-1)
    dominant_freq = xf[dominant_freq_idx]

    # Normalize for spectral entropy and bandwidth
    yf_norm = yf / (np.sum(yf, axis=-1, keepdims=True) + EPS)

    # Bandwidth (95% cumulative energy)
    cumsum = np.cumsum(yf_norm, axis=-1)
    bandwidth_idx = (cumsum >= 0.95).argmax(axis=-1)
    bandwidth = xf[bandwidth_idx]

    # Spectral entropy
    spectral_entropy = -np.sum(yf_norm * np.log2(yf_norm + EPS), axis=-1)

    # Spectral flatness
    gmean = scipy.stats.gmean(yf + EPS, axis=-1)
    spectral_flatness = gmean / (np.mean(yf + EPS, axis=-1))

    # Frequency band masks
    def band_mask(low, high):
        return (xf >= low) & (xf < high)

    def apply_band(mask):
        return np.sum(yf[..., mask], axis=-1)

    hf_mask = band_mask(15, 40)
    lf_mask = band_mask(0.5, 15)
    b0_10 = band_mask(0, 10)
    b10_20 = band_mask(10, 20)
    b20_30 = band_mask(20, 30)
    b30_40 = band_mask(30, 40)
    below_50 = band_mask(0, 50)
    above_50 = band_mask(50, xf[-1] + 1)

    hf_power = apply_band(hf_mask)
    lf_power = apply_band(lf_mask)
    hf_lf_ratio = hf_power / (lf_power + EPS)

    band_energy_0_10 = apply_band(b0_10)
    band_energy_10_20 = apply_band(b10_20)
    band_energy_20_30 = apply_band(b20_30)
    band_energy_30_40 = apply_band(b30_40)

    total_energy = sum_freq
    band_ratio_0_10 = band_energy_0_10 / (total_energy + EPS)
    band_ratio_10_20 = band_energy_10_20 / (total_energy + EPS)
    band_ratio_20_30 = band_energy_20_30 / (total_energy + EPS)
    band_ratio_30_40 = band_energy_30_40 / (total_energy + EPS)

    power_below_50Hz = apply_band(below_50)
    power_above_50Hz = apply_band(above_50)
    relative_power_below_50Hz = power_below_50Hz / (total_energy + EPS)

    # Stack all features: shape -> (samples, channels, features)
    features = np.stack(
        [
            sum_freq,
            mean_freq,
            var_freq,
            dominant_freq,
            bandwidth,
            spectral_entropy,
            spectral_flatness,
            hf_power,
            lf_power,
            hf_lf_ratio,
            band_energy_0_10,
            band_ratio_0_10,
            band_energy_10_20,
            band_ratio_10_20,
            band_energy_20_30,
            band_ratio_20_30,
            band_energy_30_40,
            band_ratio_30_40,
            power_below_50Hz,
            power_above_50Hz,
            relative_power_below_50Hz,
        ],
        axis=-1,
    )

    # Reshape to (samples, channels × features)
    features_reshaped = features.reshape(n_samples, -1)

    # Create column names
    base_names = [
        "sum_freq",
        "mean_freq",
        "variance_freq",
        "dominant_frequency",
        "bandwidth",
        "spectral_entropy",
        "spectral_flatness",
        "hf_power",
        "lf_power",
        "hf_lf_ratio",
        "band_energy_0_10",
        "band_ratio_0_10",
        "band_energy_10_20",
        "band_ratio_10_20",
        "band_energy_20_30",
        "band_ratio_20_30",
        "band_energy_30_40",
        "band_ratio_30_40",
        "power_below_50Hz",
        "power_above_50Hz",
        "relative_power_below_50Hz",
    ]
    column_names = [
        f"fft_{name}_ch{ch}" for ch in range(n_channels) for name in base_names
    ]

    feature_df = pd.DataFrame(features_reshaped, columns=column_names)
    _log_end("FFT", start, feature_df.shape)
    return feature_df


def get_statistical_features(
    ecg_data: np.ndarray,
    sfreq: float,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Extract statistical features from ECG data for each sample and channel.

    This function calculates various statistical features, including:
    - Sum
    - Mean
    - Median
    - Mode
    - Variance
    - Range
    - Min
    - Max
    - IQR
    - Skewness
    - Kurtosis
    - Peak-to-peak
    - Autocorrelation
    Args:
        ecg_data: ECG data with shape (n_samples, n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz
        n_jobs: Number of parallel jobs to run. -1 means using all processors.

    Returns:
        DataFrame containing the extracted statistical features

    Raises:
        ValueError: If input data has incorrect dimensions
    """
    assert_3_dims(ecg_data)
    start = _log_start("Statistical", ecg_data.shape[0])
    args_list = ((ecg_single, sfreq) for ecg_single in ecg_data)
    processes = _get_n_processes(n_jobs, ecg_data.shape[0])
    if processes in [0, 1]:
        results = [_stat_single_patient(*args) for args in args_list]
    else:
        logger.info(f"Starting parallel processing with {processes} CPUs")
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.starmap(_stat_single_patient, args_list)
    feature_df = pd.DataFrame(results)
    _log_end("Statistical", start, feature_df.shape)
    return feature_df


def _stat_single_patient(sample_data: np.ndarray, sfreq: float) -> dict[str, float]:
    """Extract statistical features from a single sample of ECG data.

    Args:
        sample_data: Single sample of ECG data with shape (n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz

    Returns:
        Dictionary containing the extracted statistical features
    """
    sum_ = np.sum(sample_data, axis=1)
    mean = np.mean(sample_data, axis=1)
    median = np.median(sample_data, axis=1)
    mode = scipy.stats.mode(sample_data, axis=1, keepdims=False).mode
    variance = np.var(sample_data, axis=1)
    range_ = np.ptp(sample_data, axis=1)
    min_ = np.min(sample_data, axis=1)
    max_ = np.max(sample_data, axis=1)
    iqr = np.percentile(sample_data, 75, axis=1) - np.percentile(
        sample_data, 25, axis=1
    )
    skewness = scipy.stats.skew(sample_data, axis=1)
    kurt = scipy.stats.kurtosis(sample_data, axis=1)
    peak_to_peak = max_ - min_
    autocorr = _autocorr_lag1(sample_data)
    feature_arr = np.stack(
        [
            sum_,
            mean,
            median,
            mode,
            variance,
            range_,
            min_,
            max_,
            iqr,
            skewness,
            kurt,
            peak_to_peak,
            autocorr,
        ],
        axis=1,
    )
    base_names = [
        "sum",
        "mean",
        "median",
        "mode",
        "var",
        "range",
        "min",
        "max",
        "iqr",
        "skew",
        "kurt",
        "peak_to_peak",
        "autocorr",
    ]
    column_names = [
        f"stat_{name}_ch{ch}"
        for ch in range(sample_data.shape[0])
        for name in base_names
    ]
    feature_arr = feature_arr.flatten()
    features = {
        name: value for name, value in zip(column_names, feature_arr, strict=True)
    }
    return features


def _autocorr_lag1(sample_data: np.ndarray) -> np.ndarray:
    x = sample_data[:, :-1]
    y = sample_data[:, 1:]
    x_mean = np.mean(x, axis=1, keepdims=True)
    y_mean = np.mean(y, axis=1, keepdims=True)
    numerator = np.sum((x - x_mean) * (y - y_mean), axis=1)
    denominator = np.sqrt(
        np.sum((x - x_mean) ** 2, axis=1) * np.sum((y - y_mean) ** 2, axis=1)
    )
    return numerator / denominator


def get_nonlinear_features(
    ecg_data: np.ndarray, sfreq: float, n_jobs: int = -1
) -> pd.DataFrame:
    """Extract nonlinear features from ECG data for each sample and channel.

    This function calculates 30 different nonlinear metrics per channel that capture
    complex dynamic properties of the ECG signal:
    - Sample Entropy: Measure of signal complexity and unpredictability
    - Hurst Exponent: Measure of long-term memory of the time series
    - Higuchi Fractal Dimension: Measure of the fractal dimension of the signal
    - Recurrence Rate: Measure of signal repetitions
    - DFA Alpha1/Alpha2: Detrended Fluctuation Analysis parameters
    - SD1/SD2: Poincaré plot parameters for heart rate variability
    - SD1/SD2 Ratio: Ratio of Poincaré plot parameters
    - Additional nonlinear features like Approximate Entropy and Permutation Entropy

    Args:
        ecg_data: ECG data with shape (n_samples, n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz

    Returns:
        DataFrame containing the extracted nonlinear features

    Raises:
        ValueError: If input data has incorrect dimensions
    """
    assert_3_dims(ecg_data)
    start = _log_start("Nonlinear", ecg_data.shape[0])
    args_list = (
        (sample_num, ecg_single, sfreq)
        for sample_num, ecg_single in enumerate(ecg_data)
    )
    processes = _get_n_processes(n_jobs, ecg_data.shape[0])
    if processes in [0, 1]:
        results = [_nonlinear_single_patient(*args) for args in args_list]
    else:
        logger.info(f"Starting parallel processing with {processes} CPUs")
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.starmap(_nonlinear_single_patient, args_list)
    feature_df = pd.DataFrame(results)
    _log_end("Nonlinear", start, feature_df.shape)
    return feature_df


def _nonlinear_single_patient(
    sample_num: int, sample_data: np.ndarray, sfreq: float
) -> dict[str, float]:
    logger.info(f"Processing sample number: {sample_num}...")
    features: dict[str, float] = {}
    for ch_num, ch_data in enumerate(sample_data):
        ch_feat = _nonlinear_single_channel(ch_data, sfreq, sample_num, ch_num)
        features.update(
            (f"nonlinear_{key}_ch{ch_num}", value) for key, value in ch_feat.items()
        )
    return features


def _nonlinear_single_channel(
    ch_data: np.ndarray, sfreq: float, sample_num: int, ch_num: int
) -> dict[str, float]:
    """Extract nonlinear features from a single channel of ECG data.

    Args:
        ch_data: Single channel of ECG data with shape (n_timepoints,)
        sfreq: Sampling frequency of the ECG data in Hz

    Returns:
        dict containing the extracted nonlinear features
    """
    features: dict[str, float] = {}
    features["sample_entropy"] = nolds.sampen(ch_data, emb_dim=2)
    features["hurst_exponent"] = nolds.hurst_rs(ch_data)
    # DFA (Detrended Fluctuation Analysis)
    half_len = len(ch_data) // 2
    features["dfa_alpha1"] = (
        nolds.dfa(ch_data, nvals=[4, 8, 16, 32]) if half_len > 32 else np.nan
    )
    features["dfa_alpha2"] = (
        nolds.dfa(ch_data, nvals=[64, 128, 256]) if half_len > 256 else np.nan
    )
    features["change_dfa_alpha"] = (
        nolds.dfa(ch_data[:half_len], nvals=[4, 8, 16])
        - nolds.dfa(ch_data[half_len:], nvals=[4, 8, 16])
        if half_len > 16
        else np.nan
    )

    features["embedding_dimension"] = np.nan
    embedding_dim = 3
    try:
        embedding_dim, _ = nk.complexity_dimension(ch_data, dimension_max=10)
        features["embedding_dimension"] = embedding_dim
    except IndexError as e:
        logger.warning(
            f"Error calculating embedding dimension for channel {ch_num} sample {sample_num}: {e}"
        )

    # Lyapunov
    try:
        lyap_exp = features["largest_lyapunov_exponent"] = nolds.lyap_r(
            ch_data, emb_dim=embedding_dim
        )
        features["dynamic_stability"] = np.exp(-np.abs(lyap_exp))
    except ValueError as e:
        logger.warning(
            f"Error calculating lyapunov exponent for channel {ch_num} sample {sample_num}: {e}"
        )
    features["correlation_dimension"] = nolds.corr_dim(ch_data, emb_dim=embedding_dim)
    # Too slow
    # # Fractal Dimension Higuchi
    # features["fractal_higuchi"] = nk.fractal_higuchi(ch_data, k_max="default")[
    #     0
    # ]
    # Fractal Dimension Katz
    features["fractal_katz"] = nk.fractal_katz(ch_data)[0]

    # Recurrence measures

    rqa, _ = nk.complexity_rqa(ch_data, dimension=embedding_dim)
    rec_rate = rqa["RecurrenceRate"].iat[0]
    features["recurrence_rate"] = rec_rate
    features["recurrence_variance"] = rec_rate * (1 - rec_rate)
    features["recurrence_network_measures"] = (
        rqa["Determinism"].iat[0] + rqa["Laminarity"].iat[0]
    ) / 2
    features["rqa_l_entropy"] = rqa["LEn"].iat[0]

    diffs = np.diff(ch_data)
    features["time_irreversibility"] = np.mean(diffs**3) / (np.mean(diffs**2) ** 1.5)
    features["nonlinear_variance"] = np.var(diffs**2)

    window_duration_sec = min(1, len(ch_data) / sfreq)
    window_size = int(window_duration_sec * sfreq)
    step_size = window_size // 2
    windows = np.lib.stride_tricks.sliding_window_view(ch_data, window_size)[
        ::step_size
    ]
    local_vars = np.var(windows, axis=1)
    features["dynamic_variance"] = np.var(local_vars)

    features["multiscale_entropy"], _ = nk.entropy_sample(
        ch_data, dimension=embedding_dim, scale=2
    )

    above_median = ch_data > np.median(ch_data)
    features["symbolic_dynamics"] = np.sum(np.abs(np.diff(above_median))) / (
        len(above_median) - 1
    )
    entropy1, _ = nk.entropy_sample(ch_data[:half_len], dimension=embedding_dim)
    entropy2, _ = nk.entropy_sample(ch_data[half_len:], dimension=embedding_dim)
    features["sample_entropy_change_rate"] = (
        (entropy2 - entropy1) / (entropy1 + EPS) if entropy1 != 0 else np.nan
    )
    # Shannon-Entropy
    N = len(ch_data)
    K = min(20, N // 2)  # Einbettungsdimension begrenzen
    hankel = np.zeros((N - K + 1, K))
    for i in range(K):
        hankel[:, i] = ch_data[i : i + N - K + 1]
    s = np.linalg.svd(hankel, compute_uv=False)
    s_norm = s / np.sum(s) if np.sum(s) > 0 else np.ones_like(s) / len(s)
    features["singular_spectrum_entropy"] = -np.sum(s_norm * np.log2(s_norm + EPS))

    b, a = scipy.signal.butter(2, 0.2)  # Lowpass
    filtered = scipy.signal.filtfilt(b, a, ch_data)
    entropy_orig, _ = nk.entropy_sample(ch_data, dimension=embedding_dim)
    entropy_filt, _ = nk.entropy_sample(filtered, dimension=embedding_dim)
    features["complexity_loss"] = (
        (entropy_orig - entropy_filt) / entropy_orig if entropy_orig != 0 else np.nan
    )
    return features


def get_morphological_features(
    ecg_data: np.ndarray, sfreq: float, n_jobs: int | None = -1
) -> pd.DataFrame:
    """Extract morphological features from ECG data for each sample and channel.

    This function calculates various morphological features.

    Args:
        ecg_data: ECG data with shape (n_samples, n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz
        n_jobs: Number of parallel jobs to run. -1 means using all processors.

    Returns:
        DataFrame containing the extracted morphological features

    Raises:
        ValueError: If input data has incorrect dimensions
    """
    assert_3_dims(ecg_data)
    start = _log_start("Morphological", ecg_data.shape[0])
    args_list = (
        (sample_num, ecg_single, sfreq)
        for sample_num, ecg_single in enumerate(ecg_data)
    )
    processes = _get_n_processes(n_jobs, ecg_data.shape[0])
    if processes in [0, 1]:
        results = [_morph_single_patient(*args) for args in args_list]
    else:
        logger.info(f"Starting parallel processing with {processes} CPUs")
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.starmap(_morph_single_patient, args_list)
    feature_df = pd.DataFrame(results)
    _log_end("Morphological", start, feature_df.shape)
    return feature_df


def _morph_single_patient(
    sample_num: int, sample_data: np.ndarray, sfreq: float
) -> dict[str, float]:
    """Extract morphological features from a single sample of ECG data.

    Args:
        sample_data: Single sample of ECG data with shape (n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz

    Returns:
        dict containing the extracted morphological features
    """
    logger.info(f"Processing sample number: {sample_num}...")
    features: dict[str, float] = {}
    flat_chs = np.all(np.isclose(sample_data, sample_data[:, 0:1]), axis=1)
    if np.all(flat_chs):
        logger.warning("All channels are flat lines. Skipping morphological features.")
        return features
    for ch_num, (ch_data, is_flat) in enumerate(zip(sample_data, flat_chs)):
        if is_flat:
            logger.warning(
                f"Channel {ch_num} is flat line. Skipping morphological features."
            )
            continue
        ch_feat = _morph_single_channel(ch_data, sfreq)
        features.update(
            (f"morph_{key}_ch{ch_num}", value) for key, value in ch_feat.items()
        )
    return features


def _get_r_peaks(
    ch_data: np.ndarray, sfreq: float
) -> tuple[np.ndarray | None, int, Literal[*_METHODS_FINDPEAKS]]:
    peaks_per_method: dict[Literal[*_METHODS_FINDPEAKS], np.ndarray] = {}
    max_n_peaks = 0
    for method in _METHODS_FINDPEAKS:
        _, peaks_info = nk.ecg_peaks(
            ch_data,
            sampling_rate=np.rint(sfreq).astype(int)
            if method in ["zong", "emrich2023"]
            else sfreq,
            method=method,
        )
        r_peaks: np.ndarray | None = peaks_info["ECG_R_Peaks"]
        n_r_peaks = len(r_peaks) if r_peaks is not None else 0
        if not n_r_peaks:
            logger.debug(f"No R-peaks detected for method '{method}'.")
            continue
        max_n_peaks = max(max_n_peaks, n_r_peaks)
        peaks_per_method[method] = r_peaks
        if n_r_peaks > 1:  # We need at least 2 R-peaks for some features
            return r_peaks, n_r_peaks, method
    if not max_n_peaks:
        return None, max_n_peaks, None
    for method, r_peaks in peaks_per_method.items():
        return r_peaks, max_n_peaks, method  # return first item


def _morph_single_channel(ch_data: np.ndarray, sfreq: float) -> dict[str, float]:
    """Extract morphological features from a single channel of ECG data.

    Args:
        ch_data: Single channel of ECG data with shape (n_timepoints,)
        sfreq: Sampling frequency of the ECG data in Hz

    Returns:
        dict containing the extracted morphological features
    """
    features: dict[str, float] = {}
    r_peaks, n_r_peaks, r_peak_method = _get_r_peaks(ch_data, sfreq)
    if not n_r_peaks:
        logger.warning("No R-peaks detected. Skipping morphological features.")
        return {}
    waves_dict: dict = {}
    for method in ["dwt", "prominence", "peak", "cwt"]:
        if n_r_peaks < 2 and method in {"prominence", "cwt"}:
            logger.info("Not enough R-peaks for prominence or cwt method.")
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", nk.misc.NeuroKitWarning)
                _, waves_dict = nk.ecg_delineate(
                    ch_data,
                    rpeaks=r_peaks,
                    sampling_rate=sfreq,
                    method=method,
                )
            break
        except nk.misc.NeuroKitWarning as e:
            if "Too few peaks detected" in str(e):
                logger.warning(f"Peak detection failed with method '{method}': {e}")
            else:
                raise
    if not waves_dict:
        raise ECGDelineationError("ECG delineation failed with all available methods.")

    # Extrahiere Intervalle
    p_peaks = waves_dict["ECG_P_Peaks"]
    q_peaks = waves_dict["ECG_Q_Peaks"]
    s_peaks = waves_dict["ECG_S_Peaks"]
    t_peaks = waves_dict["ECG_T_Peaks"]

    p_onsets = waves_dict["ECG_P_Onsets"]
    p_offsets = waves_dict["ECG_P_Offsets"]
    t_onsets = waves_dict["ECG_T_Onsets"]
    t_offsets = waves_dict["ECG_T_Offsets"]

    n_p_peaks = len(p_peaks) if p_peaks is not None else 0
    n_q_peaks = len(q_peaks) if q_peaks is not None else 0
    n_s_peaks = len(s_peaks) if s_peaks is not None else 0
    n_t_peaks = len(t_peaks) if t_peaks is not None else 0
    n_p_onsets = len(p_onsets) if p_onsets is not None else 0
    n_p_offsets = len(p_offsets) if p_offsets is not None else 0
    n_t_onsets = len(t_onsets) if t_onsets is not None else 0
    n_t_offsets = len(t_offsets) if t_offsets is not None else 0

    # QRS-Dauer
    if n_q_peaks and n_s_peaks:
        # Berechne durchschnittliche QRS-Dauer
        qrs_durations: list[float] = []
        max_index = min(n_q_peaks, n_s_peaks)
        for q, s in zip(q_peaks[:max_index], s_peaks[:max_index]):
            if q >= s or np.isnan(q) or np.isnan(s):
                continue
            qrs_durations.append((s - q) / sfreq * 1000)  # in ms
        if qrs_durations:
            features["qrs_duration"] = np.mean(qrs_durations)
            features["qrs_dispersion"] = np.std(qrs_durations)

    # QT-Intervall
    if n_q_peaks and n_t_peaks:
        qt_intervals = []
        max_index = min(n_q_peaks, n_t_peaks)
        for q, t in zip(q_peaks[:max_index], t_peaks[:max_index]):
            if q >= t or np.isnan(q) or np.isnan(t):
                continue
            qt_intervals.append((t - q) / sfreq * 1000)  # in ms
        if qt_intervals:
            features["qt_interval"] = np.mean(qt_intervals)
            features["qt_dispersion"] = np.std(qt_intervals)

    # PQ-Intervall
    if n_p_peaks and n_q_peaks:
        pq_intervals = []
        max_index = min(n_p_peaks, n_q_peaks)
        for p, q in zip(p_peaks[:max_index], q_peaks[:max_index]):
            if p >= q or np.isnan(p) or np.isnan(q):
                continue
            pq_intervals.append((q - p) / sfreq * 1000)  # in ms
        if pq_intervals:
            features["pq_interval"] = np.mean(pq_intervals)
            features["pq_dispersion"] = np.std(pq_intervals)

    # P-Dauer
    if n_p_onsets and n_p_offsets:
        p_durations = []
        max_index = min(n_p_onsets, n_p_offsets)
        for p_on, p_off in zip(p_onsets[:max_index], p_offsets[:max_index]):
            if p_on >= p_off or np.isnan(p_on) or np.isnan(p_off):
                continue
            p_durations.append((p_off - p_on) / sfreq * 1000)
        if p_durations:
            features["p_duration"] = np.mean(p_durations)
            features["p_dispersion"] = np.std(p_durations)

    # T-Dauer
    if n_t_onsets and n_t_offsets:
        t_durations = []
        max_index = min(n_t_onsets, n_t_offsets)
        for t_on, t_off in zip(t_onsets[:max_index], t_offsets[:max_index]):
            if t_on >= t_off or np.isnan(t_on) or np.isnan(t_off):
                continue
            t_durations.append((t_off - t_on) / sfreq * 1000)
        if t_durations:
            features["t_duration"] = np.mean(t_durations)
            features["t_dispersion"] = np.std(t_durations)

    # ST-Dauer
    if n_s_peaks and n_t_onsets:
        st_durations = []
        max_index = min(n_s_peaks, n_t_onsets)
        for s, t_on in zip(s_peaks[:max_index], t_onsets[:max_index]):
            if s >= t_on or np.isnan(s) or np.isnan(t_on):
                continue
            st_durations.append(t_on - s)
        if st_durations:
            features["st_duration"] = np.mean(st_durations) / sfreq * 1000
            features["st_dispersion"] = np.std(st_durations) / sfreq * 1000

    # RT-Dauer
    if n_r_peaks and n_t_onsets:
        rt_durations = []
        max_index = min(n_r_peaks, n_t_onsets)
        for r, t_on in zip(r_peaks[:max_index], t_onsets[:max_index]):
            if r >= t_on or np.isnan(r) or np.isnan(t_on):
                continue
            rt_durations.append((t_on - r) / sfreq * 1000)
        if rt_durations:
            features["rt_duration"] = np.mean(rt_durations)
            features["rt_dispersion"] = np.std(rt_durations)

    # Flächen (Integrale unter den Kurven)
    if n_p_onsets and n_p_offsets:
        p_areas = []
        max_index = min(n_p_onsets, n_p_offsets)
        for p_on, p_off in zip(p_onsets[:max_index], p_offsets[:max_index]):
            if p_on >= p_off or np.isnan(p_on) or np.isnan(p_off):
                continue
            p_areas.append(np.sum(np.abs(ch_data[p_on:p_off])))
        if p_areas:
            features["p_area"] = np.mean(p_areas)

    # T Area
    if n_t_onsets and n_t_offsets:
        t_areas = []
        max_index = min(n_t_onsets, n_t_offsets)
        for t_on, t_off in zip(t_onsets[:max_index], t_offsets[:max_index]):
            if t_on >= t_off or np.isnan(t_on) or np.isnan(t_off):
                continue
            t_areas.append(np.sum(np.abs(ch_data[t_on:t_off])))
        if t_areas:
            features["t_area"] = np.mean(t_areas)

    # R Slope
    if n_r_peaks and n_q_peaks:
        r_slopes = []
        max_index = min(n_r_peaks, n_q_peaks)
        for r, q in zip(r_peaks[:max_index], q_peaks[:max_index]):
            if r < q or np.isnan(r) or np.isnan(q):
                continue
            delta_y = ch_data[r] - ch_data[q]
            delta_x = (r - q) / sfreq
            if delta_x > 0:
                r_slopes.append(delta_y / delta_x)
        if r_slopes:
            features["r_slope"] = np.mean(r_slopes)

    # T Slope
    if n_t_onsets and n_t_offsets:
        t_slopes = []
        max_index = min(n_t_onsets, n_t_offsets)
        for t_on, t_off in zip(t_onsets[:max_index], t_offsets[:max_index]):
            if t_on >= t_off or np.isnan(t_on) or np.isnan(t_off):
                continue
            delta_y = ch_data[t_on] - ch_data[t_off]
            delta_x = (t_on - t_off) / sfreq
            if delta_x > 0:
                t_slopes.append(delta_y / delta_x)
        if t_slopes:
            features["t_slope"] = np.mean(t_slopes)

    # Amplituden
    if n_p_peaks:
        p_amplitudes = [ch_data[p] for p in p_peaks if not np.isnan(p)]
        if p_amplitudes:
            features["p_amplitude"] = np.mean(p_amplitudes)

    if n_q_peaks:
        q_amplitudes = [ch_data[q] for q in q_peaks if not np.isnan(q)]
        if q_amplitudes:
            features["q_amplitude"] = np.mean(q_amplitudes)

    if n_r_peaks:
        r_amplitudes = [ch_data[r] for r in r_peaks if not np.isnan(r)]
        if r_amplitudes:
            features["r_amplitude"] = np.mean(r_amplitudes)

    if n_r_peaks > 1:
        rr_intervals = np.diff(r_peaks) / sfreq
        rr_intervals = rr_intervals[~np.isnan(rr_intervals)]
        features["rr_interval_mean"] = np.mean(rr_intervals)
        features["rr_interval_std"] = np.std(rr_intervals)
        if len(rr_intervals) > 1:
            features["rr_interval_median"] = np.median(rr_intervals)
            features["rr_interval_iqr"] = np.percentile(
                rr_intervals, 75
            ) - np.percentile(rr_intervals, 25)
            features["rr_interval_skewness"] = scipy.stats.skew(rr_intervals)
            features["rr_interval_kurtosis"] = scipy.stats.kurtosis(rr_intervals)
            # SD1: short-term variability
            diff_rr = np.diff(rr_intervals)
            sd1 = np.nanstd(diff_rr / np.sqrt(2))
            # SD2: long-term variability
            sdrr = np.nanstd(rr_intervals)  # overall HRV
            interm = 2 * sdrr**2 - sd1**2
            sd2 = np.sqrt(interm) if interm > 0 else np.nan
            features["sd1"] = sd1
            features["sd2"] = sd2
            features["sd1_sd2_ratio"] = (
                sd1 / (sd2 + EPS) if not np.isnan(sd2) else np.nan
            )

    if n_s_peaks:
        s_amplitudes = [ch_data[s] for s in s_peaks if not np.isnan(s)]
        if s_amplitudes:
            features["s_amplitude"] = np.mean(s_amplitudes)

    if n_t_peaks:
        t_amplitudes = [ch_data[t] for t in t_peaks if not np.isnan(t)]
        if t_amplitudes:
            features["t_amplitude"] = np.mean(t_amplitudes)

    return features


def _get_n_processes(n_jobs: int | None, n_tasks: int) -> int:
    """Get the number of processes to use for parallel processing."""
    if n_jobs not in [-1, None]:
        return n_jobs

    if sys.version_info >= (3, 13):
        n_processes = os.process_cpu_count()
    else:
        n_processes = os.process_count()
    return min(n_processes, n_tasks)


def _log_end(feature_name: str, start_time: float, shape: tuple[int, int]) -> None:
    """Log the end of feature extraction.

    Args:
        feature_name: Name of the feature type being extracted.
        start_time: Start time of the feature extraction.
        shape: Shape of the extracted features.
    """
    logger.info(
        "Completed %s feature extraction. Shape: %s. Time taken: %.1f s",
        feature_name,
        shape,
        time.time() - start_time,
    )


def _log_start(feature_name: str, n_samples: int) -> float:
    """Log the start of feature extraction and return the current time.

    Args:
        feature_name: Name of the feature type being extracted.
        n_samples: Number of samples.

    Returns:
        Current time.
    """
    logger.info(
        "Starting %s feature extraction for %s samples...", feature_name, n_samples
    )
    return time.time()


def get_welch_features(ecg_data: np.ndarray, sfreq: float) -> pd.DataFrame:
    """Extract Welch's method power spectral density features from ECG data for each sample and channel.

    This function calculates various Welch's method features, including:
    - Log power ratio
    - Band 0-0.5 Hz
    - Band 0.5-4 Hz
    - Band 4-15 Hz
    - Band 15-40 Hz
    - Band over 40 Hz
    - Spectral entropy
    - Total power
    - Peak frequency

    Args:
        ecg_data: ECG data with shape (n_samples, n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz

    Returns:
        DataFrame containing the extracted Welch's method features

    Raises:
        ValueError: If input data has incorrect dimensions
    """
    assert_3_dims(ecg_data)
    start = _log_start("Welch", ecg_data.shape[0])
    n_samples, n_channels, n_timepoints = ecg_data.shape
    flat_data = ecg_data.reshape(
        -1, n_timepoints
    )  # Shape: (n_samples * n_channels, n_timepoints)

    # Compute Welch spectra for each channel
    psd_list = []
    freq_list = []
    for channel_data in flat_data:
        freqs, Pxx = scipy.signal.welch(
            channel_data,
            fs=sfreq,
            nperseg=sfreq,
            scaling="density",
        )
        psd_list.append(Pxx)
        freq_list.append(freqs)

    freqs = np.array(freq_list[0])
    psd_array = np.array(psd_list)  # Shape: (n_samples * n_channels, len(freqs))

    # Split into n_bins equal-sized dynamic bins (dependent on sampling frequency)
    n_bins = 10
    bins = np.zeros((psd_array.shape[0], n_bins))
    bin_freqs = np.zeros((n_bins, 2))
    for i, bin_idx in enumerate(np.array_split(np.arange(psd_array.shape[1]), n_bins)):
        bins[:, i] = np.mean(psd_array[:, bin_idx], axis=1)
        bin_freqs[i, 0] = freqs[bin_idx[0]]
        bin_freqs[i, 1] = freqs[bin_idx[-1]]

    # Create masks for hard-coded frequency bands
    mask_low = freqs <= 15
    mask_high = freqs > 15
    mask_0_0_5 = (freqs >= 0) & (freqs <= 0.5)
    mask_0_5_4 = (freqs > 0.5) & (freqs <= 4)
    mask_4_15 = (freqs > 4) & (freqs <= 15)
    mask_15_40 = (freqs > 15) & (freqs <= 40)
    mask_over_40 = freqs > 40

    low_power = np.sum(psd_array[:, mask_low], axis=1)
    high_power = np.sum(psd_array[:, mask_high], axis=1)
    log_power_ratio = np.log(high_power / (low_power + 1e-10) + 1e-10)

    band_0_0_5 = np.sum(psd_array[:, mask_0_0_5], axis=1)
    band_0_5_4 = np.sum(psd_array[:, mask_0_5_4], axis=1)
    band_4_15 = np.sum(psd_array[:, mask_4_15], axis=1)
    band_15_40 = np.sum(psd_array[:, mask_15_40], axis=1)
    band_over_40 = np.sum(psd_array[:, mask_over_40], axis=1)

    # Calculate other features
    total_power = np.sum(psd_array, axis=1)
    psd_norm = psd_array / (total_power[:, None] + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=1)
    peak_indices = np.argmax(psd_array, axis=1)
    peak_frequency = freqs[peak_indices]

    all_features = np.column_stack(
        [
            bins,
            log_power_ratio,
            band_0_0_5,
            band_0_5_4,
            band_4_15,
            band_15_40,
            band_over_40,
            spectral_entropy,
            total_power,
            peak_frequency,
        ]
    )
    all_features = all_features.reshape(n_samples, n_channels * all_features.shape[1])
    base_names = [
        *(f"bin_{low}_{high}" for low, high in bin_freqs),
        "log_power_ratio",
        "band_0_0_5",
        "band_0_5_4",
        "band_4_15",
        "band_15_40",
        "band_over_40",
        "spectral_entropy",
        "total_power",
        "peak_frequency",
    ]
    column_names = [
        f"welch_{name}_ch{ch}" for ch in range(ecg_data.shape[1]) for name in base_names
    ]
    feature_df = pd.DataFrame(all_features, columns=column_names)
    _log_end("Welch", start, feature_df.shape)
    return feature_df
