"""Feature extraction module for ECG signal analysis.

This module provides functions to extract various types of features from ECG signals,
including statistical, morphological, and nonlinear features. It supports parallel
processing for efficient computation on multi-channel ECG data.
"""

import multiprocessing
import os
import time
import warnings

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


class BaseFeature(pydantic.BaseModel):
    """Base class for feature extraction settings.

    Attributes:
        enabled: Whether this feature type should be extracted.
    """

    enabled: bool = False


class FFTArgs(BaseFeature):
    """Settings for Fast Fourier Transform (FFT) feature extraction.

    Attributes:
        enabled: Whether to compute FFT features.
    """

    enabled: bool = True


class WelchArgs(BaseFeature):
    """Settings for Welch's method power spectral density feature extraction.

    Attributes:
        enabled: Whether to compute Welch spectral features.
    """

    enabled: bool = True


class StatisticalArgs(BaseFeature):
    """Settings for statistical feature extraction.

    Attributes:
        enabled: Whether to compute statistical features.
        n_jobs: Number of parallel jobs to run. -1 means using all processors.
    """

    enabled: bool = True
    n_jobs: int = -1


class MorphologicalArgs(BaseFeature):
    """Settings for morphological feature extraction.

    Attributes:
        enabled: Whether to compute morphological features.
        n_jobs: Number of parallel jobs to run. -1 means using all processors.
    """

    enabled: bool = True
    n_jobs: int = -1


class NonlinearArgs(BaseFeature):
    """Settings for nonlinear feature extraction.

    Attributes:
        enabled: Whether to compute nonlinear features.
    """

    enabled: bool = True


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
    - Time to first peak
    - R-peak amplitude
    - Duration between peaks
    - Amplitude between peaks
    - RR interval mean
    - RR interval std
    - Heart rate
    - HRV
    - RR interval difference mean
    - RR interval median
    - RR interval IQR
    - RR interval skewness
    - RR interval kurtosis

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
        "time_to_first_peak",
        "r_peak_amp",
        "duration_between_peaks",
        "amplitude_between_peaks",
        "rr_mean",
        "rr_std",
        "heart_rate",
        "hrv",
        "rr_diff_mean",
        "rr_median",
        "rr_iqr",
        "rr_skew",
        "rr_kurt",
    ]
    column_names = [
        f"stat_{name}_ch{ch}" for ch in range(ecg_data.shape[1]) for name in base_names
    ]
    n_features = len(base_names)
    args_list = ((ecg_single, sfreq, n_features) for ecg_single in ecg_data)
    processes = _get_n_processes(n_jobs, ecg_data.shape[0])
    if processes in [0, 1]:
        results = [_stat_single_patient(*args) for args in args_list]
    else:
        logger.info(f"Starting parallel processing with {processes} CPUs")
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.starmap(_stat_single_patient, args_list)
    feature_array = np.vstack(results)
    feature_df = pd.DataFrame(feature_array, columns=column_names)
    _log_end("Statistical", start, feature_df.shape)
    return feature_df


def get_nonlinear_features(ecg_data: np.ndarray, sfreq: float) -> pd.DataFrame:
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
    base_names = [
        "sample_entropy",
        "hurst_exponent",
        "higuchi_fd",
        "recurrence_rate",
        "dfa_alpha1",
        "dfa_alpha2",
        "sd1",
        "sd2",
        "sd1_sd2_ratio",
        "approximate_entropy",
        "permutation_entropy",
        "lempel_ziv_complexity",
        "largest_lyapunov_exponent",
        "correlation_dimension",
        "fractal_dimension_katz",
        "time_irreversibility",
        "multiscale_entropy",
        "rqa_entropy",
        "recurrence_variance",
        "dynamic_stability",
        "symbolic_dynamics",
        "sample_entropy_change_rate",
        "change_dfa_alpha",
        "variance",
        "singular_spectrum_entropy",
        "dynamic_variance",
        "recurrence_network_measures",
        "delay_embedding_dimension",
        "delay_time",
        "complexity_loss",
    ]
    n_samples, n_chans, _ = ecg_data.shape
    n_features = len(base_names)
    column_names = [
        f"nonlinear_{name}_ch{ch}" for ch in range(n_chans) for name in base_names
    ]
    features = np.zeros((n_samples, n_chans * n_features))
    for sample, sample_data in enumerate(ecg_data):
        for ch, ch_data in enumerate(sample_data):
            sample_entropy = hurst_exponent = higuchi_fd = recurrence_rate = 0
            dfa_alpha1 = dfa_alpha2 = sd1 = sd2 = sd1_sd2_ratio = 0.0
            approximate_entropy = permutation_entropy = lempel_ziv_complexity = 0
            largest_lyapunov_exponent = correlation_dimension = (
                fractal_dimension_katz
            ) = 0
            time_irreversibility = multiscale_entropy = rqa_entropy = (
                recurrence_variance
            ) = 0
            dynamic_stability = symbolic_dynamics = sample_entropy_change_rate = 0
            change_dfa_alpha = nonlinear_variance = singular_spectrum_entropy = 0
            dynamic_variance = recurrence_network_measures = (
                delay_embedding_dimension
            ) = 0
            delay_time = complexity_loss = 0

            # Sample Entropy (Komplexitätsmaß)
            sample_entropy = nolds.sampen(ch_data, emb_dim=2)
            # Hurst Exponent (Langzeitabhängigkeit)
            hurst_exponent = nolds.hurst_rs(ch_data)
            # Higuchi Fractal Dimension
            higuchi_fd = nk.fractal_higuchi(ch_data, k_max="default")[0]
            # DFA (Detrended Fluctuation Analysis)
            dfa_alpha1 = nolds.dfa(ch_data, nvals=[4, 8, 16, 32])
            dfa_alpha2 = nolds.dfa(ch_data, nvals=[64, 128, 256])
            # Poincaré Plot Features
            ecg_cleaned = nk.ecg_clean(ch_data, sampling_rate=sfreq)
            _, peaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sfreq)
            rpeaks = peaks_info["ECG_R_Peaks"]
            if len(rpeaks) > 1:
                rr_intervals = np.diff(rpeaks) / sfreq
                sd1 = np.std(np.diff(rr_intervals) / np.sqrt(2))
                sd2 = np.std(rr_intervals)
                sd1_sd2_ratio = sd1 / (sd2 + 1e-10)
            # Largest Lyapunov Exponent (Lagezeitabhängigkeit)
            largest_lyapunov_exponent = nolds.lyap_r(
                ch_data, emb_dim=3, lag=1, min_tsep=10
            )
            # Correlation Dimension
            correlation_dimension = nolds.corr_dim(ch_data, emb_dim=2)

            #     # Approximate Entropy
            #     approximate_entropy = pyeeg.ap_entropy(
            #         normalized_channel, 2, 0.2 * np.std(normalized_channel)
            #     )
            #     # Permutation Entropy
            #     permutation_entropy = pyeeg.permutation_entropy(
            #         normalized_channel, 3, 1
            #     )
            #     # Lempel-Ziv Complexity
            #     binary_seq = np.array(
            #         normalized_channel > np.mean(normalized_channel)
            #     ).astype(int)
            #     binary_str = "".join(binary_seq.astype(str))
            #     lempel_ziv_complexity = pyeeg.lzc(binary_str)

            channel_features = [
                sample_entropy,
                hurst_exponent,
                higuchi_fd,
                recurrence_rate,
                dfa_alpha1,
                dfa_alpha2,
                sd1,
                sd2,
                sd1_sd2_ratio,
                approximate_entropy,
                permutation_entropy,
                lempel_ziv_complexity,
                largest_lyapunov_exponent,
                correlation_dimension,
                fractal_dimension_katz,
                time_irreversibility,
                multiscale_entropy,
                rqa_entropy,
                recurrence_variance,
                dynamic_stability,
                symbolic_dynamics,
                sample_entropy_change_rate,
                change_dfa_alpha,
                nonlinear_variance,
                singular_spectrum_entropy,
                dynamic_variance,
                recurrence_network_measures,
                delay_embedding_dimension,
                delay_time,
                complexity_loss,
            ]
            features[sample, ch * n_features : (ch + 1) * n_features] = channel_features
    feature_df = pd.DataFrame(features, columns=column_names)
    _log_end("Nonlinear", start, feature_df.shape)
    return feature_df


def get_morphological_features(
    ecg_data: np.ndarray, sfreq: float, n_jobs: int | None = -1
) -> pd.DataFrame:
    """Extract morphological features from ECG data for each sample and channel.

    This function calculates various morphological features, including:
    - QRS duration
    - QT interval
    - PQ interval
    - P duration
    - T duration
    - ST duration
    - R amplitude
    - S amplitude
    - P amplitude
    - T amplitude
    - Q amplitude
    - PQ dispersion
    - T dispersion
    - ST dispersion
    - RT dispersion
    - QRS dispersion
    - QT dispersion
    - P dispersion
    - R point amplitude
    - P area
    - QRS area
    - T area
    - R slope
    - T slope
    - RT duration
    - QRS curve length
    - T curve slope
    - R symmetry

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
    base_names = [
        "qrs_duration",
        "qt_interval",
        "pq_interval",
        "p_duration",
        "t_duration",
        "st_duration",
        "r_amplitude",
        "s_amplitude",
        "p_amplitude",
        "t_amplitude",
        "q_amplitude",
        "pq_dispersion",
        "t_dispersion",
        "st_dispersion",
        "rt_dispersion",
        "qrs_dispersion",
        "qt_dispersion",
        "p_dispersion",
        "r_point_amplitude",
        "p_area",
        "qrs_area",
        "t_area",
        "r_slope",
        "t_slope",
        "rt_duration",
        "qrs_curve_length",
        "t_curve_slope",
        "r_symmetry",
    ]
    n_chans = ecg_data.shape[1]
    n_features = len(base_names)
    column_names = [
        f"morph_{name}_ch{ch}" for ch in range(n_chans) for name in base_names
    ]

    args_list = ((ecg_single, sfreq, n_features) for ecg_single in ecg_data)
    processes = _get_n_processes(n_jobs, ecg_data.shape[0])
    if processes in [0, 1]:
        results = [_morph_single_patient(*args) for args in args_list]
    else:
        logger.info(f"Starting parallel processing with {processes} CPUs")
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.starmap(_morph_single_patient, args_list)
    features = np.vstack(results)
    feature_df = pd.DataFrame(features, columns=column_names)
    _log_end("Morphological", start, feature_df.shape)
    return feature_df


def _get_n_processes(n_jobs: int | None, n_tasks: int) -> int:
    """Get the number of processes to use for parallel processing."""
    n_processes = os.cpu_count() if n_jobs in [-1, None] else n_jobs
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


def _stat_single_patient(
    sample_data: np.ndarray, sfreq: float, n_features: int
) -> np.ndarray:
    """Extract statistical features from a single sample of ECG data.

    Args:
        sample_data: Single sample of ECG data with shape (n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz
        n_features: Number of features to extract

    Returns:
        Array containing the extracted statistical features
    """
    features = np.zeros(sample_data.shape[0] * n_features)
    for ch, ch_data in enumerate(sample_data):
        sum_ = np.sum(ch_data)
        mean = np.mean(ch_data)
        median = np.median(ch_data)
        mode_result = scipy.stats.mode(ch_data, keepdims=False)
        mode = mode_result if np.isscalar(mode_result) else mode_result[0]
        variance = np.var(ch_data)
        range_ = np.ptp(ch_data)
        min_ = np.min(ch_data)
        max_ = np.max(ch_data)
        iqr = np.percentile(ch_data, 75) - np.percentile(ch_data, 25)
        skewness = scipy.stats.skew(ch_data)
        kurt = scipy.stats.kurtosis(ch_data)
        peak_to_peak = max_ - min_
        autocorr = (
            np.corrcoef(ch_data[:-1], ch_data[1:])[0, 1] if len(ch_data) > 1 else 0
        )

        # Initialize all advanced features
        time_to_first_peak = r_peak_amplitude = duration_between_peaks = (
            amplitude_between_peaks
        ) = rr_interval_mean = rr_interval_std = hrv = rr_diff_mean = (
            rr_interval_median
        ) = rr_iqr = rr_skewness = rr_kurtosis = heart_rate = 0.0

        _, peaks_info = nk.ecg_peaks(ch_data, sampling_rate=sfreq)
        rpeaks = peaks_info["ECG_R_Peaks"]

        if len(rpeaks) > 1:
            time_to_first_peak = rpeaks[0] / sfreq
            r_peak_amplitude = ch_data[rpeaks[0]]
            last_peak = rpeaks[-1]
            duration_between_peaks = (last_peak - rpeaks[0]) / sfreq
            amplitude_between_peaks = ch_data[last_peak] - ch_data[rpeaks[0]]

            rr_intervals = np.diff(rpeaks) / sfreq
            rr_interval_mean = np.mean(rr_intervals)
            rr_interval_std = np.std(rr_intervals)
            heart_rate = 60 / (rr_interval_mean + 1e-10)
            hrv = rr_interval_std
            if len(rr_intervals) > 1:
                rr_diff_mean = np.mean(np.diff(rr_intervals))

            rr_interval_median = np.median(rr_intervals)
            rr_iqr = np.percentile(rr_intervals, 75) - np.percentile(rr_intervals, 25)
            rr_skewness = scipy.stats.skew(rr_intervals)
            rr_kurtosis = scipy.stats.kurtosis(rr_intervals)

        channel_features = [
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
            time_to_first_peak,
            r_peak_amplitude,
            duration_between_peaks,
            amplitude_between_peaks,
            rr_interval_mean,
            rr_interval_std,
            heart_rate,
            hrv,
            rr_diff_mean,
            rr_interval_median,
            rr_iqr,
            rr_skewness,
            rr_kurtosis,
        ]
        features[ch * n_features : (ch + 1) * n_features] = channel_features
    return features


def _morph_single_patient(
    sample_data: np.ndarray, sfreq: float, n_features: int
) -> np.ndarray:
    """Extract morphological features from a single sample of ECG data.

    Args:
        sample_data: Single sample of ECG data with shape (n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz
        n_features: Number of features to extract

    Returns:
        Array containing the extracted morphological features
    """
    n_chans = sample_data.shape[0]
    features = np.zeros(n_chans * n_features)
    for ch_num, ch_data in enumerate(sample_data):
        ch_feat = _morph_single_channel(ch_data, sfreq, n_features)
        features[ch_num * n_features : (ch_num + 1) * n_features] = ch_feat
    return features


def _morph_single_channel(
    ch_data: np.ndarray, sfreq: float, n_features: int
) -> np.ndarray:
    """Extract morphological features from a single channel of ECG data.

    Args:
        ch_data: Single channel of ECG data with shape (n_timepoints,)
        sfreq: Sampling frequency of the ECG data in Hz
        n_features: Number of features to extract

    Returns:
        Array containing the extracted morphological features
    """
    _, peaks_info = nk.ecg_peaks(ch_data, sampling_rate=sfreq)
    r_peaks = peaks_info["ECG_R_Peaks"]
    n_r_peaks = len(r_peaks) if r_peaks is not None else 0

    # Default-Featurewerte
    qrs_duration = qt_interval = pq_interval = p_duration = t_duration = st_duration = 0
    r_amplitude = s_amplitude = p_amplitude = t_amplitude = q_amplitude = 0
    qrs_dispersion = qt_dispersion = p_dispersion = r_point_amplitude = 0
    p_area = qrs_area = t_area = r_slope = t_slope = rt_duration = 0
    qrs_curve_length = t_curve_slope = r_symmetry = 0
    pq_dispersion = t_dispersion = st_dispersion = rt_dispersion = 0
    if n_r_peaks <= 1:
        return np.zeros(n_features)

    import warnings

    waves_dict: dict = {}
    for method in ["dwt", "peak_prominence", "peak", "cwt"]:
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
            qrs_duration = np.mean(qrs_durations)
            qrs_dispersion = np.std(qrs_durations)

    # QT-Intervall
    if n_q_peaks and n_t_peaks:
        qt_intervals = []
        max_index = min(n_q_peaks, n_t_peaks)
        for q, t in zip(q_peaks[:max_index], t_peaks[:max_index]):
            if q >= t or np.isnan(q) or np.isnan(t):
                continue
            qt_intervals.append((t - q) / sfreq * 1000)  # in ms
        if qt_intervals:
            qt_interval = np.mean(qt_intervals)
            qt_dispersion = np.std(qt_intervals)

    # PQ-Intervall
    if n_p_peaks and n_q_peaks:
        pq_intervals = []
        max_index = min(n_p_peaks, n_q_peaks)
        for p, q in zip(p_peaks[:max_index], q_peaks[:max_index]):
            if p >= q or np.isnan(p) or np.isnan(q):
                continue
            pq_intervals.append((q - p) / sfreq * 1000)  # in ms
        if pq_intervals:
            pq_interval = np.mean(pq_intervals)
            pq_dispersion = np.std(pq_intervals)

    # P-Dauer
    if n_p_onsets and n_p_offsets:
        p_durations = []
        max_index = min(n_p_onsets, n_p_offsets)
        for p_on, p_off in zip(p_onsets[:max_index], p_offsets[:max_index]):
            if p_on >= p_off or np.isnan(p_on) or np.isnan(p_off):
                continue
            p_durations.append((p_off - p_on) / sfreq * 1000)
        if p_durations:
            p_duration = np.mean(p_durations)
            p_dispersion = np.std(p_durations)

    # T-Dauer
    if n_t_onsets and n_t_offsets:
        t_durations = []
        max_index = min(n_t_onsets, n_t_offsets)
        for t_on, t_off in zip(t_onsets[:max_index], t_offsets[:max_index]):
            if t_on >= t_off or np.isnan(t_on) or np.isnan(t_off):
                continue
            t_durations.append((t_off - t_on) / sfreq * 1000)
        if t_durations:
            t_duration = np.mean(t_durations)
            t_dispersion = np.std(t_durations)

    # ST-Dauer
    if n_s_peaks and n_t_onsets:
        st_durations = []
        max_index = min(n_s_peaks, n_t_onsets)
        for s, t_on in zip(s_peaks[:max_index], t_onsets[:max_index]):
            if s >= t_on or np.isnan(s) or np.isnan(t_on):
                continue
            st_durations.append((t_on - s) / sfreq * 1000)
        if st_durations:
            st_duration = np.mean(st_durations)
            st_dispersion = np.std(st_durations)

    # RT-Dauer
    if n_r_peaks and n_t_onsets:
        rt_durations = []
        max_index = min(n_r_peaks, n_t_onsets)
        for r, t_on in zip(r_peaks[:max_index], t_onsets[:max_index]):
            if r >= t_on or np.isnan(r) or np.isnan(t_on):
                continue
            rt_durations.append((t_on - r) / sfreq * 1000)
        if rt_durations:
            rt_duration = np.mean(rt_durations)
            rt_dispersion = np.std(rt_durations)

    # Flächen (Integrale unter den Kurven)
    if n_p_onsets and n_p_offsets:
        p_areas = []
        max_index = min(n_p_onsets, n_p_offsets)
        for p_on, p_off in zip(p_onsets[:max_index], p_offsets[:max_index]):
            if p_on >= p_off or np.isnan(p_on) or np.isnan(p_off):
                continue
            p_areas.append(np.sum(np.abs(ch_data[p_on:p_off])))
        if p_areas:
            p_area = np.mean(p_areas)

    # T Area
    if n_t_onsets and n_t_offsets:
        t_areas = []
        max_index = min(n_t_onsets, n_t_offsets)
        for t_on, t_off in zip(t_onsets[:max_index], t_offsets[:max_index]):
            if t_on >= t_off or np.isnan(t_on) or np.isnan(t_off):
                continue
            t_areas.append(np.sum(np.abs(ch_data[t_on:t_off])))
        if t_areas:
            t_area = np.mean(t_areas)

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
            r_slope = np.mean(r_slopes)

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
            t_slope = np.mean(t_slopes)

    # Amplituden
    if n_p_peaks:
        p_amplitudes = [ch_data[p] for p in p_peaks if not np.isnan(p)]
        if p_amplitudes:
            p_amplitude = np.mean(p_amplitudes)

    if n_q_peaks:
        q_amplitudes = [ch_data[q] for q in q_peaks if not np.isnan(q)]
        if q_amplitudes:
            q_amplitude = np.mean(q_amplitudes)

    if n_r_peaks:
        r_amplitudes = [ch_data[r] for r in r_peaks if not np.isnan(r)]
        if r_amplitudes:
            r_amplitude = np.mean(r_amplitudes)

    if n_s_peaks:
        s_amplitudes = [ch_data[s] for s in s_peaks if not np.isnan(s)]
        if s_amplitudes:
            s_amplitude = np.mean(s_amplitudes)

    if n_t_peaks:
        t_amplitudes = [ch_data[t] for t in t_peaks if not np.isnan(t)]
        if t_amplitudes:
            t_amplitude = np.mean(t_amplitudes)

    features = np.array(
        [
            qrs_duration,
            qt_interval,
            pq_interval,
            p_duration,
            t_duration,
            st_duration,
            r_amplitude,
            s_amplitude,
            p_amplitude,
            t_amplitude,
            q_amplitude,
            pq_dispersion,
            t_dispersion,
            st_dispersion,
            rt_dispersion,
            qrs_dispersion,
            qt_dispersion,
            p_dispersion,
            r_point_amplitude,
            p_area,
            qrs_area,
            t_area,
            r_slope,
            t_slope,
            rt_duration,
            qrs_curve_length,
            t_curve_slope,
            r_symmetry,
        ]
    )
    assert n_features == features.shape[0]
    return features
