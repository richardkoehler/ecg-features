"""
feature_extraction.py - Feature-Extraktion für EKG-Daten

Dieses Modul enthält Funktionen zur Extraktion verschiedener Feature-Typen aus EKG-Daten:
- Statistische Features (Mittelwert, Median, Varianz, etc.)
- Frequenzbasierte Features (FFT, Welch-Spektrum)
- Nichtlineare Features (Entropie, Hurst-Exponent, etc.)
- Morphologische Features (QRS-Dauer, QT-Intervall, etc.)
- Patienten-Features (Alter, Geschlecht, BMI)

Die Hauptfunktion extract_all_features kombiniert alle Feature-Typen.

Author: Selina Baumgart
Date: Mai 2025
"""

import time
from enum import StrEnum
from multiprocessing import Pool, cpu_count

import mne
import mne.baseline
import neurokit2 as nk
import nolds
import numpy as np
import pandas as pd
import scipy.fft
import scipy.signal
import scipy.stats

from .logging import logger


class ECG_FEATURES(StrEnum):
    FFT = "fft"
    WELCH = "welch"
    STATISTICAL = "statistical"
    MORPHOLOGICAL = "morphological"
    NONLINEAR = "nonlinear"


class OTHER_FEATURES(StrEnum):
    PATIENT = "patient"


FEATURE_LIST = list[ECG_FEATURES | OTHER_FEATURES]

EPS = 1e-10  # Small constant for numerical stability


def get_features(
    ecg: np.ndarray,
    sfreq: int,
    sfreq_new: int | None = None,
    include_features: FEATURE_LIST = FEATURE_LIST,
) -> pd.DataFrame:
    ecg = preprocess_ecg(ecg, sfreq=sfreq, sfreq_new=sfreq_new)
    feature_list = []
    if ECG_FEATURES.MORPHOLOGICAL in include_features:
        feature_list.append(
            extract_morphological_features(ecg, sfreq_new, clean_ecg=False, n_jobs=-1)
        )
    if ECG_FEATURES.STATISTICAL in include_features:
        feature_list.append(
            extract_statistical_features(ecg, sfreq_new, clean_ecg=False, n_jobs=-1)
        )
    if ECG_FEATURES.FFT in include_features:
        feature_list.append(extract_fft_features_vectorized(ecg, sfreq_new))
    if ECG_FEATURES.WELCH in include_features:
        feature_list.append(extract_welch_features_vectorized(ecg, sfreq_new))
    if ECG_FEATURES.NONLINEAR in include_features:
        feature_list.append(extract_nonlinear_features(ecg, sfreq_new))
    return pd.concat(feature_list, axis=1)


def extract_fft_features_vectorized(
    ecg_data: np.ndarray, sampling_rate: int
) -> pd.DataFrame:
    assert ecg_data.ndim == 3, (
        "Expected input shape (n_samples, n_channels, n_timepoints)"
    )

    logger.info(f"Starting FFT feature extraction for {len(ecg_data)} samples...")

    start_time = time.time()

    n_samples, n_channels, n_timepoints = ecg_data.shape
    xf = np.fft.rfftfreq(n_timepoints, 1 / sampling_rate)  # (freqs,)
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

    elapsed = time.time() - start_time
    logger.info("Progress: %.1f%% - Elapsed: %.2fs", 100, elapsed)
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

    features_df = pd.DataFrame(features_reshaped, columns=column_names)

    logger.info("FFT feature extraction complete. Shape: %s", features_df.shape)
    return features_df


def preprocess_ecg(
    ecg_data: np.ndarray, sfreq: int, sfreq_new: int | None = None
) -> np.ndarray:
    """
    Präprozessiert EKG-Daten durch Normalisierung und Transposition.

    Diese Funktion normalisiert jede EKG-Sequenz auf den Bereich [0, 1] und
    transponiert die Dimensionen, um die für die Feature-Extraktion erforderliche
    Form zu erhalten.

    Parameters:
    -----------
    ecg_data : numpy.ndarray
        Rohe EKG-Daten mit Form (n_patients, n_times, n_leads)
    sfreq : int
        Abtastrate der EKG-Daten in Hz

    Returns:
    --------
    numpy.ndarray
        Normalisierte und transponierte EKG-Daten mit Form (n_patients, n_leads, n_times)
    """
    logger.info("Preprocessing ECG data...")
    assert ecg_data.ndim == 3
    n_patients, n_times, n_leads = ecg_data.shape
    assert n_leads < n_times  # Sanity check
    ecg_data = ecg_data.transpose(0, 2, 1)
    if sfreq_new:
        ecg_data = mne.filter.resample(
            ecg_data, up=1.0, down=sfreq / sfreq_new, axis=-1, n_jobs=1, verbose=None
        )
    else:
        sfreq_new = sfreq
    n_times = ecg_data.shape[-1]
    ecg_data = mne.filter.filter_data(
        ecg_data, sfreq_new, l_freq=0.5, h_freq=40, n_jobs=1, copy=True
    )
    ecg_data = ecg_data.reshape(n_patients, -1)
    ecg_data = mne.baseline.rescale(
        ecg_data,
        times=np.arange(ecg_data.shape[-1]),
        baseline=(None, None),
        mode="zscore",
        verbose=None,
    )
    ecg_data = ecg_data.reshape(n_patients, n_leads, n_times)
    assert ecg_data.shape == (n_patients, n_leads, n_times)
    return ecg_data


def extract_statistical_features(
    ecg_data: np.ndarray,
    sampling_rate: int,
    clean_ecg: bool,
    n_jobs: int = -1,
) -> pd.DataFrame:
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
    args = [
        (ecg_single, sampling_rate, clean_ecg, n_features) for ecg_single in ecg_data
    ]
    with Pool(processes=None if n_jobs == -1 else n_jobs) as pool:
        results = pool.starmap(_statistical_single, args)
    feature_array = np.vstack(results)
    return pd.DataFrame(feature_array, columns=column_names)


def extract_nonlinear_features(
    ecg_data: np.ndarray, sampling_rate: int
) -> pd.DataFrame:
    """
    Extrahiert nichtlineare Features aus den EKG-Daten für jede Probe und jeden Kanal.

    Diese Funktion berechnet 30 verschiedene nichtlineare Kennzahlen pro Kanal, die
    komplexe dynamische Eigenschaften des EKG-Signals erfassen:
    - Sample Entropy: Maß für die Komplexität und Unvorhersehbarkeit des Signals
    - Hurst-Exponent: Maß für die Langzeitabhängigkeit im Signal
    - Higuchi Fractal Dimension: Maß für die fraktale Dimension des Signals
    - Recurrence Rate: Maß für die Wiederholungen im Signal
    - DFA Alpha1/Alpha2: Detrended Fluctuation Analysis Parameter
    - SD1/SD2: Poincaré-Plot-Parameter für Herzfrequenzvariabilität
    - SD1/SD2 Ratio: Verhältnis der Poincaré-Plot-Parameter
    - Und weitere nichtlineare Features wie Approximate Entropy, Permutation Entropy, etc.

    Parameters:
    -----------
    ecg_data : numpy.ndarray
        EKG-Daten mit Form (n_samples, n_channels, n_timepoints)
    sampling_rate : int
        Abtastrate der EKG-Daten in Hz

    Returns:
    --------
    pandas.DataFrame
        DataFrame mit extrahierten nichtlinearen Features
    """
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
        logger.info(f"Extract nonlinear features for sample {sample}/{len(ecg_data)}")
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
            ecg_cleaned = nk.ecg_clean(ch_data, sampling_rate=sampling_rate)
            _, peaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
            rpeaks = peaks_info["ECG_R_Peaks"]
            if len(rpeaks) > 1:
                rr_intervals = np.diff(rpeaks) / sampling_rate
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

    logger.info(f"Nichtlineare Features extrahiert: {features.shape}")
    return pd.DataFrame(features, columns=column_names)


def extract_morphological_features(
    ecg_data: np.ndarray, sampling_rate: int, clean_ecg: bool, n_jobs: int = -1
) -> pd.DataFrame:
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
    args_list = [
        (ecg_single, sampling_rate, clean_ecg, n_features) for ecg_single in ecg_data[:]
    ]
    processes = cpu_count() if n_jobs in [-1, None] else n_jobs
    logger.info(f"Starte parallele Verarbeitung mit {processes} CPUs...")
    # with Pool(processes=processes) as pool:
    #     results = pool.starmap(_morph_single, args_list)
    results = []
    for i, args in enumerate(args_list):
        results.append(_morph_single(*args))
    features = np.vstack(results)
    logger.info(f"Morphologische Features extrahiert: {features.shape}")
    return pd.DataFrame(features, columns=column_names)


def extract_welch_features_vectorized(
    ecg_data: np.ndarray, sampling_rate: int
) -> pd.DataFrame:
    logger.info(f"Starting Welch feature extraction for {ecg_data.shape[0]} samples...")
    assert ecg_data.ndim == 3, "Expected shape: (n_samples, n_channels, n_timepoints)"
    start = time.time()
    n_samples, n_channels, n_timepoints = ecg_data.shape
    flat_data = ecg_data.reshape(
        -1, n_timepoints
    )  # Shape: (n_samples * n_channels, n_timepoints)

    # Compute Welch spectra for each channel
    psd_list = []
    freq_list = []
    for channel_data in flat_data:
        f, Pxx = scipy.signal.welch(
            channel_data,
            fs=sampling_rate,
            nperseg=min(256, len(channel_data)),
            scaling="spectrum",
        )
        psd_list.append(Pxx)
        freq_list.append(f)

    f = freq_list[0]
    psd_array = np.array(psd_list)  # Shape: (n_samples * n_channels, freq_len)

    # Create frequency masks
    f = np.array(f)
    mask_low = f <= 15
    mask_high = f > 15
    mask_0_0_5 = (f >= 0) & (f <= 0.5)
    mask_0_5_4 = (f > 0.5) & (f <= 4)
    mask_4_15 = (f > 4) & (f <= 15)
    mask_15_40 = (f > 15) & (f <= 40)
    mask_over_40 = f > 40

    # Welch bins
    n_bins = 10
    idx_bins = np.linspace(0, psd_array.shape[1] - 1, n_bins, dtype=int)
    welch_bins = psd_array[:, idx_bins]  # Shape: (n_samples * n_channels, 10)

    # Frequency-domain features
    low_power = np.sum(psd_array[:, mask_low], axis=1)
    high_power = np.sum(psd_array[:, mask_high], axis=1)
    log_power_ratio = np.log(high_power / (low_power + 1e-10) + 1e-10)

    band_0_0_5 = np.sum(psd_array[:, mask_0_0_5], axis=1)
    band_0_5_4 = np.sum(psd_array[:, mask_0_5_4], axis=1)
    band_4_15 = np.sum(psd_array[:, mask_4_15], axis=1)
    band_15_40 = np.sum(psd_array[:, mask_15_40], axis=1)
    band_over_40 = np.sum(psd_array[:, mask_over_40], axis=1)

    total_power = np.sum(psd_array, axis=1)
    psd_norm = psd_array / (total_power[:, None] + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=1)

    peak_indices = np.argmax(psd_array, axis=1)
    peak_frequency = f[peak_indices]

    # Combine features
    all_features = np.column_stack(
        [
            welch_bins,
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
    end = time.time()
    logger.info(f"Welch feature extraction complete. Shape: {all_features.shape}")
    logger.info(f"Time taken: {end - start:.2f} seconds")
    column_names = [f"welch_feature_{i}" for i in range(all_features.shape[1])]
    return pd.DataFrame(all_features, columns=column_names)


def extract_welch_features(ecg_data: np.ndarray, sampling_rate: int) -> pd.DataFrame:
    """
    Extrahiert Welch-Spektrum-basierte Features aus den EKG-Daten für jede Probe und jeden Kanal.

    Diese Funktion berechnet 19 verschiedene spektrale Kennzahlen pro Kanal basierend auf
    dem Welch-Verfahren zur Spektralschätzung:
    - Welch-Bin1 bis Welch-Bin10: Energie in verschiedenen Frequenzbändern
    - Log-Power-Ratio zwischen hohen und niedrigen Frequenzen
    - Bandpower in verschiedenen Frequenzbereichen (0-0.5Hz, 0.5-4Hz, 4-15Hz, 15-40Hz, >40Hz)
    - Spektrale Entropie, Gesamtleistung und Spitzenfrequenz

    Parameters:
    -----------
    ecg_data : numpy.ndarray
        EKG-Daten mit Form (n_samples, n_channels, n_timepoints)
    sampling_rate : int
        Abtastrate der EKG-Daten in Hz

    Returns:
    --------
    pandas.DataFrame
        DataFrame mit extrahierten Welch-Features
    """
    logger.info(f"Starting Welch feature extraction for {ecg_data.shape[0]} samples...")
    start = time.time()
    features = []
    for sample in ecg_data:
        sample_features = []
        for channel in sample:
            # Verwende Welch-Methode zur Spektralschätzung
            f, Pxx = scipy.signal.welch(
                channel,
                fs=sampling_rate,
                nperseg=min(256, len(channel)),
                scaling="spectrum",
            )

            # Extrahiere 10 Bins aus dem Spektrum (gleichmäßig verteilt)
            if len(Pxx) >= 10:
                indices = np.linspace(0, len(Pxx) - 1, 10, dtype=int)
                welch_bins = Pxx[indices]
            else:
                # Fallback, wenn zu wenige Punkte
                welch_bins = np.zeros(10)
                welch_bins[: len(Pxx)] = Pxx

            # Berechne das Verhältnis von hohen zu niedrigen Frequenzen
            # Definiere niedrige Frequenzen als 0-15 Hz und hohe als >15 Hz
            low_freq_idx = np.where(f <= 15)[0]
            high_freq_idx = np.where(f > 15)[0]

            if len(low_freq_idx) > 0 and len(high_freq_idx) > 0:
                low_power = np.sum(Pxx[low_freq_idx])
                high_power = np.sum(Pxx[high_freq_idx])
                # Verwende Log-Verhältnis für bessere Skalierung
                log_power_ratio = np.log(high_power / (low_power + 1e-10) + 1e-10)
            else:
                log_power_ratio = 0

            # Berechne Bandpower in verschiedenen Frequenzbereichen
            band_0_0_5 = (
                np.sum(Pxx[np.where((f >= 0) & (f <= 0.5))[0]])
                if np.any((f >= 0) & (f <= 0.5))
                else 0
            )
            band_0_5_4 = (
                np.sum(Pxx[np.where((f > 0.5) & (f <= 4))[0]])
                if np.any((f > 0.5) & (f <= 4))
                else 0
            )
            band_4_15 = (
                np.sum(Pxx[np.where((f > 4) & (f <= 15))[0]])
                if np.any((f > 4) & (f <= 15))
                else 0
            )
            band_15_40 = (
                np.sum(Pxx[np.where((f > 15) & (f <= 40))[0]])
                if np.any((f > 15) & (f <= 40))
                else 0
            )
            band_over_40 = np.sum(Pxx[np.where(f > 40)[0]]) if np.any(f > 40) else 0

            # Berechne spektrale Entropie
            psd_norm = Pxx / (np.sum(Pxx) + 1e-10)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

            # Gesamtleistung und Spitzenfrequenz
            total_power = np.sum(Pxx)
            peak_freq_idx = np.argmax(Pxx)
            peak_frequency = f[peak_freq_idx] if peak_freq_idx < len(f) else 0

            # Kombiniere alle Features
            channel_features = list(welch_bins) + [
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
            sample_features.extend(channel_features)

        features.append(sample_features)

    features_array = np.array(features)
    end = time.time()
    logger.info(f"Welch feature extraction complete. Shape: {features_array.shape}")
    logger.info(f"Time taken: {end - start:.2f} seconds")
    column_names = [f"welch_feature_{i}" for i in range(features_array.shape[1])]
    return pd.DataFrame(features_array, columns=column_names)


def extract_fft_features(ecg_data: np.ndarray, sampling_rate: int) -> pd.DataFrame:
    """
    Extrahiert FFT-basierte Features aus den EKG-Daten für jede Probe und jeden Kanal.

    Diese Funktion berechnet 25 verschiedene spektrale Kennzahlen pro Kanal basierend auf
    der Fast Fourier Transformation (FFT):
    - Grundlegende Frequenzstatistiken: Summe, Mittelwert, Varianz der Frequenzen
    - Dominante Frequenz und Bandbreite
    - Spektrale Entropie und Flachheit
    - Hochfrequenz- und Niederfrequenzleistung und deren Verhältnis
    - Bandenergie und -verhältnisse in verschiedenen Frequenzbereichen (0-10Hz, 10-20Hz, etc.)
    - Leistung unter/über 50Hz und relative Leistung unter 50Hz

    Parameters:
    -----------
    ecg_data : numpy.ndarray
        EKG-Daten mit Form (n_samples, n_channels, n_timepoints)
    sampling_rate : int
        Abtastrate der EKG-Daten in Hz

    Returns:
    --------
    pandas.DataFrame
        DataFrame mit extrahierten FFT-Features
    """
    logger.info(f"Starte FFT-Feature-Extraktion für {len(ecg_data)} Samples...")
    start_time = time.time()
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
        f"fft_{name}_ch{ch}" for ch in range(ecg_data.shape[1]) for name in base_names
    ]
    # Fortschrittsanzeige
    total_samples = len(ecg_data)
    logger.info(
        f"Verarbeite {total_samples} Samples mit jeweils {ecg_data.shape[1]} Kanälen..."
    )
    progress_step = max(1, total_samples // 10)  # Zeige Fortschritt in 10%-Schritten
    all_features = []
    for i, sample in enumerate(ecg_data):
        sample_features = []
        for channel in sample:
            n = len(channel)
            yf = scipy.fft.fft(channel)
            xf = scipy.fft.fftfreq(n, 1 / sampling_rate)

            # Verwende nur die positive Hälfte des Spektrums
            xf = xf[: n // 2]
            yf_abs = np.abs(yf[: n // 2])

            # Normalisiere das Spektrum
            yf_norm = yf_abs / np.sum(yf_abs) if np.sum(yf_abs) > 0 else yf_abs

            # Grundlegende Frequenzstatistiken
            sum_freq = np.sum(yf_abs)
            mean_freq = np.mean(yf_abs)
            variance_freq = np.var(yf_abs)

            # Dominante Frequenz und Bandbreite
            dominant_freq_idx = np.argmax(yf_abs)
            dominant_frequency = (
                xf[dominant_freq_idx] if dominant_freq_idx < len(xf) else 0
            )

            # Berechne Bandbreite (Frequenzbereich, der 95% der Leistung enthält)
            cumsum = np.cumsum(yf_norm)
            bandwidth_idx = np.where(cumsum >= 0.95)[0]
            bandwidth = xf[bandwidth_idx[0]] if len(bandwidth_idx) > 0 else 0

            # Spektrale Entropie und Flachheit
            spectral_entropy = -np.sum(yf_norm * np.log2(yf_norm + 1e-10))
            spectral_flatness = scipy.stats.gmean(yf_abs + 1e-10) / (
                np.mean(yf_abs) + 1e-10
            )

            # Hochfrequenz- und Niederfrequenzleistung
            hf_mask = (xf >= 15) & (xf <= 40)
            lf_mask = (xf >= 0.5) & (xf < 15)

            hf_power = np.sum(yf_abs[hf_mask]) if np.any(hf_mask) else 0
            lf_power = np.sum(yf_abs[lf_mask]) if np.any(lf_mask) else 0
            hf_lf_ratio = hf_power / (lf_power + 1e-10)

            # Bandenergie und -verhältnisse in verschiedenen Frequenzbereichen
            band_0_10_mask = (xf >= 0) & (xf < 10)
            band_10_20_mask = (xf >= 10) & (xf < 20)
            band_20_30_mask = (xf >= 20) & (xf < 30)
            band_30_40_mask = (xf >= 30) & (xf < 40)

            band_energy_0_10 = (
                np.sum(yf_abs[band_0_10_mask]) if np.any(band_0_10_mask) else 0
            )
            band_energy_10_20 = (
                np.sum(yf_abs[band_10_20_mask]) if np.any(band_10_20_mask) else 0
            )
            band_energy_20_30 = (
                np.sum(yf_abs[band_20_30_mask]) if np.any(band_20_30_mask) else 0
            )
            band_energy_30_40 = (
                np.sum(yf_abs[band_30_40_mask]) if np.any(band_30_40_mask) else 0
            )

            total_energy = np.sum(yf_abs)
            band_ratio_0_10 = band_energy_0_10 / (total_energy + 1e-10)
            band_ratio_10_20 = band_energy_10_20 / (total_energy + 1e-10)
            band_ratio_20_30 = band_energy_20_30 / (total_energy + 1e-10)
            band_ratio_30_40 = band_energy_30_40 / (total_energy + 1e-10)

            # Leistung unter/über 50Hz
            below_50Hz_mask = xf < 50
            above_50Hz_mask = xf >= 50

            power_below_50Hz = (
                np.sum(yf_abs[below_50Hz_mask]) if np.any(below_50Hz_mask) else 0
            )
            power_above_50Hz = (
                np.sum(yf_abs[above_50Hz_mask]) if np.any(above_50Hz_mask) else 0
            )
            relative_power_below_50Hz = power_below_50Hz / (total_energy + 1e-10)

            # Kombiniere alle Features
            channel_features = [
                sum_freq,
                mean_freq,
                variance_freq,
                dominant_frequency,
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
            ]
            sample_features.extend(channel_features)
        if i % progress_step == 0 or i == total_samples - 1:
            progress = (i + 1) / total_samples * 100
            elapsed_time = time.time() - start_time
            logger.info(
                f"Fortschritt: {progress:.1f}% ({i + 1}/{total_samples}) - Laufzeit: {elapsed_time:.1f} Sekunden"
            )
        all_features.append(sample_features)
    features_df = pd.DataFrame(all_features, columns=column_names)
    logger.info(f"FFT-Features extrahiert: {features_df.shape}")
    return features_df


def extract_all_features(
    ecg_data: np.ndarray,
    sampling_rate: int,
    include_fft: bool = True,
    include_statistical: bool = True,
    include_nonlinear: bool = True,
    include_morphological: bool = True,
    include_welch: bool = True,
    clean_ecg: bool = True,
) -> pd.DataFrame:
    """
    Extrahiert alle verfügbaren Features aus EKG-Daten und Patientendaten.

    Diese Funktion ist die Hauptschnittstelle für die Feature-Extraktion und kombiniert
    alle verfügbaren Feature-Typen in einem einzigen Aufruf. Die einzelnen Feature-Typen
    können über Parameter ein- oder ausgeschaltet werden.

    Parameters:
    -----------
    ecg_data : numpy.ndarray
        EKG-Daten mit Form (n_samples, n_channels, n_timepoints)
    sampling_rate : int, optional
        Abtastrate der EKG-Daten in Hz, Standard ist 500 Hz
    include_fft : bool, optional
        Ob FFT-Features einbezogen werden sollen, Standard ist True
    include_statistical : bool, optional
        Ob statistische Features einbezogen werden sollen, Standard ist True
    include_nonlinear : bool, optional
        Ob nichtlineare Features einbezogen werden sollen, Standard ist True
    include_morphological : bool, optional
        Ob morphologische Features einbezogen werden sollen, Standard ist True
    include_welch : bool, optional
        Ob Welch-Power-Spektrum-Features einbezogen werden sollen, Standard ist True
    clean_ecg : bool, optional
        Ob die EKG-Daten vor der Feature-Extraktion bereinigt werden sollen, Standard ist True

    Returns:
    --------
    pd.DataFrame
        DataFrame mit den extrahierten Feature-Sets, wobei die Spalten die Feature-Typen
        und die Werte die entsprechenden Feature-Werte sind.
    """
    logger.info(
        f"Extrahiere Features für {len(ecg_data)} Samples mit Sampling-Rate {sampling_rate} Hz"
    )
    features: dict[str, pd.DataFrame] = {}
    if include_fft:
        logger.info("Extrahiere FFT-Features...")
        features["fft"] = extract_fft_features(ecg_data, sampling_rate=sampling_rate)
    if include_statistical:
        logger.info("Extrahiere statistische Features...")
        features["statistical"] = extract_statistical_features(
            ecg_data, sampling_rate=sampling_rate, clean_ecg=clean_ecg
        )
    if include_nonlinear:
        logger.info("Extrahiere nichtlineare Features...")
        features["nonlinear"] = extract_nonlinear_features(
            ecg_data, sampling_rate=sampling_rate
        )
    if include_morphological:
        logger.info("Extrahiere morphologische Features...")
        features["morphological"] = extract_morphological_features(
            ecg_data, sampling_rate=sampling_rate, clean_ecg=clean_ecg
        )
    if include_welch:
        logger.info("Extrahiere Welch-Features...")
        features["welch"] = extract_welch_features(
            ecg_data, sampling_rate=sampling_rate
        )
    logger.info("Feature-Extraktion abgeschlossen:")
    for feature_type, feature_df in features.items():
        if not feature_df.empty:
            logger.info(f"  - {feature_type}: {feature_df.shape}")
    return pd.concat(features.values(), axis=1)


def _statistical_single(
    sample_data: np.ndarray, sampling_rate: int, clean_ecg: bool, n_features: int
) -> np.ndarray:
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

        if clean_ecg:
            ch_data = nk.ecg_clean(ch_data, sampling_rate=sampling_rate)
        _, peaks_info = nk.ecg_peaks(ch_data, sampling_rate=sampling_rate)
        rpeaks = peaks_info["ECG_R_Peaks"]

        if len(rpeaks) > 1:
            time_to_first_peak = rpeaks[0] / sampling_rate
            r_peak_amplitude = ch_data[rpeaks[0]]
            last_peak = rpeaks[-1]
            duration_between_peaks = (last_peak - rpeaks[0]) / sampling_rate
            amplitude_between_peaks = ch_data[last_peak] - ch_data[rpeaks[0]]

            rr_intervals = np.diff(rpeaks) / sampling_rate
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


def _morph_single(
    sample_data: np.ndarray,
    sampling_rate: int,
    clean_ecg: bool,
    n_features: int,
):
    n_chans = sample_data.shape[0]
    features = np.zeros(n_chans * n_features)
    for ch_num, ch_data in enumerate(sample_data):
        if clean_ecg:
            ch_data = nk.ecg_clean(ch_data, sampling_rate=sampling_rate)

        _, peaks_info = nk.ecg_peaks(ch_data, sampling_rate=sampling_rate)
        r_peaks = peaks_info["ECG_R_Peaks"]
        n_r_peaks = len(r_peaks) if r_peaks is not None else 0

        # Default-Featurewerte
        qrs_duration = qt_interval = pq_interval = p_duration = t_duration = (
            st_duration
        ) = 0
        r_amplitude = s_amplitude = p_amplitude = t_amplitude = q_amplitude = 0
        qrs_dispersion = qt_dispersion = p_dispersion = r_point_amplitude = 0
        p_area = qrs_area = t_area = r_slope = t_slope = rt_duration = 0
        qrs_curve_length = t_curve_slope = r_symmetry = 0
        pq_dispersion = t_dispersion = st_dispersion = rt_dispersion = 0
        if n_r_peaks > 1:
            _, waves_dict = nk.ecg_delineate(
                ch_data,
                rpeaks=r_peaks,
                sampling_rate=sampling_rate,
                method="dwt",
            )

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
                    qrs_durations.append((s - q) / sampling_rate * 1000)  # in ms
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
                    qt_intervals.append((t - q) / sampling_rate * 1000)  # in ms
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
                    pq_intervals.append((q - p) / sampling_rate * 1000)  # in ms
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
                    p_durations.append((p_off - p_on) / sampling_rate * 1000)
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
                    t_durations.append((t_off - t_on) / sampling_rate * 1000)
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
                    st_durations.append((t_on - s) / sampling_rate * 1000)
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
                    rt_durations.append((t_on - r) / sampling_rate * 1000)
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
                    delta_x = (r - q) / sampling_rate
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
                    delta_x = (t_on - t_off) / sampling_rate
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

        features[ch_num * n_features : (ch_num + 1) * n_features] = [
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
    return features
