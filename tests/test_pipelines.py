"""Unit tests for the ECG feature extraction pipeline."""

import neurokit2 as nk
import numpy as np
import pandas as pd
import pytest

import pte_ecg


def test_get_features_basic(test_data: tuple[np.ndarray, int]):
    """Test basic functionality of get_features with default settings."""

    ecg_data, sfreq = test_data

    settings = pte_ecg.Settings()

    # Extract features with default settings
    features = pte_ecg.get_features(ecg_data, sfreq=sfreq, settings=settings)

    # Basic assertions
    assert isinstance(features, pd.DataFrame)
    assert len(features.columns) > 0  # At least one feature extracted

    # Check for common feature prefixes
    feature_columns = set(features.columns)
    assert any(col.startswith("fft_") for col in feature_columns)  # FFT features


def test_get_features_custom_settings(test_data: tuple[np.ndarray, int]):
    """Test get_features with custom settings."""
    ecg_data, sfreq = test_data

    # Create custom settings with only statistical features enabled
    settings = pte_ecg.Settings()
    settings.preprocessing.resample.enabled = True
    settings.preprocessing.resample.sfreq_new = sfreq / 2

    settings.preprocessing.bandpass.enabled = True
    settings.preprocessing.bandpass.l_freq = 0.5
    settings.preprocessing.bandpass.h_freq = sfreq / 5

    settings.preprocessing.notch.enabled = True
    settings.preprocessing.notch.freq = sfreq / 6

    settings.preprocessing.normalize.enabled = True
    settings.preprocessing.normalize.mode = "zscore"

    settings.features.fft.enabled = True
    settings.features.morphological.enabled = False
    settings.features.nonlinear.enabled = False
    settings.features.welch.enabled = False
    settings.features.statistical.enabled = False

    # Extract features
    features = pte_ecg.get_features(ecg=ecg_data, sfreq=sfreq, settings=settings)

    # Check that only FFT features were extracted
    assert all(col.startswith("fft_") for col in features.columns)


def test_get_features_invalid_input():
    """Test get_features with invalid input."""
    # Test with wrong dimensions
    with pytest.raises(ValueError):
        pte_ecg.get_features(np.random.randn(100), sfreq=360)  # 1D input

    with pytest.raises(ValueError):
        pte_ecg.get_features(np.random.randn(10, 10), sfreq=360)  # 2D input

    # Test with invalid sfreq
    with pytest.raises(ValueError):
        pte_ecg.get_features(np.random.randn(5, 1, 1000), sfreq=0)  # Zero sfreq

    # Test with invalid settings type
    with pytest.raises(TypeError):
        pte_ecg.get_features(np.random.randn(5, 1, 1000), sfreq=360, settings=1)

    # Test with invalid settings value
    with pytest.raises(ValueError):
        pte_ecg.get_features(np.random.randn(5, 1, 1000), sfreq=360, settings="random")


def test_get_features_empty_features(test_data: tuple[np.ndarray, int]):
    """Test get_features with no features enabled."""
    ecg_data, sfreq = test_data
    settings = pte_ecg.Settings()

    # Disable all features
    for feature in [
        settings.features.fft,
        settings.features.morphological,
        settings.features.nonlinear,
        settings.features.statistical,
        settings.features.welch,
    ]:
        feature.enabled = False

    # Should raise ValueError when no features are enabled
    with pytest.raises(ValueError, match="No features were extracted"):
        pte_ecg.get_features(ecg=ecg_data, sfreq=sfreq, settings=settings)


@pytest.fixture
def test_data() -> tuple[np.ndarray, int]:
    """Generate synthetic ECG data for testing.

    Returns:
        np.ndarray: ECG data with shape (n_recordings, n_channels, n_timepoints).
    """
    n_recordings = 1
    sfreq = 100
    duration = 10

    ecg_data = np.zeros((n_recordings, duration * sfreq, 12))
    for i in range(n_recordings):
        ecg_data[i] = nk.ecg_simulate(
            duration=duration,
            sampling_rate=sfreq,
            noise=0.01,
            heart_rate=70,
            method="multileads",
            random_state=i,
        )
    ecg_data = np.transpose(ecg_data, (0, 2, 1))
    return ecg_data, sfreq


if __name__ == "__main__":
    pytest.main([__file__])
