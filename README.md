![ECG Trace](docs/ecg_trace.svg)

# PTE-ECG

![License](https://img.shields.io/github/license/richardkoehler/pte-ecg)
<!-- ![PyPI version](https://img.shields.io/pypi/v/pte-ecg?color=blue) -->
<!-- ![Build Status](https://img.shields.io/github/actions/workflow/status/richardkoehler/pte-ecg/python-package.yml?branch=main) -->

**Python Tools for Electrophysiology (PTE) - ECG**

A Python package for extracting features from ECG signals.

This package aims at providing an extensible and pluggable interface to extract features from raw ECG, while also providing reasonable default values for preprocessing.

## Table of Contents
- [✨ Highlights](#highlights)
- [🚀 Installation](#installation)
- [💻 Development setup](#development-setup)
- [🩺 Usage](#usage)
- [📄 License](#license)
- [🤝 Contributing](#contributing)


## Highlights

- 🔩Configurable and pluggable feature extraction pipeline
- ⚡️Efficient processing of multi-subject, multi-channel data
- 🛠️ Preprocessing methods
  - Resampling
  - Bandpass filtering
  - Notch filtering
  - Normalization
- 📊 Feature extraction methods
  - FFT-based features
  - Morphological features
  - Nonlinear features
  - Statistical features
  - Welch's method-based features


## 🚀 Installation

### Using pip

```bash
pip install git+https://github.com/richardkoehler/pte-ecg.git
```

### Using uv

```bash
uv add git+https://github.com/richardkoehler/pte-ecg.git
```

## Development setup

```bash
# Clone the repository
git clone https://github.com/richardkoehler/pte-ecg.git
cd pte-ecg

# Install with pip
pip install -e .

# Or install with uv
uv sync
```

## Usage

Here's a basic example of how to use the package to extract features from ECG data:

```python
import numpy as np
import pte_ecg

# Generate some synthetic ECG data (replace with your actual data)
# Shape should be (n_channels, n_samples)
sfreq = 1000  # Sampling frequency in Hz
ecg_data = np.random.randn(1, 10000)  # 1 channel, 10 seconds at 1000 Hz

# Use default settings
settings = "default"

# Or use custom setting
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

print(f"Extracted {len(features.columns)} features:\n{features.head()}")
```

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit an issue if you find a bug or have a feature request. Pull requests are also welcome.
