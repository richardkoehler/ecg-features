# PTE-ECG

A Python package for extracting features from ECG signals, designed for pulmonary thromboembolism (PTE) analysis.

## Installation

### Using pip

```bash
pip install git+https://github.com/richardkoehler/pte-ecg.git
```

### Using uv

```bash
uv add git+https://github.com/richardkoehler/pte-ecg.git
```

### From source

```bash
# Clone the repository
git clone https://github.com/richardkoehler/pte-ecg.git
cd pte-ecg

# Install with pip
pip install .

# Or install with uv
uv pip install .
```

## Usage

Here's a basic example of how to use the package to extract features from ECG data:

```python
import numpy as np
from pte_ecg.pipelines import get_features, Settings

# Generate some synthetic ECG data (replace with your actual data)
# Shape should be (n_channels, n_samples)
sfreq = 1000  # Sampling frequency in Hz
ecg_data = np.random.randn(1, 10000)  # 1 channel, 10 seconds at 1000 Hz

# Create custom settings (optional)
settings = Settings()
settings.features.fft.enabled = True
settings.features.morphological.enabled = False
settings.features.nonlinear.enabled = False
settings.features.welch.enabled = False
settings.features.statistical.enabled = False

# Extract features
features = get_features(ecg_data, sfreq, settings=settings)

print(f"Extracted {len(features.columns)} features:")
print(features.head())
```

## Features

- Multiple feature extraction methods:
  - FFT-based features
  - Morphological features
  - Nonlinear features
  - Statistical features
  - Welch's method for power spectral density
- Configurable feature extraction pipeline
- Efficient processing of multi-channel ECG data

## Documentation

For detailed documentation, please refer to the [documentation](https://github.com/richardkoehler/pte-ecg).

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
