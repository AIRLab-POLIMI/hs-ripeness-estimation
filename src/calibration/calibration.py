"""
A class structure for implementing geometric and radiometric calibration of spectral data.

Classes:
- `Calibration`: Base class for implementing calibration of spectral data
- `GeometricCalibration`: Derived class for implementing geometric calibration of spectral data
- `RadiometricCalibration`: Derived class for implementing radiometric calibration of spectral data

"""
from abc import ABC, abstractmethod


class Calibration(ABC):
    def __init__(self):
        self.reference_data = None
        self.params = None

    @abstractmethod
    def compute_params(self):
        # Compute calibration parameters here
        pass

    @abstractmethod
    def load_params(self, cfg):
        # Load calibration parameters from file here
        pass

    @abstractmethod
    def save_params(self, filepath):
        # Load calibration parameters from file here
        pass

    @abstractmethod
    def apply_calibration(self, spectral_data):
        # Apply calibration to the data here
        pass

