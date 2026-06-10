"""Stage 3 transformer variants preserved as experimental or archived references."""

from .carey_tcn_fourier import CareyTCNFourierClassifier, CareyTCNFourierNetwork
from .fripp_patchtst_lite import FrippPatchTSTLiteClassifier, FrippPatchTSTLiteNetwork
from .wilson_tcn_multiscale import WilsonTCNMultiscaleClassifier, WilsonTCNMultiscaleNetwork
from .zappa_mmoe import ZappaMMoEClassifier, ZappaMMoENetwork

__all__ = [
    "CareyTCNFourierClassifier",
    "CareyTCNFourierNetwork",
    "FrippPatchTSTLiteClassifier",
    "FrippPatchTSTLiteNetwork",
    "WilsonTCNMultiscaleClassifier",
    "WilsonTCNMultiscaleNetwork",
    "ZappaMMoEClassifier",
    "ZappaMMoENetwork",
]
