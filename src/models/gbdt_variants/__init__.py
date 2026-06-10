"""GBDT variants inspired by archived transformer ideas."""

from .ancestral_lgbm_multiscale import AncestralLGBMMultiscaleClassifier
from .common import CareyLGBMFourierClassifier
from .common import CareyXGBFourierClassifier
from .common import FrippLGBMSparseClassifier
from .common import FrippXGBSparseClassifier
from .common import WilsonLGBMMultiscaleClassifier
from .common import WilsonXGBMultiscaleClassifier
from .common import ZappaLGBMChainClassifier
from .common import ZappaLGBMRankerClassifier
from .common import ZappaXGBMultiOutputClassifier
from .carey_lgbm_temporal_v2 import CareyLGBMTemporalV2Classifier
from .discipline_lgbm_sparse import DisciplineLGBMSparseClassifier
from .juxtapose_lgbm_chain import JuxtaposeLGBMChainClassifier
from .spiral_lgbm_temporal import SpiralLGBMTemporalClassifier
from .wilson_lgbm_multiscale_v2 import WilsonLGBMMultiscaleV2Classifier
from .zappa_lgbm_chain_v2 import ZappaLGBMChainV2Classifier

__all__ = [
    "AncestralLGBMMultiscaleClassifier",
    "CareyLGBMFourierClassifier",
    "CareyLGBMTemporalV2Classifier",
    "CareyXGBFourierClassifier",
    "DisciplineLGBMSparseClassifier",
    "FrippLGBMSparseClassifier",
    "FrippXGBSparseClassifier",
    "JuxtaposeLGBMChainClassifier",
    "SpiralLGBMTemporalClassifier",
    "WilsonLGBMMultiscaleClassifier",
    "WilsonLGBMMultiscaleV2Classifier",
    "WilsonXGBMultiscaleClassifier",
    "ZappaLGBMChainClassifier",
    "ZappaLGBMChainV2Classifier",
    "ZappaLGBMRankerClassifier",
    "ZappaXGBMultiOutputClassifier",
]
