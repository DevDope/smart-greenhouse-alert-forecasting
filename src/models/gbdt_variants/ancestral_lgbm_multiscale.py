"""Official paper name for the Wilson-inspired multiscale v2 variant."""

from .wilson_lgbm_multiscale_v2 import WilsonLGBMMultiscaleV2Classifier


class AncestralLGBMMultiscaleClassifier(WilsonLGBMMultiscaleV2Classifier):
    model_name = "ancestral_lgbm_multiscale"


__all__ = ["AncestralLGBMMultiscaleClassifier"]
