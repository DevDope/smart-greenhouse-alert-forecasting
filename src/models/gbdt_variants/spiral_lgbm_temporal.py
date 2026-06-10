"""Official paper name for the Carey-inspired temporal v2 variant."""

from .carey_lgbm_temporal_v2 import CareyLGBMTemporalV2Classifier


class SpiralLGBMTemporalClassifier(CareyLGBMTemporalV2Classifier):
    model_name = "spiral_lgbm_temporal"


__all__ = ["SpiralLGBMTemporalClassifier"]
