"""Model registry and factory."""

from __future__ import annotations

from typing import Any, Callable

from ..constants import ACTIVE_MODELS, ARCHIVED_MODELS, EXPERIMENTAL_MODELS, SKELETON_MODELS
from .base import BaseModel
from .archive.transformers_stage3.carey_tcn_fourier import CareyTCNFourierClassifier
from .archive.transformers_stage3.fripp_patchtst_lite import FrippPatchTSTLiteClassifier
from .archive.transformers_stage3.wilson_tcn_multiscale import WilsonTCNMultiscaleClassifier
from .archive.transformers_stage3.zappa_mmoe import ZappaMMoEClassifier
from .deep.gru import GRUClassifier
from .deep.tcn import TCNClassifier
from .gbdt_variants.ancestral_lgbm_multiscale import AncestralLGBMMultiscaleClassifier
from .gbdt_variants.carey_lgbm_fourier import CareyLGBMFourierClassifier
from .gbdt_variants.carey_lgbm_temporal_v2 import CareyLGBMTemporalV2Classifier
from .gbdt_variants.carey_xgb_fourier import CareyXGBFourierClassifier
from .gbdt_variants.discipline_lgbm_sparse import DisciplineLGBMSparseClassifier
from .gbdt_variants.fripp_lgbm_sparse import FrippLGBMSparseClassifier
from .gbdt_variants.fripp_xgb_sparse import FrippXGBSparseClassifier
from .gbdt_variants.juxtapose_lgbm_chain import JuxtaposeLGBMChainClassifier
from .gbdt_variants.juxtapose_lgbm_relation_v2 import JuxtaposeLGBMRelationV2Classifier
from .gbdt_variants.spiral_lgbm_temporal import SpiralLGBMTemporalClassifier
from .gbdt_variants.wilson_lgbm_multiscale import WilsonLGBMMultiscaleClassifier
from .gbdt_variants.wilson_lgbm_multiscale_v2 import WilsonLGBMMultiscaleV2Classifier
from .gbdt_variants.wilson_xgb_multiscale import WilsonXGBMultiscaleClassifier
from .gbdt_variants.zappa_lgbm_chain import ZappaLGBMChainClassifier
from .gbdt_variants.zappa_lgbm_chain_v2 import ZappaLGBMChainV2Classifier
from .gbdt_variants.zappa_lgbm_ranker import ZappaLGBMRankerClassifier
from .gbdt_variants.zappa_xgb_multioutput import ZappaXGBMultiOutputClassifier
from .skeletons import build_skeleton
from .tabular.lightgbm_model import LightGBMClassifier
from .tabular.logistic import LogisticClassifier
from .tabular.naive import NaiveClassifier
from .tabular.xgboost_model import XGBoostClassifier
from .transformers.tft_base import TFTBaseClassifier
from .transformers.tft_carey import TFTCareyClassifier
from .transformers.tft_fripp import TFTFrippClassifier
from .transformers.tft_wilson import TFTWilsonClassifier
from .transformers.tft_zappa import TFTZappaClassifier
from .transformers.vanilla_transformer import VanillaTransformerClassifier

ModelFactory = Callable[[dict[str, Any]], BaseModel]


def _build_naive(config: dict[str, Any]) -> BaseModel:
    return NaiveClassifier(**config.get("params", {}))


def _build_logistic(config: dict[str, Any]) -> BaseModel:
    return LogisticClassifier(**config.get("params", {}))


def _build_xgboost(config: dict[str, Any]) -> BaseModel:
    return XGBoostClassifier(**config.get("params", {}))


def _build_lightgbm(config: dict[str, Any]) -> BaseModel:
    return LightGBMClassifier(**config.get("params", {}))


def _build_gru(config: dict[str, Any]) -> BaseModel:
    return GRUClassifier(**config.get("params", {}))


def _build_tcn(config: dict[str, Any]) -> BaseModel:
    return TCNClassifier(**config.get("params", {}))


def _build_vanilla_transformer(config: dict[str, Any]) -> BaseModel:
    return VanillaTransformerClassifier(**config.get("params", {}))


def _build_tft_base(config: dict[str, Any]) -> BaseModel:
    return TFTBaseClassifier(**config.get("params", {}))


def _build_tft_wilson(config: dict[str, Any]) -> BaseModel:
    return TFTWilsonClassifier(**config.get("params", {}))


def _build_tft_zappa(config: dict[str, Any]) -> BaseModel:
    return TFTZappaClassifier(**config.get("params", {}))


def _build_tft_carey(config: dict[str, Any]) -> BaseModel:
    return TFTCareyClassifier(**config.get("params", {}))


def _build_tft_fripp(config: dict[str, Any]) -> BaseModel:
    return TFTFrippClassifier(**config.get("params", {}))


def _build_zappa_mmoe(config: dict[str, Any]) -> BaseModel:
    return ZappaMMoEClassifier(**config.get("params", {}))


def _build_carey_tcn_fourier(config: dict[str, Any]) -> BaseModel:
    return CareyTCNFourierClassifier(**config.get("params", {}))


def _build_fripp_patchtst_lite(config: dict[str, Any]) -> BaseModel:
    return FrippPatchTSTLiteClassifier(**config.get("params", {}))


def _build_wilson_tcn_multiscale(config: dict[str, Any]) -> BaseModel:
    return WilsonTCNMultiscaleClassifier(**config.get("params", {}))


def _build_zappa_xgb_multioutput(config: dict[str, Any]) -> BaseModel:
    return ZappaXGBMultiOutputClassifier(**config.get("params", {}))


def _build_zappa_lgbm_chain(config: dict[str, Any]) -> BaseModel:
    return ZappaLGBMChainClassifier(**config.get("params", {}))


def _build_juxtapose_lgbm_chain(config: dict[str, Any]) -> BaseModel:
    return JuxtaposeLGBMChainClassifier(**config.get("params", {}))


def _build_juxtapose_lgbm_relation_v2(config: dict[str, Any]) -> BaseModel:
    return JuxtaposeLGBMRelationV2Classifier(**config.get("params", {}))


def _build_zappa_lgbm_ranker(config: dict[str, Any]) -> BaseModel:
    return ZappaLGBMRankerClassifier(**config.get("params", {}))


def _build_carey_lgbm_fourier(config: dict[str, Any]) -> BaseModel:
    return CareyLGBMFourierClassifier(**config.get("params", {}))


def _build_carey_xgb_fourier(config: dict[str, Any]) -> BaseModel:
    return CareyXGBFourierClassifier(**config.get("params", {}))


def _build_carey_lgbm_temporal_v2(config: dict[str, Any]) -> BaseModel:
    return CareyLGBMTemporalV2Classifier(**config.get("params", {}))


def _build_spiral_lgbm_temporal(config: dict[str, Any]) -> BaseModel:
    return SpiralLGBMTemporalClassifier(**config.get("params", {}))


def _build_fripp_xgb_sparse(config: dict[str, Any]) -> BaseModel:
    return FrippXGBSparseClassifier(**config.get("params", {}))


def _build_fripp_lgbm_sparse(config: dict[str, Any]) -> BaseModel:
    return FrippLGBMSparseClassifier(**config.get("params", {}))


def _build_discipline_lgbm_sparse(config: dict[str, Any]) -> BaseModel:
    return DisciplineLGBMSparseClassifier(**config.get("params", {}))


def _build_wilson_xgb_multiscale(config: dict[str, Any]) -> BaseModel:
    return WilsonXGBMultiscaleClassifier(**config.get("params", {}))


def _build_wilson_lgbm_multiscale(config: dict[str, Any]) -> BaseModel:
    return WilsonLGBMMultiscaleClassifier(**config.get("params", {}))


def _build_wilson_lgbm_multiscale_v2(config: dict[str, Any]) -> BaseModel:
    return WilsonLGBMMultiscaleV2Classifier(**config.get("params", {}))


def _build_ancestral_lgbm_multiscale(config: dict[str, Any]) -> BaseModel:
    return AncestralLGBMMultiscaleClassifier(**config.get("params", {}))


def _build_zappa_lgbm_chain_v2(config: dict[str, Any]) -> BaseModel:
    return ZappaLGBMChainV2Classifier(**config.get("params", {}))


_ACTIVE_REGISTRY: dict[str, ModelFactory] = {
    "naive": _build_naive,
    "logistic": _build_logistic,
    "xgboost": _build_xgboost,
    "lightgbm": _build_lightgbm,
    "gru": _build_gru,
    "tcn": _build_tcn,
    "vanilla_transformer": _build_vanilla_transformer,
    "tft_base": _build_tft_base,
    "discipline_lgbm_sparse": _build_discipline_lgbm_sparse,
    "juxtapose_lgbm_chain": _build_juxtapose_lgbm_chain,
    "juxtapose_lgbm_relation_v2": _build_juxtapose_lgbm_relation_v2,
}

_EXPERIMENTAL_REGISTRY: dict[str, ModelFactory] = {
    "tft_wilson": _build_tft_wilson,
    "tft_zappa": _build_tft_zappa,
    "tft_carey": _build_tft_carey,
    "tft_fripp": _build_tft_fripp,
    "zappa_mmoe": _build_zappa_mmoe,
    "fripp_patchtst_lite": _build_fripp_patchtst_lite,
    "zappa_xgb_multioutput": _build_zappa_xgb_multioutput,
    "zappa_lgbm_ranker": _build_zappa_lgbm_ranker,
    "carey_lgbm_fourier": _build_carey_lgbm_fourier,
    "spiral_lgbm_temporal": _build_spiral_lgbm_temporal,
    "carey_xgb_fourier": _build_carey_xgb_fourier,
    "fripp_xgb_sparse": _build_fripp_xgb_sparse,
    "wilson_xgb_multiscale": _build_wilson_xgb_multiscale,
    "wilson_lgbm_multiscale": _build_wilson_lgbm_multiscale,
    "ancestral_lgbm_multiscale": _build_ancestral_lgbm_multiscale,
    "zappa_lgbm_chain_v2": _build_zappa_lgbm_chain_v2,
}

_ARCHIVED_REGISTRY: dict[str, ModelFactory] = {
    "carey_tcn_fourier": _build_carey_tcn_fourier,
    "wilson_tcn_multiscale": _build_wilson_tcn_multiscale,
}

_COMPATIBILITY_REGISTRY: dict[str, ModelFactory] = {
    "fripp_lgbm_sparse": _build_fripp_lgbm_sparse,
    "zappa_lgbm_chain": _build_zappa_lgbm_chain,
    "wilson_lgbm_multiscale_v2": _build_wilson_lgbm_multiscale_v2,
    "carey_lgbm_temporal_v2": _build_carey_lgbm_temporal_v2,
}

_REGISTRY: dict[str, ModelFactory] = {
    **_ACTIVE_REGISTRY,
    **_EXPERIMENTAL_REGISTRY,
    **_ARCHIVED_REGISTRY,
    **_COMPATIBILITY_REGISTRY,
}

for model_name in SKELETON_MODELS:
    _REGISTRY[model_name] = lambda config, model_name=model_name: build_skeleton(
        model_name=model_name,
        params=config.get("params", {}),
    )


def create_model(model_name: str, config: dict[str, Any]) -> BaseModel:
    if model_name not in _REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'.")
    return _REGISTRY[model_name](config)


def list_models() -> dict[str, str]:
    return {
        **{name: "active" for name in ACTIVE_MODELS},
        **{name: "experimental" for name in EXPERIMENTAL_MODELS},
        **{name: "archived" for name in ARCHIVED_MODELS},
        **{name: "skeleton" for name in SKELETON_MODELS},
    }
