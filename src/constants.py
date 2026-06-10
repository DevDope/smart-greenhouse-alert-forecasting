"""Project-wide constants."""

TASKS = (
    "water_stress_60m",
    "water_stress_120m",
    "water_excess_60m",
    "pollination_window_60m",
    "photosynthesis_opportunity_60m",
    "soil_atmosphere_mismatch_60m",
)

ACTIVE_MODELS = (
    "naive",
    "logistic",
    "xgboost",
    "lightgbm",
    "gru",
    "tcn",
    "vanilla_transformer",
    "tft_base",
    "discipline_lgbm_sparse",
    "juxtapose_lgbm_chain",
    "juxtapose_lgbm_relation_v2",
)

EXPERIMENTAL_MODELS = (
    "tft_wilson",
    "tft_zappa",
    "tft_carey",
    "tft_fripp",
    "zappa_mmoe",
    "fripp_patchtst_lite",
    "zappa_xgb_multioutput",
    "zappa_lgbm_ranker",
    "carey_lgbm_fourier",
    "spiral_lgbm_temporal",
    "carey_xgb_fourier",
    "fripp_xgb_sparse",
    "wilson_xgb_multiscale",
    "wilson_lgbm_multiscale",
    "ancestral_lgbm_multiscale",
    "zappa_lgbm_chain_v2",
)

ARCHIVED_MODELS = (
    "carey_tcn_fourier",
    "wilson_tcn_multiscale",
)

IMPLEMENTED_MODELS = ACTIVE_MODELS + EXPERIMENTAL_MODELS + ARCHIVED_MODELS

PAPER_BENCHMARK_MODELS = (
    "lightgbm",
    "xgboost",
    "discipline_lgbm_sparse",
    "juxtapose_lgbm_chain",
    "juxtapose_lgbm_relation_v2",
    "vanilla_transformer",
    "tcn",
    "tft_base",
)

MODEL_RENAME_ALIASES = {
    "fripp_lgbm_sparse": "discipline_lgbm_sparse",
    "zappa_lgbm_chain": "juxtapose_lgbm_chain",
    "wilson_lgbm_multiscale_v2": "ancestral_lgbm_multiscale",
    "carey_lgbm_temporal_v2": "spiral_lgbm_temporal",
}

SKELETON_MODELS = (
    "minirocket",
    "inceptiontime",
    "patchtst",
    "chronos_bolt",
    "sarimax",
    "prophet",
    "vanilla_tft",
)
