"""Adaptive agronomic label thresholds and validation helpers."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import pandas as pd

_ADAPTIVE_MODES = {"fixed", "percentile", "hybrid"}
_DEFAULT_STATS_QUANTILES = (0.20, 0.30, 0.40, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95)
_DEFAULT_VALIDATION = {
    "prevalence_min": 0.03,
    "prevalence_max": 0.25,
    "stability_max_delta": 0.12,
    "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
}
_DEFAULT_ADAPTIVE_SEARCH = {"percentile_step": 0.05, "max_iterations": 18}
_ADAPTIVE_PERCENTILE_ORDER = {
    "photosynthesis_opportunity": [("vpd_min_percentile", "vpd_max_percentile")],
    "pollination_window": [
        ("temp_low_percentile", "temp_high_percentile"),
        ("rh_low_percentile", "rh_high_percentile"),
        ("vpd_low_percentile", "vpd_high_percentile"),
    ],
}
_ADAPTIVE_ALARM_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "photosynthesis_opportunity": {
        "radiation_min": {
            "threshold_name": "ppfd_good",
            "role": "radiation_proxy",
            "percentile_name": "radiation_min_percentile",
            "default_percentile": 0.75,
            "direction": "lower_bound",
            "combine": "max",
        },
        "vpd_min": {
            "threshold_name": "vpd_opt_low",
            "role": "vpd",
            "percentile_name": "vpd_min_percentile",
            "default_percentile": 0.30,
            "direction": "lower_bound",
            "combine": "max",
        },
        "vpd_max": {
            "threshold_name": "vpd_opt_high",
            "role": "vpd",
            "percentile_name": "vpd_max_percentile",
            "default_percentile": 0.90,
            "direction": "upper_bound",
            "combine": "min",
        },
        "soil_moisture_min": {
            "threshold_name": "soil_moisture_warning",
            "role": "soil_moisture",
            "percentile_name": "soil_moisture_min_percentile",
            "default_percentile": 0.40,
            "direction": "lower_bound",
            "combine": "max",
        },
        "temp_max": {
            "threshold_name": "temp_high_functional",
            "role": "air_temperature",
            "percentile_name": "temp_max_percentile",
            "default_percentile": 0.95,
            "direction": "upper_bound",
            "combine": "min",
        },
    },
    "pollination_window": {
        "temp_low": {
            "threshold_name": "temp_low_functional",
            "role": "air_temperature",
            "percentile_name": "temp_low_percentile",
            "default_percentile": 0.20,
            "direction": "lower_bound",
            "combine": "max",
        },
        "temp_high": {
            "threshold_name": "temp_high_functional",
            "role": "air_temperature",
            "percentile_name": "temp_high_percentile",
            "default_percentile": 0.90,
            "direction": "upper_bound",
            "combine": "min",
        },
        "rh_low": {
            "threshold_name": "rh_low_functional",
            "role": "relative_humidity",
            "percentile_name": "rh_low_percentile",
            "default_percentile": 0.30,
            "direction": "lower_bound",
            "combine": "max",
        },
        "rh_high": {
            "threshold_name": "rh_high_functional",
            "role": "relative_humidity",
            "percentile_name": "rh_high_percentile",
            "default_percentile": 0.90,
            "direction": "upper_bound",
            "combine": "min",
        },
        "vpd_low": {
            "threshold_name": "vpd_opt_low",
            "role": "vpd",
            "percentile_name": "vpd_low_percentile",
            "default_percentile": 0.20,
            "direction": "lower_bound",
            "combine": "max",
        },
        "vpd_high": {
            "threshold_name": "vpd_opt_high",
            "role": "vpd",
            "percentile_name": "vpd_high_percentile",
            "default_percentile": 0.80,
            "direction": "upper_bound",
            "combine": "min",
        },
        "radiation_min": {
            "threshold_name": "ppfd_min_useful",
            "role": "radiation_proxy",
            "percentile_name": "radiation_min_percentile",
            "default_percentile": 0.40,
            "direction": "lower_bound",
            "combine": "max",
        },
    },
}


def _future_matrix(series: pd.Series, horizon_steps: int) -> pd.DataFrame:
    return pd.concat([series.shift(-step) for step in range(1, horizon_steps + 1)], axis=1)


def adaptive_mode(rule: dict[str, Any], alarm_name: str) -> str:
    adaptive_cfg = rule.get("adaptive", {})
    options_cfg = rule.get("options", {})
    raw_mode = adaptive_cfg.get("mode", options_cfg.get("threshold_mode", options_cfg.get("adaptive_mode", "fixed")))
    mode = str(raw_mode).strip().lower()
    if mode not in _ADAPTIVE_MODES:
        raise ValueError(
            f"Task '{rule.get('alarm_name', alarm_name)}' configured unsupported adaptive mode '{raw_mode}'."
        )
    if alarm_name not in _ADAPTIVE_ALARM_SPECS:
        return "fixed"
    return mode


def percentile_label(value: float) -> str:
    return f"p{int(round(float(value) * 100)):02d}"


def clamp_percentile(value: float) -> float:
    return float(min(0.99, max(0.01, value)))


def _merge_nested(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(merged[key], value)
        else:
            merged[key] = value
    return merged


def internal_validation_config(rule: dict[str, Any]) -> dict[str, Any]:
    adaptive_cfg = rule.get("adaptive", {})
    override = adaptive_cfg.get("validation", {})
    config = _merge_nested(_DEFAULT_VALIDATION, override if isinstance(override, dict) else {})
    split_ratios = config.get("split_ratios", {})
    train_ratio = float(split_ratios.get("train", 0.6))
    val_ratio = float(split_ratios.get("val", 0.2))
    test_ratio = float(split_ratios.get("test", 0.2))
    total = train_ratio + val_ratio + test_ratio
    if total <= 0.0:
        raise ValueError("Adaptive validation split ratios must add up to a positive value.")
    config["split_ratios"] = {
        "train": train_ratio / total,
        "val": val_ratio / total,
        "test": test_ratio / total,
    }
    config["prevalence_min"] = float(config["prevalence_min"])
    config["prevalence_max"] = float(config["prevalence_max"])
    config["stability_max_delta"] = float(config["stability_max_delta"])
    return config


def adaptive_search_config(rule: dict[str, Any]) -> dict[str, Any]:
    adaptive_cfg = rule.get("adaptive", {})
    override = adaptive_cfg.get("search", {})
    config = dict(_DEFAULT_ADAPTIVE_SEARCH)
    if isinstance(override, dict):
        config.update(override)
    config["percentile_step"] = float(config["percentile_step"])
    config["max_iterations"] = int(config["max_iterations"])
    return config


def percentile_lookup(
    series: pd.Series,
    *,
    task_name: str,
    role: str,
    quantile: float,
    cache: dict[tuple[str, float], float],
) -> float:
    quantile = clamp_percentile(quantile)
    key = (role, round(quantile, 6))
    if key not in cache:
        usable = series.dropna()
        if usable.empty:
            raise ValueError(
                f"Task '{task_name}' cannot compute percentile {percentile_label(quantile)} for '{role}' "
                "because the reference series has no usable values."
            )
        cache[key] = float(usable.quantile(quantile))
    return cache[key]


def collect_dataset_statistics(
    *,
    task_name: str,
    roles: Iterable[str],
    signal_lookup: Callable[[str], pd.Series],
    resolved_columns: dict[str, str],
    quantile_cache: dict[tuple[str, float], float],
    quantiles: Iterable[float],
) -> dict[str, Any]:
    statistics: dict[str, Any] = {}
    unique_quantiles = sorted({clamp_percentile(float(value)) for value in quantiles})
    for role in roles:
        series = signal_lookup(role)
        usable = series.dropna()
        if usable.empty:
            continue
        statistics[role] = {
            "column": resolved_columns.get(role),
            "count": int(usable.shape[0]),
            "mean": float(usable.mean()),
            "std": float(usable.std(ddof=0)) if usable.shape[0] > 1 else 0.0,
            "quantiles": {
                percentile_label(quantile): percentile_lookup(
                    series,
                    task_name=task_name,
                    role=role,
                    quantile=quantile,
                    cache=quantile_cache,
                )
                for quantile in unique_quantiles
            },
        }
    return statistics


def evaluate_target_distribution(
    target: pd.Series,
    *,
    validation_cfg: dict[str, Any],
    embargo_steps: int,
) -> dict[str, Any]:
    usable = target.astype(float).reset_index(drop=True)
    n_rows = int(len(usable))
    split_ratios = validation_cfg["split_ratios"]
    evaluation = {
        "overall_positive_rate": float(usable.mean()) if n_rows > 0 else 0.0,
        "split_positive_rates": {},
        "split_counts": {},
        "degenerate_splits": {},
        "embargo_steps": int(max(0, embargo_steps)),
    }
    if n_rows == 0:
        evaluation["stability_range"] = None
        evaluation["prevalence_ok"] = False
        evaluation["stable_ok"] = False
        evaluation["degenerate_any_split"] = True
        evaluation["passes"] = False
        return evaluation

    split_labels = pd.Series(["dropped"] * n_rows)
    usable_rows = n_rows - (2 * max(0, embargo_steps))
    if usable_rows < 9:
        thirds = max(1, n_rows // 3)
        split_labels.iloc[:thirds] = "train"
        split_labels.iloc[thirds : min(n_rows, 2 * thirds)] = "val"
        split_labels.iloc[min(n_rows, 2 * thirds) :] = "test"
    else:
        train_n = max(1, int(usable_rows * float(split_ratios["train"])))
        val_n = max(1, int(usable_rows * float(split_ratios["val"])))
        test_n = usable_rows - train_n - val_n
        if test_n < 1:
            test_n = 1
            val_n = max(1, usable_rows - train_n - test_n)
        train_end = train_n - 1
        val_start = train_end + embargo_steps + 1
        val_end = min(n_rows - 1, val_start + val_n - 1)
        test_start = min(n_rows, val_end + embargo_steps + 1)
        split_labels.iloc[: train_end + 1] = "train"
        split_labels.iloc[val_start : val_end + 1] = "val"
        split_labels.iloc[test_start:] = "test"

    stable_rates: list[float] = []
    degenerate_any = False
    for split_name in ("train", "val", "test"):
        split_values = usable[split_labels == split_name]
        evaluation["split_counts"][split_name] = int(len(split_values))
        if split_values.empty:
            evaluation["split_positive_rates"][split_name] = None
            evaluation["degenerate_splits"][split_name] = True
            degenerate_any = True
            continue
        positive_rate = float(split_values.mean())
        evaluation["split_positive_rates"][split_name] = positive_rate
        degenerate = bool(split_values.nunique(dropna=True) < 2)
        evaluation["degenerate_splits"][split_name] = degenerate
        degenerate_any = degenerate_any or degenerate
        stable_rates.append(positive_rate)

    stability_range = max(stable_rates) - min(stable_rates) if stable_rates else None
    evaluation["stability_range"] = float(stability_range) if stability_range is not None else None
    evaluation["degenerate_any_split"] = degenerate_any
    evaluation["prevalence_ok"] = bool(
        validation_cfg["prevalence_min"]
        <= evaluation["overall_positive_rate"]
        <= validation_cfg["prevalence_max"]
    )
    evaluation["stable_ok"] = bool(
        stability_range is not None and stability_range <= validation_cfg["stability_max_delta"]
    )
    evaluation["passes"] = bool(
        evaluation["prevalence_ok"] and evaluation["stable_ok"] and not evaluation["degenerate_any_split"]
    )
    return evaluation


def adaptive_score(
    *,
    validation: dict[str, Any],
    validation_cfg: dict[str, Any],
    selected_percentiles: dict[str, float],
    requested_percentiles: dict[str, float],
) -> float:
    target_rate = (validation_cfg["prevalence_min"] + validation_cfg["prevalence_max"]) / 2.0
    overall = float(validation["overall_positive_rate"])
    prevalence_penalty = max(0.0, validation_cfg["prevalence_min"] - overall) + max(
        0.0, overall - validation_cfg["prevalence_max"]
    )
    stability_range = validation.get("stability_range")
    stability_penalty = 0.0
    if stability_range is None:
        stability_penalty += 1.0
    else:
        stability_penalty += max(0.0, float(stability_range) - validation_cfg["stability_max_delta"])
    degeneracy_penalty = sum(1 for value in validation["degenerate_splits"].values() if value)
    anchor_penalty = float(
        sum(abs(selected_percentiles[key] - requested_percentiles[key]) for key in requested_percentiles)
        / max(1, len(requested_percentiles))
    )
    centered_penalty = abs(overall - target_rate)
    return (
        (1000.0 * degeneracy_penalty)
        + (400.0 * prevalence_penalty)
        + (150.0 * stability_penalty)
        + (15.0 * centered_penalty)
        + anchor_penalty
    )


def enforce_percentile_order(
    alarm_name: str,
    selected_percentiles: dict[str, float],
    *,
    minimum_gap: float = 0.02,
) -> dict[str, float]:
    updated = {key: clamp_percentile(value) for key, value in selected_percentiles.items()}
    for low_key, high_key in _ADAPTIVE_PERCENTILE_ORDER.get(alarm_name, []):
        low_value = updated[low_key]
        high_value = updated[high_key]
        if high_value <= low_value:
            midpoint = (low_value + high_value) / 2.0
            low_value = clamp_percentile(midpoint - (minimum_gap / 2.0))
            high_value = clamp_percentile(midpoint + (minimum_gap / 2.0))
            if high_value <= low_value:
                high_value = clamp_percentile(low_value + minimum_gap)
            if high_value <= low_value:
                low_value = clamp_percentile(high_value - minimum_gap)
            updated[low_key] = low_value
            updated[high_key] = high_value
    return updated


def resolve_adaptive_alarm(
    *,
    task_name: str,
    alarm_name: str,
    rule: dict[str, Any],
    thresholds_cfg: dict[str, Any],
    adaptive_mode_value: str,
    horizon_steps: int,
    window_size_steps: int,
    signal_lookup: Callable[[str], pd.Series],
    fixed_threshold_lookup: Callable[[str, str], float],
    build_mask: Callable[[dict[str, float], bool], pd.Series],
    resolved_columns: dict[str, str],
) -> dict[str, Any]:
    alarm_specs = _ADAPTIVE_ALARM_SPECS[alarm_name]
    validation_cfg = internal_validation_config(rule)
    search_cfg = adaptive_search_config(rule)
    quantile_cache: dict[tuple[str, float], float] = {}

    relevant_roles = sorted({spec["role"] for spec in alarm_specs.values()})
    requested_percentiles = {
        spec["percentile_name"]: clamp_percentile(
            float(thresholds_cfg.get(spec["percentile_name"], spec["default_percentile"]))
        )
        for spec in alarm_specs.values()
    }
    requested_percentiles = enforce_percentile_order(alarm_name, requested_percentiles)

    def adaptive_thresholds(
        selected_percentiles: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        candidate_thresholds: dict[str, float] = {}
        threshold_sources: dict[str, dict[str, Any]] = {}
        for spec_name, spec in alarm_specs.items():
            fixed_value = fixed_threshold_lookup(spec["threshold_name"], spec["role"])
            percentile_name = spec["percentile_name"]
            percentile_value = selected_percentiles[percentile_name]
            percentile_threshold = percentile_lookup(
                signal_lookup(spec["role"]),
                task_name=task_name,
                role=spec["role"],
                quantile=percentile_value,
                cache=quantile_cache,
            )
            if adaptive_mode_value == "percentile":
                final_threshold = percentile_threshold
            elif spec["combine"] == "max":
                final_threshold = max(fixed_value, percentile_threshold)
            else:
                final_threshold = min(fixed_value, percentile_threshold)
            candidate_thresholds[spec["threshold_name"]] = float(final_threshold)
            threshold_sources[spec["threshold_name"]] = {
                "spec_name": spec_name,
                "mode": adaptive_mode_value,
                "combine": spec["combine"],
                "role": spec["role"],
                "fixed_value": float(fixed_value),
                "percentile_name": percentile_name,
                "percentile_value": float(percentile_value),
                "percentile_threshold": float(percentile_threshold),
                "final_threshold": float(final_threshold),
            }
        return candidate_thresholds, threshold_sources

    def evaluate_percentiles(selected_percentiles: dict[str, float]) -> dict[str, Any]:
        selected_percentiles = enforce_percentile_order(alarm_name, selected_percentiles)
        candidate_thresholds, threshold_sources = adaptive_thresholds(selected_percentiles)
        mask = build_mask(candidate_thresholds, True)
        target = (_future_matrix(mask.astype(float), horizon_steps) > 0.0).any(axis=1).astype("float")
        validation = evaluate_target_distribution(
            target,
            validation_cfg=validation_cfg,
            embargo_steps=max(horizon_steps, int(window_size_steps)),
        )
        return {
            "selected_percentiles": selected_percentiles,
            "thresholds": candidate_thresholds,
            "threshold_sources": threshold_sources,
            "mask": mask,
            "target": target,
            "validation": validation,
            "score": adaptive_score(
                validation=validation,
                validation_cfg=validation_cfg,
                selected_percentiles=selected_percentiles,
                requested_percentiles=requested_percentiles,
            ),
        }

    current = evaluate_percentiles(requested_percentiles)
    best = current
    visited = {tuple(round(value, 6) for value in current["selected_percentiles"].values())}
    percentile_step = float(search_cfg["percentile_step"])
    max_iterations = int(search_cfg["max_iterations"])

    def shifted_percentiles(
        base_percentiles: dict[str, float],
        *,
        target_spec: str | None,
        direction_name: str,
    ) -> dict[str, float]:
        updated = dict(base_percentiles)
        target_specs = [target_spec] if target_spec is not None else list(alarm_specs.keys())
        for spec_name in target_specs:
            spec = alarm_specs[spec_name]
            percentile_name = spec["percentile_name"]
            if spec["direction"] == "lower_bound":
                delta = -percentile_step if direction_name == "relax" else percentile_step
            else:
                delta = percentile_step if direction_name == "relax" else -percentile_step
            updated[percentile_name] = clamp_percentile(updated[percentile_name] + delta)
        return enforce_percentile_order(alarm_name, updated)

    for _ in range(max_iterations):
        if best["validation"]["passes"]:
            break
        candidates: list[dict[str, Any]] = []
        for direction_name in ("relax", "tighten"):
            for target_spec in [None, *alarm_specs.keys()]:
                candidate_percentiles = shifted_percentiles(
                    best["selected_percentiles"],
                    target_spec=target_spec,
                    direction_name=direction_name,
                )
                key = tuple(round(candidate_percentiles[name], 6) for name in sorted(candidate_percentiles))
                if key in visited:
                    continue
                visited.add(key)
                candidates.append(evaluate_percentiles(candidate_percentiles))
        if not candidates:
            break
        candidate_best = min(candidates, key=lambda payload: payload["score"])
        if candidate_best["score"] > best["score"] + 1e-9:
            break
        best = candidate_best

    stats_quantiles = set(_DEFAULT_STATS_QUANTILES)
    stats_quantiles.update(requested_percentiles.values())
    stats_quantiles.update(best["selected_percentiles"].values())
    dataset_statistics = collect_dataset_statistics(
        task_name=task_name,
        roles=relevant_roles,
        signal_lookup=signal_lookup,
        resolved_columns=resolved_columns,
        quantile_cache=quantile_cache,
        quantiles=stats_quantiles,
    )
    return {
        "mask": best["mask"],
        "resolved_thresholds": best["thresholds"],
        "adaptive_adjusted": bool(best["selected_percentiles"] != requested_percentiles),
        "adaptive_requested_percentiles": requested_percentiles,
        "adaptive_selected_percentiles": best["selected_percentiles"],
        "adaptive_validation": best["validation"],
        "adaptive_threshold_sources": best["threshold_sources"],
        "dataset_statistics": dataset_statistics,
    }
