"""Task label generation from YAML rules."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .agronomic_adaptive import adaptive_mode, resolve_adaptive_alarm

_BINARY_CLASS_NAMES = ["0", "1"]

_AGRONOMIC_COLUMN_ALIASES: dict[str, list[str]] = {
    "soil_moisture": ["hum_suelo", "soil_moisture"],
    "air_temperature": ["temp_aire_media", "air_temperature"],
    "relative_humidity": ["hum_aire_media", "relative_humidity"],
    "vpd": ["vpd_media", "vpd"],
    "radiation_primary": ["ppfd", "radiation_proxy"],
    "radiation_fallback": ["lux", "light_lux"],
    "soil_temperature": ["temp_suelo", "soil_temperature"],
    "outside_precipitation": [
        "outside_openmeteo_precipitation",
        "outside_nasa_PRECTOTCORR",
        "outside_fused_precipitation_mean",
    ],
    "outside_precipitation_rolling_6h": [
        "outside_openmeteo_precipitation_rolling_sum_6h",
        "outside_nasa_precipitation_rolling_sum_6h",
        "outside_fused_precipitation_rolling_sum_6h_mean",
    ],
    "outside_precipitation_rolling_24h": [
        "outside_openmeteo_precipitation_rolling_sum_24h",
        "outside_nasa_precipitation_rolling_sum_24h",
        "outside_fused_precipitation_rolling_sum_24h_mean",
    ],
    "outside_wind_speed": [
        "outside_openmeteo_wind_speed_10m",
        "outside_nasa_WS10M",
        "outside_nasa_WS2M",
        "outside_fused_wind_speed_10m_mean",
    ],
    "outside_radiation": [
        "outside_openmeteo_shortwave_radiation",
        "outside_nasa_ALLSKY_SFC_SW_DWN",
        "outside_fused_shortwave_radiation_mean",
    ],
    "outside_radiation_rolling_6h": [
        "outside_openmeteo_radiation_rolling_sum_6h",
        "outside_nasa_radiation_rolling_sum_6h",
        "outside_fused_radiation_rolling_sum_6h_mean",
    ],
    "outside_daylight_flag": [
        "outside_openmeteo_daylight_flag",
        "outside_nasa_daylight_flag",
        "outside_fused_daylight_flag_max",
    ],
}


def _future_matrix(series: pd.Series, horizon_steps: int) -> pd.DataFrame:
    columns = [series.shift(-step) for step in range(1, horizon_steps + 1)]
    return pd.concat(columns, axis=1)


def _unique_preserve_order(values: Iterable[str | None]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        candidate = str(value)
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def _threshold_series(
    signal_cache: dict[str, pd.Series],
    role: str,
) -> pd.Series:
    if role not in signal_cache:
        raise ValueError(f"Threshold role '{role}' was requested before its signal was resolved.")
    return signal_cache[role]


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _find_usable_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate not in df.columns:
            continue
        if _coerce_numeric(df[candidate]).notna().sum() > 0:
            return candidate
    return None


def _missing_column_error(
    *,
    task_name: str,
    role: str,
    candidates: list[str],
) -> ValueError:
    expected = ", ".join(f"'{candidate}'" for candidate in candidates)
    return ValueError(
        f"Task '{task_name}' requires a usable column for '{role}'. "
        f"Tried: {expected}. The canonical dataset does not expose any of them with numeric values."
    )


def _resolve_signal(
    df: pd.DataFrame,
    *,
    task_name: str,
    role: str,
    columns_cfg: dict[str, Any],
) -> tuple[pd.Series, str, bool]:
    if role == "radiation_proxy":
        primary_candidates = _unique_preserve_order(
            [
                columns_cfg.get("radiation_primary"),
                columns_cfg.get("radiation_proxy"),
                *_AGRONOMIC_COLUMN_ALIASES["radiation_primary"],
            ]
        )
        fallback_candidates = _unique_preserve_order(
            [
                columns_cfg.get("radiation_fallback"),
                *_AGRONOMIC_COLUMN_ALIASES["radiation_fallback"],
            ]
        )
        primary_column = _find_usable_column(df, primary_candidates)
        if primary_column is not None:
            return _coerce_numeric(df[primary_column]), primary_column, False
        fallback_column = _find_usable_column(df, fallback_candidates)
        if fallback_column is not None:
            return _coerce_numeric(df[fallback_column]), fallback_column, True
        raise _missing_column_error(
            task_name=task_name,
            role=role,
            candidates=[*primary_candidates, *fallback_candidates],
        )

    configured = columns_cfg.get(role)
    configured_candidates = configured if isinstance(configured, list) else [configured]
    candidates = _unique_preserve_order([*configured_candidates, *_AGRONOMIC_COLUMN_ALIASES.get(role, [])])
    resolved_column = _find_usable_column(df, candidates)
    if resolved_column is None:
        raise _missing_column_error(task_name=task_name, role=role, candidates=candidates)
    return _coerce_numeric(df[resolved_column]), resolved_column, False


def _resolve_threshold(
    raw_value: Any,
    *,
    task_name: str,
    threshold_name: str,
    reference_series: pd.Series,
) -> float:
    if isinstance(raw_value, dict):
        mode = str(raw_value.get("mode", "value")).lower()
        if mode == "quantile":
            quantile = float(raw_value["quantile"])
            if not 0.0 <= quantile <= 1.0:
                raise ValueError(
                    f"Task '{task_name}' configured threshold '{threshold_name}' with invalid quantile {quantile}."
                )
            usable = reference_series.dropna()
            if usable.empty:
                raise ValueError(
                    f"Task '{task_name}' cannot resolve quantile threshold '{threshold_name}' "
                    "because the reference series has no usable values."
                )
            return float(usable.quantile(quantile))
        if mode == "value":
            return float(raw_value["value"])
        raise ValueError(
            f"Task '{task_name}' configured unsupported threshold mode '{mode}' for '{threshold_name}'."
        )
    if raw_value is None:
        raise ValueError(f"Task '{task_name}' is missing required threshold '{threshold_name}' in YAML.")
    return float(raw_value)


def _resolve_radiation_threshold(
    *,
    task_name: str,
    threshold_name: str,
    raw_value: Any,
    radiation_column: str,
    reference_series: pd.Series,
    thresholds_cfg: dict[str, Any],
) -> float:
    resolved = _resolve_threshold(
        raw_value,
        task_name=task_name,
        threshold_name=threshold_name,
        reference_series=reference_series,
    )
    if radiation_column in {"lux", "light_lux"}:
        lux_per_ppfd = thresholds_cfg.get("lux_per_ppfd")
        if lux_per_ppfd is None:
            raise ValueError(
                f"Task '{task_name}' fell back to '{radiation_column}', but YAML does not define 'lux_per_ppfd'."
            )
        return float(resolved * float(lux_per_ppfd))
    return resolved


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    if window < 2:
        raise ValueError("Rolling slope requires window >= 2.")

    def _slope(values: np.ndarray) -> float:
        array = np.asarray(values, dtype=float)
        mask = np.isfinite(array)
        if mask.sum() < 2:
            return np.nan
        y = array[mask]
        x = np.arange(len(array), dtype=float)[mask]
        x_centered = x - x.mean()
        denominator = float(np.dot(x_centered, x_centered))
        if denominator <= 0.0:
            return 0.0
        return float(np.dot(x_centered, y - y.mean()) / denominator)

    return _coerce_numeric(series).rolling(window=window, min_periods=2).apply(_slope, raw=True)


def _agronomic_alarm_mask(
    df: pd.DataFrame,
    *,
    task_name: str,
    rule: dict[str, Any],
    horizon_steps: int,
    window_size_steps: int,
) -> tuple[pd.Series, dict[str, Any]]:
    alarm_name = str(rule.get("alarm_name", "")).strip()
    if not alarm_name:
        raise ValueError(f"Task '{task_name}' requires rule.alarm_name for agronomic alarms.")

    columns_cfg = rule.get("columns", {})
    thresholds_cfg = rule.get("thresholds", {})
    options_cfg = rule.get("options", {})
    adaptive_mode_value = adaptive_mode(rule, alarm_name)

    signal_cache: dict[str, pd.Series] = {}
    resolved_columns: dict[str, str] = {}
    resolved_thresholds: dict[str, float] = {}
    radiation_fallback_used = False

    def signal(role: str) -> pd.Series:
        nonlocal radiation_fallback_used
        if role not in signal_cache:
            resolved_series, resolved_column, used_fallback = _resolve_signal(
                df,
                task_name=task_name,
                role=role,
                columns_cfg=columns_cfg,
            )
            signal_cache[role] = resolved_series
            resolved_columns[role] = resolved_column
            if role == "radiation_proxy":
                radiation_fallback_used = used_fallback
        return signal_cache[role]

    def multi_signal_pairs(role: str) -> list[tuple[pd.Series, str]]:
        configured = columns_cfg.get(role)
        configured_candidates = configured if isinstance(configured, list) else [configured]
        candidates = _unique_preserve_order([*configured_candidates, *_AGRONOMIC_COLUMN_ALIASES.get(role, [])])
        resolved: list[tuple[pd.Series, str]] = []
        for candidate in candidates:
            if candidate not in df.columns:
                continue
            series = _coerce_numeric(df[candidate])
            if series.notna().sum() == 0:
                continue
            resolved.append((series, candidate))
            resolved_columns[f"{role}__{len(resolved)}"] = candidate
        if not resolved:
            raise _missing_column_error(task_name=task_name, role=role, candidates=candidates)
        return resolved

    def multi_signal(role: str) -> list[pd.Series]:
        return [series for series, _ in multi_signal_pairs(role)]

    def any_threshold(role: str, operator: str, value: float) -> pd.Series:
        masks = []
        for series in multi_signal(role):
            if operator == ">=":
                masks.append(series >= value)
            elif operator == "<=":
                masks.append(series <= value)
            else:
                raise ValueError(f"Unsupported threshold operator: {operator}")
        return pd.concat(masks, axis=1).any(axis=1)

    def any_column_threshold(role: str, operator: str, default_value: float, by_column: dict[str, Any]) -> pd.Series:
        masks = []
        for series, column_name in multi_signal_pairs(role):
            value = float(by_column.get(column_name, default_value))
            if operator == ">=":
                masks.append(series >= value)
            elif operator == "<=":
                masks.append(series <= value)
            else:
                raise ValueError(f"Unsupported threshold operator: {operator}")
        return pd.concat(masks, axis=1).any(axis=1)

    def threshold(name: str, reference_role: str) -> float:
        signal(reference_role)
        if name not in resolved_thresholds:
            raw_value = thresholds_cfg.get(name)
            if reference_role == "radiation_proxy":
                resolved_thresholds[name] = _resolve_radiation_threshold(
                    task_name=task_name,
                    threshold_name=name,
                    raw_value=raw_value,
                    radiation_column=resolved_columns["radiation_proxy"],
                    reference_series=_threshold_series(signal_cache, reference_role),
                    thresholds_cfg=thresholds_cfg,
                )
            else:
                resolved_thresholds[name] = _resolve_threshold(
                    raw_value,
                    task_name=task_name,
                    threshold_name=name,
                    reference_series=_threshold_series(signal_cache, reference_role),
                )
        return resolved_thresholds[name]

    slope_window = int(options_cfg.get("soil_slope_window_steps", 3))

    def build_mask_from_thresholds(
        alarm_thresholds: dict[str, float],
        force_temperature_cap: bool = False,
    ) -> pd.Series:
        if alarm_name == "water_stress":
            soil = signal("soil_moisture")
            vpd = signal("vpd")
            air_temperature = signal("air_temperature")
            dryness = soil <= alarm_thresholds["soil_moisture_low"]
            atmospheric_demand = (
                (vpd >= alarm_thresholds["vpd_high"])
                | (air_temperature >= alarm_thresholds["temp_high_functional"])
            )
            mask = dryness & atmospheric_demand
            if bool(options_cfg.get("require_negative_soil_moisture_slope", False)):
                mask = mask & (_rolling_slope(soil, slope_window) < 0.0)
            return mask.fillna(False)

        if alarm_name == "water_excess":
            soil = signal("soil_moisture")
            vpd = signal("vpd")
            mask = (soil >= alarm_thresholds["soil_moisture_high"]) & (vpd <= alarm_thresholds["vpd_low"])
            if bool(options_cfg.get("require_high_relative_humidity", False)):
                relative_humidity = signal("relative_humidity")
                mask = mask & (relative_humidity >= alarm_thresholds["rh_high_functional"])
            return mask.fillna(False)

        if alarm_name == "pollination_window":
            air_temperature = signal("air_temperature")
            relative_humidity = signal("relative_humidity")
            vpd = signal("vpd")
            radiation = signal("radiation_proxy")
            return (
                air_temperature.between(
                    alarm_thresholds["temp_low_functional"],
                    alarm_thresholds["temp_high_functional"],
                    inclusive="both",
                )
                & relative_humidity.between(
                    alarm_thresholds["rh_low_functional"],
                    alarm_thresholds["rh_high_functional"],
                    inclusive="both",
                )
                & vpd.between(
                    alarm_thresholds["vpd_opt_low"],
                    alarm_thresholds["vpd_opt_high"],
                    inclusive="both",
                )
                & (radiation >= alarm_thresholds["ppfd_min_useful"])
            ).fillna(False)

        if alarm_name == "pollination_window_expert":
            air_temperature = signal("air_temperature")
            relative_humidity = signal("relative_humidity")
            radiation = signal("radiation_proxy")
            mask = (
                air_temperature.between(
                    alarm_thresholds["temp_low_functional"],
                    alarm_thresholds["temp_high_functional"],
                    inclusive="both",
                )
                & (relative_humidity >= alarm_thresholds["rh_pollination_min"])
                & (relative_humidity > alarm_thresholds["rh_pollination_block"])
                & (radiation >= alarm_thresholds["ppfd_min_useful"])
            )
            if bool(options_cfg.get("require_vpd_range", False)):
                vpd = signal("vpd")
                mask = mask & vpd.between(
                    alarm_thresholds["vpd_opt_low"],
                    alarm_thresholds["vpd_opt_high"],
                    inclusive="both",
                )
            return mask.fillna(False)

        if alarm_name == "photosynthesis_opportunity":
            soil = signal("soil_moisture")
            vpd = signal("vpd")
            radiation = signal("radiation_proxy")
            mask = (
                (radiation >= alarm_thresholds["ppfd_good"])
                & vpd.between(
                    alarm_thresholds["vpd_opt_low"],
                    alarm_thresholds["vpd_opt_high"],
                    inclusive="both",
                )
                & (soil >= alarm_thresholds["soil_moisture_warning"])
            )
            if force_temperature_cap or bool(options_cfg.get("require_temperature_cap", False)):
                air_temperature = signal("air_temperature")
                mask = mask & (air_temperature <= alarm_thresholds["temp_high_functional"])
            return mask.fillna(False)

        if alarm_name == "soil_atmosphere_mismatch":
            soil = signal("soil_moisture")
            vpd = signal("vpd")
            radiation = signal("radiation_proxy")
            negative_slope = _rolling_slope(soil, slope_window) < 0.0
            return (
                (vpd >= alarm_thresholds["vpd_stress"])
                & (radiation >= alarm_thresholds["ppfd_high"])
                & ((soil <= alarm_thresholds["soil_moisture_warning"]) | negative_slope)
            ).fillna(False)

        if alarm_name == "heat_water_stress":
            air_temperature = signal("air_temperature")
            mask = air_temperature.between(
                alarm_thresholds["temp_stress_low"],
                alarm_thresholds["temp_stress_high"],
                inclusive="both",
            )
            if bool(options_cfg.get("require_low_relative_humidity", False)):
                relative_humidity = signal("relative_humidity")
                mask = mask & (relative_humidity <= alarm_thresholds["rh_low_functional"])
            if bool(options_cfg.get("require_high_vpd", False)):
                vpd = signal("vpd")
                mask = mask & (vpd >= alarm_thresholds["vpd_high"])
            return mask.fillna(False)

        if alarm_name == "water_saturation_stress":
            soil = signal("soil_moisture")
            return (soil > alarm_thresholds["soil_moisture_saturation"]).fillna(False)

        if alarm_name == "rain_risk":
            masks = [
                any_threshold("outside_precipitation", ">=", alarm_thresholds["rain_current_min"]),
                any_threshold("outside_precipitation_rolling_6h", ">=", alarm_thresholds["rain_rolling_6h_min"]),
            ]
            if bool(options_cfg.get("include_rolling_24h", True)):
                masks.append(
                    any_threshold("outside_precipitation_rolling_24h", ">=", alarm_thresholds["rain_rolling_24h_min"])
                )
            return pd.concat(masks, axis=1).any(axis=1).fillna(False)

        if alarm_name == "wind_risk":
            return any_column_threshold(
                "outside_wind_speed",
                ">=",
                alarm_thresholds["wind_speed_high"],
                thresholds_cfg.get("wind_speed_high_by_column", {}),
            ).fillna(False)

        if alarm_name == "cloudy_low_radiation":
            low_radiation = any_threshold("outside_radiation", "<=", alarm_thresholds["outside_radiation_low"])
            daylight = any_threshold("outside_daylight_flag", ">=", alarm_thresholds["outside_daylight_min"])
            mask = low_radiation & daylight
            if bool(options_cfg.get("include_rolling_6h", True)):
                mask = mask | any_threshold(
                    "outside_radiation_rolling_6h",
                    "<=",
                    alarm_thresholds["outside_radiation_rolling_6h_low"],
                )
            return mask.fillna(False)

        if alarm_name == "high_radiation_stress":
            high_radiation = any_threshold("outside_radiation", ">=", alarm_thresholds["outside_radiation_high"])
            air_temperature = signal("air_temperature")
            mask = high_radiation & (air_temperature >= alarm_thresholds["temp_high_functional"])
            if bool(options_cfg.get("include_vpd_stress", True)):
                vpd = signal("vpd")
                mask = mask | (high_radiation & (vpd >= alarm_thresholds["vpd_stress"]))
            return mask.fillna(False)

        raise ValueError(f"Unsupported agronomic alarm '{alarm_name}' in task '{task_name}'.")

    if adaptive_mode_value != "fixed":
        adaptive_payload = resolve_adaptive_alarm(
            task_name=task_name,
            alarm_name=alarm_name,
            rule=rule,
            thresholds_cfg=thresholds_cfg,
            adaptive_mode_value=adaptive_mode_value,
            horizon_steps=horizon_steps,
            window_size_steps=window_size_steps,
            signal_lookup=signal,
            fixed_threshold_lookup=threshold,
            build_mask=build_mask_from_thresholds,
            resolved_columns=resolved_columns,
        )
        metadata = {
            "alarm_name": alarm_name,
            "adaptive_mode": adaptive_mode_value,
            **{key: value for key, value in adaptive_payload.items() if key != "mask"},
            "resolved_columns": resolved_columns,
            "required_columns": sorted(resolved_columns.values()),
            "radiation_fallback_used": radiation_fallback_used,
            "radiation_column": resolved_columns.get("radiation_proxy"),
            "soil_slope_window_steps": slope_window,
        }
        return adaptive_payload["mask"], metadata

    if alarm_name == "water_stress":
        fixed_thresholds = {
            "soil_moisture_low": threshold("soil_moisture_low", "soil_moisture"),
            "vpd_high": threshold("vpd_high", "vpd"),
            "temp_high_functional": threshold("temp_high_functional", "air_temperature"),
        }
    elif alarm_name == "water_excess":
        fixed_thresholds = {
            "soil_moisture_high": threshold("soil_moisture_high", "soil_moisture"),
            "vpd_low": threshold("vpd_low", "vpd"),
        }
        if bool(options_cfg.get("require_high_relative_humidity", False)):
            fixed_thresholds["rh_high_functional"] = threshold("rh_high_functional", "relative_humidity")
    elif alarm_name == "pollination_window":
        fixed_thresholds = {
            "temp_low_functional": threshold("temp_low_functional", "air_temperature"),
            "temp_high_functional": threshold("temp_high_functional", "air_temperature"),
            "rh_low_functional": threshold("rh_low_functional", "relative_humidity"),
            "rh_high_functional": threshold("rh_high_functional", "relative_humidity"),
            "vpd_opt_low": threshold("vpd_opt_low", "vpd"),
            "vpd_opt_high": threshold("vpd_opt_high", "vpd"),
            "ppfd_min_useful": threshold("ppfd_min_useful", "radiation_proxy"),
        }
    elif alarm_name == "pollination_window_expert":
        fixed_thresholds = {
            "temp_low_functional": threshold("temp_low_functional", "air_temperature"),
            "temp_high_functional": threshold("temp_high_functional", "air_temperature"),
            "rh_pollination_min": threshold("rh_pollination_min", "relative_humidity"),
            "rh_pollination_block": threshold("rh_pollination_block", "relative_humidity"),
            "ppfd_min_useful": threshold("ppfd_min_useful", "radiation_proxy"),
        }
        if bool(options_cfg.get("require_vpd_range", False)):
            fixed_thresholds["vpd_opt_low"] = threshold("vpd_opt_low", "vpd")
            fixed_thresholds["vpd_opt_high"] = threshold("vpd_opt_high", "vpd")
    elif alarm_name == "photosynthesis_opportunity":
        fixed_thresholds = {
            "ppfd_good": threshold("ppfd_good", "radiation_proxy"),
            "vpd_opt_low": threshold("vpd_opt_low", "vpd"),
            "vpd_opt_high": threshold("vpd_opt_high", "vpd"),
            "soil_moisture_warning": threshold("soil_moisture_warning", "soil_moisture"),
        }
        if bool(options_cfg.get("require_temperature_cap", False)):
            fixed_thresholds["temp_high_functional"] = threshold("temp_high_functional", "air_temperature")
    elif alarm_name == "soil_atmosphere_mismatch":
        fixed_thresholds = {
            "vpd_stress": threshold("vpd_stress", "vpd"),
            "ppfd_high": threshold("ppfd_high", "radiation_proxy"),
            "soil_moisture_warning": threshold("soil_moisture_warning", "soil_moisture"),
        }
    elif alarm_name == "heat_water_stress":
        fixed_thresholds = {
            "temp_stress_low": threshold("temp_stress_low", "air_temperature"),
            "temp_stress_high": threshold("temp_stress_high", "air_temperature"),
        }
        if bool(options_cfg.get("require_low_relative_humidity", False)):
            fixed_thresholds["rh_low_functional"] = threshold("rh_low_functional", "relative_humidity")
        if bool(options_cfg.get("require_high_vpd", False)):
            fixed_thresholds["vpd_high"] = threshold("vpd_high", "vpd")
    elif alarm_name == "water_saturation_stress":
        fixed_thresholds = {
            "soil_moisture_saturation": threshold("soil_moisture_saturation", "soil_moisture"),
        }
    elif alarm_name == "rain_risk":
        fixed_thresholds = {
            "rain_current_min": threshold("rain_current_min", "outside_precipitation"),
            "rain_rolling_6h_min": threshold("rain_rolling_6h_min", "outside_precipitation_rolling_6h"),
        }
        if bool(options_cfg.get("include_rolling_24h", True)):
            fixed_thresholds["rain_rolling_24h_min"] = threshold(
                "rain_rolling_24h_min",
                "outside_precipitation_rolling_24h",
            )
    elif alarm_name == "wind_risk":
        fixed_thresholds = {
            "wind_speed_high": threshold("wind_speed_high", "outside_wind_speed"),
        }
    elif alarm_name == "cloudy_low_radiation":
        fixed_thresholds = {
            "outside_radiation_low": threshold("outside_radiation_low", "outside_radiation"),
            "outside_daylight_min": threshold("outside_daylight_min", "outside_daylight_flag"),
        }
        if bool(options_cfg.get("include_rolling_6h", True)):
            fixed_thresholds["outside_radiation_rolling_6h_low"] = threshold(
                "outside_radiation_rolling_6h_low",
                "outside_radiation_rolling_6h",
            )
    elif alarm_name == "high_radiation_stress":
        fixed_thresholds = {
            "outside_radiation_high": threshold("outside_radiation_high", "outside_radiation"),
            "temp_high_functional": threshold("temp_high_functional", "air_temperature"),
        }
        if bool(options_cfg.get("include_vpd_stress", True)):
            fixed_thresholds["vpd_stress"] = threshold("vpd_stress", "vpd")
    else:
        raise ValueError(f"Unsupported agronomic alarm '{alarm_name}' in task '{task_name}'.")

    metadata = {
        "alarm_name": alarm_name,
        "adaptive_mode": adaptive_mode_value,
        "adaptive_adjusted": False,
        "resolved_columns": resolved_columns,
        "required_columns": sorted(resolved_columns.values()),
        "resolved_thresholds": fixed_thresholds,
        "radiation_fallback_used": radiation_fallback_used,
        "radiation_column": resolved_columns.get("radiation_proxy"),
        "soil_slope_window_steps": slope_window,
    }
    return build_mask_from_thresholds(fixed_thresholds), metadata


def _generate_legacy_labels(
    df: pd.DataFrame,
    *,
    task_name: str,
    rule: dict[str, Any],
    horizon_steps: int,
) -> tuple[pd.Series, list[str], dict[str, Any]]:
    if rule["source_column"] not in df.columns:
        raise ValueError(
            f"Task '{task_name}' requires source column '{rule['source_column']}', "
            "but it is not available in the canonical dataset."
        )
    source = df[rule["source_column"]]
    if source.notna().sum() == 0:
        raise ValueError(
            f"Task '{task_name}' cannot generate labels because '{rule['source_column']}' has no usable values."
        )

    if rule["type"] == "any_above_threshold":
        future_values = _future_matrix(source, horizon_steps)
        target = (future_values > float(rule["threshold"])).any(axis=1).astype("float")
        class_names = list(_BINARY_CLASS_NAMES)
    elif rule["type"] == "any_below_threshold":
        future_values = _future_matrix(source, horizon_steps)
        target = (future_values < float(rule["threshold"])).any(axis=1).astype("float")
        class_names = list(_BINARY_CLASS_NAMES)
    elif rule["type"] == "future_delta_above_threshold":
        future_values = _future_matrix(source, horizon_steps)
        deltas = future_values.subtract(source, axis=0)
        target = (deltas >= float(rule["threshold"])).any(axis=1).astype("float")
        class_names = list(_BINARY_CLASS_NAMES)
    elif rule["type"] == "future_bucket":
        target_values = source.shift(-horizon_steps)
        bins = [-float("inf"), *rule["bins"], float("inf")]
        labels_cfg = rule["labels"]
        buckets = pd.cut(target_values, bins=bins, labels=labels_cfg, right=False)
        mapping = {label: index for index, label in enumerate(labels_cfg)}
        target = buckets.map(mapping).astype("float")
        class_names = list(labels_cfg)
    else:
        raise ValueError(f"Unsupported label rule: {rule['type']}")

    return target, class_names, {
        "rule_type": rule["type"],
        "source_column": rule["source_column"],
    }


def generate_labels(
    df: pd.DataFrame,
    task_config: dict[str, Any],
    data_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    timestamp_column = data_config["timestamp_column"]
    frequency = pd.Timedelta(data_config["frequency"])
    horizon_steps = int(pd.Timedelta(minutes=task_config["horizon_minutes"]) / frequency)
    if horizon_steps <= 0:
        raise ValueError(f"Task '{task_config['name']}' resolved a non-positive horizon: {horizon_steps}.")

    rule = task_config["rule"]
    timestamps = pd.to_datetime(df[timestamp_column])
    labels = pd.DataFrame({timestamp_column: timestamps})

    if rule["type"] == "agronomic_alarm":
        alarm_mask, alarm_metadata = _agronomic_alarm_mask(
            df,
            task_name=task_config["name"],
            rule=rule,
            horizon_steps=horizon_steps,
            window_size_steps=int(task_config.get("window_size_steps", horizon_steps)),
        )
        target = (_future_matrix(alarm_mask.astype(float), horizon_steps) > 0.0).any(axis=1).astype("float")
        class_names = list(_BINARY_CLASS_NAMES)
        metadata = {
            "rule_type": rule["type"],
            **alarm_metadata,
            "current_alarm_rate": float(alarm_mask.mean()),
        }
    else:
        target, class_names, metadata = _generate_legacy_labels(
            df,
            task_name=task_config["name"],
            rule=rule,
            horizon_steps=horizon_steps,
        )

    labels[task_config["label_column"]] = target
    labels["label_end"] = timestamps + (horizon_steps * frequency)
    return labels, {
        "task_name": task_config["name"],
        "label_column": task_config["label_column"],
        "class_names": class_names,
        "horizon_steps": horizon_steps,
        "positive_rate": float(target.dropna().mean()) if len(class_names) == 2 else None,
        **metadata,
    }
