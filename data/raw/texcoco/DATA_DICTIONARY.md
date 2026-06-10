# Texcoco Enriched Dataset Dictionary

This dictionary describes the enriched dataset used by the reviewer reproduction package. It describes the benchmark-ready file, not the preliminary raw sensor export.

## File

- Archive: `data/raw/texcoco/texcoco.rar`
- Expected extracted file: `data/raw/texcoco/proto_enriched_outside_weather.csv`
- Rows: `14,316`
- Columns: `111`
- Valid timestamp period: `2025-09-11 06:37:14` to `2025-10-30 23:50:05`
- Internal sampling interval: approximately `5 min`
- External weather frequency: hourly, aligned to local hourly blocks
- Site: greenhouse in Texcoco, State of Mexico, Mexico
- Crop: tomato greenhouse crop canopy
- Time zone: `America/Mexico_City`
- External weather coordinates: latitude `19.496304`, longitude `-98.865113`

External weather variables are hourly. They are assigned to internal observations by local hourly block, so multiple internal observations may share the same external-weather value within one hour.

## Canonical Internal Variables

| Raw column | Canonical name | Description |
| --- | --- | --- |
| `fecha` + `hora` | `timestamp` | Local timestamp |
| `hum_suelo` | `soil_moisture` | Soil moisture |
| `temp_aire_media` | `air_temperature` | Mean internal air temperature |
| `hum_aire_media` | `relative_humidity` | Mean internal relative humidity |
| `ppfd` | `radiation_proxy` | Internal radiation / photosynthetic photon flux proxy |
| `temp_suelo` | `soil_temperature` | Soil temperature |
| `vpd_media` | `vpd` | Internal vapor pressure deficit |
| `lux` | `light_lux` | Internal illuminance |
| `ec25_uScm` | `ec25` | Corrected electrical conductivity |
| `ph` | `ph` | Associated pH measurement |
| `irrigation_lph` | `irrigation_lph` | Irrigation flow when available |
| `trigger.irrigation_event` | `irrigation_event_observed` | Observed irrigation event flag when available |

CO2 and ventilation variables are not used by this benchmark and are not candidate features for Exp19.

## Open-Meteo Weather Families

Columns using the `outside_openmeteo_` prefix include:

| Family | Examples |
| --- | --- |
| Temperature and humidity | `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `vpd_kpa` |
| Precipitation | `precipitation`, daily sums, 6 h / 24 h / 48 h rolling sums |
| Radiation | `shortwave_radiation`, daily radiation, 3 h / 6 h / 12 h rolling sums |
| Wind | `wind_speed_10m`, `wind_direction_10m` |
| Evapotranspiration | `et0_fao_evapotranspiration` |
| Lagged weather | temperature and humidity lags of 1 h, 3 h, and 6 h |
| Inside-outside contrasts | temperature, humidity, and VPD deltas |

## NASA POWER Weather Families

Columns using the `outside_nasa_` prefix include:

| Family | Examples |
| --- | --- |
| Temperature and humidity | `T2M`, `RH2M`, `T2MDEW`, `vpd_kpa` |
| Precipitation | `PRECTOTCORR`, daily sums, 6 h / 24 h / 48 h rolling sums |
| Radiation | `ALLSKY_SFC_SW_DWN`, `ALLSKY_SFC_PAR_TOT`, rolling radiation sums |
| Wind | `WS2M`, `WS10M`, `WD10M` |
| Pressure | `PS` |
| Lagged weather | temperature and humidity lags of 1 h, 3 h, and 6 h |
| Inside-outside contrasts | temperature, humidity, and VPD deltas |

## Astronomical Context

The enriched file includes astronomical context derived for the greenhouse location and local timestamp. Exp19 mainly uses solar-position and day-length context:

- `outside_solar_elevation_angle`
- `outside_day_length_hours`
- `outside_minutes_since_sunrise`
- `outside_minutes_until_sunset`

## Weather-Source Views

Exp19 uses three views over the same enriched file:

| View | Config | Description |
| --- | --- | --- |
| `openmeteo` | `configs/data/texcoco_enriched_openmeteo.yaml` | Internal sensors plus Open-Meteo external weather |
| `nasa` | `configs/data/texcoco_enriched_nasa.yaml` | Internal sensors plus NASA POWER external weather |
| `fused` | `configs/data/texcoco_enriched_fused.yaml` | Internal sensors plus selected Open-Meteo, NASA POWER, and source-gap variables |

The fused view is defined by YAML transformations over Open-Meteo and NASA columns. It is not a separate raw data source.

## Alert Families

Exp19 evaluates seven operational proxy alert families:

| Alert family | Main variables |
| --- | --- |
| `cloudy_low_radiation` | External radiation, daylight flag, rolling radiation summaries |
| `heat_water_stress` | Internal air temperature, humidity, and VPD context |
| `high_radiation_stress` | Radiation, internal temperature, and VPD |
| `pollination_window` | Temperature, relative humidity, radiation proxy, optional VPD context |
| `rain_risk` | External precipitation and precipitation accumulations |
| `water_saturation_stress` | Soil moisture |
| `wind_risk` | External wind speed |

The labels are operational decision-support proxies. They do not directly measure crop damage, biological outcomes, yield loss, greenhouse impact, or verified management action.
