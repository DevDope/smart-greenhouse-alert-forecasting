"""Experiment summary helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..evaluation.reports import collect_metric_rows, write_summary_report
from ..utils.paths import ensure_dir
from ..utils.reproducibility import make_run_id
from .run_experiment import prepare_all_datasets


EXP19_MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "zappa_lgbm_chain": "juxtapose_wrapper",
    "juxtapose_lgbm_relation_v2": "juxtapose_native_v2",
    "gru_seq48": "GRU (seq=48)",
    "tft_base_seq48": "TFT (seq=48)",
    "fripp_lgbm_sparse": "discipline_wrapper",
    "tcn_seq48": "TCN (seq=48)",
    "juxtapose_lgbm_chain": "juxtapose_native_v1",
    "discipline_lgbm_sparse": "discipline_native",
    "vanilla_transformer_seq48": "Vanilla Transformer (seq=48)",
    "logistic": "Logistic",
    "naive": "Naive",
}

EXP19_RANK_METRICS = ["pr_auc", "f1", "brier_score", "ece"]
EXP19_EXPECTED_SOURCES = {"fused", "nasa", "openmeteo"}
EXP19_EXPECTED_ALERTS = {
    "cloudy_low_radiation",
    "heat_water_stress",
    "high_radiation_stress",
    "pollination_window",
    "rain_risk",
    "water_saturation",
    "wind_risk",
}

EXP19_MODEL_FAMILIES = {
    "naive": "baseline",
    "logistic": "baseline",
    "xgboost": "gbdt",
    "lightgbm": "gbdt",
    "fripp_lgbm_sparse": "discipline_wrapper",
    "zappa_lgbm_chain": "juxtapose_wrapper",
    "discipline_lgbm_sparse": "discipline_native",
    "juxtapose_lgbm_chain": "juxtapose_native",
    "juxtapose_lgbm_relation_v2": "juxtapose_native",
    "gru_seq48": "sequence",
    "tcn_seq48": "sequence",
    "vanilla_transformer_seq48": "sequence",
    "tft_base_seq48": "sequence",
}

EXP19_MODEL_LINEAGES = {
    "fripp_lgbm_sparse": "wrapper",
    "zappa_lgbm_chain": "wrapper",
    "discipline_lgbm_sparse": "native",
    "juxtapose_lgbm_chain": "native",
    "juxtapose_lgbm_relation_v2": "native",
}

JOURNAL_FIGURE_DPI = 600
JOURNAL_MODEL_LABELS = {
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "juxtapose_wrapper": "Juxtapose wrapper",
    "juxtapose_native_v2": "Juxtapose native v2",
    "GRU seq48": "GRU (seq=48)",
    "GRU (seq=48)": "GRU (seq=48)",
    "TFT base seq48": "TFT (seq=48)",
    "TFT (seq=48)": "TFT (seq=48)",
    "discipline_wrapper": "Discipline wrapper",
    "TCN seq48": "TCN (seq=48)",
    "TCN (seq=48)": "TCN (seq=48)",
    "juxtapose_native_v1": "Juxtapose native v1",
    "discipline_native": "Discipline native",
    "Vanilla Transformer": "Vanilla Transformer (seq=48)",
    "Vanilla Transformer (seq=48)": "Vanilla Transformer (seq=48)",
    "Logistic": "Logistic",
    "Naive": "Naive",
}
JOURNAL_ALERT_LABELS = {
    "cloudy_low_radiation": "Cloudy low radiation",
    "heat_water_stress": "Heat-water stress",
    "high_radiation_stress": "High-radiation stress",
    "pollination_window": "Pollination window",
    "rain_risk": "Rain risk",
    "water_saturation_stress": "Water-saturation stress",
    "water_saturation": "Water saturation",
    "wind_risk": "Wind risk",
}
JOURNAL_SOURCE_LABELS = {
    "fused": "Fused",
    "nasa": "NASA POWER",
    "openmeteo": "Open-Meteo",
}
JOURNAL_LINEAGE_COLORS = {
    "baseline": "#4C78A8",
    "native": "#54A24B",
    "wrapper": "#F58518",
    "other": "#8A8A8A",
}


def _model_label(model: str) -> str:
    return EXP19_MODEL_LABELS.get(model, model)


def _model_family(model: str) -> str:
    return EXP19_MODEL_FAMILIES.get(model, "other")


def _model_lineage(model: str) -> str:
    return EXP19_MODEL_LINEAGES.get(model, "baseline")


def _alert_family(task: str) -> str:
    name = str(task)
    for suffix in ("_openmeteo", "_nasa", "_fused"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.endswith("_60m"):
        name = name[:-4]
    if name.endswith("_expert"):
        name = name[:-7]
    return name


def _view_config(resolved_config: dict[str, Any], view: dict[str, Any]) -> dict[str, Any]:
    config = dict(resolved_config)
    config.pop("weather_source_views", None)
    config["name"] = f"{resolved_config['name']}__{view['name']}"
    config["weather_source_view"] = view["name"]
    config["data"] = view["data"]
    config["tasks"] = view["tasks"]
    return config


def _write_frame(frame: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    csv_path = output_dir / f"{stem}.csv"
    frame.to_csv(csv_path, index=False)
    md_path = output_dir / f"{stem}.md"
    if frame.empty:
        md_path.write_text("_empty_", encoding="utf-8")
    else:
        try:
            markdown = frame.to_markdown(index=False)
        except ImportError:
            markdown = frame.to_csv(index=False)
        md_path.write_text(markdown, encoding="utf-8")
    return csv_path


def _xlsx_column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xlsx_cell(row: int, column: int, value: Any) -> str:
    ref = f"{_xlsx_column_name(column)}{row}"
    if pd.isna(value):
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _xlsx_sheet_xml(frame: pd.DataFrame) -> str:
    rows = []
    headers = list(frame.columns)
    header_cells = "".join(_xlsx_cell(1, index + 1, header) for index, header in enumerate(headers))
    rows.append(f'<row r="1">{header_cells}</row>')
    for row_index, (_, row) in enumerate(frame.iterrows(), start=2):
        cells = "".join(_xlsx_cell(row_index, column_index + 1, row[column]) for column_index, column in enumerate(headers))
        rows.append(f'<row r="{row_index}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(rows)}</sheetData>'
        "</worksheet>"
    )


def _write_xlsx(sheets: dict[str, pd.DataFrame], output_path: Path) -> Path:
    safe_sheets = {name[:31]: frame for name, frame in sheets.items()}
    sheet_entries = []
    workbook_rels = []
    content_overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    with ZipFile(output_path, "w", ZIP_DEFLATED) as archive:
        for index, (sheet_name, frame) in enumerate(safe_sheets.items(), start=1):
            sheet_entries.append(
                f'<sheet name="{escape(sheet_name)}" sheetId="{index}" r:id="rId{index}"/>'
            )
            workbook_rels.append(
                f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
            )
            content_overrides.append(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _xlsx_sheet_xml(frame))
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f'{"".join(content_overrides)}'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets>{"".join(sheet_entries)}</sheets>'
            "</workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'{"".join(workbook_rels)}'
            "</Relationships>",
        )
        archive.writestr(
            "docProps/core.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>exp19 leaderboard</dc:title></cp:coreProperties>',
        )
        archive.writestr(
            "docProps/app.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
            "<Application>smart-greenhouse-alert-forecasting</Application></Properties>",
        )
    return output_path


def _weather_source_tables(metrics_root: Path, tables_root: Path, base_run_id: str, views: list[str]) -> Path:
    output_dir = ensure_dir(tables_root / base_run_id)
    frames: list[pd.DataFrame] = []
    for view_name in views:
        frame = collect_metric_rows(metrics_root, run_id=f"{base_run_id}__{view_name}")
        if frame.empty:
            continue
        frame["weather_source_view"] = view_name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _write_frame(combined, output_dir, "weather_source_summary")
    if combined.empty:
        return output_dir / "weather_source_summary.csv"

    test = combined[combined["split"] == "test"].copy()
    metric_columns = [
        column
        for column in ["pr_auc", "f1", "brier_score", "ece", "training_seconds", "inference_seconds"]
        if column in test.columns
    ]
    source_rank = (
        test.groupby(["weather_source_view", "task", "model"], as_index=False)[metric_columns]
        .mean(numeric_only=True)
        .sort_values(["task", "pr_auc", "f1"], ascending=[True, False, False])
        if metric_columns
        else pd.DataFrame()
    )
    _write_frame(source_rank, output_dir, "weather_source_model_ranking")
    if {"weather_source_view", "task", "model", "pr_auc", "f1"}.issubset(source_rank.columns):
        best_by_source = (
            source_rank.sort_values(["weather_source_view", "task", "pr_auc", "f1"], ascending=[True, True, False, False])
            .groupby(["weather_source_view", "task"], as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        _write_frame(best_by_source, output_dir, "weather_source_best_by_task")
        export_exp19_leaderboard_tables(combined, output_dir)
    return output_dir / "weather_source_summary.csv"


def _rank_models(frame: pd.DataFrame, group_columns: list[str] | None = None) -> pd.DataFrame:
    group_columns = group_columns or []
    metric_columns = [column for column in EXP19_RANK_METRICS if column in frame.columns]
    if not {"task", "model", "pr_auc", "f1"}.issubset(frame.columns):
        return pd.DataFrame()
    summary = (
        frame.groupby(group_columns + ["model"], as_index=False)[metric_columns]
        .mean(numeric_only=True)
        .rename(columns={"brier_score": "brier"})
    )
    task_winners = (
        frame.sort_values(group_columns + ["task", "pr_auc", "f1"], ascending=[True] * len(group_columns) + [True, False, False])
        .groupby(group_columns + ["task"], as_index=False)
        .head(1)
    )
    top3 = (
        frame.sort_values(group_columns + ["task", "pr_auc", "f1"], ascending=[True] * len(group_columns) + [True, False, False])
        .groupby(group_columns + ["task"], as_index=False)
        .head(3)
    )
    winner_counts = task_winners.groupby(group_columns + ["model"]).size().rename("wins").reset_index()
    top3_counts = top3.groupby(group_columns + ["model"]).size().rename("top3").reset_index()
    summary = summary.merge(winner_counts, on=group_columns + ["model"], how="left")
    summary = summary.merge(top3_counts, on=group_columns + ["model"], how="left")
    summary["wins"] = summary["wins"].fillna(0).astype(int)
    summary["top3"] = summary["top3"].fillna(0).astype(int)
    summary["model_label"] = summary["model"].map(_model_label)
    summary["model_lineage"] = summary["model"].map(_model_lineage)
    preferred_rank_sort = ["pr_auc", "f1", "brier", "ece", "model_label"]
    rank_sort = group_columns + [column for column in preferred_rank_sort if column in summary.columns]
    ascending_map = {"pr_auc": False, "f1": False, "brier": True, "ece": True, "model_label": True}
    ascending = [True] * len(group_columns) + [ascending_map[column] for column in rank_sort[len(group_columns):]]
    summary = summary.sort_values(rank_sort, ascending=ascending).reset_index(drop=True)
    if group_columns:
        summary["rank"] = summary.groupby(group_columns).cumcount() + 1
    else:
        summary["rank"] = range(1, len(summary) + 1)
    ordered = group_columns + [
        "rank",
        "model",
        "model_label",
        "model_lineage",
        "pr_auc",
        "f1",
        "brier",
        "ece",
        "wins",
        "top3",
    ]
    return summary[[column for column in ordered if column in summary.columns]]


def _degenerate_alerts(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"weather_source_view", "alert", "model", "pr_auc", "f1"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    grouped = frame.groupby(["weather_source_view", "alert"], as_index=False).agg(
        model_count=("model", "nunique"),
        pr_auc_max=("pr_auc", "max"),
        f1_max=("f1", "max"),
    )
    grouped["status"] = grouped.apply(
        lambda row: "invalid_or_degenerate"
        if float(row["pr_auc_max"] or 0.0) == 0.0 and float(row["f1_max"] or 0.0) == 0.0
        else "valid",
        axis=1,
    )
    return grouped.sort_values(["weather_source_view", "alert"]).reset_index(drop=True)


def _coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    test = frame[frame["split"] == "test"].copy() if "split" in frame.columns else frame.copy()
    if test.empty:
        return pd.DataFrame([{"status": "empty"}])
    sources = set(test.get("weather_source_view", pd.Series(dtype=str)).dropna().astype(str))
    alerts = set(test.get("alert", pd.Series(dtype=str)).dropna().astype(str))
    models = set(test.get("model", pd.Series(dtype=str)).dropna().astype(str))
    expected_runs = len(sources or EXP19_EXPECTED_SOURCES) * len(alerts or EXP19_EXPECTED_ALERTS) * len(models)
    observed_runs = len(test[["weather_source_view", "alert", "model"]].drop_duplicates()) if {"weather_source_view", "alert", "model"}.issubset(test.columns) else len(test)
    return pd.DataFrame(
        [
            {
                "status": "complete" if expected_runs == observed_runs else "incomplete",
                "observed_runs": observed_runs,
                "expected_runs": expected_runs,
                "sources": len(sources),
                "alerts": len(alerts),
                "models": len(models),
                "failed_runs": 0,
                "pending_runs": max(expected_runs - observed_runs, 0),
            }
        ]
    )


def _q1_label_validity_by_alert_source(frame: pd.DataFrame) -> pd.DataFrame:
    split_column = "split" if "split" in frame.columns else None
    group_columns = ["weather_source_view", "alert"] + ([split_column] if split_column else [])
    validity = frame.groupby(group_columns, as_index=False).agg(
        model_count=("model", "nunique"),
        pr_auc_max=("pr_auc", "max"),
        f1_max=("f1", "max"),
        pr_auc_mean=("pr_auc", "mean"),
        f1_mean=("f1", "mean"),
    )
    validity["status"] = validity.apply(
        lambda row: "invalid_or_degenerate"
        if float(row["pr_auc_max"] or 0.0) == 0.0 and float(row["f1_max"] or 0.0) == 0.0
        else "valid",
        axis=1,
    )
    return validity.sort_values(group_columns).reset_index(drop=True)


def _weather_ablation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.copy()
    if valid.empty:
        return pd.DataFrame()
    metrics = [column for column in ["pr_auc", "f1", "brier_score", "ece"] if column in valid.columns]
    summary = valid.groupby("weather_source_view", as_index=False)[metrics].mean(numeric_only=True)
    winner_counts = (
        valid.sort_values(["weather_source_view", "alert", "pr_auc", "f1"], ascending=[True, True, False, False])
        .groupby(["weather_source_view", "alert"], as_index=False)
        .head(1)
        .groupby(["weather_source_view", "model"])
        .size()
        .rename("alert_wins")
        .reset_index()
    )
    top_model = (
        winner_counts.sort_values(["weather_source_view", "alert_wins", "model"], ascending=[True, False, True])
        .groupby("weather_source_view", as_index=False)
        .head(1)
    )
    summary = summary.merge(top_model, on="weather_source_view", how="left")
    summary["top_model_label"] = summary["model"].fillna("").map(_model_label)
    summary["top_model_lineage"] = summary["model"].fillna("").map(_model_lineage)
    return summary.rename(columns={"brier_score": "brier"}).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def _family_comparison_exp19(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    working["model_family_q1"] = working["model"].map(_model_family)
    metrics = [
        column
        for column in ["pr_auc", "f1", "brier_score", "ece", "training_seconds", "inference_seconds"]
        if column in working.columns
    ]
    summary = working.groupby("model_family_q1", as_index=False)[metrics].mean(numeric_only=True)
    best = (
        working.sort_values(["model_family_q1", "pr_auc", "f1"], ascending=[True, False, False])
        .groupby("model_family_q1", as_index=False)
        .head(1)[["model_family_q1", "model"]]
        .rename(columns={"model": "best_model"})
    )
    summary = summary.merge(best, on="model_family_q1", how="left")
    summary["best_model_label"] = summary["best_model"].fillna("").map(_model_label)
    summary["best_model_lineage"] = summary["best_model"].fillna("").map(_model_lineage)
    return summary.rename(columns={"brier_score": "brier"}).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def _calibration_summary_exp19(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"model", "ece", "brier_score"}.issubset(frame.columns):
        return pd.DataFrame()
    summary = (
        frame.groupby("model", as_index=False)[["ece", "brier_score", "pr_auc", "f1"]]
        .mean(numeric_only=True)
        .rename(columns={"brier_score": "brier"})
    )
    summary["model_label"] = summary["model"].map(_model_label)
    summary["model_lineage"] = summary["model"].map(_model_lineage)
    summary["calibration_status"] = summary["ece"].apply(
        lambda value: "strong" if value <= 0.05 else ("watch" if value <= 0.20 else "weak")
    )
    return summary.sort_values(["ece", "brier", "pr_auc"], ascending=[True, True, False]).reset_index(drop=True)


def _cost_performance_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    metrics = [
        column
        for column in ["pr_auc", "f1", "training_seconds", "inference_seconds", "artifact_size_kb"]
        if column in frame.columns
    ]
    summary = frame.groupby("model", as_index=False)[metrics].mean(numeric_only=True)
    summary["model_label"] = summary["model"].map(_model_label)
    summary["model_lineage"] = summary["model"].map(_model_lineage)
    if "training_seconds" in summary.columns:
        summary["pr_auc_per_train_second"] = summary.apply(
            lambda row: row["pr_auc"] / row["training_seconds"]
            if pd.notna(row.get("training_seconds")) and row["training_seconds"] > 0
            else pd.NA,
            axis=1,
        )
    return summary.sort_values(["pr_auc", "f1"], ascending=[False, False]).reset_index(drop=True)


def _model_alert_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"alert", "model", "pr_auc"}.issubset(frame.columns):
        return pd.DataFrame()
    metrics = [column for column in ["pr_auc", "f1", "brier_score", "ece"] if column in frame.columns]
    summary = frame.groupby(["alert", "model"], as_index=False)[metrics].mean(numeric_only=True)
    summary["model_label"] = summary["model"].map(_model_label)
    summary["model_lineage"] = summary["model"].map(_model_lineage)
    return summary.rename(columns={"brier_score": "brier"}).sort_values(["alert", "pr_auc"], ascending=[True, False])


def _journal_model_label(label: Any) -> str:
    text = str(label)
    return JOURNAL_MODEL_LABELS.get(text, text.replace("_", " ").title())


def _journal_alert_label(label: Any) -> str:
    text = str(label)
    return JOURNAL_ALERT_LABELS.get(text, text.replace("_", " ").title())


def _journal_source_label(label: Any) -> str:
    text = str(label)
    return JOURNAL_SOURCE_LABELS.get(text, text.replace("_", " ").title())


def _format_metric(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _format_time(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    value = float(value)
    if value < 0.01:
        return f"{value:.4f}s"
    if value < 10:
        return f"{value:.2f}s"
    return f"{value:.0f}s"


def _apply_journal_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": JOURNAL_FIGURE_DPI,
            "savefig.dpi": JOURNAL_FIGURE_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_journal_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=JOURNAL_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return png_path


def _plot_no_data(output_path: Path) -> Path:
    _apply_journal_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.axis("off")
    ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    return _save_journal_figure(fig, output_path)


def _plot_q1_bar(frame: pd.DataFrame, label_column: str, value_column: str, output_path: Path, title: str = "") -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = frame[[label_column, value_column]].dropna().sort_values(value_column, ascending=True)
    if plot_frame.empty:
        return _plot_no_data(output_path)

    _apply_journal_style()
    labels = plot_frame[label_column].map(_journal_model_label)
    fig, ax = plt.subplots(figsize=(7.2, max(4.2, 0.34 * len(plot_frame) + 1.0)))
    colors = ["#2F6F9F" if rank == len(plot_frame) - 1 else "#B7C9D6" for rank in range(len(plot_frame))]
    bars = ax.barh(labels, plot_frame[value_column], color=colors, edgecolor="#263238", linewidth=0.35)
    for bar, value in zip(bars, plot_frame[value_column], strict=False):
        ax.text(
            bar.get_width() + 0.006,
            bar.get_y() + bar.get_height() / 2,
            _format_metric(value),
            va="center",
            ha="left",
            fontsize=7.5,
        )
    max_value = float(plot_frame[value_column].max())
    ax.set_xlim(0, min(1.0, max_value + 0.09))
    ax.set_xlabel(value_column.replace("_", "-").upper() if value_column == "pr_auc" else value_column.replace("_", " ").title())
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.20, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, pad=4)
    fig.tight_layout(pad=0.8)
    return _save_journal_figure(fig, output_path)


def _plot_q1_heatmap(
    frame: pd.DataFrame,
    index_column: str,
    column_column: str,
    value_column: str,
    output_path: Path,
    title: str = "",
    *,
    cbar_label: str | None = None,
    model_order: list[str] | None = None,
    alert_order: list[str] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = frame.pivot_table(index=index_column, columns=column_column, values=value_column, aggfunc="mean")
    if pivot.empty:
        return _plot_no_data(output_path)

    if model_order:
        kept = [label for label in model_order if label in pivot.index]
        pivot = pivot.reindex(kept)
    if alert_order:
        kept_columns = [label for label in alert_order if label in pivot.columns]
        pivot = pivot.reindex(columns=kept_columns)

    _apply_journal_style()
    fig_width = max(6.8, 0.75 * len(pivot.columns) + 3.0)
    fig_height = max(4.2, 0.34 * len(pivot.index) + 1.7)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([_journal_alert_label(label) for label in pivot.columns], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_journal_model_label(label) for label in pivot.index])
    for row_idx, row_label in enumerate(pivot.index):
        for column_idx, column_label in enumerate(pivot.columns):
            value = pivot.loc[row_label, column_label]
            if pd.isna(value):
                continue
            text_color = "white" if float(value) >= 0.62 else "#1A1A1A"
            ax.text(column_idx, row_idx, _format_metric(value), ha="center", va="center", fontsize=6.2, color=text_color)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title:
        ax.set_title(title, pad=4)
    cbar = fig.colorbar(image, ax=ax, fraction=0.030, pad=0.025)
    cbar.ax.set_ylabel(cbar_label or value_column.replace("_", " ").title(), rotation=90, va="center")
    fig.tight_layout(pad=0.8)
    return _save_journal_figure(fig, output_path)


def _plot_q1_scatter(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    required = {"pr_auc", "training_seconds", "model_label"}
    if frame.empty or not required.issubset(frame.columns):
        return _plot_no_data(output_path)

    _apply_journal_style()
    plot_frame = frame.dropna(subset=["pr_auc", "training_seconds"]).copy()
    plot_frame = plot_frame[plot_frame["training_seconds"] > 0]
    if plot_frame.empty:
        return _plot_no_data(output_path)

    artifact = plot_frame.get("artifact_size_kb", pd.Series([pd.NA] * len(plot_frame), index=plot_frame.index))
    artifact_numeric = pd.to_numeric(artifact, errors="coerce")
    if artifact_numeric.notna().any():
        min_size = float(artifact_numeric.min())
        max_size = float(artifact_numeric.max())
        denom = max(max_size - min_size, 1.0)
        plot_frame["_marker_size"] = 42 + 210 * (artifact_numeric.fillna(min_size) - min_size) / denom
    else:
        plot_frame["_marker_size"] = 72

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for lineage, subset in plot_frame.groupby("model_lineage", dropna=False):
        label = str(lineage) if pd.notna(lineage) else "other"
        ax.scatter(
            subset["training_seconds"],
            subset["pr_auc"],
            s=subset["_marker_size"],
            color=JOURNAL_LINEAGE_COLORS.get(label, JOURNAL_LINEAGE_COLORS["other"]),
            alpha=0.86,
            edgecolor="#222222",
            linewidth=0.45,
            label=label.title(),
        )

    label_positions = {
        "XGBoost": (5.5, 0.812),
        "LightGBM": (23.0, 0.752),
        "Juxtapose wrapper": (115.0, 0.795),
        "Juxtapose native v2": (260.0, 0.750),
        "GRU (seq=48)": (2.5, 0.768),
        "TFT (seq=48)": (480.0, 0.690),
        "Discipline wrapper": (0.12, 0.735),
        "TCN (seq=48)": (22.0, 0.678),
        "Juxtapose native v1": (380.0, 0.648),
        "Discipline native": (0.12, 0.662),
        "Vanilla Transformer (seq=48)": (46.0, 0.644),
        "Logistic": (1.6, 0.585),
        "Naive": (0.0025, 0.232),
    }
    for _, row in plot_frame.iterrows():
        label = _journal_model_label(row["model_label"])
        compact_label = {
            "Juxtapose wrapper": "Juxtapose\nwrapper",
            "Juxtapose native v2": "Juxtapose\nnative v2",
            "Juxtapose native v1": "Juxtapose\nnative v1",
            "Discipline wrapper": "Discipline\nwrapper",
            "Discipline native": "Discipline\nnative",
            "GRU (seq=48)": "GRU\n(seq=48)",
            "TCN (seq=48)": "TCN\n(seq=48)",
            "TFT (seq=48)": "TFT\n(seq=48)",
            "Vanilla Transformer (seq=48)": "Vanilla\nTransformer\n(seq=48)",
        }.get(label, label)
        xy = (row["training_seconds"], row["pr_auc"])
        text = f"{compact_label}\n{_format_metric(row['pr_auc'])}"
        if label in label_positions:
            xytext = label_positions[label]
            ax.annotate(
                text,
                xy,
                xytext=xytext,
                textcoords="data",
                fontsize=6.7,
                ha="left" if xytext[0] >= float(row["training_seconds"]) else "right",
                va="center",
                arrowprops={"arrowstyle": "-", "color": "#666666", "linewidth": 0.45, "shrinkA": 2.5, "shrinkB": 2.5},
            )
            continue
        ax.annotate(
            text,
            xy,
            fontsize=6.7,
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="center",
        )

    ax.set_xscale("log")
    ax.set_xlim(max(float(plot_frame["training_seconds"].min()) * 0.55, 0.0008), float(plot_frame["training_seconds"].max()) * 2.0)
    ax.set_ylim(max(0.18, float(plot_frame["pr_auc"].min()) - 0.04), min(0.83, float(plot_frame["pr_auc"].max()) + 0.04))
    ax.set_xlabel("Training time (s, log scale)")
    ax.set_ylabel("Mean PR-AUC")
    ax.grid(alpha=0.20, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", title="Model lineage", title_fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.98)
    return _save_journal_figure(fig, output_path)


def _plot_source_winners(frame: pd.DataFrame, output_path: Path) -> Path:
    required = {"weather_source_view", "model_label", "pr_auc"}
    if frame.empty or not required.issubset(frame.columns):
        return _plot_no_data(output_path)
    _apply_journal_style()
    plot_frame = frame.copy().sort_values("pr_auc", ascending=True)
    labels = plot_frame["weather_source_view"].map(_journal_source_label)
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    bars = ax.barh(labels, plot_frame["pr_auc"], color="#76A5AF", edgecolor="#263238", linewidth=0.35)
    for bar, (_, row) in zip(bars, plot_frame.iterrows(), strict=False):
        ax.text(
            bar.get_width() + 0.006,
            bar.get_y() + bar.get_height() / 2,
            f"{_journal_model_label(row['model_label'])}  {_format_metric(row['pr_auc'])}",
            va="center",
            ha="left",
            fontsize=7.4,
        )
    ax.set_xlim(0, min(1.0, float(plot_frame["pr_auc"].max()) + 0.13))
    ax.set_xlabel("Mean PR-AUC")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.20, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.8)
    return _save_journal_figure(fig, output_path)


def _plot_calibration_bars(frame: pd.DataFrame, output_path: Path) -> Path:
    required = {"model_label", "ece"}
    if frame.empty or not required.issubset(frame.columns):
        return _plot_no_data(output_path)
    _apply_journal_style()
    plot_frame = frame.dropna(subset=["ece"]).copy().sort_values("ece", ascending=True)
    labels = plot_frame["model_label"].map(_journal_model_label)
    fig, ax = plt.subplots(figsize=(7.2, max(3.8, 0.34 * len(plot_frame) + 1.0)))
    bars = ax.barh(labels, plot_frame["ece"], color="#9D7660", edgecolor="#263238", linewidth=0.35)
    for bar, value in zip(bars, plot_frame["ece"], strict=False):
        ax.text(bar.get_width() + 0.006, bar.get_y() + bar.get_height() / 2, _format_metric(value), va="center", ha="left", fontsize=7.5)
    ax.set_xlim(0, min(1.0, float(plot_frame["ece"].max()) + 0.08))
    ax.set_xlabel("Expected calibration error")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.20, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.8)
    return _save_journal_figure(fig, output_path)


def _load_prevalence_by_alert_source(output_dir: Path) -> pd.DataFrame:
    candidates = [
        output_dir / "label_prevalence_by_split.csv",
        output_dir / "prevalence_by_split.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            frame = pd.read_csv(candidate)
            required = {"weather_source_view", "alert", "split", "positive_pct"}
            if required.issubset(frame.columns):
                test = frame[frame["split"].astype(str).str.lower() == "test"].copy()
                return (
                    test.groupby(["weather_source_view", "alert"], as_index=False)["positive_pct"]
                    .mean(numeric_only=True)
                    .sort_values(["weather_source_view", "alert"])
                )
    return pd.DataFrame()


def _plot_label_validity_heatmap(validity: pd.DataFrame, output_dir: Path, output_path: Path) -> Path:
    required = {"weather_source_view", "alert", "status"}
    if validity.empty or not required.issubset(validity.columns):
        return _plot_no_data(output_path)

    prevalence = _load_prevalence_by_alert_source(output_dir)
    use_prevalence = not prevalence.empty
    if use_prevalence:
        plot_frame = validity.merge(prevalence, on=["weather_source_view", "alert"], how="left")
        value_column = "positive_pct"
        cbar_label = "Test positive labels (%)"
        vmax = max(1.0, float(plot_frame[value_column].max()))
    else:
        plot_frame = validity.copy()
        value_column = "pr_auc_max"
        cbar_label = "Best test PR-AUC"
        vmax = 1.0

    pivot = plot_frame.pivot_table(index="alert", columns="weather_source_view", values=value_column, aggfunc="mean")
    status = plot_frame.pivot_table(
        index="alert",
        columns="weather_source_view",
        values="status",
        aggfunc=lambda values: "invalid_or_degenerate" if "invalid_or_degenerate" in set(values) else "valid",
    )
    alert_order = [alert for alert in JOURNAL_ALERT_LABELS if alert in pivot.index]
    source_order = [source for source in ["fused", "nasa", "openmeteo"] if source in pivot.columns]
    pivot = pivot.reindex(index=alert_order, columns=source_order)
    status = status.reindex(index=alert_order, columns=source_order)

    _apply_journal_style()
    fig, ax = plt.subplots(figsize=(6.8, max(3.9, 0.42 * len(pivot.index) + 1.5)))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([_journal_source_label(label) for label in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_journal_alert_label(label) for label in pivot.index])
    for row_idx, row_label in enumerate(pivot.index):
        for column_idx, column_label in enumerate(pivot.columns):
            value = pivot.loc[row_label, column_label]
            cell_status = status.loc[row_label, column_label]
            if pd.isna(value):
                continue
            invalid = cell_status == "invalid_or_degenerate"
            label = f"{float(value):.1f}%" if use_prevalence else _format_metric(value)
            annotation = f"{label}\ninvalid" if invalid else label
            text_color = "#B2182B" if invalid else ("white" if float(value) >= 0.62 * vmax else "#1A1A1A")
            ax.text(column_idx, row_idx, annotation, ha="center", va="center", fontsize=6.8, color=text_color)
            if invalid:
                ax.add_patch(Rectangle((column_idx - 0.5, row_idx - 0.5), 1, 1, fill=False, edgecolor="#B2182B", linewidth=1.2))
    cbar = fig.colorbar(image, ax=ax, fraction=0.040, pad=0.030)
    cbar.ax.set_ylabel(cbar_label, rotation=90, va="center")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout(pad=0.8)
    return _save_journal_figure(fig, output_path)


def _write_journal_captions(output_dir: Path, figure_paths: list[Path]) -> Path:
    captions_path = output_dir / "captions_journal.md"
    path_by_name = {path.stem: path for path in figure_paths}
    lines = [
        "# Journal Figure Captions",
        "",
        "All figures are exported as vector PDF and 600 dpi PNG. The plotted values are read from the benchmark summary tables without changing the underlying metrics.",
        "",
        "## Figure 1",
        f"**File:** `{path_by_name.get('figure_1_global_prauc_ranking', Path('figure_1_global_prauc_ranking.png')).name}`",
        "",
        "Non-degenerate global model ranking for the Texcoco greenhouse alert benchmark. Bars show mean PR-AUC after excluding the degenerate `rain_risk` target from model promotion.",
        "",
        "## Figure 2",
        f"**File:** `{path_by_name.get('figure_2_label_prevalence_validity_heatmap', Path('figure_2_label_prevalence_validity_heatmap.png')).name}`",
        "",
        "Label prevalence and validity audit by alert family and weather-source view. Cells report test-set positive prevalence when the prevalence table is available; otherwise, cells report the best test PR-AUC. Invalid cells identify degenerate labels excluded from model promotion.",
        "",
        "## Figure 3",
        f"**File:** `{path_by_name.get('figure_3_weather_source_winners', Path('figure_3_weather_source_winners.png')).name}`",
        "",
        "Best source-level model by weather-source view, showing the winning model and mean PR-AUC for fused, NASA POWER, and Open-Meteo views.",
        "",
        "## Figure 4",
        f"**File:** `{path_by_name.get('figure_4_model_alert_prauc_heatmap', Path('figure_4_model_alert_prauc_heatmap.png')).name}`",
        "",
        "Model-by-alert PR-AUC heatmap for non-degenerate alert families. Cells show mean PR-AUC aggregated across weather-source views for each model-alert pair.",
        "",
        "## Figure 5",
        f"**File:** `{path_by_name.get('figure_5_calibration_summary', Path('figure_5_calibration_summary.png')).name}`",
        "",
        "Calibration summary for selected models. Bars report expected calibration error, emphasizing that discrimination and probability reliability must be interpreted separately.",
        "",
        "## Figure 6",
        f"**File:** `{path_by_name.get('figure_6_cost_performance_scatter', Path('figure_6_cost_performance_scatter.png')).name}`",
        "",
        "Cost-performance trade-off across model families. Points show mean PR-AUC versus training time on a logarithmic scale; marker area reflects artifact size and color indicates model lineage.",
        "",
    ]
    captions_path.write_text("\n".join(lines), encoding="utf-8")
    return captions_path


def _export_q1_figures(outputs: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    figure_dir = ensure_dir(output_dir / "figures_journal")
    model_order = (
        outputs["model_global_summary_sin_rain"].sort_values("rank")["model_label"].tolist()
        if "model_global_summary_sin_rain" in outputs and "rank" in outputs["model_global_summary_sin_rain"].columns
        else None
    )
    alert_order = [
        "cloudy_low_radiation",
        "heat_water_stress",
        "high_radiation_stress",
        "pollination_window",
        "water_saturation_stress",
        "wind_risk",
    ]
    model_alert = outputs.get("model_alert_summary", outputs.get("model_ranking_with_alert_rank", pd.DataFrame()))
    generated = [
        _plot_q1_bar(
            outputs["model_global_summary_sin_rain"],
            "model_label",
            "pr_auc",
            figure_dir / "figure_1_global_prauc_ranking.png",
        ),
        _plot_label_validity_heatmap(
            outputs["label_validity_by_alert_source"],
            output_dir,
            figure_dir / "figure_2_label_prevalence_validity_heatmap.png",
        ),
        _plot_source_winners(
            outputs["model_by_source_summary"][outputs["model_by_source_summary"]["rank"] == 1],
            figure_dir / "figure_3_weather_source_winners.png",
        ),
        _plot_q1_heatmap(
            model_alert,
            "model_label",
            "alert",
            "pr_auc",
            figure_dir / "figure_4_model_alert_prauc_heatmap.png",
            cbar_label="Mean PR-AUC",
            model_order=model_order,
            alert_order=alert_order,
        ),
        _plot_calibration_bars(
            outputs["calibration_summary_exp19"].head(8),
            figure_dir / "figure_5_calibration_summary.png",
        ),
        _plot_q1_scatter(outputs["cost_performance_summary"], figure_dir / "figure_6_cost_performance_scatter.png"),
    ]
    captions_path = _write_journal_captions(figure_dir, generated)
    print("Saved journal figure files:")
    for path in generated:
        print(str(path))
        print(str(path.with_suffix(".pdf")))
    print(str(captions_path))
    return generated


def export_exp19_leaderboard_tables(summary: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir = ensure_dir(output_dir)
    test = summary[summary["split"] == "test"].copy() if "split" in summary.columns else summary.copy()
    if test.empty:
        return {}
    test["alert"] = test["task"].map(_alert_family)

    degenerate = _degenerate_alerts(test)
    invalid_alerts = set(degenerate.loc[degenerate["status"] == "invalid_or_degenerate", "alert"])
    valid_test = test[~test["alert"].isin(invalid_alerts)].copy()

    global_ranking = _rank_models(test)
    global_without_degenerate = _rank_models(valid_test)
    source_ranking = _rank_models(valid_test, group_columns=["weather_source_view"])
    label_validity = _q1_label_validity_by_alert_source(test)
    weather_ablation = _weather_ablation_summary(valid_test)
    family_comparison = _family_comparison_exp19(valid_test)
    calibration_summary = _calibration_summary_exp19(valid_test)
    cost_performance = _cost_performance_summary(valid_test)
    model_alert_summary = _model_alert_summary(valid_test)
    alert_winners = (
        test.sort_values(["weather_source_view", "alert", "pr_auc", "f1"], ascending=[True, True, False, False])
        .groupby(["weather_source_view", "alert"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    if not alert_winners.empty:
        alert_winners["model_label"] = alert_winners["model"].map(_model_label)
        alert_winners["model_lineage"] = alert_winners["model"].map(_model_lineage)
        alert_winners = alert_winners[
            [
                "weather_source_view",
                "alert",
                "task",
                "model",
                "model_label",
                "model_lineage",
                "pr_auc",
                "f1",
                "brier_score",
                "ece",
            ]
        ].rename(columns={"brier_score": "brier"})
        alert_winners = alert_winners.merge(
            degenerate[["weather_source_view", "alert", "status"]],
            on=["weather_source_view", "alert"],
            how="left",
        )

    outputs = {
        "run_coverage": _coverage_summary(test),
        "model_global_summary": global_ranking,
        "model_global_summary_sin_rain": global_without_degenerate,
        "model_by_source_summary": source_ranking,
        "model_ranking_with_alert_rank": alert_winners,
        "degenerate_alerts": degenerate,
        "label_validity_by_alert_source": label_validity,
        "weather_ablation_summary": weather_ablation,
        "family_comparison_exp19": family_comparison,
        "calibration_summary_exp19": calibration_summary,
        "cost_performance_summary": cost_performance,
        "model_alert_summary": model_alert_summary,
    }
    for stem, frame in outputs.items():
        _write_frame(frame, output_dir, stem)
    q1_figures = pd.DataFrame({"figure_path": [str(path) for path in _export_q1_figures(outputs, output_dir)]})
    outputs["q1_figures"] = q1_figures
    _write_frame(q1_figures, output_dir, "q1_figures")
    _write_xlsx(
        {
            "Ranking_Global": global_ranking,
            "Global_Sin_Rain": global_without_degenerate,
            "Ranking_Fuente": source_ranking,
            "Ganadores_Alerta": alert_winners,
            "Corridas_Completas": test,
            "Degenerate_Alerts": degenerate,
            "Label_Validity": label_validity,
            "Weather_Ablation": weather_ablation,
            "Family_Comparison": family_comparison,
            "Calibration": calibration_summary,
            "Cost_Performance": cost_performance,
        },
        output_dir / "exp19_all_models_all_alerts_resultados.xlsx",
    )
    return outputs


def summarize_results(resolved_config: dict[str, Any], run_id: str | None = None) -> Path:
    results_dir = Path(resolved_config["artifacts"]["results_dir"])
    if "weather_source_views" in resolved_config:
        base_run_id = run_id or make_run_id(resolved_config["name"], resolved_config)
        view_names: list[str] = []
        for view in resolved_config["weather_source_views"]:
            view_name = str(view["name"])
            view_names.append(view_name)
            view_resolved = _view_config(resolved_config, view)
            canonical_dataset, prepared_datasets = prepare_all_datasets(view_resolved)
            write_summary_report(
                metrics_root=results_dir / "metrics",
                tables_root=results_dir / "tables",
                run_id=f"{base_run_id}__{view_name}",
                prepared_datasets=prepared_datasets,
                canonical_dataset=canonical_dataset,
                figures_root=results_dir / "figures",
            )
        return _weather_source_tables(results_dir / "metrics", results_dir / "tables", base_run_id, view_names)

    canonical_dataset, prepared_datasets = prepare_all_datasets(resolved_config)
    return write_summary_report(
        metrics_root=results_dir / "metrics",
        tables_root=results_dir / "tables",
        run_id=run_id,
        prepared_datasets=prepared_datasets,
        canonical_dataset=canonical_dataset,
        figures_root=results_dir / "figures",
    )
