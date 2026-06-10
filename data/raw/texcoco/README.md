# Texcoco Data Folder

This folder contains the compressed dataset used by the benchmark.

Before running the benchmark, extract:

```text
texcoco.rar
```

in this same directory so the following file exists:

```text
proto_enriched_outside_weather.csv
```

The pipeline intentionally reads the extracted CSV from a relative path:

```text
data/raw/texcoco/proto_enriched_outside_weather.csv
```

Use `python scripts/reviewer_tui.py` to verify the archive hash and whether the CSV is present.
