"""Compatibility shim for the archived Wilson Stage 3 variant."""

from __future__ import annotations

from ..archive.transformers_stage3.wilson_tcn_multiscale import WilsonTCNMultiscaleClassifier, WilsonTCNMultiscaleNetwork

__all__ = ["WilsonTCNMultiscaleClassifier", "WilsonTCNMultiscaleNetwork"]
