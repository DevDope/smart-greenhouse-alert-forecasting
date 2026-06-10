"""Compatibility shim for the archived Zappa Stage 3 variant."""

from __future__ import annotations

from ..archive.transformers_stage3.zappa_mmoe import ZappaMMoEClassifier, ZappaMMoENetwork

__all__ = ["ZappaMMoEClassifier", "ZappaMMoENetwork"]
