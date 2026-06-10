"""Compatibility shim for the archived Carey Stage 3 variant."""

from __future__ import annotations

from ..archive.transformers_stage3.carey_tcn_fourier import CareyTCNFourierClassifier, CareyTCNFourierNetwork

__all__ = ["CareyTCNFourierClassifier", "CareyTCNFourierNetwork"]
