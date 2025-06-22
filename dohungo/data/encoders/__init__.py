"""Board position encoders for dohungo."""
from __future__ import annotations

from .base import Encoder
from .sevenplane import SevenPlaneEncoder
from .simple import SimpleEncoder

__all__ = ["Encoder", "SevenPlaneEncoder", "SimpleEncoder"] 