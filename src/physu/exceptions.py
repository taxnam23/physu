"""Custom exception hierarchy for physical domain and library errors in physu."""

from __future__ import annotations


class PhysuError(Exception):
    """Base exception for all physu-specific errors."""


class PhysicalDomainError(PhysuError, ValueError):
    """Raised when an input violates a physical domain constraint.

    Examples include negative time or mass where non-physical, zero radius or
    denominator, or negative discriminants that yield non-real solutions.
    """


__all__ = ["PhysicalDomainError", "PhysuError"]
