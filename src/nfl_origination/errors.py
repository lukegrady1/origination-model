"""Exception hierarchy mapped to CLI exit codes (spec section 15)."""

from __future__ import annotations


class OriginationError(Exception):
    """Base class for expected, user-facing failures."""

    exit_code: int = 1


class InvalidInputError(OriginationError):
    """Invalid CLI input or configuration (exit code 2)."""

    exit_code = 2


class MissingDataError(OriginationError):
    """Missing or ineligible data (exit code 3)."""

    exit_code = 3


class ModelValidationError(OriginationError):
    """Failed numerical or model validation (exit code 4)."""

    exit_code = 4
