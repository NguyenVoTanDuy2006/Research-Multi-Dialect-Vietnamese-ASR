"""Shared exception hierarchy for the ViMD ASR toolkit."""


class VimdASRError(Exception):
    """Base class for expected, user-actionable toolkit errors.

    Catch this when a caller wants to handle any known application error while
    allowing unexpected programming errors (for example ``TypeError``) through.
    Subclasses also inherit an appropriate built-in exception type so existing
    code catching ``ValueError``, ``LookupError`` or ``RuntimeError`` keeps
    working.
    """


class ConfigError(VimdASRError, ValueError):
    """Experiment configuration is missing, malformed, or inconsistent."""


class AmbiguousSampleError(VimdASRError, LookupError):
    """A sample lookup resolved to multiple dataset records."""


class DatasetIntegrityError(VimdASRError, RuntimeError):
    """Dataset metadata, audio payloads, or split membership are inconsistent."""


class CheckpointMismatchError(VimdASRError, RuntimeError):
    """A model checkpoint does not match the requested base/model state."""


class ResumeMismatchError(VimdASRError, RuntimeError):
    """A cached run or training checkpoint belongs to another experiment."""


class IncompleteEvaluationError(VimdASRError, RuntimeError):
    """An evaluation lacks required predictions or comparable sample IDs."""

