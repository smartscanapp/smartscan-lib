

from enum import Enum
from typing import Dict, Optional, Union

class ErrorCode(Enum):
    """Enum for error codes used by client."""

    BELOW_SIMILARITY_THRESHOLD = "BELOW_SIMILARITY_THRESHOLD"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    PROTOTYPE_GENERATION_ERROR = "PROTOTYPE_GENERATION_ERROR"
    INVALID_MODEL_PATH = "INVALID_MODEL_PATH"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"

class SmartScanError(Exception):
    """Base class for all SmartScan related errors."""

    def __init__(self, message: str, code: Optional[ErrorCode] = None, details: Optional[Union[Dict, str, object]] = None):
        if details is None:
            details = {}
        self.message = message
        self.code = code
        self.details = details
        super().__init__(message)
