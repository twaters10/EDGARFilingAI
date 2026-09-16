"""SEC EDGAR access: identifiers, endpoints, throttling, caching, and the client."""

from .cache import CacheStats, ResponseCache
from .client import EdgarClient, EdgarHTTPError
from .identifiers import (
    Company,
    InvalidCIKError,
    TickerResolver,
    UnknownTickerError,
    normalize_cik,
    to_archives_cik,
    to_data_api_cik,
)
from .throttle import RateLimiter

__all__ = [
    "CacheStats",
    "Company",
    "EdgarClient",
    "EdgarHTTPError",
    "InvalidCIKError",
    "RateLimiter",
    "ResponseCache",
    "TickerResolver",
    "UnknownTickerError",
    "normalize_cik",
    "to_archives_cik",
    "to_data_api_cik",
]
