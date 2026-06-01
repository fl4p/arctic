class ArcticException(Exception):
    pass


class NoDataFoundException(ArcticException):
    pass


class UnhandledDtypeException(ArcticException):
    pass


class LibraryNotFoundException(ArcticException):
    pass


class DuplicateSnapshotException(ArcticException):
    pass


class StoreNotInitializedException(ArcticException):
    pass


class OptimisticLockException(ArcticException):
    pass


class QuotaExceededException(ArcticException):
    pass


class UnsupportedPickleStoreVersion(ArcticException):
    pass


class DataIntegrityException(ArcticException):
    """
    Base class for data integrity issues.
    """
    pass


class ArcticSerializationException(ArcticException):
    pass


class ConcurrentModificationException(DataIntegrityException):
    pass


class UnorderedDataException(DataIntegrityException, ValueError):
    # Also a ValueError: unordered / non-monotonic input data is a value error, and callers (and
    # tests) have long caught ValueError for this. Multiple inheritance keeps the specific type and
    # the ArcticException/DataIntegrityException lineage while making `except ValueError` catch it.
    pass


class OverlappingDataException(DataIntegrityException):
    pass


class AsyncArcticException(ArcticException):
    pass


class RequestDurationException(AsyncArcticException):
    pass
