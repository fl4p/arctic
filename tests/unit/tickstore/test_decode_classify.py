"""Unit tests for the bucket-decode failure classifier (_decode_or_classify / _reprobe_bucket).

On a tickstore bucket decode failure (e.g. a pco / LnQ16_pco "Corruption error") the reader re-fetches the
SAME doc from mongo by _id and retries the decode once, to tell a TRANSIENT torn/short read (re-fetch decodes
clean) from CORRUPT-AT-REST data (re-fetch fails identically), and re-raises an error that NAMES the bucket.
This is what makes a downstream failure log (lib.exec.cluster) actionable instead of an opaque 'pco error'."""
import pytest

from arctic.exceptions import ArcticException
from arctic.tickstore.tickstore import (_decode_or_classify, _reprobe_bucket,
                                        SYMBOL, START, END, ID, INDEX_COMPRESSION, VERSION)

DOC = {SYMBOL: 'BTC/NGN@luno', START: '2024-06-01', END: '2024-06-15', ID: 'deadbeef',
       INDEX_COMPRESSION: 'pco', VERSION: 4}


class _Coll:
    """Minimal mongo collection stand-in: find_one returns the doc handed at construction (or None)."""
    name = 'cc_l2_ft2'

    def __init__(self, fresh):
        self._fresh = fresh

    def find_one(self, _query):
        return self._fresh


def test_success_passes_through_untouched():
    # the happy path must not touch the collection at all
    sentinel = object()
    assert _decode_or_classify(None, DOC, 'index', lambda d: sentinel) is sentinel


def test_transient_when_refetch_decodes_clean():
    calls = {'n': 0}

    def decode(d):                      # fail on the original doc, succeed on the re-fetched copy
        calls['n'] += 1
        if calls['n'] == 1:
            raise RuntimeError("pco error: expected trailing bits at end of page to be empty")
        return 'ok'

    with pytest.raises(ArcticException) as ei:
        _decode_or_classify(_Coll(fresh=DOC), DOC, 'index', decode)
    msg = str(ei.value)
    assert 'CLASSIFY=TRANSIENT' in msg
    assert 'symbol=BTC/NGN@luno' in msg and '_id=deadbeef' in msg and '[index]' in msg


def test_corrupt_at_rest_when_refetch_fails_identically():
    def decode(_d):
        raise RuntimeError("pco error: expected trailing bits at end of page to be empty")

    with pytest.raises(ArcticException) as ei:
        _decode_or_classify(_Coll(fresh=DOC), DOC, 'column:volBid_20p', decode)
    assert 'CLASSIFY=CORRUPT-AT-REST' in str(ei.value)
    assert 'column:volBid_20p' in str(ei.value)


def test_unknown_when_refetch_returns_none():
    def decode(_d):
        raise RuntimeError("pco error: ...")

    with pytest.raises(ArcticException) as ei:
        _decode_or_classify(_Coll(fresh=None), DOC, 'index', decode)
    assert 'CLASSIFY=UNKNOWN' in str(ei.value)


def test_unknown_when_no_collection_no_db_decode():
    # a no-DB decode (collection None, e.g. the _decode_bucket_no_db test path) can still fail-classify
    def decode(_d):
        raise RuntimeError("pco error: ...")

    verdict, _detail = _reprobe_bucket(None, DOC, decode)
    assert verdict == 'CLASSIFY=UNKNOWN'


def test_probe_never_masks_original_when_refetch_errors():
    # if the re-fetch itself raises, we must still classify UNKNOWN (never let the probe's error escape)
    class _Boom:
        name = 'x'

        def find_one(self, _q):
            raise RuntimeError('mongo down')

    def decode(_d):
        raise RuntimeError("pco error: ...")

    with pytest.raises(ArcticException) as ei:
        _decode_or_classify(_Boom(), DOC, 'index', decode)
    assert 'CLASSIFY=UNKNOWN' in str(ei.value)
