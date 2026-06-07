from __future__ import print_function, annotations

import copy
import datetime
import logging
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime as dt, timedelta
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pymongo
from bson.binary import Binary
from pymongo import ReadPreference
from pymongo.errors import OperationFailure
from pymongo.results import InsertManyResult

from .coding import nparray_varint_encode, nparray_varint_decode, decode_logQ16_10_dzv, \
    LnQ16_VQL, LnQ16_zlib, binary_compressors, binary_decompressors, LnQ16, LnQ16_pco, \
    pco_encode, pco_decode

try:
    from pandas.core.frame import _arrays_to_mgr
except ImportError:
    # Deprecated since pandas 0.23.4
    from pandas.core.internals.construction import arrays_to_mgr as _arrays_to_mgr

try:
    from pandas.api.types import infer_dtype
except ImportError:
    from pandas.lib import infer_dtype

from ..date import DateRange, to_pandas_closed_closed, mktz, datetime_to_ms, ms_to_datetime, CLOSED_CLOSED, to_dt, \
    utc_dt_to_local_dt
from ..decorators import mongo_retry
from ..exceptions import OverlappingDataException, NoDataFoundException, UnorderedDataException, \
    UnhandledDtypeException, ArcticException
from .._util import indent, mongo_count

try:
    from lz4.block import compress as lz4_compress, decompress as lz4_decompress


    def lz4_compressHC(_str):  # ordinary function so name appears in profile
        return lz4_compress(_str, mode='high_compression')
except ImportError as e:
    from lz4 import compress as lz4_compress, compressHC as lz4_compressHC, decompress as lz4_decompress

PD_VER = pd.__version__
logger = logging.getLogger(__name__)

# Example-Schema:
# --------------
# {ID: ObjectId('52b1d39eed5066ab5e87a56d'),
#  SYMBOL: u'symbol'
#  INDEX: Binary('...', 0),
#  IMAGE_DOC: { IMAGE:  {
#                          'ASK': 10.
#                          ...
#                        }
#              's': <sequence_no>
#              't': DateTime(...)
#             }
#  COLUMNS: {
#   'ACT_FLAG1': {
#        DATA: Binary('...', 0),
#        DTYPE: u'U1',
#        ROWMASK: Binary('...', 0)},
#   'ACVOL_1': {
#        DATA: Binary('...', 0),
#        DTYPE: u'float64',
#        ROWMASK: Binary('...', 0)},
#               ...
#    }
#  START: DateTime(...),
#  END: DateTime(...),
#  END_SEQ: 31553879L,
#  SEGMENT: 1386933906826L,
#  SHA: 1386933906826L,
#  VERSION: 3,
# }

TICK_STORE_TYPE = 'TickStoreV3'

ID = '_id'
SYMBOL = 'sy'
INDEX = 'i'
START = 's'
END = 'e'
START_SEQ = 'sS'
END_SEQ = 'eS'
SEGMENT = 'se'
SHA = 'sh'
IMAGE_DOC = 'im'
IMAGE = 'i'

COLUMNS = 'cs'
DATA = 'd'
DTYPE = 't'
CODEC = 'c'
IMAGE_TIME = 't'
ROWMASK = 'm'
GAIN = 'g'  # factor, (pre)scaler, gain,

COUNT = 'c'
VERSION = 'v'
INDEX_PRECISION = 'p'
INDEX_COMPRESSION = 'd'

META = 'md'

CHUNK_VERSION_NUMBER = 3
CHUNK_VERSION_NUMBER_MAX = 4

import warnings
import pandas as pd

# turns out that pandas 3 allows that: pd.DatetimeIndex(np.uint64(ns), tz=utc)
warnings.filterwarnings(
    action='ignore', category=FutureWarning,
    message=r".*Indexing a timezone-aware DatetimeIndex with a timezone-naive datetime is deprecated and will raise KeyError in a future version.*")

# version.parse(pandas.__version__) > version.parse('1.0')
# IS_PANDAS_1x = version.parse(pandas.__version__) > version.parse('1.0')


codec_registry = {}


def register_codec(name, obj):
    # validate with raises (not asserts) so registration is still checked under `python -O`;
    # the name is persisted as the on-disk CODEC field and must round-trip on read.
    if not (hasattr(obj, 'encode') and callable(obj.encode)):
        raise ValueError("codec %r must have a callable encode()" % name)
    if not (hasattr(obj, 'decode') and callable(obj.decode)):
        raise ValueError("codec %r must have a callable decode()" % name)
    if not (hasattr(obj, 'rtol_max') and 0 <= obj.rtol_max <= 0.05):
        raise ValueError("codec %r must define rtol_max in [0, 0.05]" % name)
    if not (hasattr(obj, 'rtol_reg') and 0 <= obj.rtol_reg <= 0.0005):
        raise ValueError("codec %r must define rtol_reg in [0, 0.0005]" % name)
    if name in codec_registry:
        raise ValueError("codec %r already registered" % name)
    if len(name) >= 16:
        raise ValueError("codec name %r too long (must be < 16 chars)" % name)
    codec_registry[name] = obj


LnQ25VQLgz = LnQ16_VQL(loq_loss=25, comp='gz')
register_codec('LnQ25VQLgz', LnQ25VQLgz)  # for l2 data, general purpose [1e-13, 2e+27]
register_codec('LnQ15VQLlz4', LnQ16_VQL(loq_loss=15, comp='lz4'))  # for price data, fast
register_codec('LnQ15gz', LnQ16_zlib(loq_loss=15))  # for signed trade qty, no VQL (no auto-corr)
# TODO like LnQ25VQLgz but smaller prescale to trade small number precision for bigger range (up to 2e35 or so)
# this one: LnQ16_VQL(loq_loss=25, comp='zlib', log_prescale=24, loq_preadd=1e-8)

LnQ20VQLgz = LnQ16_VQL(loq_loss=20, comp='gz', log_prescale=20, loq_preadd=1e-12)
register_codec('LnQ20VQLgz', LnQ20VQLgz)  # for l2 data, general purpose [1e-15, 2e+32]

register_codec('LnQ15br9', LnQ16('br_9', 15))  # for qty, 115ppm, abs(x) >= 0.73e-11
# Deprecated for writes: superseded by LnQ185pco (same loss=1.85/prescale=0 grid -> identical ~18ppm
# error and dynamic range, but ~6-15% smaller on real px and far cheaper to (de)code). Kept registered
# so buckets already written with it still decode; only new writes are blocked.
_LnQ185VQLgz = LnQ16_VQL('gz', 1.85, 0, 0)  # px 16ppm, down to 1e-45, then overflow
_LnQ185VQLgz.deprecated = 'LnQ185pco'
register_codec('LnQ185VQLgz', _LnQ185VQLgz)

# brotli q10 + smallest window (lgwin=10) on the VQL pipeline: ~24-30% smaller than LnQ20VQLgz at
# ~232ppm (loss=30 uses the full 250ppm budget). Decode ~1.9x the column-data cost of gz (the slower
# brotli decompress); index/rowmask are unaffected (lz4). Best size/decode point found on real L2
# data (beats lzma on decode, zstd on size). See test/coding/find_coders.py in the jnb repo.
_LnQ30brW10q10 = LnQ16_VQL('br_w10q10', loq_loss=30, log_prescale=20, loq_preadd=1e-12)
_LnQ30brW10q10.rtol_max = 250e-6  # loss=30 inherent error ~232ppm; default 200ppm would fail verify
register_codec('LnQ30brW10q10', _LnQ30brW10q10)

# Same loss=30 grid (~232ppm) as LnQ30brW10q10, but the int stream goes through pcodec (the `pco`
# crate) instead of delta+varint+brotli. ~equal size on L2, but encodes ~37x cheaper and decodes ~3x
# faster -- the better write-once/read-many choice. Requires the optional `pcodec` package; the codec
# only imports it on first encode/decode, so registration here is import-safe without it.
_LnQ30pco = LnQ16_pco(loq_loss=30, log_prescale=20, loq_preadd=1e-12)
_LnQ30pco.rtol_max = 250e-6  # loss=30 inherent error ~232ppm; default 200ppm would fail verify
register_codec('LnQ30pco', _LnQ30pco)

# Signed pcodec codec for trade qty (carries a sign): same grid as LnQ15br9 (loss=15, ~115ppm) but the
# signed int stream goes through pcodec. On realistic qty (lognormal magnitudes + autocorrelated sign)
# it is ~14% smaller than LnQ15br9 and ~10x cheaper to encode. loss=15 -> 115ppm < 200ppm default, so
# no rtol_max override. Requires the optional `pcodec` package (lazily imported on first use).
register_codec('LnQ15pcoS', LnQ16_pco(loq_loss=15, log_prescale=37, loq_preadd=1e-4, signed=True))

# Wide-range pco codec for px: the pco twin of LnQ185VQLgz (loss=1.85, prescale=0, preadd=0 -> ln_q16
# takes log(x) directly, so no x*2**prescale step to overflow). Covers the full positive float32 range
# (~1e-45 .. 3.4e38) at ~18ppm, unlike LnQ30pco (prescale=20) which overflows above ~3.2e32. Use this
# for price columns that need wide dynamic range; LnQ30pco stays the choice for bounded L2/general data.
register_codec('LnQ185pco', LnQ16_pco(loq_loss=1.85, log_prescale=0, loq_preadd=0))


def index_to_ns(series, dtype=np.int64):
    try:
        # python 3 & PeriodIndex
        idx_dt = series.index.tz_convert('UTC').tz_localize(None).astype('datetime64[ns]')
        assert isinstance(idx_dt, pd.DatetimeIndex), type(idx_dt)
        idx = idx_dt.values.astype(dtype)
        # assert idx[1] > idx[0], idx[:4]
        return idx
    except TypeError:
        # python 0.x
        return series.index.tz_convert('UTC').tz_localize(None).values.astype('datetime64[ns]').astype(dtype)


def dt2ns(dt):
    if isinstance(dt, pd.Timestamp):
        return dt.value  # int
    elif isinstance(dt, np.datetime64):
        return dt.astype('datetime64[ns]').view(int)
    elif isinstance(dt, datetime.datetime):
        return pd.Timestamp(dt).value
    else:
        raise TypeError('dt must be pd.Timestamp or np.datetime64 or datetime.datetime')


def ns2dt(ns, **kwargs):
    return pd.to_datetime(ns, unit='ns', **kwargs)


def _progressbar(it, count=None, prefix="", size=60, out=sys.stdout):  # Python3.6+
    count = count or len(it)
    start = time.time()  # time estimate start

    def show(j):
        x = int(size * j / count)
        # time estimate calculation and string
        remaining = ((time.time() - start) / j) * (count - j)
        mins, sec = divmod(remaining, 60)  # limited to minutes
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(f" [{u'█' * x}{('.' * (size - x))}] {prefix} {j}/{count} Est wait {time_str}", end='\r', file=out,
              flush=True)

    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)


def _reprobe_bucket(collection, doc, decode_fn):
    """On a bucket decode failure, re-fetch the SAME doc from mongo by _id and retry the decode ONCE to
    distinguish a TRANSIENT torn/short read (re-fetch decodes clean -> the bytes at rest are fine, the
    in-flight copy was truncated) from CORRUPT-AT-REST data (re-fetch fails identically). Best-effort and
    fully self-contained: it must NEVER raise -- any probe failure (incl. no collection, e.g. a no-DB decode)
    just downgrades the verdict to UNKNOWN so the ORIGINAL decode error is never masked. Returns
    (verdict, detail) strings."""
    if collection is None:
        return 'CLASSIFY=UNKNOWN', 'no collection on this reader (cannot re-fetch)'
    _id = doc.get(ID)
    if _id is None:
        return 'CLASSIFY=UNKNOWN', 'no _id on the projected doc (cannot re-fetch)'
    try:
        fresh = collection.find_one({ID: _id})  # full doc; decode_fn touches only its own field
    except Exception as e:
        return 'CLASSIFY=UNKNOWN', 're-fetch errored: %r' % (e,)
    if fresh is None:
        return 'CLASSIFY=UNKNOWN', 're-fetch returned no doc (deleted/rewritten since)'
    try:
        decode_fn(fresh)
    except Exception as e:
        return 'CLASSIFY=CORRUPT-AT-REST', 're-fetch decoded identically-bad: %r' % (e,)
    return 'CLASSIFY=TRANSIENT', 're-fetch decoded clean -> torn/short read in flight (data at rest is OK)'


def _decode_or_classify(collection, doc, what, decode_fn):
    """Run a (pco-prone) bucket decode; on ANY failure, enrich the opaque codec error with the bucket's
    identity (symbol + date-range + _id + lib) and a TRANSIENT-vs-CORRUPT classification from a single mongo
    re-fetch probe, then re-raise (chaining the original). Turns a bare 'pco error: ...' into a record that
    NAMES the bad bucket and says whether a re-read fixes it -- captured verbatim by callers (e.g.
    lib.exec.cluster's per-task failure log). `decode_fn(d)` decodes from a doc `d` so the SAME logic runs on
    the original and on the re-fetched copy; `collection` is the reader's mongo collection (None in a no-DB
    decode -> verdict UNKNOWN). The success path never touches `collection`."""
    try:
        return decode_fn(doc)
    except Exception as orig:
        verdict, detail = _reprobe_bucket(collection, doc, decode_fn)
        raise ArcticException(
            'tickstore decode failed [%s]: symbol=%s bucket=[%s .. %s] _id=%s idx_codec=%s ver=%s lib=%s '
            '-- %s (%s); original: %r' % (
                what, doc.get(SYMBOL), doc.get(START), doc.get(END), doc.get(ID),
                doc.get(INDEX_COMPRESSION, 'lz4'), doc.get(VERSION),
                getattr(collection, 'name', '?'), verdict, detail, orig)) from orig


class TickStore(object):

    @classmethod
    def initialize_library(cls, arctic_lib, **kwargs):
        TickStore(arctic_lib)._ensure_index()

    @mongo_retry
    def _ensure_index(self):
        collection = self._collection
        collection.create_index([(START, pymongo.ASCENDING)], background=True)
        collection.create_index([(END, pymongo.ASCENDING)], background=True)
        # (SYMBOL, START, END), name "sy_1_s_1_e_1". Serves the per-symbol date-range reads (via its
        # (SYMBOL, START) prefix -- so it replaces the old standalone sy_1_s_1 index) and
        # symbols_in_range(date_range=...): leading on SYMBOL lets the $group use a DISTINCT_SCAN
        # (hop between distinct symbols rather than scan every chunk), with START/END covered too,
        # so it stays an index-only scan that never loads chunk payloads.
        collection.create_index([(SYMBOL, pymongo.ASCENDING),
                                 (START, pymongo.ASCENDING),
                                 (END, pymongo.ASCENDING)], background=True)

        self._metadata.create_index([(SYMBOL, pymongo.ASCENDING)], background=True, unique=True)

    def __init__(self, arctic_lib, chunk_size=100000, index_precision='ms', verify_codec=True):
        """
        Parameters
        ----------
        arctic_lib : ArcticLibraryBinding
            Arctic Library
        chunk_size : int
            Number of ticks to store in a document before splitting to another document.
            if the library was obtained through get_library then set with: self._chuck_size = 10000
        """
        self._arctic_lib = arctic_lib
        # Do we allow reading from secondaries
        self._allow_secondary = self._arctic_lib.arctic._allow_secondary
        self._chunk_size = chunk_size
        # assert index_precision in ('ms', 's')
        self._index_precision = index_precision
        self._verify_codec = verify_codec
        self._reset()

    @mongo_retry
    def _reset(self):
        # The default collections
        self._collection = self._arctic_lib.get_top_level_collection()
        self._metadata = self._collection.metadata

    def __getstate__(self):
        return {'arctic_lib': self._arctic_lib, 'chunk_size': self._chunk_size,
                'index_precision': self._index_precision,
                'verify_codec': self._verify_codec}

    def __setstate__(self, state):
        return TickStore.__init__(self, state['arctic_lib'], chunk_size=state['chunk_size'],
                                  index_precision=state['index_precision'],
                                  verify_codec=state['verify_codec'])

    def __str__(self):
        return """<%s at %s>
%s""" % (self.__class__.__name__, hex(id(self)), indent(str(self._arctic_lib), 4))

    def __repr__(self):
        return str(self)

    def delete(self, symbol, date_range=None):
        """
        Delete all chunks for a symbol.

        Which are, for the moment, fully contained in the passed in
        date_range.

        Parameters
        ----------
        symbol : `str`
            symbol name for the item
        date_range : `date.DateRange`
            DateRange to delete ticks in
        """
        query = {SYMBOL: symbol}
        date_range = to_pandas_closed_closed(date_range)
        if date_range is not None:
            assert date_range.start and date_range.end
            query[START] = {'$gte': date_range.start}
            query[END] = {'$lte': date_range.end}
        else:
            # delete metadata on complete deletion
            self._metadata.delete_one({SYMBOL: symbol})
        return self._collection.delete_many(query)

    def list_symbols(self, date_range=None, columns=None, regex=None) -> list:
        """
        List the symbols in the store, optionally restricted to those that have data
        overlapping ``date_range`` and/or hold the given column(s) and/or match ``regex``.

        With no arguments this returns *every* symbol via a single covered DISTINCT_SCAN
        over the ``(SYMBOL, ...)`` index; its cost is proportional to the number of symbols
        and can't be reduced further. Passing a filter returns only the relevant subset,
        which is much cheaper than listing them all.

        Parameters
        ----------
        date_range : `date.DateRange`
            Only consider chunks that overlap this range. A chunk overlaps
            ``[start, end]`` iff ``chunk.start <= end`` and ``chunk.end >= start``.
            ``None`` (or open-ended bounds) means unbounded on that side, i.e.
            the whole stored history.
        columns : `str` or `list` of `str`
            If given, only return symbols that have at least one in-range chunk
            containing *every* one of these columns. Since columns are stored
            per-chunk, a symbol qualifies as long as one of its overlapping
            chunks carries all the requested columns.
        regex : `str`
            If given, only return symbols whose name matches this regular
            expression (Mongo ``$regex``).

        Returns
        -------
        list of `str`
            Sorted list of distinct symbol names.

        Notes
        -----
        All paths key off the ``(SYMBOL, START, END)`` index (``sy_1_s_1_e_1``, created by
        ``_ensure_index``; on a pre-existing library create it once by hand). It must lead with
        SYMBOL: that is what lets Mongo enumerate symbols with a DISTINCT_SCAN (one key per symbol)
        rather than a COLLSCAN that fetches every chunk document.

        A ``regex`` is handled in two cheap steps rather than one slow ``$match`` over the whole
        range. ``'^[A-Z](XBT|BTC).+'`` has no literal prefix, so it can't bound the index, and
        ``distinct(SYMBOL, {SYMBOL: regex})`` falls back to a fetch/scan. Instead the candidates are
        enumerated with ``[{$group: sy}, {$match: regex}]`` -- a DISTINCT_SCAN whose regex runs on the
        small grouped set -- and the date overlap is then resolved for just those candidates with a
        single ``{SYMBOL: {$in: candidates}, ...}`` aggregation (the ``$in`` lets the index drive one
        seek per candidate). Doing the overlap per-symbol with separate queries instead would cost a
        network round-trip per candidate.
        """
        if isinstance(columns, str):
            columns = [columns]

        date_range = to_pandas_closed_closed(date_range)
        start = date_range.start if date_range is not None else None
        end = date_range.end if date_range is not None else None
        has_time_bound = start is not None or end is not None

        coll = self._collection

        def overlap_match(extra):
            # A chunk overlaps [start, end] iff chunk.START <= end and chunk.END >= start.
            m = dict(extra)
            if end is not None:
                m[START] = {'$lte': end}
            if start is not None:
                m[END] = {'$gte': start}
            return m

        if columns is not None:
            # Strict semantics: a *single* chunk must both overlap the range and carry every
            # requested column. $exists on a cs.<col> sub-path is not index-coverable so this scans;
            # it is the non-hot path. Combine everything into one server-side $match + $group.
            match = {}
            if regex is not None:
                match[SYMBOL] = {'$regex': regex}
            for c in columns:
                match['%s.%s' % (COLUMNS, c)] = {'$exists': True}
            pipeline = [{'$match': overlap_match(match)}, {'$group': {'_id': '$' + SYMBOL}}]
            return sorted(doc['_id'] for doc in coll.aggregate(pipeline, allowDiskUse=True))

        if regex is not None:
            # Enumerate candidates index-only: $group on SYMBOL is a DISTINCT_SCAN (one key per
            # symbol); the regex is applied AFTER the group, on the small grouped set, so it does
            # not defeat the index the way distinct(SYMBOL, {sy: regex}) does.
            candidates = [doc['_id'] for doc in coll.aggregate(
                [{'$group': {'_id': '$' + SYMBOL}}, {'$match': {'_id': {'$regex': regex}}}],
                allowDiskUse=True)]
            if not has_time_bound:
                return sorted(candidates)
            if not candidates:
                return []
            # Resolve the date overlap for just those candidates in a single round trip; $in lets
            # the (SYMBOL, START, END) index seek per candidate instead of scanning the whole range.
            base = {SYMBOL: {'$in': candidates}}
        elif not has_time_bound:
            # No regex, no time bound -> plain DISTINCT_SCAN over every symbol.
            return sorted(coll.distinct(SYMBOL))
        else:
            # No regex but a time bound -> overlap over all symbols (uses the START/END indexes).
            base = {}

        pipeline = [{'$match': overlap_match(base)}, {'$group': {'_id': '$' + SYMBOL}}]
        return sorted(doc['_id'] for doc in coll.aggregate(pipeline, allowDiskUse=True))

    def symbols_in_range(self, *args, **kwargs):
        """Deprecated alias for :meth:`list_symbols` (same signature)."""
        warnings.warn("TickStore.symbols_in_range is deprecated; use list_symbols",
                      DeprecationWarning, stacklevel=2)
        return self.list_symbols(*args, **kwargs)

    def _mongo_date_range_query(self, symbol, date_range):
        # Handle date_range
        if not date_range:
            date_range = DateRange()

        # We're assuming CLOSED_CLOSED on these Mongo queries
        assert date_range.interval == CLOSED_CLOSED

        # Since we only index on the start of the chunk,
        # we do a pre-flight aggregate query to find the point where the
        # earliest relevant chunk starts.

        start_range = {}
        first_dt = last_dt = None
        if date_range.start:
            assert date_range.start.tzinfo
            start = date_range.start

            # If all chunks start inside of the range, we default to capping to our
            # range so that we don't fetch any chunks from the beginning of time
            start_range['$gte'] = start

            match = self._symbol_query(symbol)
            match.update({'s': {'$lte': start}})

            result = self._collection.aggregate([
                # Only look at the symbols we are interested in and chunks that
                # start before our start datetime
                {'$match': match},
                # Throw away everything but the start of every chunk and the symbol
                {'$project': {'_id': 0, 's': 1, 'sy': 1}},
                # For every symbol, get the latest chunk start (that is still before
                # our sought start)
                {'$group': {'_id': '$sy', 'start': {'$max': '$s'}}},
                {'$sort': {'start': 1}},
            ])
            # Now we need to get the earliest start of the chunk that still spans the start point.
            # Since we got them sorted by start, we just need to fetch their ends as well and stop
            # when we've seen the first such chunk
            try:
                for candidate in result:
                    chunk = self._collection.find_one({'s': candidate['start'], 'sy': candidate['_id']}, {'e': 1})
                    if chunk['e'].replace(tzinfo=mktz('UTC')) >= start:
                        start_range['$gte'] = candidate['start'].replace(tzinfo=mktz('UTC'))
                        break
            except StopIteration:
                pass

        # Find the end bound
        if date_range.end:
            # If we have an end, we are only interested in the chunks that start before the end.
            assert date_range.end.tzinfo
            last_dt = date_range.end
        else:
            logger.warning("No end provided.  Loading a month for: {}:{}".format(symbol, first_dt))
            if not first_dt:
                first_doc = self._collection.find_one(self._symbol_query(symbol),
                                                      projection={START: 1, ID: 0},
                                                      sort=[(START, pymongo.ASCENDING)])
                if not first_doc:
                    raise NoDataFoundException()

                first_dt = first_doc[START]
            last_dt = first_dt + timedelta(days=30)
        if last_dt:
            start_range['$lte'] = last_dt

        # Return chunks in the specified range
        if not start_range:
            return {}
        return {START: start_range}

    def _symbol_query(self, symbol):
        if isinstance(symbol, str):
            query = {SYMBOL: symbol}
        elif symbol is not None:
            query = {SYMBOL: {'$in': symbol}}
        else:
            query = {}
        return query

    def _read_preference(self, allow_secondary):
        """ Return the mongo read preference given an 'allow_secondary' argument
        """
        allow_secondary = self._allow_secondary if allow_secondary is None else allow_secondary
        return ReadPreference.NEAREST if allow_secondary else ReadPreference.PRIMARY

    def _fetch_and_decode(self, symbol, date_range, columns, allow_secondary, include_images, show_progress,
                          _target_tick_count):
        rtn = {}
        column_set = set()

        multiple_symbols = not isinstance(symbol, str)

        date_range = to_pandas_closed_closed(date_range)
        query = self._symbol_query(symbol)
        query.update(self._mongo_date_range_query(symbol, date_range))

        projection = TickStore._read_projection(columns)
        if columns:
            column_set.update([c for c in columns if c != 'SYMBOL'])

        column_dtypes = {}

        ticks_read = 0
        bytes_i = 0  # compressed index size
        bytes_d = 0
        bytes_m = 0
        data_coll = self._collection.with_options(read_preference=self._read_preference(allow_secondary))

        cursor = data_coll.find(query, projection=projection).sort([(START, pymongo.ASCENDING)], )
        num = mongo_count(data_coll, filter=query)
        if show_progress:
            if num > 20:
                cursor = _progressbar(cursor, prefix=symbol, count=num)

        buckets = []
        for doc in cursor:
            assert doc[SYMBOL] == symbol
            data = self._decode_bucket(doc, column_set, column_dtypes,
                                       multiple_symbols or (columns is not None and 'SYMBOL' in columns),
                                       include_images, columns)
            buckets.append(data)
            bytes_i += len(doc[INDEX])
            bytes_d += sum(len(c[DATA]) for c in doc[COLUMNS].values())
            bytes_m += sum(len(c[ROWMASK]) for c in doc[COLUMNS].values())

        assert len(buckets) == num

        # sort buckets in case they arrived out-of-order (might happen with multiprocessing)
        # buckets = sorted(buckets, key=lambda b: b[INDEX][0])

        t = -(2 << 62)  # int64 min
        for data in buckets:
            # check for overlapping blocks
            if len(data[INDEX]) == 0:
                assert not data.get(ROWMASK)
                assert not data.get(COLUMNS)
                continue
            if data[INDEX][0] < t:
                # Chunks are validated non-overlapping on write and fetched START-sorted, so this is
                # defence-in-depth. Log the block layout for forensics (through the logger, not stdout,
                # and only when this actually fires) and fail clearly rather than returning shuffled
                # ticks. Equal-range blocks are usually just duplicates -- group them so the log shows
                # whether that is the case, without the old per-block pickle asserts (which raised
                # AssertionError before this ValueError, and compiled out entirely under `python -O`).
                grouped = defaultdict(list)
                for b in buckets:
                    if len(b[INDEX]):
                        grouped[(b[INDEX][0], b[INDEX][-1])].append(b)
                logger.error("%s %s: overlapping/out-of-order blocks. Ranges (start, end, count):\n%s",
                             self._arctic_lib, symbol,
                             '\n'.join('%s  x%d' % (ms_to_iso((s, e)), len(g)) for (s, e), g in grouped.items()))
                raise ValueError('overlap block: prev ends at %s, curr start at %s' % ms_to_iso((t, data[INDEX][0])))

            assert data[INDEX][0] <= data[INDEX][-1]
            t = data[INDEX][-1]

            for k, v in data.items():
                try:
                    rtn[k].append(v)
                except KeyError:
                    rtn[k] = [v]

            # For testing
            ticks_read += len(data[INDEX])

            if _target_tick_count and ticks_read > _target_tick_count:
                break

        self.last_read_bytes = bytes_i, bytes_d, bytes_m

        return rtn, column_dtypes

    def read(self, symbol, date_range=None, columns=None, include_images=False, allow_secondary=None,
             _target_tick_count=0, show_progress=False):
        """
        Read data for the named symbol. Returns a ``pandas.DataFrame`` of the stored ticks
        (see the Returns section); use ``read_metadata`` for the per-symbol metadata.

        Parameters
        ----------
        symbol : `str`
            symbol name for the item
        date_range : `date.DateRange`
            Returns ticks in the specified DateRange
        columns : `list` of `str`
            Columns (fields) to return from the tickstore
        include_images : `bool`
            Should images (/snapshots) be included in the read
        allow_secondary : `bool` or `None`
            Override the default behavior for allowing reads from secondary members of a cluster:
            `None` : use the settings from the top-level `Arctic` object used to query this version store.
            `True` : allow reads from secondary members
            `False` : only allow reads from primary members

        Returns
        -------
        pandas.DataFrame of data
        """
        perf_start = dt.now()

        date_range = to_pandas_closed_closed(date_range)

        rtn, column_dtypes = self._fetch_and_decode(symbol, date_range, columns,
                                                    allow_secondary=allow_secondary,
                                                    include_images=include_images,
                                                    show_progress=show_progress,
                                                    _target_tick_count=_target_tick_count)

        if not rtn:
            raise NoDataFoundException("No Data found for {} in range: {}".format(symbol, date_range))

        multiple_symbols = not isinstance(symbol, str)

        # sort buckets in case they arrived out of order

        rtn = self._pad_and_fix_dtypes(rtn, column_dtypes)

        # index = pd.to_datetime(np.concatenate(rtn[INDEX]), utc=True, unit='ms') # this is slow
        idx_ns = TickStore._index_ms_to_ns(rtn[INDEX])
        index = pd.DatetimeIndex(idx_ns, tz=datetime.timezone.utc)  # this has ~zero cost!
        # and we already validated our index during write

        if columns is None:
            columns = [x for x in rtn.keys() if x not in (INDEX, 'SYMBOL')]
        if multiple_symbols and 'SYMBOL' not in columns:
            columns = ['SYMBOL', ] + columns

        if len(index) > 0:
            arrays = [np.concatenate(rtn[k]) for k in columns]
        else:
            arrays = [[] for _ in columns]

        if multiple_symbols:
            sort = np.argsort(index, kind='mergesort')
            index = index[sort]
            arrays = [a[sort] for a in arrays]

        t = (dt.now() - perf_start).total_seconds()
        logger.info("Got data in %s secs, creating DataFrame..." % t)
        rtn = TickStore._arrays_to_dataframe(arrays, columns, index)
        # Present data in the user's default TimeZone
        rtn.index = rtn.index.tz_convert(mktz())

        t = (dt.now() - perf_start).total_seconds()
        ticks = len(rtn)
        rate = int(ticks / t) if t != 0 else float("nan")
        logger.info("%d rows in %s secs: %s ticks/sec" % (ticks, t, rate))
        if not rtn.index.is_monotonic_increasing:
            # The write path and _fetch_and_decode's cross-bucket overlap check should guarantee a
            # monotonic result, so this is defence-in-depth. Recover by sorting (stable, to preserve
            # the relative order of equal timestamps) rather than crashing the read.
            nmi = np.argmin(np.diff(rtn.index.values))
            logger.error("TimeSeries data is out of order for %s, sorting! First disorder near:\n%s",
                         symbol, '\n'.join(map(str, rtn.index.values[nmi - 2:nmi + 10])))
            rtn = rtn.sort_index(kind='mergesort')

        if date_range:
            # FIXME: support DateRange.interval...
            pi = self._index_quant_ns
            #if dt2ns(date_range.end) % pi or dt2ns(date_range.start) % pi:
            #    warnings.warn('read(): DateRange timestamps are sub index-precision %s' % (self._index_precision))
            # ^ TODO enable warning if CLOSED_CLOSED
            # if date_range.start > index[0] and date_range.end < index[-1]:
            # rtn1 = rtn.loc[date_range.start:date_range.end] # this is very sloooow
            # rtn = TickStore._fast_time_slice(rtn, date_range.start, date_range.end)
            # Bound off the *final* frame index, not the original idx_ns: for a multi-symbol read
            # the rows were argsorted (and the non-monotonic recovery above may have re-sorted them),
            # so idx_ns no longer matches rtn's order -- searchsorted on it would return wrong bounds.
            # rtn.index is a tz-aware DatetimeIndex; .values is datetime64[ns] in UTC, so view(int64)
            # is ns-since-epoch UTC, matching dt2ns() (Timestamp.value). Sorted by construction here.
            final_ns = rtn.index.values.view(np.int64)
            istart = np.searchsorted(final_ns, dt2ns(date_range.start), side='left')
            iend = np.searchsorted(final_ns, dt2ns(date_range.end), side='right')
            rtn = rtn.iloc[istart:iend]  # this is fast

            # assert len(rtn1) == len(rtn)
            # assert rtn1.index[0] ==rtn.index[0], (rtn1.index, rtn.index)
            # assert (rtn1.index == rtn.index).all()
            # pd.testing.assert_frame_equal(rtn1, rtn, check_exact=True)

        return rtn

    def get_blocks(self, symbol, date_range=None, allow_secondary=None):
        date_range = to_pandas_closed_closed(date_range)

        multiple_symbols = not isinstance(symbol, str)

        date_range = to_pandas_closed_closed(date_range)
        query = self._symbol_query(symbol)
        assert not date_range, "not implemented"
        #query.update(self._mongo_date_range_query(symbol, date_range))

        projection = dict([(SYMBOL, 1),
                               (INDEX_PRECISION, 1),
                               (START, 1),
                               (END, 1),
                               (VERSION, 1)
                               ])

        data_coll = self._collection.with_options(read_preference=self._read_preference(allow_secondary))

        cursor = data_coll.find(query, projection=projection).sort([(START, pymongo.ASCENDING)], )

        buckets = []
        for doc in cursor:
            assert doc[SYMBOL] == symbol
            buckets.append((utc_dt_to_local_dt(doc[START]), utc_dt_to_local_dt(doc[END])))

        return buckets

    @staticmethod
    def _fast_time_slice(df, index_start, index_end):
        idx_ns = index_to_ns(df, np.int64)
        istart = np.searchsorted(idx_ns, dt2ns(index_start), side='left')
        iend = np.searchsorted(idx_ns, dt2ns(index_end), side='right')
        # istart = np.searchsorted(df.index, index_start, side='left')
        # iend = np.searchsorted(df.index, index_end, side='right')
        return df.iloc[istart:iend]

    def read_metadata(self, symbol):
        """
        Read metadata for the specified symbol

        Parameters
        ----------
        symbol : `str`
            symbol name for the item

        Returns
        -------
        dict
        """
        return self._metadata.find_one({SYMBOL: symbol})[META]

    def _pad_and_fix_dtypes(self, cols, column_dtypes):
        # Pad out Nones with empty arrays of appropriate dtypes
        rtn = {}
        index = cols[INDEX]
        full_length = len(index)
        for k, v in cols.items():
            if k != INDEX and k != 'SYMBOL':
                col_len = len(v)
                if col_len < full_length:
                    v = ([None, ] * (full_length - col_len)) + v
                    assert len(v) == full_length
                for i, arr in enumerate(v):
                    if arr is None:
                        #  Replace Nones with appropriate-length empty arrays
                        v[i] = TickStore._empty(len(index[i]), column_dtypes.get(k))
                    else:
                        # Promote to appropriate dtype only if we can safely cast all the values
                        # This avoids the case with strings where None is cast as 'None'.
                        # Casting the object to a string is not worthwhile anyway as Pandas changes the
                        # dtype back to objectS
                        if (i == 0 or v[i].dtype != v[i - 1].dtype) and np.can_cast(v[i].dtype, column_dtypes[k],
                                                                                    casting='safe'):
                            v[i] = v[i].astype(column_dtypes[k], casting='safe')

            rtn[k] = v
        return rtn

    _INDEX_PREC_TO_MS = {"2s": 2000, "3s": 3000, "4s": 4000, "5s": 5000,
                         "10s": 10_000, "20s": 20_000, "30s": 30_000, "60s": 60_000, "1min": 60_000}

    @staticmethod
    def _set_or_promote_dtype(column_dtypes, c, dtype):
        existing_dtype = column_dtypes.get(c)
        if existing_dtype is None or existing_dtype != dtype:
            # Promote ints to floats - as we can't easily represent NaNs
            if np.issubdtype(dtype, np.integer):  # <- this was `int`
                dtype = np.dtype('f8')  # float64
            column_dtypes[c] = np.promote_types(column_dtypes.get(c, dtype), dtype)

    def _prepend_image(self, document, im, rtn_length, column_dtypes, column_set, columns):
        image = im[IMAGE]
        first_dt = im[IMAGE_TIME]
        if not first_dt.tzinfo:
            first_dt = first_dt.replace(tzinfo=mktz('UTC'))
        document[INDEX] = np.insert(document[INDEX], 0, np.uint64(datetime_to_ms(first_dt)))
        for field in image:
            if field == INDEX:
                continue
            if columns and field not in columns:
                continue
            if field not in document or document[field] is None:
                col_dtype = np.dtype(str if isinstance(image[field], str) else 'f8')
                document[field] = TickStore._empty(rtn_length, dtype=col_dtype)
                column_dtypes[field] = col_dtype
                column_set.add(field)
            val = image[field]
            document[field] = np.insert(document[field], 0, document[field].dtype.type(val))
        # Now insert rows for fields in document that are not in the image
        for field in set(document).difference(set(image)):
            if field == INDEX:
                continue
            logger.debug("Field %s is missing from image!" % field)
            if document[field] is not None:
                val = np.nan
                document[field] = np.insert(document[field], 0, document[field].dtype.type(val))
        return document

    @staticmethod
    def _arrays_to_dataframe(arrays, columns, index):
        # Build the result DataFrame via pandas' internal arrays_to_mgr (fast, avoids per-column
        # copies). Its signature changed across pandas majors and must be dispatched by version:
        #   0.x      -> columns passed positionally twice
        #   1.x/2.x  -> typ='block'
        #   >= 3.0   -> the `typ` argument was removed (BlockManager is the only manager); passing it
        #               raises "arrays_to_mgr() got an unexpected keyword argument 'typ'".
        if pd.__version__.startswith("0."):
            mgr = _arrays_to_mgr(arrays, columns, index, columns, dtype=None)
        elif int(pd.__version__.split(".")[0]) >= 3:
            mgr = _arrays_to_mgr(arrays, columns, index, dtype=None)
        else:
            mgr = _arrays_to_mgr(arrays, columns, index, dtype=None, typ="block")
        return pd.DataFrame(mgr)

    @staticmethod
    def _index_ms_to_ns(index_arrays):
        # rtn[INDEX] holds uint64 millisecond arrays (pco_decode / cumsum). numpy has no common
        # integer dtype for uint64 * int64, so `uint64_array * np.int64(1_000_000)` silently
        # promotes to float64 -- whose ULP at ns magnitudes (~1.5e18) is hundreds of ns. That
        # rounds sub-second timestamps on the way out (e.g. ...460001 ms -> ...460000999936 ns,
        # i.e. .001ms read back as .000999936); whole-second values stay exact, which is why only
        # ms-resolution reads were corrupted. view (not astype) as int64: zero-cost, bit-exact, and
        # it preserves the pre-epoch wraparound the uint64 layout encodes, keeping the product integer.
        #
        # CRUCIAL: view each array to int64 BEFORE concatenating. The per-bucket index arrays are not
        # all the same dtype -- v3 buckets decode to uint64 (np.frombuffer) while v4 buckets decode to
        # int64 (nparray_varint_decode). np.concatenate of mixed uint64+int64 has no common integer
        # type, so numpy upcasts the WHOLE result to float64; a later .view(np.int64) then reinterprets
        # those float64 bit patterns as int64 -> garbage timestamps (e.g. year 2124/2262) and a
        # non-monotonic index. This only bites reads that span the v3->v4 boundary; pure-v3 or pure-v4
        # reads stay single-dtype and were unaffected. Viewing each array first keeps concatenate on a
        # uniform int64 dtype (bit-exact for both uint64 and int64 inputs).
        return np.concatenate([a.view(np.int64) for a in index_arrays]) * np.int64(1000_000)

    @staticmethod
    def _read_projection(columns):
        # Fields the read path must fetch from Mongo for _decode_bucket to reconstruct a bucket.
        # INDEX_COMPRESSION ('d') MUST be here: it selects the index codec ('pco' vs the lz4/byte
        # codecs), and 'pco' is now the default writer. If it is dropped from the projection, Mongo
        # omits it, doc.get(INDEX_COMPRESSION, 'lz4') silently falls back to lz4, and _decode_bucket
        # tries to lz4-decompress pco-encoded index bytes -> LZ4BlockError on every real read.
        proj = {SYMBOL: 1,
                INDEX: 1,
                INDEX_PRECISION: 1,
                INDEX_COMPRESSION: 1,
                START: 1,
                END: 1,
                VERSION: 1,
                IMAGE_DOC: 1}
        if columns:
            for c in columns:
                proj[COLUMNS + '.%s' % c] = 1
        else:
            proj[COLUMNS] = 1
        return proj

    def _decode_bucket(self, doc, column_set, column_dtypes, include_symbol, include_images, columns):
        rtn = {}
        if doc[VERSION] != 3 and doc[VERSION] != CHUNK_VERSION_NUMBER_MAX:
            raise ArcticException("Unhandled document version: %s" % doc[VERSION])
        # if doc.get(INDEX_PRECISION, 'ms') != self._index_precision:
        #    raise ArcticException("Unexpected index precision: %s" % doc.get(INDEX_PRECISION, 'ms'))
        # np.cumsum copies the read-only array created with frombuffer.
        # INDEX_COMPRESSION selects the index codec; absent -> 'lz4' (the legacy default). 'pco' stores
        # the uint64 delta array directly with pcodec (no varint/lz4 wrapping); any other value is a
        # byte->byte compressor from binary_decompressors wrapping the version-keyed varint/raw layout.
        def _decode_index(d):
            ic = d.get(INDEX_COMPRESSION, 'lz4')
            if ic == 'pco':
                return pco_decode(d[INDEX])  # uint64 deltas (idx[0] is the absolute first stamp)
            decompress = binary_decompressors.get(ic)
            if decompress is None:
                raise ArcticException("Unknown index compression: %s" % ic)
            buf = decompress(d[INDEX])
            return nparray_varint_decode(buf) if d[VERSION] == 4 else np.frombuffer(buf, dtype='uint64')
        idx = _decode_or_classify(getattr(self, '_collection', None), doc, 'index', _decode_index)
        # Defence-in-depth on read: idx[0] is the absolute first stamp and idx[1:] are the deltas,
        # stored as int64 reinterpreted to uint64 (so the old `idx[1:].min() >= 0` check on uint64 was
        # always true and caught nothing). View the deltas back as signed: a negative one means a
        # corrupted/hand-written bucket -> fail clearly rather than return shuffled ticks via cumsum wrap.
        if len(idx) > 1 and idx[1:].view(np.int64).min() < 0:
            raise ArcticException("non-monotonic index in stored bucket (corrupted delta)")
        rtn[INDEX] = np.cumsum(idx)
        del idx

        ns_prec = 1
        ip = doc.get(INDEX_PRECISION, 'ms')
        if ip == 's' or ip == '1s':
            rtn[INDEX] *= 1000
            ns_prec = 1000_000_000
        elif ip == 'ms' or ip == '1ms':
            ns_prec = 1000_000
        else:
            s = TickStore._INDEX_PREC_TO_MS[ip]
            if s is None:
                s = pd.to_timedelta(ip).total_seconds() * 1000
                assert s >= 1 and int(s) == s  # TODO add support for sub-ms precision
                s = int(s)
                TickStore._INDEX_PREC_TO_MS[ip] = s

            rtn[INDEX] *= s
            ns_prec = 1000_000 * s

        start_ns = dt2ns(doc[START])
        end_ns = dt2ns(doc[END])
        assert start_ns % ns_prec == 0, (start_ns / ns_prec)
        assert end_ns % ns_prec == 0, (end_ns / ns_prec)
        # rtn[INDEX] is uint64; for pre-epoch timestamps the first/last stamps are negative int64
        # values reinterpreted to uint64, so `uint64_scalar * 1000_000` overflows (wraps mod 2^64)
        # and the check spuriously fails. View as int64 first (zero-cost, matches the layout) -- the
        # signed ms values * 1e6 fit comfortably in int64.
        idx_i64 = rtn[INDEX].view(np.int64)
        assert idx_i64[0] * 1000_000 == start_ns
        assert idx_i64[-1] * 1000_000 == end_ns, (self._index_precision, idx_i64[-1] * 1000_000, end_ns)

        doc_length = len(rtn[INDEX])
        column_set.update(doc[COLUMNS].keys())

        # get the mask for the columns we're about to load
        union_mask = np.zeros((doc_length + 7) // 8, dtype='uint8')
        for c in column_set:
            try:
                coldata = doc[COLUMNS][c]
                # the or below will make a copy of this read-only array
                mask = np.frombuffer(lz4_decompress(coldata[ROWMASK]), dtype='uint8')
                union_mask = union_mask | mask
            except KeyError:
                rtn[c] = None
        union_mask = np.unpackbits(union_mask)[:doc_length].astype('bool')
        rtn_length = np.sum(union_mask)

        rtn[INDEX] = rtn[INDEX][union_mask]
        if include_symbol:
            rtn['SYMBOL'] = [doc[SYMBOL], ] * rtn_length

        # Unpack each requested column in turn
        for c in column_set:
            try:
                coldata = doc[COLUMNS][c]
                codec = coldata.get(CODEC)
                dtype = np.dtype(coldata[DTYPE])
                # values ends up being copied by pandas before being returned to the user. However, we
                # copy it into a bytearray here for safety.
                if codec:
                    if codec == 'logQ16_10_dzv':
                        values = decode_logQ16_10_dzv(coldata[DATA])
                    elif codec == 'log28Q16_10_dzv':
                        values = decode_logQ16_10_dzv(coldata[DATA], prescale=28)
                    elif codec == 'loq16_10_dzv1':
                        values = decode_logQ16_10_dzv(coldata[DATA], prescale=24, preadd=.001, comp='zlib')
                    else:
                        coder = codec_registry.get(codec)
                        if coder is None:
                            raise ArcticException("unknown codec: %s" % codec)
                        # LnQ16_pco and any registry codec decode here; classify torn-vs-corrupt on failure
                        values = _decode_or_classify(
                            getattr(self, '_collection', None), doc, 'column:%s' % c,
                            lambda d, _c=c, _coder=coder: _coder.decode(d[COLUMNS][_c][DATA]))
                    if values.dtype != dtype:
                        values = values.astype(dtype)
                else:
                    values = np.frombuffer(bytearray(lz4_decompress(coldata[DATA])), dtype=dtype)
                if GAIN in coldata:
                    if values.dtype != np.float16:
                        raise ArcticException(
                            "column %s has GAIN set but stored dtype is %s (expected float16)" % (c, values.dtype))
                    values = values.astype(np.float32) * coldata[GAIN]
                    dtype = np.float32
                TickStore._set_or_promote_dtype(column_dtypes, c, dtype)
                rtn[c] = TickStore._empty(rtn_length, dtype=column_dtypes[c])
                # unpackbits will make a copy of the read-only array created by frombuffer
                rowmask = np.unpackbits(np.frombuffer(lz4_decompress(coldata[ROWMASK]),
                                                      dtype='uint8'))[:doc_length].astype('bool')
                rowmask = rowmask[union_mask]
                rtn[c][rowmask] = values
            except KeyError:
                rtn[c] = None

        if include_images and doc.get(IMAGE_DOC, {}).get(IMAGE, {}):
            rtn = self._prepend_image(rtn, doc[IMAGE_DOC], rtn_length, column_dtypes, column_set, columns)
        return rtn

    @staticmethod
    def _empty(length, dtype):
        if dtype is not None and np.issubdtype(dtype, np.floating):
            # float16/float32/float64 all get a NaN-filled array of their own dtype, so gaps in a
            # (gain-)downcast float column read back as NaN rather than uninitialised object garbage.
            rtn = np.empty(length, dtype)
            rtn[:] = np.nan
            return rtn
        else:
            return np.empty(length, dtype=np.object_)

    def stats(self):
        """
        Return storage statistics about the library

        Returns
        -------
        dictionary of storage stats
        """
        res = {}
        db = self._collection.database
        conn = db.connection
        res['sharding'] = {}
        try:
            sharding = conn.config.databases.find_one({'_id': db.name})
            if sharding:
                res['sharding'].update(sharding)
            res['sharding']['collections'] = list(conn.config.collections.find(
                {'_id': {'$regex': '^' + db.name + r"\..*"}}))
        except OperationFailure:
            # Access denied
            pass
        res['dbstats'] = db.command('dbstats')
        res['chunks'] = db.command('collstats', self._collection.name)
        res['totals'] = {'count': res['chunks']['count'],
                         'size': res['chunks']['size'],
                         }
        return res

    def _assert_nonoverlapping_data(self, symbol, start, end):
        #
        # Imagine we're trying to insert a tick bucket like:
        #      |S------ New-B -------------- E|
        #  |---- 1 ----| |----- 2 -----| |----- 3 -----|
        #
        # S = New-B Start
        # E = New-B End
        # New-B overlaps with existing buckets 1,2,3
        #
        # All we need to do is find the bucket who's start is immediately before (E)
        # If that document's end is > S, then we know it overlaps
        # with this bucket.
        doc = self._collection.find_one({SYMBOL: symbol,
                                         START: {'$lt': end}
                                         },
                                        projection={START: 1,
                                                    END: 1,
                                                    '_id': 0},
                                        sort=[(START, pymongo.DESCENDING)])
        if doc:
            if not doc[END].tzinfo:
                doc[END] = doc[END].replace(tzinfo=mktz('UTC'))
            if not start.tzinfo:
                start = start.replace(tzinfo=mktz('UTC'))
            if doc[END] > start:
                raise OverlappingDataException(
                    symbol + " Document already exists with start:{} end:{} in the range of our start:{} end:{}".format(
                        doc[START], doc[END], start, end))

    def write(self, symbol, data, initial_image=None, metadata=None, to_dtype=None, codec=None,
              index_compressor='pco'):
        """
        Writes a list of market data events.

        Parameters
        ----------
        symbol : `str`
            symbol name for the item
        data : list of dicts or a pandas.DataFrame
            List of ticks to store to the tick-store.
            if a list of dicts, each dict must contain a 'index' datetime
            if a pandas.DataFrame the index must be a Timestamp that can be converted to a datetime.
            Index names will not be preserved.
        initial_image : dict
            Dict of the initial image at the start of the document. If this contains a 'index' entry it is
            assumed to be the time of the timestamp of the index
        metadata: dict
            optional user defined metadata - one per symbol
        index_compressor : str
            codec for the timestamp index. 'pco' (default; pcodec, lossless, ~0.9 B/val on irregular
            tick indices vs ~1.7 for lz4, near-free on a regular grid) or 'lz4' (the legacy layout).
            The codec is recorded per-bucket, so reads pick it up automatically and existing lz4 buckets
            keep reading unchanged. Note 'pco' is the default, so the `pcodec` package is required to
            write (and to read newly-written data); pass index_compressor='lz4' to avoid the dependency.
        """
        pandas = isinstance(data, pd.DataFrame)

        # Check for overlapping data
        start, end = TickStore._get_time_start_end(data, self._index_quant_ns)
        self._assert_nonoverlapping_data(symbol, start, end)

        if pandas:
            # assert codec is None
            # assert self._index_precision is None or self._index_precision == 'ms'
            buckets = self._pandas_to_buckets(data, symbol, initial_image, to_dtype=to_dtype, codec=codec,
                                              index_compressor=index_compressor)
        else:
            buckets = self._to_buckets(data, symbol, initial_image, to_dtype=to_dtype, codec=codec,
                                       index_compressor=index_compressor)

        self._write(buckets)

        if metadata:
            self._metadata.replace_one({SYMBOL: symbol},
                                       {SYMBOL: symbol, META: metadata},
                                       upsert=True)

    @staticmethod
    def _get_buckets_size_stats_idm(buckets):
        return (
            sum(len(b["i"]) for b in buckets),
            sum(len(c["d"]) for b in buckets for c in b['cs'].values()),
            sum(len(c["m"]) for b in buckets for c in b['cs'].values()),
        )

    @staticmethod
    def _get_buckets_size_by_col(buckets):
        last_write_bytes_per_col = defaultdict(int)
        for b in buckets:
            assert isinstance(b, dict), type(b)
            for cn, cd in b['cs'].items():
                last_write_bytes_per_col[cn] += len(cd['d']) + len(cd['m'])
            last_write_bytes_per_col['_i'] += len(b['i'])
        return last_write_bytes_per_col

    def _write(self, buckets):
        start = dt.now()

        self.last_write_bytes_per_col = TickStore._get_buckets_size_by_col(buckets)
        self.last_write_bytes = TickStore._get_buckets_size_stats_idm(buckets)

        # this goes to pymong.bulk._execute_command
        res: InsertManyResult = mongo_retry(self._collection.insert_many)(buckets)
        assert res.acknowledged
        assert len(res.inserted_ids) == len(buckets)

        t = (dt.now() - start).total_seconds()
        ticks = len(buckets) * self._chunk_size
        rate = int(ticks / t) if t != 0 else float("nan")
        logger.debug("%d buckets in %s: approx %s ticks/sec" % (len(buckets), t, rate))

    def _pandas_to_buckets(self, x, symbol, initial_image, to_dtype=None, codec=None, index_compressor='pco'):
        rtn = []
        idx_prec = self._index_quant_ns
        for i in range(0, len(x), self._chunk_size):
            bucket, initial_image = TickStore._to_bucket_pandas(x.iloc[i:i + self._chunk_size], symbol, initial_image,
                                                                index_precision=idx_prec,
                                                                to_dtype=to_dtype, codec=codec,
                                                                verify_codec=self._verify_codec,
                                                                index_compressor=index_compressor)
            rtn.append(bucket)
        return rtn

    @property
    def _index_quant_ns(self):
        return TickStore._time_quant_to_ns(self._index_precision)

    @staticmethod
    def _time_quant_to_ns(prec: str | int):
        if prec is None or prec == 'ms':
            i = 1_000_000  # ms->ns
        elif prec == 's':
            i = 1_000_000_000  # s->ns
        elif isinstance(prec, str) and prec[0].isdigit():
            i = np.int64(pd.to_timedelta(prec).value)
            assert i > 0
        else:
            raise ValueError("unrecognized index_precision %s" % prec)
        return i

    def _to_buckets(self, x, symbol, initial_image, to_dtype=None, codec=None, index_compressor='pco'):
        rtn = []
        idx_prec = self._index_quant_ns
        for i in range(0, len(x), self._chunk_size):
            bucket, initial_image = TickStore._to_bucket(x[i:i + self._chunk_size], symbol, initial_image,
                                                         index_precision=idx_prec,
                                                         to_dtype=to_dtype, codec=codec,
                                                         verify_codec=self._verify_codec,
                                                         index_compressor=index_compressor)
            rtn.append(bucket)
        return rtn

    @staticmethod
    def _to_ms(date):
        if isinstance(date, dt):
            if not date.tzinfo:
                logger.warning('WARNING: treating naive datetime as UTC in write path')
            return datetime_to_ms(date)
        return date

    @staticmethod
    def _str_dtype(dtype):
        """
        Represent dtypes without byte order, as earlier Java tickstore code doesn't support explicit byte order.
        float16/float32 only occur on the (Python-only) gain/to_dtype downcast write path and produce
        version-4 chunks that the Java reader never sees.
        """
        assert dtype.byteorder != '>'
        if dtype.kind == 'i':
            assert dtype.itemsize == 8
            return 'int64'
        elif dtype.kind == 'f':
            if dtype.itemsize == 8:
                return 'float64'
            elif dtype.itemsize == 4:
                return 'float32'
            elif dtype.itemsize == 2:
                return 'float16'
            else:
                raise UnhandledDtypeException("Bad float dtype '%s'" % dtype)
        elif dtype.kind == 'U':
            return 'U%d' % (dtype.itemsize / 4)
        else:
            raise UnhandledDtypeException("Bad dtype '%s'" % dtype)

    @staticmethod
    def _ensure_supported_dtypes(array, to_dtype=None):

        gain = 0
        if to_dtype and array.dtype != to_dtype:
            gain = max(np.max(array) / np.finfo(to_dtype).max,
                       np.min(array) / np.finfo(to_dtype).min)
            if gain > 1:
                array = (array / gain).astype(to_dtype)
                gain = float(gain)
            else:
                array = array.astype(to_dtype)
                gain = 0

        # We only support these types for now, as we need to read them in Java
        if array.dtype.kind == 'i':
            array = array.astype('<i8')
        elif array.dtype.kind == 'f':
            array = array.astype('<f%d' % array.dtype.itemsize)
        elif array.dtype.kind in ('O', 'U', 'S'):
            if array.dtype.kind == 'O' and infer_dtype(array) not in ['unicode', 'string', 'bytes']:
                # `string` in python2 and `bytes` in python3
                raise UnhandledDtypeException("Casting object column to string failed")
            try:
                array = array.astype(np.unicode_)
            except (UnicodeDecodeError, SystemError):
                # `UnicodeDecodeError` in python2 and `SystemError` in python3
                array = np.array([s.decode('utf-8') for s in array])
            except:
                raise UnhandledDtypeException("Only unicode and utf8 strings are supported.")
        else:
            raise UnhandledDtypeException(
                "Unsupported dtype '%s' - only int64, float64 and U are supported" % array.dtype)
        # Everything is little endian in tickstore
        if array.dtype.byteorder != '<':
            array = array.astype(array.dtype.newbyteorder('<'))
        return array, gain

    @staticmethod
    def _pandas_compute_final_image(df, image, end):
        # Compute the final image with forward fill of df applied to the image
        final_image = copy.copy(image)
        last_values = df.ffill().tail(1).to_dict()
        last_dict = {i: list(a.values())[0] for i, a in last_values.items()}
        final_image.update(last_dict)
        final_image['index'] = end
        return final_image

    @staticmethod
    def _pandas_to_bucket(df, symbol, initial_image, to_dtype=None):
        rtn = {SYMBOL: symbol, VERSION: CHUNK_VERSION_NUMBER, COLUMNS: {}, COUNT: len(df)}
        end = to_dt(df.index[-1].to_pydatetime())
        if initial_image:
            if 'index' in initial_image:
                start = min(to_dt(df.index[0].to_pydatetime()), initial_image['index'])
            else:
                start = to_dt(df.index[0].to_pydatetime())
            image_start = initial_image.get('index', start)
            rtn[IMAGE_DOC] = {IMAGE_TIME: image_start, IMAGE: initial_image}
            final_image = TickStore._pandas_compute_final_image(df, initial_image, end)
        else:
            start = to_dt(df.index[0].to_pydatetime())
            final_image = {}
        rtn[END] = end
        rtn[START] = start

        logger.warning("NB treating all values as 'exists' - no longer sparse")
        rowmask = Binary(lz4_compressHC(np.packbits(np.ones(len(df), dtype='uint8')).tobytes()))

        index_name = df.index.names[0] or "index"
        if PD_VER < '0.23.0':
            recs = df.to_records(convert_datetime64=False)
        else:
            recs = df.to_records()

        for col in df:
            array, gain = TickStore._ensure_supported_dtypes(recs[col], to_dtype=to_dtype)

            col_data = {
                DATA: Binary(lz4_compressHC(array.tobytes())),
                ROWMASK: rowmask,
                DTYPE: TickStore._str_dtype(array.dtype),
            }
            if gain:
                col_data[GAIN] = gain
                rtn[VERSION] = max(rtn[VERSION], CHUNK_VERSION_NUMBER_MAX)
            rtn[COLUMNS][col] = col_data
        # Derive the index in UTC ms from a tz-safe source. recs[index_name] (from df.to_records())
        # is an object/Timestamp array for tz-aware indices on this pandas version and has no .astype
        # to datetime64. index_to_ns already does tz_convert('UTC').tz_localize(None) -> datetime64[ns];
        # convert ns -> ms (floor, matching the naive datetime64[ms] cast for the already-working case).
        idx_ms = (index_to_ns(df, np.int64) // 1_000_000).astype('uint64')
        rtn[INDEX] = Binary(
            lz4_compressHC(np.concatenate(
                ([idx_ms[0]], np.diff(idx_ms))).tobytes()))
        return rtn, final_image

    @staticmethod
    def _encode(v, col_name, to_dtype, codec, verify, dbg_ctx):
        # keep the caller's original values so verification measures the codec against the true input,
        # not an intermediate. (codec + gain-scaled to_dtype downcast is rejected below, so the only
        # lossy stage compared here is the codec itself.)
        orig_dtype = v.dtype
        v_orig = v if verify else None
        v, gain = TickStore._ensure_supported_dtypes(v, to_dtype=to_dtype)
        enc = None
        codec_sel = None
        if codec:
            # dzv codecs are write-deprecated (superseded by the registry codecs); old data still
            # decodes via the dzv branch in _decode_bucket.
            if codec in ('logQ16_10_dzv', 'log28Q16_10_dzv', 'loq16_10_dzv1'):
                raise ArcticException("deprecated codec: %s" % codec)
            else:
                codec_sel = codec[col_name] if isinstance(codec, dict) else codec
                if codec_sel is not None:
                    coder = codec_registry.get(codec_sel)
                    if coder is None:
                        raise ValueError("Unsupported codec: %s" % codec_sel)
                    _dep = getattr(coder, 'deprecated', None)
                    if _dep:
                        raise ArcticException(
                            "codec %s is deprecated for writes%s; existing data still decodes"
                            % (codec_sel, (" -- use %s instead" % _dep) if isinstance(_dep, str) else ""))
                    if v.dtype != np.float32:
                        # The registry codecs operate on float32 (ln_q16 asserts it). Anything else here
                        # is either a non-float32 column or a gain-scaled to_dtype downcast (e.g. float16)
                        # -- itself a separate lossy stage that the codec can't represent. Reject up front
                        # with a clear message rather than crash deep inside coder.encode() (or, for a
                        # gain downcast, silently store the downcast loss unverified).
                        raise ArcticException(
                            "codec %s requires float32 input but column %s is %s (after to_dtype=%s); "
                            "cast the column to float32 or drop the codec %s"
                            % (codec_sel, col_name, v.dtype, to_dtype, dbg_ctx))
                    try:
                        enc = coder.encode(v)
                    except Exception:
                        logger.error("error coding %s %s", codec_sel, dbg_ctx)
                        raise
                    if verify:
                        # Compare what the reader gets (codec decode) against the original input.
                        # codec + gain downcast is rejected above, so no gain re-application is needed.
                        d = coder.decode(enc)
                        code_err = np.nanmax(abs(d - v_orig) / (abs(v_orig) + coder.rtol_reg))
                        if not code_err < coder.rtol_max:
                            raise ArcticException("codec %s rtol %s exceeds %s %s"
                                                  % (codec_sel, round(code_err, 6), coder.rtol_max, dbg_ctx))

        # lz4 lossless is the no-codec fast-path (gz/brotli would be too slow here). Keep whichever of
        # lossy-codec / lossless-lz4 is smaller. Decoupled from `verify` so disabling verification never
        # leaves a column un-encoded.
        # Skip the probe when the codec already won decisively: lz4-HC won't shrink this numeric data below
        # ~1/3 of its raw size, so if the codec output is already smaller than that, raw lz4 can't beat it
        # and we avoid the v.tobytes() alloc + full-image compression. v.nbytes is free (len * itemsize).
        if not codec_sel or len(enc) * 3 > v.nbytes:
            buf2 = lz4_compressHC(v.tobytes())
            if not codec_sel or len(buf2) < len(enc):
                enc = buf2
                codec_sel = None
            del buf2

        # DTYPE semantics differ by path:
        #  - codec: store the logical (pre-cast) dtype; decode reconstructs to float32 then upcasts to it.
        #  - no-codec: store the physical dtype of the bytes (the possibly gain-downcast v), because the
        #    reader does np.frombuffer(DATA, dtype=DTYPE) and needs an exact match.
        dtype = TickStore._str_dtype(orig_dtype if codec_sel else v.dtype)

        return enc, codec_sel, gain, dtype

    @staticmethod
    def _get_time_start_end(ticks: list | pd.DataFrame, index_precision) -> Tuple[pd.Timestamp, pd.Timestamp]:
        assert isinstance(index_precision, (np.integer, int))

        if isinstance(ticks, list):
            start = ticks[0]['index']
            end = ticks[-1]['index']
        elif isinstance(ticks, pd.DataFrame):
            start = ticks.index[0]
            end = ticks.index[-1]
        else:
            raise UnhandledDtypeException("Can't persist type %s to tickstore" % type(ticks))

        ceil = lambda i: -(-np.int64(i) // np.int64(index_precision)) * np.int64(index_precision)
        ceil_dt_utc = lambda dt: ns2dt(ceil(dt2ns(dt)), utc=True)

        # assume_utc = lambda dt: dt.astimezone(TickStore.TZ_UTC) if dt.tzinfo else dt.replace(tzinfo=TickStore.TZ_UTC)
        def to_pdts_utc(dt):
            if isinstance(dt, (int, np.integer)):
                # classic arctic behavior (mktz returns local tz)
                # TODO tzinfo is actually irelevant since we are using a timestamp, which is well-defined without tz
                return ceil_dt_utc(ms_to_datetime(dt, mktz()).astimezone(TickStore.TZ_UTC))
            elif isinstance(dt, np.datetime64):
                raise ValueError("passing np.datetime64 no longer supported")
            elif dt.tzinfo is None:
                raise ValueError("Must specify a TimeZone on incoming data")
            elif isinstance(dt, pd.Timestamp) or isinstance(dt, datetime.datetime):
                # before converting to pydatetime it is important to forward adjust to index resolution
                # (otherwise we'll floor on the time axis wich we don't want)
                # datetime.datetime has no ns resolution!
                # A plain tz-aware datetime.datetime has no .tz_convert; promote to pd.Timestamp first.
                # (pd.Timestamp.tz_convert is a no-op-equivalent for an already-Timestamp value here.)
                return ceil_dt_utc(pd.Timestamp(dt).tz_convert(TickStore.TZ_UTC))
            else:
                raise ValueError("Unsupported datetime type: %s" % type(dt))

        start = to_pdts_utc(start)
        end = to_pdts_utc(end)

        return start, end

    TZ_UTC = datetime.timezone.utc

    @staticmethod
    def _bucket_head(ticks, symbol, initial_image, index_precision: int, codec, index_vlq=True):
        assert isinstance(index_precision, (np.integer, int)) and index_precision > 0, (index_precision,
                                                                                        type(index_precision))
        is_ms = index_precision == 1_000_000

        rtn = {SYMBOL: symbol,
               VERSION: CHUNK_VERSION_NUMBER if is_ms and not index_vlq and not codec else CHUNK_VERSION_NUMBER + 1,
               COLUMNS: {}, COUNT: len(ticks)}

        is_ms = index_precision == 1_000_000

        if not is_ms:
            if index_precision == 1_000_000_000:
                rtn[INDEX_PRECISION] = 's'
            elif (index_precision % 1_000_000_000) == 0:
                rtn[INDEX_PRECISION] = str(index_precision // 1_000_000_000) + 's'
            else:
                rtn[INDEX_PRECISION] = index_precision

        start, end = TickStore._get_time_start_end(ticks, index_precision)

        # pd.Timestamp is a reasonable timestamp format to work with here because it has ns precision
        # and caries timezone info
        assert isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp)
        assert end.tzinfo == start.tzinfo
        # assert start.tzinfo == TickStore.TZ_UTC, (start.tzinfo, TickStore.TZ_UTC)
        if start > end:
            # first tick is after the last tick -> the input is not in ascending time order.
            # surface this as UnorderedDataException (same contract as the per-tick ordering check
            # below), not a bare ValueError.
            raise UnorderedDataException('start %s > end %s' % (start, end))

        # TODO write at test for thath (failing with floats)
        # start = datetime.datetime.fromtimestamp(ceil(start.timestamp() * 1e9) * 1e-9, tz=TickStore.TZ_UTC)
        # end = datetime.datetime.fromtimestamp(ceil(end.timestamp() * 1e9) * 1e-9, tz=TickStore.TZ_UTC)
        # start = ns2dt(ceil(dt2ns(start)), utc=True)
        # end = ns2dt(ceil(dt2ns(end)), utc=True)

        # to_ns = lambda t:np.datetime64(t).astype('datetime64[ns]').astype(np.int64)
        # ns_to_dt = lambda ns, tzi: np.datetime64(int(ns), 'ns').astype(datetime.datetime).replace(tzinfo=tzi)
        # start = ns_to_dt(ceil(to_ns(start)), tzi=start.tzinfo)
        # end = ns_to_dt(ceil(to_ns(start)), end.tzinfo)

        if initial_image:
            image_start = initial_image.get('index', start)
            if image_start > start:
                raise UnorderedDataException("Image timestamp is after first tick: %s > %s" % (
                    image_start, start))
            start = min(start, image_start)
            rtn[IMAGE_DOC] = {IMAGE_TIME: image_start, IMAGE: initial_image}

        rtn[END] = end
        rtn[START] = start

        return rtn, index_vlq

    @staticmethod
    def _to_bucket(ticks, symbol, initial_image, index_precision='ms', to_dtype=None, codec=None, verify_codec=True,
                   varint_coding=False, index_compressor='pco'):
        # index_precision may be given as an ns int (internal callers) or as a precision spec
        # ('ms', 's', '<n>s', ...) -> normalize to ns. varint_coding selects the on-disk index layout:
        # True -> varint-encoded deltas (version-4 chunk), False (default) -> raw uint64 deltas
        # (version-3 chunk). _decode_bucket picks the matching reader off the chunk VERSION.
        if not isinstance(index_precision, (np.integer, int)):
            index_precision = TickStore._time_quant_to_ns(index_precision)
        assert index_precision >= 1000_000

        rtn, index_vlq = TickStore._bucket_head(ticks, symbol, initial_image, index_precision, codec,
                                                index_vlq=bool(varint_coding))
        # tr = [to_dt(ticks[0]['index']), to_dt(ticks[-1]['index'])]
        tr = [rtn[START], rtn[END]]

        data = {}
        rowmask = {}

        final_image = copy.copy(initial_image) if initial_image else {}
        for i, t in enumerate(ticks):
            if initial_image:
                final_image.update(t)
            for k, v in t.items():
                if k != 'index' and v != v:
                    # NaN -> treat as absent, mirroring the pandas path's dropna. The position is left
                    # unmasked, so it reads back as NaN, and a non-finite value never reaches a lossy
                    # codec (which would otherwise fail-fast in _assert_finite).
                    continue
                try:
                    if k != 'index':
                        rowmask[k][i] = 1
                    else:
                        if isinstance(v, (int, np.integer)):
                            v *= 1000_000  # ms -> ns
                        elif isinstance(v, pd.Timestamp):
                            v = v.value
                        elif isinstance(v, datetime.datetime):
                            # tz-aware datetime.datetime index values: promote to pd.Timestamp to
                            # get the UTC ns value (pd.Timestamp.value is tz-independent ns-since-epoch).
                            v = pd.Timestamp(v).value
                        else:
                            raise ValueError("Unsupported type %s" % type(v))
                        v = -(-v // index_precision) * index_precision  # ns ceil
                        v //= 1000_000  # ns -> ms (we made sure index_precision<=1000_000)
                        # v = TickStore._to_ms(v)
                        if data[k][-1] > v:
                            raise UnorderedDataException("Timestamps out-of-order: %s > %s" % (
                                ms_to_datetime(data[k][-1]), t))
                    data[k].append(v)
                except KeyError:
                    if k != 'index':
                        rowmask[k] = np.zeros(len(ticks), dtype='uint8')
                        rowmask[k][i] = 1
                    data[k] = [v]

        rowmask = dict([(k, Binary(lz4_compressHC(np.packbits(v).tobytes()))) for k, v in rowmask.items()])

        for k, v in data.items():
            if k != 'index':
                v = np.array(v)
                enc, codec_sel, gain, dtype = TickStore._encode(v, k, to_dtype, codec, verify_codec,
                                                                dbg_ctx=(symbol, k, tr))

                rtn[COLUMNS][k] = {DATA: Binary(enc),
                                   DTYPE: dtype,
                                   ROWMASK: rowmask[k]}
                if codec_sel:
                    rtn[COLUMNS][k][CODEC] = codec_sel

                if gain:
                    rtn[COLUMNS][k][GAIN] = gain
                    rtn[VERSION] = max(rtn[VERSION], CHUNK_VERSION_NUMBER_MAX)

        is_ms = index_precision == 1_000_000
        assert is_ms or index_precision == 1_000_000_000

        if isinstance(data['index'][0], np.datetime64):
            idx = np.concatenate(([data['index'][0].astype(np.int64)], np.diff(data['index']).astype(np.int64)))
            s = np.int64(index_precision)
        else:
            # assume ms integers
            idx = np.concatenate((np.int64([data['index'][0]]), np.diff(data['index']).astype(np.int64)))
            assert index_precision >= 1_000_000
            assert (index_precision % 1_000_000) == 0
            s = np.int64(index_precision) // 1000_000  # index granularity in *ms* (integer; float // here
            # rounds the subsequent ceil up by one for exact-boundary timestamps)
        if s != 1:
            assert idx.dtype == np.int64
            idx = -(-idx // s)  # causal conversion ns -> ms, s
        if idx.dtype != np.uint64:
            idx = idx.astype(np.uint64)
        # idx[0]*s is the first *tick* in ms. START is the bucket start, which an initial_image can
        # legitimately push earlier than the first tick (image time <= first tick), so the invariant is
        # START <= first tick, with equality when there is no earlier image.
        assert dt2ns(rtn[START]) // 1000_000 <= idx[0] * s
        # assert dt2ns(rtn[END])//1000_000 == sum(idx) * s
        # The index layout must match what _decode_bucket infers from the chunk VERSION: version 4 ->
        # varint, version 3 -> raw uint64. index_vlq drives the version in _bucket_head, but a gain
        # (to_dtype downcast) or codec can have bumped VERSION to 4 since then, so key off the final
        # VERSION here rather than index_vlq alone.
        if index_compressor == 'pco':
            rtn[INDEX_COMPRESSION] = 'pco'
            rtn[VERSION] = max(rtn[VERSION], CHUNK_VERSION_NUMBER_MAX)
            rtn[INDEX] = Binary(pco_encode(idx))
        else:
            assert index_compressor == 'lz4', ("unsupported index_compressor %s" % index_compressor)
            use_varint = rtn[VERSION] == CHUNK_VERSION_NUMBER_MAX
            rtn[INDEX] = Binary(lz4_compressHC(nparray_varint_encode(idx) if use_varint else idx.tobytes()))
        return rtn, final_image

    @staticmethod
    def _to_bucket_pandas(df, symbol, initial_image, index_precision, to_dtype=None, codec=None, verify_codec=True,
                          binary_class=Binary, index_compressor='pco'):
        rtn, index_vlq = TickStore._bucket_head(df, symbol, initial_image, index_precision, codec)
        tr = [to_dt(df.index[0]), to_dt(df.index[-1])]

        # assert len(df) > 1, len(df)

        assert not initial_image

        idx = index_to_ns(df, np.int64)  # can't use np.uint64 here for timestamps before unix epoch

        assert isinstance(index_precision, (np.integer, int)), (index_precision, type(index_precision))
        assert index_precision > 0
        s = np.int64(index_precision)

        if s != 1:
            assert idx.dtype == np.int64
            idx = -(-idx // s)  # causal conversion ns -> ms, s

        assert idx.dtype == np.int64
        assert dt2ns(rtn[START]) == idx[0] * s
        assert dt2ns(rtn[END]) == idx[-1] * s

        # internally np.diff converts to int64 and raises on overflow
        # also there is no safe check for monotony with uint64 input ?
        idx = np.diff(idx, prepend=np.int64(0))
        if len(idx) > 1 and idx[1:].min() < 0:
            # UnorderedDataException is a ValueError, so `except ValueError` callers are unaffected;
            # this just unifies the type with _bucket_head / the ticks path ordering guards.
            raise UnorderedDataException("non monotonic index")
        assert idx.dtype == np.int64

        # if idx[0] < 0:
        #    idx[0] = (2<<(64-1)) + idx[0] # equal to `idx[0].view(np.uint64)` buf faster
        # assert idx[0] >= 0

        # now it is safe to to an unsafe reinterpreting cast of the int64 to uint64
        idx = idx.view(np.uint64)  # no cost O(0)

        # INDEX_COMPRESSION selects the INDEX codec only. The rowmask bitmaps are always lz4: the read
        # path (_decode_bucket) lz4-decompresses them unconditionally, and pco is a numeric-array codec
        # that cannot compress the packed-bit bytes anyway.
        #   'pco'  -> the uint64 delta array straight through pcodec (lossless, ~0.9 B/val on irregular
        #             tick indices vs ~1.7 for lz4(diff); near-free on a regular grid).
        #   else   -> a byte->byte compressor from binary_compressors wrapping the version-keyed layout
        #             (varint deltas for v4, raw uint64 for v3).
        row_compressor = binary_compressors['lz4']
        if index_compressor == 'pco':
            assert INDEX_COMPRESSION not in rtn
            rtn[INDEX_COMPRESSION] = 'pco'
            rtn[VERSION] = max(rtn[VERSION], CHUNK_VERSION_NUMBER_MAX)
            rtn[INDEX] = binary_class(pco_encode(idx))
        else:
            compressor = binary_compressors.get(index_compressor)
            assert compressor is not None, ("unknown compressor %s" % index_compressor)
            if index_compressor != 'lz4':
                assert INDEX_COMPRESSION not in rtn
                rtn[INDEX_COMPRESSION] = index_compressor
            rtn[INDEX] = binary_class(compressor(nparray_varint_encode(idx) if index_vlq else idx.tobytes()))

        for k in df.columns:
            val = df[k].values
            rm = ~np.isnan(val)
            val = val[rm]  # dropna

            if len(val) == 0:
                continue

            enc, codec_sel, gain, dtype = TickStore._encode(val, k, to_dtype, codec, verify_codec,
                                                            dbg_ctx=(symbol, k, tr))
            rtn[COLUMNS][k] = {DATA: binary_class(enc),
                               DTYPE: dtype,
                               ROWMASK: binary_class(row_compressor(np.packbits(rm).tobytes()))}
            if codec_sel:
                rtn[COLUMNS][k][CODEC] = codec_sel
            if gain:
                rtn[COLUMNS][k][GAIN] = gain
                rtn[VERSION] = max(rtn[VERSION], CHUNK_VERSION_NUMBER_MAX)

        return rtn, {}

    def max_date(self, symbol):
        """
        Return the maximum datetime stored for a particular symbol

        Parameters
        ----------
        symbol : `str`
            symbol name for the item
        """
        res = self._collection.find_one({SYMBOL: symbol}, projection={ID: 0, END: 1},
                                        sort=[(START, pymongo.DESCENDING)])
        if res is None:
            raise NoDataFoundException("No Data found for {}".format(symbol))
        return utc_dt_to_local_dt(res[END])

    def min_date(self, symbol):
        """
        Return the minimum datetime stored for a particular symbol

        Parameters
        ----------
        symbol : `str`
            symbol name for the item
        """
        res = self._collection.find_one({SYMBOL: symbol}, projection={ID: 0, START: 1},
                                        sort=[(START, pymongo.ASCENDING)])
        if res is None:
            raise NoDataFoundException("No Data found for {}".format(symbol))
        return utc_dt_to_local_dt(res[START])

    def total_rows(self, symbol,  date_range=None ):
        query = {SYMBOL: symbol}
        date_range = to_pandas_closed_closed(date_range)
        if date_range is not None:
            assert date_range.start and date_range.end
            query[START] = {'$gte': date_range.start}
            query[END] = {'$lte': date_range.end}
        res = self._collection.aggregate([
            {"$match": query},
            {"$group": {"_id": "$" + SYMBOL, "sum": {"$sum": "$" + COUNT}}}
        ])
        if res is None:
            raise NoDataFoundException("No Data found for {}".format(symbol))
        return next(res)['sum']


from functools import partial


def to_iso(t, **kwargs) -> Union[str, List[str]]:
    # iso 8601
    if isinstance(t, list) or isinstance(t, map) or isinstance(t, tuple):
        return type(t)(map(partial(to_iso, **kwargs), t))
    return pd.Timestamp(t, **kwargs).isoformat().replace('+00:00', 'Z')


def ms_to_iso(t, **kwargs) -> Union[str, List[str]]:
    return to_iso(t, unit='ms', **kwargs)
