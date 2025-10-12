from __future__ import print_function, annotations

import copy
import datetime
import logging
import lzma
import warnings
import zlib
from collections import defaultdict
from datetime import datetime as dt, timedelta
from functools import partial

import lz4
import numpy as np
import pandas as pd
import pymongo
import pytz
from bson.binary import Binary
from pymongo import ReadPreference
from pymongo.errors import OperationFailure
from pymongo.results import InsertManyResult

from .coding import nparray_varint_encode, nparray_varint_decode, encode_logQ16_10_dzv, decode_logQ16_10_dzv, \
    LnQ16_VQL, LnQ16_zlib, binary_compressors

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
from .._util import indent

try:
    from lz4.block import compress as lz4_compress, decompress as lz4_decompress

    def lz4_compressHC(_str): # ordinary function so name appears in profile
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


# version.parse(pandas.__version__) > version.parse('1.0')
# IS_PANDAS_1x = version.parse(pandas.__version__) > version.parse('1.0')


codec_registry = {}

def register_codec(name, obj):
    assert hasattr(obj, 'encode') and callable(obj.encode)
    assert hasattr(obj, 'decode') and callable(obj.decode)
    assert hasattr(obj, 'rtol_max') and 0 <= obj.rtol_max <= 0.05
    assert hasattr(obj, 'rtol_reg') and 0 <= obj.rtol_reg <= 0.0005
    assert name not in codec_registry
    assert len(name) < 16
    codec_registry[name] = obj

LnQ25VQLgz = LnQ16_VQL(loq_loss=25, comp='gz')
register_codec('LnQ25VQLgz',LnQ25VQLgz) # for l2 data, general purpose [1e-13, 2e+27]
register_codec('LnQ15VQLlz4', LnQ16_VQL(loq_loss=15, comp='lz4')) # for price data, fast
register_codec('LnQ15gz', LnQ16_zlib(loq_loss=15)) # for signed trade qty, no VQL (no auto-corr)
# TODO like LnQ25VQLgz but smaller prescale to trade small number precision for bigger range (up to 2e35 or so)
# this one: LnQ16_VQL(loq_loss=25, comp='zlib', log_prescale=24, loq_preadd=1e-8)

LnQ20VQLgz = LnQ16_VQL(loq_loss=20, comp='gz', log_prescale=20, loq_preadd=1e-12)
register_codec('LnQ20VQLgz', LnQ20VQLgz) # for l2 data, general purpose [1e-15, 2e+32]

def index_to_ns(series, dtype):
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

class TickStore(object):

    @classmethod
    def initialize_library(cls, arctic_lib, **kwargs):
        TickStore(arctic_lib)._ensure_index()

    @mongo_retry
    def _ensure_index(self):
        collection = self._collection
        collection.create_index([(SYMBOL, pymongo.ASCENDING),
                                 (START, pymongo.ASCENDING)], background=True)
        collection.create_index([(START, pymongo.ASCENDING)], background=True)

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
        #assert index_precision in ('ms', 's')
        self._index_precision = index_precision
        self._verify_codec = verify_codec
        self._reset()

    @mongo_retry
    def _reset(self):
        # The default collections
        self._collection = self._arctic_lib.get_top_level_collection()
        self._metadata = self._collection.metadata

    def __getstate__(self):
        return {'arctic_lib': self._arctic_lib, 'chunk_size': self._chunk_size, 'index_precision': self._index_precision,
                'verify_codec': self._verify_codec}

    def __setstate__(self, state):
        return TickStore.__init__(self, state['arctic_lib'], chunk_size=state['chunk_size'], index_precision=state['index_precision'],
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

    def list_symbols(self, date_range=None):
        return self._collection.distinct(SYMBOL)

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

    def read(self, symbol, date_range=None, columns=None, include_images=False, allow_secondary=None,
             _target_tick_count=0, show_progress=False):
        """
        Read data for the named symbol.  Returns a VersionedItem object with
        a data and metdata element (as passed into write).

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
        rtn = {}
        column_set = set()

        multiple_symbols = not isinstance(symbol, str)

        date_range = to_pandas_closed_closed(date_range)
        query = self._symbol_query(symbol)
        query.update(self._mongo_date_range_query(symbol, date_range))

        import sys
        import time

        def progressbar(it, count=None, prefix="", size=60, out=sys.stdout):  # Python3.6+
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

            #show(0.1)  # avoid div/0
            for i, item in enumerate(it):
                yield item
                show(i + 1)
            print("\n", flush=True, file=out)

        if columns:
            projection = dict([(SYMBOL, 1),
                               (INDEX, 1),
                               (INDEX_PRECISION, 1),
                               (START, 1),
                               (VERSION, 1),
                               (IMAGE_DOC, 1)] +
                              [(COLUMNS + '.%s' % c, 1) for c in columns])
            column_set.update([c for c in columns if c != 'SYMBOL'])
        else:
            projection = dict([(SYMBOL, 1),
                               (INDEX, 1),
                               (INDEX_PRECISION, 1),
                               (START, 1),
                               (VERSION, 1),
                               (COLUMNS, 1),
                               (IMAGE_DOC, 1)])

        column_dtypes = {}
        ticks_read = 0
        bytes_i = 0  # compressed index size
        bytes_d = 0
        bytes_m = 0
        data_coll = self._collection.with_options(read_preference=self._read_preference(allow_secondary))

        cursor = data_coll.find(query, projection=projection).sort([(START, pymongo.ASCENDING)], )
        if show_progress:
            num = data_coll.count(query)
            if num > 20:
                cursor = progressbar(cursor, prefix=symbol, count=num)
        for b in cursor:
            data = self._read_bucket(b, column_set, column_dtypes,
                                     multiple_symbols or (columns is not None and 'SYMBOL' in columns),
                                     include_images, columns)
            for k, v in data.items():
                try:
                    rtn[k].append(v)
                except KeyError:
                    rtn[k] = [v]
            # For testing
            ticks_read += len(data[INDEX])
            bytes_i += len(b['i'])
            bytes_d += sum(len(c['d']) for c in b['cs'].values())
            bytes_m += sum(len(c['m']) for c in b['cs'].values())

            if _target_tick_count and ticks_read > _target_tick_count:
                break

        self.last_read_bytes = bytes_i, bytes_d, bytes_m

        if not rtn:
            raise NoDataFoundException("No Data found for {} in range: {}".format(symbol, date_range))
        rtn = self._pad_and_fix_dtypes(rtn, column_dtypes)

        index = pd.to_datetime(np.concatenate(rtn[INDEX]), utc=True, unit='ms')
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
        if pd.__version__.startswith("0.") or pd.__version__.startswith("1."):
            if  pd.__version__.startswith("1."):
                mgr = _arrays_to_mgr(arrays, columns, index, dtype=None, typ='block')
            else:
                mgr = _arrays_to_mgr(arrays, columns, index, columns, dtype=None)
        else:
            # if pd.__version__
            # new argument typ is mandatory            
            mgr = _arrays_to_mgr(arrays, columns, index, dtype=None, typ="block") # TODO array ?

        rtn = pd.DataFrame(mgr)
        # Present data in the user's default TimeZone
        rtn.index = rtn.index.tz_convert(mktz())

        t = (dt.now() - perf_start).total_seconds()
        ticks = len(rtn)
        rate = int(ticks / t) if t != 0 else float("nan")
        logger.info("%d rows in %s secs: %s ticks/sec" % (ticks, t, rate))
        if not rtn.index.is_monotonic_increasing:
            logger.error("TimeSeries data is out of order, sorting!")
            rtn = rtn.sort_index(kind='mergesort')
        if date_range:
            # FIXME: support DateRange.interval...
            pi =  self._index_precision_int
            if int(date_range.end.timestamp() * 1e9) % pi or int(date_range.start.timestamp() * 1e9) % pi:
                warnings.warn('read(): DateRange timestamps are sub index-precision %s' % (self._index_precision))

            rtn = rtn.loc[date_range.start:date_range.end]
        return rtn

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
            if np.issubdtype(dtype, np.integer): # <- this was `int`
                dtype = np.dtype('f8') # float64
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

    def _read_bucket(self, doc, column_set, column_dtypes, include_symbol, include_images, columns):
        rtn = {}
        if doc[VERSION] != 3 and doc[VERSION] != CHUNK_VERSION_NUMBER_MAX:
            raise ArcticException("Unhandled document version: %s" % doc[VERSION])
        #if doc.get(INDEX_PRECISION, 'ms') != self._index_precision:
        #    raise ArcticException("Unexpected index precision: %s" % doc.get(INDEX_PRECISION, 'ms'))
        # np.cumsum copies the read-only array created with frombuffer
        buf = lz4_decompress(doc[INDEX])
        if doc[VERSION] == 4:
            rtn[INDEX] = np.cumsum(nparray_varint_decode(buf))
        else:
            rtn[INDEX] = np.cumsum(np.frombuffer(buf, dtype='uint64'))
        del buf

        ip = doc.get(INDEX_PRECISION, 'ms')
        if ip == 's' or ip == '1s':
            rtn[INDEX] *= 1000
        elif ip == 'ms' or ip == '1ms':
            pass
        else:
            s = TickStore._INDEX_PREC_TO_MS[ip]
            if s is None:
                s = pd.to_timedelta(ip).total_seconds() * 1000
                assert s >= 1 and int(s) == s # TODO add support for sub-ms precision
                s = int(s)
                TickStore._INDEX_PREC_TO_MS[ip] = s

            rtn[INDEX] *= s


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
                        values = coder.decode(coldata[DATA])
                    if values.dtype != dtype:
                        values = values.astype(dtype)
                else:
                    values = np.frombuffer(bytearray(lz4_decompress(coldata[DATA])), dtype=dtype)
                if GAIN in coldata:
                    assert values.dtype == np.float16  # TODO
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
        if dtype is not None and dtype == np.float64:
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
                    "Document already exists with start:{} end:{} in the range of our start:{} end:{}".format(
                        doc[START], doc[END], start, end))

    def write(self, symbol, data, initial_image=None, metadata=None, to_dtype=None, codec=None):
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
        """
        pandas = False
        # Check for overlapping data
        if isinstance(data, list):
            start = data[0]['index']
            end = data[-1]['index']
        elif isinstance(data, pd.DataFrame):
            start = data.index[0].to_pydatetime()
            end = data.index[-1].to_pydatetime()
            pandas = True
        else:
            raise UnhandledDtypeException("Can't persist type %s to tickstore" % type(data))
        self._assert_nonoverlapping_data(symbol, to_dt(start), to_dt(end))

        if pandas:
            #assert codec is None
            #assert self._index_precision is None or self._index_precision == 'ms'
            buckets = self._pandas_to_buckets(data, symbol, initial_image, to_dtype=to_dtype, codec=codec)
        else:
            buckets = self._to_buckets(data, symbol, initial_image, to_dtype=to_dtype, codec=codec)

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

        res: InsertManyResult = mongo_retry(self._collection.insert_many)(buckets)
        assert res.acknowledged
        assert len(res.inserted_ids) == len(buckets)

        t = (dt.now() - start).total_seconds()
        ticks = len(buckets) * self._chunk_size
        rate = int(ticks / t) if t != 0 else float("nan")
        logger.debug("%d buckets in %s: approx %s ticks/sec" % (len(buckets), t, rate))

    def _pandas_to_buckets(self, x, symbol, initial_image, to_dtype=None, codec=None):
        rtn = []
        idx_prec = self._index_precision_int
        for i in range(0, len(x), self._chunk_size):
            bucket, initial_image = TickStore._to_bucket_pandas(x.iloc[i:i + self._chunk_size], symbol, initial_image,
                                                                index_precision=idx_prec,
                                                                to_dtype=to_dtype, codec=codec, verify_codec=self._verify_codec)
            rtn.append(bucket)
        return rtn

    @property
    def _index_precision_int(self):
        return TickStore.index_precision_to_ns(self._index_precision)

    @staticmethod
    def index_precision_to_ns(prec:str|int):
        if prec is None or prec == 'ms':
            i = 1_000_000  # ms->ns
        elif prec == 's':
            i = 1_000_000_000  # s->ns
        elif isinstance(prec, str) and prec[0].isdigit():
            i = np.int64(pd.to_timedelta(prec).total_seconds() * 1e9)
            assert i > 0
        else:
            raise ValueError("unrecognized index_precision %s" % prec)
        return i


    def _to_buckets(self, x, symbol, initial_image, to_dtype=None, codec=None):
        rtn = []
        idx_prec = self._index_precision_int
        for i in range(0, len(x), self._chunk_size):
            bucket, initial_image = TickStore._to_bucket(x[i:i + self._chunk_size], symbol, initial_image,
                                                         index_precision=idx_prec,
                                                         to_dtype=to_dtype, codec=codec, verify_codec=self._verify_codec)
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
        """
        assert dtype.byteorder != '>'
        if dtype.kind == 'i':
            assert dtype.itemsize == 8
            return 'int64'
        elif dtype.kind == 'f':
            assert dtype.itemsize == 8
            return 'float64'
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
        rtn[INDEX] = Binary(
            lz4_compressHC(np.concatenate(
                ([recs[index_name][0].astype('datetime64[ms]').view('uint64')],
                 np.diff(
                     recs[index_name].astype('datetime64[ms]').view('uint64')))).tobytes()))
        return rtn, final_image

    @staticmethod
    def _encode(v, col_name, to_dtype, codec, verify, dbg_ctx):
        v, gain = TickStore._ensure_supported_dtypes(v, to_dtype=to_dtype)
        enc = None
        codec_sel = None
        if codec:
            if codec == 'logQ16_10_dzv' or codec == 'log28Q16_10_dzv':
                raise ArcticException("deprecated codec")
            elif codec == 'loq16_10_dzv1':
                # raise ArcticException("deprecated codec")
                # todo this one is too slow!
                enc = encode_logQ16_10_dzv(v, prescale=24, preadd=.001, comp='zlib')
                if verify:
                    code_err = np.nanmax(
                        abs(decode_logQ16_10_dzv(enc, prescale=24, preadd=.001, comp='zlib') - v) / (v + 1e-10))
                    assert code_err < 250e-6, (round(code_err, 6), dbg_ctx)
                codec_sel = codec
            else:
                # todo select codec based on column name
                codec_sel = codec[col_name] if isinstance(codec, dict) else codec
                if codec_sel is not None:
                    coder = codec_registry.get(codec_sel)
                    if coder is None:
                        raise ValueError("Unsupported codec: %s" % codec)
                    try:
                        enc = coder.encode(v)
                    except Exception as e:
                        print('error coding', codec_sel, dbg_ctx)
                        raise
                    if verify:
                        code_err = np.nanmax(abs(coder.decode(enc) - v) / (abs(v) + coder.rtol_reg))
                        assert code_err < coder.rtol_max, (round(code_err, 6), dbg_ctx)

        if verify:
            buf2 = lz4_compressHC(v.tobytes())  # this is pretty fast, so just try if we are better
            if not codec_sel or len(buf2) < len(enc):
                enc = buf2
                codec_sel = None
            del buf2

        return enc, codec_sel, gain

    TZ_UTC = datetime.timezone.utc

    @staticmethod
    def _bucket_head(ticks, symbol, initial_image, index_precision:int, codec):
        index_vlq = True
        assert isinstance(index_precision, (np.integer,int)) and index_precision > 0, (index_precision, type(index_precision))
        is_ms = index_precision == 1_000_000

        rtn = {SYMBOL: symbol,
               VERSION: CHUNK_VERSION_NUMBER if is_ms and not index_vlq and not codec else CHUNK_VERSION_NUMBER + 1,
               COLUMNS: {}, COUNT: len(ticks)}


        if not is_ms:
            if index_precision == 1_000_000_000:
                rtn[INDEX_PRECISION] = 's'
            elif (index_precision % 1_000_000_000) == 0:
                rtn[INDEX_PRECISION] = str(index_precision//1_000_000_000) + 's'
            else:
                rtn[INDEX_PRECISION] = index_precision

        if not isinstance(ticks, pd.DataFrame):
            start:datetime.datetime= to_dt(ticks[0]['index'])
            end: datetime.datetime = to_dt(ticks[-1]['index'])
        else:
            start, end = to_dt(ticks.index[0]), to_dt(ticks.index[-1])
        assert isinstance(start, datetime.datetime) and isinstance(end, datetime.datetime)

        # this is actually not necessary, because we convert to timestamp later and store without tzinfo
        assume_utc = lambda dt: dt.astimezone(TickStore.TZ_UTC) if dt.tzinfo else dt.replace(tzinfo=TickStore.TZ_UTC)
        start = assume_utc(start)
        end = assume_utc(end)
        assert end.tzinfo == start.tzinfo == TickStore.TZ_UTC
        if not (start < end):
            raise ValueError('start >= end')

        ceil = lambda i: -(-np.int64(i) // np.int64(index_precision)) * np.int64(index_precision)
        start = datetime.datetime.fromtimestamp(ceil(start.timestamp() * 1e9) * 1e-9, tz=TickStore.TZ_UTC)
        end = datetime.datetime.fromtimestamp(ceil(end.timestamp() * 1e9) * 1e-9, tz=TickStore.TZ_UTC)

        #to_ns = lambda t:np.datetime64(t).astype('datetime64[ns]').astype(np.int64)
        #ns_to_dt = lambda ns, tzi: np.datetime64(int(ns), 'ns').astype(datetime.datetime).replace(tzinfo=tzi)
        #start = ns_to_dt(ceil(to_ns(start)), tzi=start.tzinfo)
        #end = ns_to_dt(ceil(to_ns(start)), end.tzinfo)

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
    def _to_bucket(ticks, symbol, initial_image, index_precision:str|int='ms', to_dtype=None, codec=None, verify_codec=True):
        rtn, index_vlq = TickStore._bucket_head(ticks, symbol, initial_image, index_precision, codec)
        tr = [to_dt(ticks[0]['index']), to_dt(ticks[-1]['index'])]

        data = {}
        rowmask = {}

        final_image = copy.copy(initial_image) if initial_image else {}
        for i, t in enumerate(ticks):
            if initial_image:
                final_image.update(t)
            for k, v in t.items():
                try:
                    if k != 'index':
                        rowmask[k][i] = 1
                    else:
                        v = TickStore._to_ms(v)
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
                enc, codec_sel, gain = TickStore._encode(v, k, to_dtype, codec, verify_codec, dbg_ctx=(symbol, k, tr))

                rtn[COLUMNS][k] = {DATA: Binary(enc),
                                   DTYPE: TickStore._str_dtype(v.dtype),
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
            s = np.int64(index_precision) / 1_000_000
        if s != 1:
            assert idx.dtype == np.int64
            idx = -(-idx // s)  # causal conversion ns -> ms, s
        if idx.dtype != np.uint64:
            idx = idx.astype(np.uint64)  # TODO  use .view()
        rtn[INDEX] = Binary(lz4_compressHC(nparray_varint_encode(idx) if index_vlq else idx.tobytes()))
        return rtn, final_image

    @staticmethod
    def _to_bucket_pandas(df, symbol, initial_image, index_precision, to_dtype=None, codec=None, verify_codec=True,
                          binary_class=Binary, index_compressor='lz4'):
        rtn, index_vlq = TickStore._bucket_head(df, symbol, initial_image, index_precision, codec)
        tr = [to_dt(df.index[0]), to_dt(df.index[-1])]

        assert len(df) > 1, len(df)

        assert not initial_image

        idx = index_to_ns(df, np.int64) # can't use np.uint64 here for timestamps before unix epoch

        assert isinstance(index_precision, (np.integer,int)), (index_precision, type(index_precision))
        assert index_precision > 0
        s = np.int64(index_precision)

        if s != 1:
            assert idx.dtype == np.int64
            idx = -(-idx // s)  # causal conversion ns -> ms, s

        assert idx.dtype == np.int64
        # internally np.diff converts to int64 and raises on overflow
        # also there is no safe check for monotony with uint64 input ?
        idx = np.diff(idx, prepend=np.int64(0))
        if  idx[1:].min() < 0:
            raise ValueError("non monotonic index")
        assert idx.dtype == np.int64
        #if idx[0] < 0:
        #    idx[0] = (2<<(64-1)) + idx[0] # equal to `idx[0].view(np.uint64)` buf faster
        #assert idx[0] >= 0

        # now it is safe to to an unsafe reinterpreting cast of the int64 to uint64
        idx = idx.view(np.uint64) # no cost O(0)



        compressor = dict(
            lz4=lz4_compressHC,
            gz=zlib.compress, #partial(zlib.compress, wbits=15),
            lzma=lambda d: lzma.compress(d, format=lzma.FORMAT_ALONE, preset=lzma.PRESET_EXTREME),
            n=lambda d: d,
        ).get(index_compressor)
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

            enc, codec_sel, gain = TickStore._encode(val, k, to_dtype, codec, verify_codec, dbg_ctx=(symbol, k, tr))
            rtn[COLUMNS][k] = {DATA: binary_class(enc),
                               DTYPE: TickStore._str_dtype(val.dtype),
                               ROWMASK: binary_class(compressor(np.packbits(rm).tobytes()))}
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
