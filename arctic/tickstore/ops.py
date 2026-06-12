"""Symbol-level operations built on TickStore's storage internals (bucket docs + metadata),
kept policy-free: no host-app caches, locks or codecs in here -- a host app wraps these and
adds its own invalidation/locking (cf. the write/delete observer seam in tickstore.py).

All functions take TickStore instances; none of them decodes a tick payload except where noted.
"""
import logging
import time

import pandas as pd
from pymongo import InsertOne
from pymongo.results import UpdateResult

from ..exceptions import NoDataFoundException
from .tickstore import TickStore, SYMBOL, START, END, META

logger = logging.getLogger(__name__)


def _utc(t):
    t = pd.Timestamp(t)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def rename_symbol(ts: TickStore, symbol: str, new_symbol: str, overwrite_meta: bool = False):
    """Rename a symbol by re-tagging its metadata + data documents (a pure tag move, no
    decode/re-encode). When the target already holds data, the rename is refused unless every
    target bucket is time-disjoint from the source's [min, max] range -- src-before-dst,
    dst-before-src, or src falling into a gap between dst segments -- so the merged symbol can
    never violate the bucket non-overlap invariant read() enforces.

    NOTE: this is the data-visibility boundary for temp-then-publish overwrite flows; a host
    app with external caches must invalidate both names after this returns."""
    meta = ts._metadata.find_one({SYMBOL: symbol})
    meta_ex = ts._metadata.find_one({SYMBOL: new_symbol})

    del_meta_sym = None
    if meta and meta_ex and meta[META] == meta_ex[META]:
        logger.info('%s meta equal, dont touch', symbol)
        meta = None
        del_meta_sym = symbol  # just delete src meta (and use the existing dst)

    if meta and meta_ex:
        if overwrite_meta:
            del_meta_sym = new_symbol  # delete the dst meta
        else:
            logger.error('meta src %s\nmeta dst %s', meta, meta_ex)
            raise RuntimeError("cannot rename symbol %s -> %s due to existing metadata (%s)"
                               % (symbol, new_symbol, meta_ex))

    data_ex = ts._collection.find_one({SYMBOL: new_symbol}, {"_id": 1})

    if data_ex:
        tr = ts.min_date(symbol), ts.max_date(symbol)
        clash = ts._collection.find_one(
            {SYMBOL: new_symbol, START: {"$lte": tr[1]}, END: {"$gte": tr[0]}}, {"_id": 1, START: 1, END: 1})
        if clash:
            raise ValueError("cannot rename symbol %s -> %s due to overlapping data (%s)"
                             % (symbol, new_symbol, clash))

    if del_meta_sym:
        ts._metadata.delete_one({SYMBOL: del_meta_sym})  # delete one meta

    res_meta = None
    if meta:
        # noinspection PyProtectedMember
        res_meta = ts._metadata.update_one({SYMBOL: symbol}, {"$set": {SYMBOL: new_symbol}})
        assert res_meta.modified_count == 1

    # noinspection PyProtectedMember
    res_data: UpdateResult = ts._collection.update_many({SYMBOL: symbol}, {"$set": {SYMBOL: new_symbol}})

    if not (res_data.acknowledged and (not meta or res_meta.acknowledged) and res_data.modified_count > 0):
        ctx = (symbol, new_symbol, meta and res_meta.modified_count, res_data.modified_count,
               res_data.matched_count, res_data.acknowledged, res_data.upserted_id)
        raise RuntimeError('rename error ' + str(ctx))


def copy_symbol(ts_src: TickStore, ts_dst: TickStore, symbol: str, new_symbol: "str | None" = None):
    """Fast block-level copy of a symbol into another (or the same) library: raw bucket +
    metadata documents, no decode/re-encode. Refuses when the destination already holds data
    for the target name."""
    try:
        ts_dst.min_date(new_symbol or symbol)
        raise ValueError("cannot copy, %s exists in %s" % (symbol, ts_dst))
    except NoDataFoundException:
        pass

    # copy meta
    doc = ts_src._metadata.find_one(ts_src._symbol_query(symbol))
    if new_symbol:
        doc[SYMBOL] = new_symbol
    ts_dst._metadata.insert_one(doc)

    # copy data
    docs = list(ts_src._collection.find(ts_src._symbol_query(symbol)))
    for doc in docs:
        doc.pop("_id", None)  # let dst assign fresh ids (uniqueness is per-collection)
        if new_symbol:
            doc[SYMBOL] = new_symbol
    assert len(docs) > 0, "nothing to copy"
    ts_dst._collection.bulk_write([InsertOne(d) for d in docs])


def bucket_bounds(ts: TickStore, symbol: str):
    """[(start, end), ...] of every data bucket for symbol, sorted by start. Cheap: projects
    the bucket bounds only, never decodes the compressed tick payload."""
    cur = (ts._collection
           .find(ts._symbol_query(symbol), projection={START: 1, END: 1, "_id": 0})
           .sort(START, 1))
    return [(d[START], d[END]) for d in cur]


def find_bucket_overlap(bounds):
    """Pure check over [(start, end), ...] data-bucket bounds: the first time-interleaved pair
    ((s1, e1), (s2, e2)) or None. Sorts internally. A bucket merely touching the previous end is
    fine -- TickStore.read() only rejects a bucket starting strictly before the previous one ends."""
    bounds = sorted(bounds)
    return next(((bounds[i - 1], b) for i, b in enumerate(bounds) if i and b[0] < bounds[i - 1][1]), None)


def assert_buckets_nonoverlapping(ts: TickStore, symbol: str):
    """Defence-in-depth after publishing a symbol: assert no two data buckets' time spans
    interleave -- the invariant TickStore.read() enforces, whose violation makes the symbol
    unreadable ('overlap block: ...'). Cheap: bounds-only projection."""
    clash = find_bucket_overlap(bucket_bounds(ts, symbol))
    if clash:
        raise ValueError("%s: overlapping data buckets %s | %s" % (symbol, *clash))


def wait_until_range_matches(ts: TickStore, symbol: str, start, end, attempts: int = 8):
    """Poll until the stored [min_date, max_date] of symbol equals [start, end] (UTC-compared)
    -- mongodb writes are sometimes not instantly visible to the next read. Raises ValueError
    when the range never settles."""
    for i in range(attempts):
        try:
            if _utc(ts.min_date(symbol)) == _utc(start) and _utc(ts.max_date(symbol)) == _utc(end):
                return True
        except NoDataFoundException:
            pass
        time.sleep(.1 * 1.5 ** i)
    raise ValueError("time range of %s never matched (q %s != ref %s) and (q %s != ref %s)" % (
        symbol, ts.min_date(symbol), start, ts.max_date(symbol), end))
