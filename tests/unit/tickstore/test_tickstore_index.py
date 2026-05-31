"""Ported from jnb test/unit/lib/data/arctic_test.py.

The pure (no-Mongo) tests that exercise TickStore's index/time handling and the numeric invariants the
delta-index encoding relies on. The live-Mongo write/read tests from that file were not ported.
"""
import datetime

import numpy as np
import pandas as pd
import pytest

from arctic.tickstore.tickstore import TickStore, dt2ns


def test_fast_time_slice():
    _fast_time_slice = TickStore._fast_time_slice

    i0 = pd.to_datetime('2018').tz_localize('UTC')
    _1s = pd.to_timedelta('1s')
    s = pd.Series([0, 1, 2, 3, 4], index=[i0, i0, i0 + _1s, i0 + 2 * _1s, i0 + 2 * _1s])
    assert (_fast_time_slice(s, i0, i0) == s[i0:i0]).all()

    def assert_eq(s, start, end):
        pd.testing.assert_series_equal(_fast_time_slice(s, start, end), s[start:end])

    for i in range(len(s)):
        for j in range(len(s)):
            assert_eq(s, i0 + i * _1s, i0 + j * _1s)


def test_tz():
    # locale-independent tz arithmetic + the property arctic actually relies on: dt2ns is
    # timezone-invariant and TickStore.TZ_UTC is UTC. (The original jnb test also asserted things
    # about the runner's *local* tz via time.timezone; those are environment-dependent and dropped.)
    sgtTZObject = datetime.timezone(datetime.timedelta(hours=5), name="SGT")

    dt_sgt = datetime.datetime(1900, 1, 1, 2, tzinfo=sgtTZObject)
    dt_utc = datetime.datetime(1900, 1, 1, 2, tzinfo=datetime.timezone.utc)

    assert dt_utc.tzinfo != dt_sgt.tzinfo
    assert (dt_utc.timestamp() - dt_sgt.timestamp()) / 3600 == 5
    assert dt_sgt.timestamp() == dt_sgt.astimezone(datetime.timezone.utc).timestamp()

    assert TickStore.TZ_UTC == datetime.timezone.utc
    assert dt2ns(dt_sgt) == dt2ns(dt_sgt.astimezone(datetime.timezone.utc))


def test_numpy_diff_uint64_overflow():
    # the index delta encoding relies on these exact uint64 diff / overflow semantics
    idx = np.diff(np.uint64([1, 0]), prepend=np.uint64(0))
    assert idx[0] == 1
    assert idx[1] == (2 << 63) - 1

    idx = np.diff(np.uint64([3, 2, 0]), prepend=np.uint64(0))
    assert idx[0] == 3
    assert idx[1] == (2 << 63) - 1
    assert idx[2] == (2 << 63) - 2

    with pytest.raises(OverflowError):
        np.diff(np.uint64([(2 << 63), 0]), prepend=np.uint64(0))

    idx = np.diff(np.uint64([(2 << 63) - 1, 0]), prepend=np.uint64(0))
    assert idx[0] == (2 << 63) - 1
    assert idx[1] == 1

    idx = np.diff(np.uint64([(2 << 63) - 1, 1]), prepend=np.uint64(0))
    assert idx[1] == 2

    idx = np.diff(np.uint64([(2 << 63) - 2, 0]), prepend=np.uint64(0))
    assert idx[0] == (2 << 63) - 2
    assert idx[1] == 2


def test_floor_divide():
    # the index-ceil trick used throughout the write path is -(-x // q) * q
    assert -(-2.0001 // 1) == 3
    assert -(-2.0000000000001 // 1) == 3

    assert -(-0 // 2) * 2 == 0
    assert -(-.00001 // 2) * 2 == 2
    assert -(-(-.00001) // 2) * 2 == -0
    assert int(-0) == 0
    assert -(-(-1.1) // 2) * 2 == -0
    assert -(-(-1.9999) // 2) * 2 == -0
    assert -(-(-2.000) // 2) * 2 == -2
    assert -(-(-2) // 2) * 2 == -2
    assert -(-(-2.000001) // 2) * 2 == -2
    assert -(-(-3) // 2) * 2 == -2
    assert -(-(-4) // 2) * 2 == -4

    ii = np.int64([0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 14, 15, 19, 20, 21, ])
    assert (-(-ii // 2) * 2 == [0, 2, 2, 4, 4, 6, 6, 10, 10, 12, 14, 16, 20, 20, 22, ]).all()

    ii = np.int64([-1, -2, -3, -4, -5, - 9])
    assert (-(-ii // 2) * 2 == [0, -2, -2, -4, -4, -8]).all()

    ii = np.int64([-29, -21, -20, -19, -15, -11, -10, -9, -1, 0, 1, 5, 9, 15, 19, 20, 21])
    assert (-(-ii // 10) * 10 == [-20, -20, -20, -10, -10, -10, -10, -0, -0, 0, 10, 10, 10, 20, 20, 20, 30]).all()
