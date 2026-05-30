"""Regression tests for the TickStore lossy-codec write path (TickStore.write(..., codec=)).

Each test targets a specific finding from the codec review:
  #1 lossless lz4 fallback decoupled from `verify` (no more enc=None when verify is off)
  #2 failed verification raises ArcticException (survives python -O) instead of asserting
  #3 lz4 is the always-available lossless fast-path, chosen when smaller than the codec
  #4 the raw-lz4 probe is skipped when the codec already won decisively
  #5 the log quantizers/codecs fail-fast on non-finite input instead of silently corrupting
  #6 codec + a gain-scaled to_dtype downcast is rejected (mutually exclusive lossy stages)
  #7 the deprecated dzv codecs raise on write; their decode path is preserved
"""
import types

import numpy as np
import pandas as pd
import pytest

from arctic.exceptions import ArcticException
from arctic.tickstore.tickstore import TickStore, codec_registry, register_codec
from arctic.tickstore.coding import lz4_decompress, ln_q16, log_q16, LnQ16, LnQ16_VQL, LnQ16_zlib

DBG = ('sym', 'col', None)


# --------------------------------------------------------------------------- #
# stub codecs (registered only for these tests)
# --------------------------------------------------------------------------- #
class _BadCodec:
    """Encodes exactly but decodes 50% off -> must fail verification."""
    rtol_max = 1e-3
    rtol_reg = 1e-10

    def encode(self, arr):
        return arr.astype('<f4').tobytes()

    def decode(self, buf):
        return np.frombuffer(buf, dtype='<f4') * np.float32(1.5)


class _BigButExact:
    """Exact, but emits 10x the raw bytes -> raw lz4 must win the size guard."""
    rtol_max = 1e-3
    rtol_reg = 1e-10

    def encode(self, arr):
        return arr.astype('<f4').tobytes() * 10

    def decode(self, buf):
        return np.frombuffer(buf[:len(buf) // 10], dtype='<f4')


class _Passthru16:
    """Near-lossless codec that accepts float16 (unlike the ln_* codecs), so we can pair a codec
    with to_dtype=float16 and exercise the gain path inside verification."""
    rtol_max = 1e-5
    rtol_reg = 1e-10

    def encode(self, arr):
        return arr.tobytes()  # stored as-is (float16)

    def decode(self, buf):
        return np.frombuffer(buf, dtype='<f2')


@pytest.fixture(autouse=True)
def _stub_codecs():
    codec_registry['_bad'] = _BadCodec()
    codec_registry['_bigexact'] = _BigButExact()
    codec_registry['_passthru16'] = _Passthru16()
    yield
    for n in ('_bad', '_bigexact', '_passthru16'):
        codec_registry.pop(n, None)


# --------------------------------------------------------------------------- #
# #1 lossless fallback decoupled from verify
# --------------------------------------------------------------------------- #
def test_no_codec_verify_off_still_encodes():
    # was: enc=None -> Binary(None) crash when verify_codec=False and no codec
    v = np.array([100.0, 101.5, 99.25], dtype=np.float32)
    enc, codec_sel, gain, _ = TickStore._encode(v, 'c', None, None, False, DBG)
    assert enc is not None
    assert codec_sel is None
    assert gain == 0
    assert lz4_decompress(enc) == v.tobytes()  # lossless


def test_no_codec_verify_on_encodes_losslessly():
    v = np.array([100.0, 101.5, 99.25], dtype=np.float32)
    enc, codec_sel, _, _ = TickStore._encode(v, 'c', None, None, True, DBG)
    assert codec_sel is None
    assert lz4_decompress(enc) == v.tobytes()


# --------------------------------------------------------------------------- #
# #2 failed verification raises (not assert)
# --------------------------------------------------------------------------- #
def test_failed_verification_raises_arctic_exception():
    v = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    with pytest.raises(ArcticException):
        TickStore._encode(v, 'c', None, '_bad', True, DBG)


def test_good_codec_passes_verification():
    v = np.linspace(100.0, 101.0, 500).astype(np.float32)
    enc, _, _, _ = TickStore._encode(v, 'c', None, 'LnQ15VQLlz4', True, DBG)
    assert enc is not None  # did not raise


# --------------------------------------------------------------------------- #
# #3 lz4 is the always-available lossless fast-path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('verify', [True, False])
def test_lz4_fallback_wins_when_smaller(verify):
    # exact codec emits 10x the bytes -> lz4 must win and clear codec_sel, regardless of verify.
    v = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float32)
    enc, codec_sel, _, _ = TickStore._encode(v, 'c', None, '_bigexact', verify, DBG)
    assert codec_sel is None, verify
    assert lz4_decompress(enc) == v.tobytes()


# --------------------------------------------------------------------------- #
# #4 raw-lz4 probe skipped when the codec wins decisively
# --------------------------------------------------------------------------- #
def test_lz4_probe_skipped_when_codec_wins(monkeypatch):
    import arctic.tickstore.tickstore as ts
    calls = {'n': 0}
    orig = ts.lz4_compressHC
    monkeypatch.setattr(ts, 'lz4_compressHC', lambda b: (calls.__setitem__('n', calls['n'] + 1), orig(b))[1])

    v = np.full(2000, 50000.0, dtype=np.float32)  # constant -> codec output is a few bytes
    enc, codec_sel, _, _ = TickStore._encode(v, 'c', None, 'LnQ15VQLlz4', False, DBG)
    assert codec_sel == 'LnQ15VQLlz4'
    assert len(enc) * 3 <= v.nbytes  # sanity: decisive win
    assert calls['n'] == 0           # fallback probe was skipped


def test_lz4_probe_runs_for_no_codec(monkeypatch):
    import arctic.tickstore.tickstore as ts
    calls = {'n': 0}
    orig = ts.lz4_compressHC
    monkeypatch.setattr(ts, 'lz4_compressHC', lambda b: (calls.__setitem__('n', calls['n'] + 1), orig(b))[1])

    v = np.full(2000, 50000.0, dtype=np.float32)
    TickStore._encode(v, 'c', None, None, True, DBG)
    assert calls['n'] >= 1  # no codec -> probe always runs


# --------------------------------------------------------------------------- #
# #5 non-finite inputs fail fast
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_codecs_reject_non_finite(bad):
    arr = np.array([1.0, bad, 3.0], dtype=np.float32)
    for coder in (LnQ16_zlib(15), LnQ16_VQL(comp='lz4'), LnQ16('gz', 15)):
        with pytest.raises(ValueError):
            coder.encode(arr)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_quantizers_reject_non_finite(bad):
    arr = np.array([1.0, bad, 3.0], dtype=np.float32)
    with pytest.raises(ValueError):
        ln_q16(arr)
    with pytest.raises(ValueError):
        log_q16(arr)


def test_codec_clean_roundtrip_unaffected():
    arr = np.array([1.0, 2.0, 3.5, 0.001], dtype=np.float32)
    c = LnQ16_zlib(15)
    assert np.allclose(c.decode(c.encode(arr)), arr, rtol=1e-3)


# --------------------------------------------------------------------------- #
# #6 codec + gain-scaled to_dtype downcast is rejected
# --------------------------------------------------------------------------- #
def test_codec_plus_gain_downcast_rejected():
    # A gain-scaled to_dtype downcast (here float16, since the values exceed its 65504 max) is itself
    # a lossy stage, and the registry codecs require float32 input anyway -> the two are mutually
    # exclusive. _encode must fail fast rather than store the downcast loss unverified (or crash deep
    # inside coder.encode()).
    orig = np.array([70000.0, 140000.0, 98000.0], dtype=np.float64)
    _, gain = TickStore._ensure_supported_dtypes(orig.copy(), to_dtype=np.float16)
    assert gain > 1  # sanity: this to_dtype really does gain-scale

    with pytest.raises(ArcticException):
        TickStore._encode(orig, 'c', np.float16, '_passthru16', True, DBG)


def test_no_to_dtype_keeps_gain_zero():
    v = np.linspace(100.0, 101.0, 400).astype(np.float32)
    _, _, gain, _ = TickStore._encode(v, 'c', None, 'LnQ15VQLlz4', True, DBG)
    assert gain == 0  # no downcast -> verification reference unchanged from before


@pytest.mark.parametrize('arr, to_dtype, why', [
    (np.array([1.0, 2.0, 3.0], dtype=np.float64), None, 'float64 column, no cast'),
    (np.array([1.0, 2.0, 3.0], dtype=np.float64), np.float16, 'downcast that fits float16 (gain == 0)'),
    (np.array([70000.0, 140000.0], dtype=np.float64), np.float16, 'gain-scaled downcast (gain > 1)'),
    (np.array([1, 2, 3], dtype=np.int64), None, 'int64 column'),
])
def test_codec_rejects_non_float32_input(arr, to_dtype, why):
    # registry codecs require float32; any other input must fail fast with a clear ArcticException
    # instead of an opaque assert deep inside the codec (or storing a downcast unverified).
    with pytest.raises(ArcticException):
        TickStore._encode(arr, 'c', to_dtype, 'LnQ15VQLlz4', True, DBG)


def test_codec_accepts_float32_input():
    v = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    enc, codec_sel, gain, dtype = TickStore._encode(v, 'c', None, 'LnQ15VQLlz4', True, DBG)
    assert enc is not None and gain == 0 and dtype == 'float32'


# --------------------------------------------------------------------------- #
# #7 deprecated dzv codecs: write rejected, decode preserved
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('dep', ['logQ16_10_dzv', 'log28Q16_10_dzv', 'loq16_10_dzv1'])
def test_deprecated_dzv_codecs_raise_on_write(dep):
    v = np.array([100.0, 101.0, 102.0], dtype=np.float32)
    with pytest.raises(ArcticException):
        TickStore._encode(v, 'c', None, dep, True, DBG)


def test_dzv_encode_removed_decode_kept():
    import arctic.tickstore.tickstore as ts
    assert not hasattr(ts, 'encode_logQ16_10_dzv')  # write path removed
    assert hasattr(ts, 'decode_logQ16_10_dzv')      # read path preserved for old data


# --------------------------------------------------------------------------- #
# register_codec validates with raises (survives python -O), not asserts
# --------------------------------------------------------------------------- #
def test_register_codec_rejects_invalid():
    good = _Passthru16()

    with pytest.raises(ValueError):  # name too long (>= 16 chars); persisted as on-disk CODEC field
        register_codec('x' * 16, good)
    assert 'x' * 16 not in codec_registry

    class _NoRtol:
        def encode(self, a): return b''
        def decode(self, b): return None
    with pytest.raises(ValueError):  # missing rtol_max / rtol_reg
        register_codec('_nortol', _NoRtol())
    assert '_nortol' not in codec_registry

    try:
        register_codec('_ok_reg', good)
        assert '_ok_reg' in codec_registry
        with pytest.raises(ValueError):  # duplicate name
            register_codec('_ok_reg', good)
    finally:
        codec_registry.pop('_ok_reg', None)


# --------------------------------------------------------------------------- #
# float16 / gain to_dtype downcast storage (no codec): DTYPE must match the stored
# bytes so write -> read round-trips, instead of crashing in _str_dtype / frombuffer.
# --------------------------------------------------------------------------- #
def _decode_bucket_no_db(doc):
    """Decode a single bucket doc without a live-Mongo TickStore (only _index_precision is used)."""
    fake = types.SimpleNamespace(_index_precision='ms')
    return TickStore._decode_bucket(fake, doc, set(), {}, False, False, None)


def test_str_dtype_float_widths():
    assert TickStore._str_dtype(np.dtype('<f8')) == 'float64'
    assert TickStore._str_dtype(np.dtype('<f4')) == 'float32'
    assert TickStore._str_dtype(np.dtype('<f2')) == 'float16'
    assert TickStore._str_dtype(np.dtype('<i8')) == 'int64'


def test_empty_nan_fills_all_float_widths():
    for dt in (np.float16, np.float32, np.float64):
        a = TickStore._empty(4, np.dtype(dt))
        assert a.dtype == dt
        assert np.isnan(a).all()
    # non-float -> object array (unchanged behaviour)
    assert TickStore._empty(3, np.dtype('U2')).dtype == np.object_


def test_encode_reports_stored_dtype():
    # no downcast: physical dtype == logical dtype == float64, bytes round-trip losslessly
    v64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    enc, codec_sel, gain, dtype = TickStore._encode(v64, 'c', None, None, True, DBG)
    assert (codec_sel, gain, dtype) == (None, 0, 'float64')
    assert np.frombuffer(lz4_decompress(enc), dtype='<f8').tolist() == v64.tolist()

    # gain-scaled downcast: stored bytes AND DTYPE must both be float16 (the bug: DTYPE was 'float64')
    big = np.array([70000.0, 140000.0, 98000.0], dtype=np.float64)
    enc, codec_sel, gain, dtype = TickStore._encode(big, 'c', np.float16, None, True, DBG)
    assert codec_sel is None and gain > 1 and dtype == 'float16'
    stored = np.frombuffer(lz4_decompress(enc), dtype='<f2')
    assert stored.dtype == np.float16 and len(stored) == 3


def test_float16_gain_downcast_roundtrip_pandas():
    idx = pd.date_range('2020-01-01', periods=5, freq='s', tz='UTC')
    vals = np.array([70000.0, 140000.0, 98000.0, 12345.0, 0.5], dtype=np.float64)
    df = pd.DataFrame({'px': vals}, index=idx)

    doc, _ = TickStore._to_bucket_pandas(df, 'sym', None, index_precision=1_000_000, to_dtype=np.float16)
    col = doc['cs']['px']
    assert col['t'] == 'float16'   # DTYPE matches stored bytes
    assert 'g' in col              # GAIN recorded

    out = _decode_bucket_no_db(doc)
    got = out['px']
    assert got.dtype == np.float32  # gain path upcasts to float32 on read

    gain = col['g']
    expected = (vals / gain).astype(np.float16).astype(np.float32) * gain
    np.testing.assert_array_equal(got, expected)


def test_float16_gain_downcast_roundtrip_list():
    vals = [70000.0, 140000.0, 98000.0, 12345.0, 0.5]
    ticks = [{'index': pd.Timestamp('2020-01-01 00:00:%02d' % i, tz='UTC'), 'px': v}
             for i, v in enumerate(vals)]
    doc, _ = TickStore._to_bucket(ticks, 'sym', None, index_precision=1_000_000, to_dtype=np.float16)
    col = doc['cs']['px']
    assert col['t'] == 'float16'
    assert 'g' in col

    out = _decode_bucket_no_db(doc)
    got = out['px']
    assert got.dtype == np.float32
    gain = col['g']
    expected = (np.array(vals) / gain).astype(np.float16).astype(np.float32) * gain
    np.testing.assert_array_equal(got, expected)


def test_no_downcast_still_float64_roundtrip_pandas():
    idx = pd.date_range('2020-01-01', periods=4, freq='s', tz='UTC')
    vals = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    df = pd.DataFrame({'px': vals}, index=idx)
    doc, _ = TickStore._to_bucket_pandas(df, 'sym', None, index_precision=1_000_000)
    col = doc['cs']['px']
    assert col['t'] == 'float64'
    assert 'g' not in col
    out = _decode_bucket_no_db(doc)
    np.testing.assert_array_equal(out['px'], vals)


# --------------------------------------------------------------------------- #
# #4 decode rejects a non-monotonic (corrupted) index instead of returning shuffled ticks
# --------------------------------------------------------------------------- #
def test_decode_rejects_corrupted_negative_delta():
    from arctic.tickstore.coding import nparray_varint_encode, lz4_compressHC
    from arctic.tickstore.tickstore import CHUNK_VERSION_NUMBER_MAX, VERSION, INDEX
    # idx[0] is the absolute start, idx[1:] are deltas stored as int64-viewed-as-uint64; a negative
    # delta (here -50) means a non-monotonic index -> decode must raise before cumsum reshuffles it.
    idx = np.array([1000, -50], dtype=np.int64).view(np.uint64)
    doc = {VERSION: CHUNK_VERSION_NUMBER_MAX, INDEX: lz4_compressHC(nparray_varint_encode(idx))}
    with pytest.raises(ArcticException):
        _decode_bucket_no_db(doc)


# --------------------------------------------------------------------------- #
# #5 list write path drops NaN like the pandas path (instead of storing it / crashing a lossy codec)
# --------------------------------------------------------------------------- #
def test_list_path_treats_nan_as_absent():
    pxs = [100.0, float('nan'), 102.0]
    ticks = [{'index': pd.Timestamp('2020-01-01 00:00:%02d' % i, tz='UTC'), 'px': px, 'qty': 1.0 + i}
             for i, px in enumerate(pxs)]
    doc, _ = TickStore._to_bucket(ticks, 'sym', None, index_precision=1_000_000)
    out = _decode_bucket_no_db(doc)
    px = out['px']
    assert len(px) == 3 and np.isnan(px[1])           # masked NaN reads back as NaN
    assert px[0] == 100.0 and px[2] == 102.0          # no codec -> lossless


def test_list_path_nan_with_lossy_codec_does_not_raise():
    # previously a NaN + lossy codec hard-errored in _assert_finite; now the NaN is dropped first
    pxs = [100.0, float('nan'), 102.0]
    ticks = [{'index': pd.Timestamp('2020-01-01 00:00:%02d' % i, tz='UTC'), 'px': px, 'qty': 1.0 + i}
             for i, px in enumerate(pxs)]
    doc, _ = TickStore._to_bucket(ticks, 'sym', None, index_precision=1_000_000,
                                  codec='LnQ15VQLlz4', to_dtype=np.float32)
    out = _decode_bucket_no_db(doc)
    px = out['px']
    assert len(px) == 3 and np.isnan(px[1])
    assert px[0] == pytest.approx(100.0, rel=1e-3)
    assert px[2] == pytest.approx(102.0, rel=1e-3)


# --------------------------------------------------------------------------- #
# ported from jnb test/unit/lib/data/arctic_test.py (those went through jnb's TickStoreClient +
# mongomock; here they run at the bucket level against the arctic TickStore directly, no Mongo).
# --------------------------------------------------------------------------- #
def test_codec_roundtrip_prices_pandas():
    # was jnb arctic_test.py::test_roundtrip_prices. The codec needs float32 input (to_dtype), and
    # LnQ15VQLlz4 is lossy, so compare with a tolerance rather than exact frame equality.
    idx = pd.date_range('2021-01-01', periods=3, freq='s', tz='UTC')
    df = pd.DataFrame({'px': [1.0, 2.0, 3.0]}, index=idx)
    doc, _ = TickStore._to_bucket_pandas(df, 'SYM', None, index_precision=1_000_000,
                                         codec='LnQ15VQLlz4', to_dtype=np.float32)
    out = _decode_bucket_no_db(doc)
    np.testing.assert_allclose(out['px'], df['px'].values, rtol=2e-4)


def test_write_requires_tz_pandas():
    # was jnb arctic_test.py::test_write_requires_tz -- a tz-naive index must be rejected.
    df = pd.DataFrame({'px': [1.0]}, index=pd.to_datetime(['2021-01-01']))  # tz-naive
    with pytest.raises(ValueError):
        TickStore._to_bucket_pandas(df, 'SYM', None, index_precision=1_000_000)
