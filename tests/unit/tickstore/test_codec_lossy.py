"""Regression tests for the lossy column codecs and the TickStore._encode path.

Covers the behaviours hardened during the codec review/search:
  * every registered codec round-trips positive data within its declared rtol_max
  * LnQ30brW10q10 (brotli q10/lgwin10, loss=30) round-trips < 250ppm and beats LnQ20VQLgz on size
  * the br_w10q10 compressor round-trips raw bytes
  * non-finite input is rejected (ValueError) instead of silently corrupting     (#5)
  * deprecated dzv codecs raise on write but still decode old data                (#7)
  * a codec whose error exceeds rtol_max raises ArcticException, not AssertionError(#2)
  * _encode always produces an lz4 fallback for the no-codec path, even verify=off (#1)
  * full write->read bucket serialization round-trips with the new codec
"""
import datetime
import types

import numpy as np
import pandas as pd
import pytest

from arctic.exceptions import ArcticException
from arctic.tickstore import tickstore as ts
from arctic.tickstore.tickstore import TickStore, codec_registry
from arctic.tickstore.coding import (
    ln_q16, log_q16, binary_compressors, binary_decompressors, lz4_decompress,
)


def _logspace(n, lo=1e2, hi=1e8, seed=0):
    rng = np.random.default_rng(seed)
    return (10 ** rng.uniform(np.log10(lo), np.log10(hi), n)).astype(np.float32)


def _autocorr_price(n, seed=1):
    # smooth, positive, autocorrelated series (delta+vql friendly)
    rng = np.random.default_rng(seed)
    return (50000 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))).astype(np.float32)


try:
    import pcodec  # noqa: F401  -- optional dependency for the LnQ*pco codecs
    HAVE_PCO = True
except ImportError:
    HAVE_PCO = False

# codecs that need the optional `pcodec` package; skipped when it isn't installed
_PCO_CODECS = {'LnQ30pco', 'LnQ15pcoS'}


# --- every registered codec round-trips within its declared tolerance ------------
@pytest.mark.parametrize("name", sorted(codec_registry))
def test_registered_codec_roundtrip(name):
    if name in _PCO_CODECS and not HAVE_PCO:
        pytest.skip("pcodec not installed")
    code = codec_registry[name]
    a = _logspace(20000)
    dec = code.decode(code.encode(a))
    rtol = float(np.max(np.abs(dec - a) / (np.abs(a) + code.rtol_reg)))
    assert rtol < code.rtol_max, (name, rtol, code.rtol_max)


# --- the new codec ---------------------------------------------------------------
def test_LnQ30brW10q10_roundtrip_and_budget():
    code = codec_registry['LnQ30brW10q10']
    assert code.rtol_max == 250e-6  # raised from the 200ppm default for loss=30
    a = _logspace(50000)
    enc = code.encode(a)
    dec = code.decode(enc)
    rtol = float(np.max(np.abs(dec - a) / (np.abs(a) + code.rtol_reg)))
    assert rtol < 250e-6


def test_LnQ30brW10q10_smaller_than_baseline():
    a = _autocorr_price(50000)  # realistic autocorrelated data
    new = len(codec_registry['LnQ30brW10q10'].encode(a))
    base = len(codec_registry['LnQ20VQLgz'].encode(a))
    assert new < base, (new, base)


def test_br_w10q10_compressor_roundtrips():
    assert 'br_w10q10' in binary_compressors and 'br_w10q10' in binary_decompressors
    payload = b'\x00\x01\x02\x03' * 5000
    assert binary_decompressors['br_w10q10'](binary_compressors['br_w10q10'](payload)) == payload


# --- LnQ30pco (pcodec back-end, same loss=30 grid as LnQ30brW10q10) ---------------
pco_required = pytest.mark.skipif(not HAVE_PCO, reason="pcodec not installed")


def test_LnQ30pco_registered_and_budget():
    code = codec_registry['LnQ30pco']
    assert code.rtol_max == 250e-6  # loss=30 grid, same as LnQ30brW10q10


@pco_required
def test_LnQ30pco_roundtrip_within_budget():
    code = codec_registry['LnQ30pco']
    a = _logspace(50000)
    dec = code.decode(code.encode(a))
    rtol = float(np.max(np.abs(dec - a) / (np.abs(a) + code.rtol_reg)))
    assert rtol < 250e-6


@pco_required
def test_LnQ30pco_same_error_as_brW10q10():
    # identical ln_q16 quantization grid -> identical error, only the entropy stage differs
    a = _autocorr_price(50000)
    pco = codec_registry['LnQ30pco']
    br = codec_registry['LnQ30brW10q10']
    e_pco = float(np.max(np.abs(pco.decode(pco.encode(a)) - a) / (np.abs(a) + pco.rtol_reg)))
    e_br = float(np.max(np.abs(br.decode(br.encode(a)) - a) / (np.abs(a) + br.rtol_reg)))
    assert abs(e_pco - e_br) < 1e-6, (e_pco, e_br)


@pco_required
def test_LnQ30pco_smaller_than_baseline():
    a = _autocorr_price(50000)
    new = len(codec_registry['LnQ30pco'].encode(a))
    base = len(codec_registry['LnQ20VQLgz'].encode(a))
    assert new < base, (new, base)


@pco_required
def test_LnQ30pco_rejects_nonfinite():
    a = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    with pytest.raises(ValueError):
        codec_registry['LnQ30pco'].encode(a)


def test_LnQ30pco_unavailable_raises_importerror_on_use():
    # registration is import-safe without pcodec; a *use* without it must fail clearly, not silently.
    if HAVE_PCO:
        pytest.skip("pcodec installed; cannot exercise the missing-dependency path")
    with pytest.raises(ImportError):
        codec_registry['LnQ30pco'].encode(_logspace(100))


@pco_required
def test_LnQ15pcoS_signed_roundtrip():
    # signed trade-qty codec: quantize magnitude, carry the sign; both signs must round-trip < 200ppm.
    code = codec_registry['LnQ15pcoS']
    rng = np.random.default_rng(0)
    mag = (10 ** rng.normal(0, 1.2, 50000)).astype(np.float32)
    a = (mag * np.where(rng.random(50000) < 0.5, -1, 1)).astype(np.float32)
    assert (a < 0).any() and (a > 0).any()
    dec = code.decode(code.encode(a))
    assert np.array_equal(np.sign(dec), np.sign(a))            # sign preserved exactly
    rtol = float(np.max(np.abs(dec - a) / (np.abs(a) + code.rtol_reg)))
    assert rtol < code.rtol_max


@pco_required
def test_LnQ30pco_rejects_negative_input():
    # the positive (default) pco codec must reject signed input rather than silently mis-store it.
    a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    with pytest.raises(AssertionError):
        codec_registry['LnQ30pco'].encode(a)


@pco_required
def test_full_bucket_roundtrip_LnQ30pco():
    n = 5000
    idx = pd.date_range('2026-01-01', periods=n, freq='5s', tz='UTC')
    df = pd.DataFrame({'volAt40': _logspace(n, seed=3), 'vwapBid': _autocorr_price(n)}, index=idx)
    bucket, _ = TickStore._to_bucket_pandas(df, 'SYM', None, 1_000_000,
                                            codec='LnQ30pco', verify_codec=True)
    fake_self = types.SimpleNamespace(_index_precision='ms')
    rtn = TickStore._decode_bucket(fake_self, bucket, set(df.columns), {},
                                   include_symbol=False, include_images=False, columns=None)
    for c in df.columns:
        orig = df[c].values.astype(np.float64)
        got = np.asarray(rtn[c], dtype=np.float64)
        rtol = float(np.max(np.abs(got - orig) / (np.abs(orig) + 1e-10)))
        assert rtol < 250e-6, (c, rtol)


# --- #5 non-finite input is rejected, not silently corrupted ---------------------
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_quantizers_reject_nonfinite(bad):
    a = np.array([1.0, bad, 3.0], dtype=np.float32)
    with pytest.raises(ValueError):
        ln_q16(a)
    with pytest.raises(ValueError):
        log_q16(a)


@pytest.mark.parametrize("name", ['LnQ20VQLgz', 'LnQ15gz', 'LnQ15br9', 'LnQ30brW10q10'])
def test_codecs_reject_nonfinite(name):
    a = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    with pytest.raises(ValueError):
        codec_registry[name].encode(a)


# --- #7 deprecated dzv codecs raise on write, but decode helper still present -----
@pytest.mark.parametrize("codec", ['logQ16_10_dzv', 'log28Q16_10_dzv', 'loq16_10_dzv1'])
def test_dzv_codecs_deprecated_on_write(codec):
    a = _logspace(1000)
    with pytest.raises(ArcticException):
        TickStore._encode(a, 'px', None, codec, True, dbg_ctx=('s', 'px', None))


def test_dzv_encode_helper_removed_but_decode_kept():
    assert not hasattr(ts, 'encode_logQ16_10_dzv')   # write helper dropped
    assert hasattr(ts, 'decode_logQ16_10_dzv')       # read helper retained for old data


# --- #2 rtol failure raises ArcticException (survives python -O), not assert ------
def test_bad_codec_raises_not_assert():
    class _BadCodec:
        rtol_max = 1e-3
        rtol_reg = 1e-10
        def encode(self, arr):
            return arr.astype('<f4').tobytes()
        def decode(self, buf):
            return np.frombuffer(buf, dtype='<f4') * 1.5  # 50% error -> way over rtol_max
    codec_registry['badcodec_t'] = _BadCodec()
    try:
        with pytest.raises(ArcticException):
            TickStore._encode(_logspace(1000), 'px', None, 'badcodec_t', True, dbg_ctx=('s', 'px', None))
    finally:
        del codec_registry['badcodec_t']


# --- #1 no-codec path always yields a usable lz4 buffer, even with verify off -----
@pytest.mark.parametrize("verify", [True, False])
def test_no_codec_fallback_yields_lz4(verify):
    a = _logspace(2000)
    out = TickStore._encode(a, 'px', None, None, verify, dbg_ctx=('s', 'px', None))
    enc, codec_sel = out[0], out[1]
    assert codec_sel is None
    assert enc is not None
    assert lz4_decompress(enc) == a.tobytes()  # lossless lz4 fast-path


# --- #6 codec verify is measured against the original; gain/to_dtype downcast rejected -----
def test_codec_rejects_non_float32_input():
    # registry codecs operate on float32; a float64 column must be rejected up front, not crash deep
    # inside coder.encode (and verify must compare against the true input, not an intermediate).
    a = _logspace(1000).astype(np.float64)
    with pytest.raises(ArcticException):
        TickStore._encode(a, 'px', None, 'LnQ20VQLgz', True, dbg_ctx=('s', 'px', None))


def test_codec_plus_to_dtype_downcast_rejected():
    # codec + gain-scaled float16 downcast = two stacked lossy stages; reject rather than store the
    # downcast loss unverified.
    a = _logspace(1000)
    with pytest.raises(ArcticException):
        TickStore._encode(a, 'px', np.float16, 'LnQ20VQLgz', True, dbg_ctx=('s', 'px', None))


def test_codec_float32_input_ok():
    a = _logspace(1000)  # float32
    out = TickStore._encode(a, 'px', None, 'LnQ20VQLgz', True, dbg_ctx=('s', 'px', None))
    assert out[1] == 'LnQ20VQLgz'  # codec kept, verify passed against original


# --- index compressor (INDEX_COMPRESSION): pco + the generalized read path -------
def _irregular_index(n, seed=3):
    """Bursty ms-resolution tick index with jitter + occasional gaps (not a regular grid)."""
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(40, n).astype('int64')
    m = rng.random(n) < 0.02
    gaps[m] += rng.integers(1000, 60000, m.sum())
    idx_ms = np.cumsum(gaps) + 1_700_000_000_000
    return pd.to_datetime(idx_ms, unit='ms', utc=True)


def _index_roundtrip(index_compressor, codec=None, n=20000):
    idx = _irregular_index(n)
    df = pd.DataFrame({'volAt40': _logspace(n, seed=4), 'vwapBid': _autocorr_price(n)}, index=idx)
    bucket, _ = TickStore._to_bucket_pandas(df, 'SYM', None, 1_000_000, codec=codec,
                                            verify_codec=True, index_compressor=index_compressor)
    fake_self = types.SimpleNamespace(_index_precision='ms')
    rtn = TickStore._decode_bucket(fake_self, bucket, set(df.columns), {},
                                   include_symbol=False, include_images=False, columns=None)
    got = pd.to_datetime(np.asarray(rtn['i']) * 1_000_000, utc=True)
    return bucket, df, idx, got, rtn


@pytest.mark.parametrize("index_compressor", ['lz4', 'gz', 'lzma'])
def test_index_compressor_byte_codecs_roundtrip(index_compressor):
    # lz4 is the legacy default; gz/lzma exercise the generalized read path (previously the read
    # hardcoded lz4 and silently broke a non-lz4 index -- regression guard for that fix).
    bucket, df, idx, got, _ = _index_roundtrip(index_compressor)
    assert np.array_equal(got.view('int64'), idx.view('int64'))
    # only a non-default codec records the field
    assert bucket.get('d') == (None if index_compressor == 'lz4' else index_compressor)


@pco_required
def test_index_compressor_pco_roundtrip_lossless():
    bucket, df, idx, got, _ = _index_roundtrip('pco')
    assert bucket['d'] == 'pco'                      # INDEX_COMPRESSION recorded
    assert np.array_equal(got.view('int64'), idx.view('int64'))   # lossless, exact


@pco_required
def test_index_compressor_pco_smaller_than_lz4():
    big = 50000
    b_pco, *_ = _index_roundtrip('pco', n=big)
    b_lz4, *_ = _index_roundtrip('lz4', n=big)
    assert len(b_pco['i']) < len(b_lz4['i']), (len(b_pco['i']), len(b_lz4['i']))


@pco_required
def test_index_pco_composes_with_column_pco():
    # pco index + pco column in the same bucket: independent codecs, both round-trip.
    bucket, df, idx, got, rtn = _index_roundtrip('pco', codec='LnQ30pco')
    assert bucket['d'] == 'pco'
    assert np.array_equal(got.view('int64'), idx.view('int64'))
    for c in df.columns:
        orig = df[c].values.astype(np.float64)
        rtol = float(np.max(np.abs(np.asarray(rtn[c], dtype=np.float64) - orig) / (np.abs(orig) + 1e-10)))
        assert rtol < 250e-6, (c, rtol)


def test_legacy_bucket_without_index_compression_reads_as_lz4():
    # a bucket missing the INDEX_COMPRESSION field must still decode (defaults to lz4) -- back-compat.
    bucket, df, idx, got, _ = _index_roundtrip('lz4')
    assert 'd' not in bucket  # legacy layout: field absent
    assert np.array_equal(got.view('int64'), idx.view('int64'))


@pco_required
def test_index_pco_via_ticks_path():
    n = 2000
    idx = _irregular_index(n, seed=9)
    px = _autocorr_price(n)
    ticks = [{'index': pd.Timestamp(t), 'px': float(p)} for t, p in zip(idx, px)]
    bucket, _ = TickStore._to_bucket(ticks, 'SYM', None, index_precision=1_000_000, index_compressor='pco')
    assert bucket['d'] == 'pco'
    fake_self = types.SimpleNamespace(_index_precision='ms')
    rtn = TickStore._decode_bucket(fake_self, bucket, {'px'}, {},
                                   include_symbol=False, include_images=False, columns=None)
    got = pd.to_datetime(np.asarray(rtn['i']) * 1_000_000, utc=True)
    assert np.array_equal(got.view('int64'), idx.view('int64'))


def _apply_mongo_projection(doc, projection):
    """Mimic a MongoDB inclusion projection so a field omitted from the projection is genuinely
    absent from the doc handed to _decode_bucket -- which is exactly how the read path loses the
    INDEX_COMPRESSION marker when it isn't projected."""
    out = {}
    nested = {}
    for key in projection:
        if '.' in key:
            top, sub = key.split('.', 1)
            nested.setdefault(top, set()).add(sub)
        elif key in doc:
            out[key] = doc[key]
    for top, subs in nested.items():
        if top in doc:
            out[top] = {k: v for k, v in doc[top].items() if k in subs}
    return out


@pco_required
@pytest.mark.parametrize("columns", [None, ['volAt40', 'vwapBid']])
def test_pco_bucket_survives_read_projection(columns):
    # Regression guard for the dropped INDEX_COMPRESSION projection field. pco is the default index
    # writer, so a real read() must decode a pco bucket *after* it passes through the exact field set
    # read() fetches from Mongo (TickStore._read_projection). If 'd' is missing from the projection,
    # the projected doc loses it, _decode_bucket falls back to lz4 and raises LZ4BlockError on the
    # pco-encoded index -- the production failure, reproduced here with no live Mongo.
    bucket, df, idx, _, _ = _index_roundtrip('pco')
    assert bucket['d'] == 'pco'
    projected = _apply_mongo_projection(bucket, TickStore._read_projection(columns))
    assert projected.get('d') == 'pco', "read projection dropped INDEX_COMPRESSION"
    fake_self = types.SimpleNamespace(_index_precision='ms')
    col_set = set(df.columns) if columns is None else set(columns)
    rtn = TickStore._decode_bucket(fake_self, projected, col_set, {},
                                   include_symbol=False, include_images=False, columns=columns)
    got = pd.to_datetime(np.asarray(rtn['i']) * 1_000_000, utc=True)
    assert np.array_equal(got.view('int64'), idx.view('int64'))


@pco_required
@pytest.mark.parametrize("index_compressor", ['pco', 'lz4'])
def test_ms_index_to_ns_no_float_precision_loss(index_compressor):
    # read() turns the decoded uint64 *ms* index into ns. Doing `uint64_array * np.int64(1_000_000)`
    # promotes to float64 (no common uint64/int64 integer dtype in numpy), and at ns magnitudes
    # float64's ULP is hundreds of ns -- so sub-second timestamps come back rounded
    # (.001ms -> .000999936). Whole-second values stay exact, which is why only ms-resolution reads
    # broke. This guards the integer (view-as-int64) conversion read() now uses.
    #
    # pick ms timestamps whose ns value is an odd multiple of 1e6 (2-adic valuation below float64's
    # ULP exponent at this magnitude) so the float64 path is guaranteed to corrupt them:
    idx = pd.to_datetime(['2017-01-01T01:01:00.001Z', '2017-01-01T01:02:00.001Z',
                          '2017-01-01T01:03:00.007Z'])
    df = pd.DataFrame({'a': [1., 2., 3.]}, index=idx).astype(np.float32)
    bucket, _ = TickStore._to_bucket_pandas(df, 'SYM', None, 1_000_000, index_compressor=index_compressor)
    fake_self = types.SimpleNamespace(_index_precision='ms')
    rtn = TickStore._decode_bucket(fake_self, bucket, set(df.columns), {},
                                   include_symbol=False, include_images=False, columns=None)

    ns = TickStore._index_ms_to_ns([rtn['i']])
    assert ns.dtype == np.int64                                   # stayed integer, no float promotion
    got = pd.DatetimeIndex(ns, tz=datetime.timezone.utc)
    assert (got == idx).all(), (list(got.astype(str)), list(idx.astype(str)))

    if rtn['i'].dtype == np.uint64:
        # the uint64 layout (pco, and the v3 raw-uint64 lz4 layout) is what triggered the bug: the
        # old `uint64_array * np.int64(...)` promotes to float64 and loses the .001ms. (The v4 varint
        # lz4 layout decodes to int64 and was never affected -- which is why pco-as-default exposed it.)
        naive = np.concatenate([rtn['i']]) * np.int64(1_000_000)
        assert naive.dtype == np.float64
        assert not (pd.DatetimeIndex(naive.astype('int64'), tz=datetime.timezone.utc) == idx).all()


@pco_required
@pytest.mark.parametrize("index_compressor", ['pco', 'lz4'])
@pytest.mark.parametrize("label,prec_ns", [('ms', 1_000_000), ('s', 1_000_000_000)])
def test_pre_epoch_index_roundtrip(index_compressor, label, prec_ns):
    # Pre-1970 timestamps are negative int64 ns; the index layout stores them as int64 reinterpreted
    # to uint64. The decode-time sanity asserts did `uint64_scalar * 1_000_000`, which OVERFLOWS
    # (wraps mod 2^64) for those wrapped values and raised AssertionError. They must view as int64
    # first. Guard the full pre-epoch round-trip (index + column data).
    idx = pd.to_datetime(['1854-01-01T00:00:00Z', '1854-01-01T00:00:02Z', '1854-01-01T00:00:04Z'])
    assert pd.Timestamp(idx[0]).value < 0                            # genuinely pre-epoch
    df = pd.DataFrame({'a': [1., 2., 3.]}, index=idx).astype(np.float64)
    bucket, _ = TickStore._to_bucket_pandas(df, 'SYM', None, prec_ns, index_compressor=index_compressor)
    fake_self = types.SimpleNamespace(_index_precision=label)
    rtn = TickStore._decode_bucket(fake_self, bucket, set(df.columns), {},
                                   include_symbol=False, include_images=False, columns=None)
    got = pd.DatetimeIndex(TickStore._index_ms_to_ns([rtn['i']]), tz=datetime.timezone.utc)
    assert (got == idx.ceil('1%s' % label)).all(), (list(got.astype(str)), list(idx.astype(str)))
    assert np.allclose(rtn['a'], df['a'].values)


def test_arrays_to_dataframe_pandas_version_compat():
    # read() builds its result DataFrame via pandas' internal arrays_to_mgr, whose signature changed
    # across pandas majors (0.x positional / 1.x-2.x typ='block' / >=3.0 typ removed). This guards
    # that the version dispatch calls it correctly on the *installed* pandas -- it would have caught
    # the pandas-3.0 break (`arrays_to_mgr() got an unexpected keyword argument 'typ'`).
    index = pd.DatetimeIndex(np.array([1, 2, 3], dtype='int64') * 1_000_000_000, tz='UTC')
    arrays = [np.array([1., 2., 3.]), np.array([4., 5., 6.])]
    df = TickStore._arrays_to_dataframe(arrays, ['a', 'b'], index)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['a', 'b']
    assert (df.index == index).all()
    assert df['a'].tolist() == [1., 2., 3.] and df['b'].tolist() == [4., 5., 6.]


@pco_required
def test_lnq185vqlgz_write_deprecated_but_still_decodes():
    # LnQ185VQLgz is superseded by LnQ185pco (same grid/error, smaller, faster). New writes must be
    # blocked, but it stays in the registry so buckets already written with it keep decoding.
    assert 'LnQ185VQLgz' in codec_registry
    assert getattr(codec_registry['LnQ185VQLgz'], 'deprecated', None) == 'LnQ185pco'

    n = 2000
    idx = pd.date_range('2024-01-01', periods=n, freq='1s', tz='UTC')
    df = pd.DataFrame({'px': _autocorr_price(n)}, index=idx)
    with pytest.raises(ArcticException):
        TickStore._to_bucket_pandas(df, 'SYM', None, 1_000_000, codec='LnQ185VQLgz')

    # decode path is unaffected: the codec object still round-trips (mirrors reading old data)
    coder = codec_registry['LnQ185VQLgz']
    x = df['px'].values.astype(np.float32)
    got = coder.decode(coder.encode(x)).astype(np.float64)
    rtol = float(np.max(np.abs(got - x.astype(np.float64)) / (np.abs(x) + 1e-10)))
    assert rtol < 250e-6


# --- full write->read bucket serialization (no mongo) ----------------------------
def test_full_bucket_roundtrip_LnQ30brW10q10():
    n = 5000
    idx = pd.date_range('2026-01-01', periods=n, freq='5s', tz='UTC')
    df = pd.DataFrame({'volAt40': _logspace(n, seed=3), 'vwapBid': _autocorr_price(n)}, index=idx)
    bucket, _ = TickStore._to_bucket_pandas(df, 'SYM', None, 1_000_000,
                                            codec='LnQ30brW10q10', verify_codec=True)
    fake_self = types.SimpleNamespace(_index_precision='ms')
    rtn = TickStore._decode_bucket(fake_self, bucket, set(df.columns), {},
                                   include_symbol=False, include_images=False, columns=None)
    for c in df.columns:
        orig = df[c].values.astype(np.float64)
        got = np.asarray(rtn[c], dtype=np.float64)
        rtol = float(np.max(np.abs(got - orig) / (np.abs(orig) + 1e-10)))
        assert rtol < 250e-6, (c, rtol)
