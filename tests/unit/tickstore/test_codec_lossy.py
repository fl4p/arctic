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


# --- every registered codec round-trips within its declared tolerance ------------
@pytest.mark.parametrize("name", sorted(codec_registry))
def test_registered_codec_roundtrip(name):
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
