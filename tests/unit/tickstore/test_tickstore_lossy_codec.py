"""Regression tests for the TickStore lossy-codec write path (TickStore.write(..., codec=)).

Each test targets a specific finding from the codec review:
  #1 lossless lz4 fallback decoupled from `verify` (no more enc=None when verify is off)
  #2 failed verification raises ArcticException (survives python -O) instead of asserting
  #3 lz4 is the always-available lossless fast-path, chosen when smaller than the codec
  #4 the raw-lz4 probe is skipped when the codec already won decisively
  #5 the log quantizers/codecs fail-fast on non-finite input instead of silently corrupting
  #6 verification includes the to_dtype/gain downcast loss (measured vs the original input)
  #7 the deprecated dzv codecs raise on write; their decode path is preserved
"""
import numpy as np
import pytest

from arctic.exceptions import ArcticException
from arctic.tickstore.tickstore import TickStore, codec_registry
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
    enc, codec_sel, gain = TickStore._encode(v, 'c', None, None, False, DBG)
    assert enc is not None
    assert codec_sel is None
    assert gain == 0
    assert lz4_decompress(enc) == v.tobytes()  # lossless


def test_no_codec_verify_on_encodes_losslessly():
    v = np.array([100.0, 101.5, 99.25], dtype=np.float32)
    enc, codec_sel, _ = TickStore._encode(v, 'c', None, None, True, DBG)
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
    enc, _, _ = TickStore._encode(v, 'c', None, 'LnQ15VQLlz4', True, DBG)
    assert enc is not None  # did not raise


# --------------------------------------------------------------------------- #
# #3 lz4 is the always-available lossless fast-path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('verify', [True, False])
def test_lz4_fallback_wins_when_smaller(verify):
    # exact codec emits 10x the bytes -> lz4 must win and clear codec_sel, regardless of verify.
    v = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float32)
    enc, codec_sel, _ = TickStore._encode(v, 'c', None, '_bigexact', verify, DBG)
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
    enc, codec_sel, _ = TickStore._encode(v, 'c', None, 'LnQ15VQLlz4', False, DBG)
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
# #6 verification includes the to_dtype/gain downcast loss
# --------------------------------------------------------------------------- #
def test_verify_includes_gain_downcast_loss():
    # values > float16 max (65504) force gain-scaling on the downcast.
    orig = np.array([70000.0, 140000.0, 98000.0], dtype=np.float64)

    # the OLD reference (decode vs the already-downcast v) would have passed: the codec is lossless
    # on the float16 values, so the downcast loss was hidden.
    v_cast, gain = TickStore._ensure_supported_dtypes(orig.copy(), to_dtype=np.float16)
    assert gain > 1
    c = codec_registry['_passthru16']
    d = c.decode(c.encode(v_cast))
    old_err = np.nanmax(abs(d - v_cast) / (abs(v_cast) + c.rtol_reg))
    assert old_err < c.rtol_max  # would have been declared "fine"

    # the NEW reference measures decode*gain against the original input, so the float16 loss is
    # included and verification raises.
    with pytest.raises(ArcticException):
        TickStore._encode(orig, 'c', np.float16, '_passthru16', True, DBG)


def test_no_to_dtype_keeps_gain_zero():
    v = np.linspace(100.0, 101.0, 400).astype(np.float32)
    _, _, gain = TickStore._encode(v, 'c', None, 'LnQ15VQLlz4', True, DBG)
    assert gain == 0  # no downcast -> verification reference unchanged from before


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
