import io
import zlib

import numpy as np
import pandas as pd

from .int_coding import varint_encode_unsigned, varint_decode_unsigned, zigzag_encode_inplace, zigzag_decode_inplace

try:
    from lz4.block import compress as lz4_compress, decompress as lz4_decompress

    lz4_compressHC = lambda _str: lz4_compress(_str, mode='high_compression')
except ImportError as e:
    from lz4 import compress as lz4_compress, compressHC as lz4_compressHC, decompress as lz4_decompress


# import pyximport  # cython
# pyximport.install(
#    setup_args={"include_dirs": np.get_include()},
#    reload_support=True,
#    language_level=3)
# instead: python setup.py build_ext --inplace


def nparray_varint_encode(arr: np.array):
    iob = io.BytesIO()
    varint_encode_unsigned(arr, iob.write)
    iob.seek(0)
    return iob.read(-1)


def nparray_varint_decode(buf):
    return varint_decode_unsigned(buf, (1 << 64) - 1)


def numpy_fill(arr):
    df = pd.Series(arr)
    df.ffill(inplace=True)
    return df.values


def log_q16(x, loss=10, prescale=16, preadd=1):
    f = np.round(np.log10(x * (2 ** prescale) + preadd) * (2 ** 16 / loss))
    return (f).astype('int')


def exp_q16(x, loss=10, prescale=16, preadd=1):
    return (np.power(10, (x / (2 ** 16 / loss))) - preadd) / (2 ** prescale)


def encode_logQ16_10_dzv(arr, delta_order=1, prescale=16, preadd=1, comp='lz4'):
    i = log_q16(arr, prescale=prescale, preadd=preadd)
    for _ in range(delta_order):
        i = np.diff(i, prepend=0)
    zigzag_encode_inplace(i)
    assert i.min() >= 0, i.min()
    buf = nparray_varint_encode(i.astype(np.uint64))
    buf = lz4_compressHC(buf) if comp == 'lz4' else zlib.compress(buf)
    return buf


def decode_logQ16_10_dzv(buf: bytes, delta_order=1, prescale=16, preadd=1, comp='lz4'):
    buf = lz4_decompress(buf) if comp == 'lz4' else zlib.decompress(buf)
    i = nparray_varint_decode(buf)
    zigzag_decode_inplace(i)
    for _ in range(delta_order):
        i = np.cumsum(i)
    f = exp_q16(i, prescale=prescale, preadd=preadd)
    return f


LOQ_PRESCALE = 37
LOQ_PREADD = .0001


def ln_q16(x, loss=15, prescale=LOQ_PRESCALE, preadd=LOQ_PREADD):
    assert x.dtype == np.float32
    f = np.round(np.log(x * np.float32(2 ** prescale) + np.float32(preadd)) * np.float32(2 ** 16 / loss))
    assert f.dtype == np.float32
    return f


def e_q16(x, loss=15, prescale=LOQ_PRESCALE, preadd=LOQ_PREADD):
    # e^x is faster than 10^x (and 2^x on some machines/older numpy?)!
    assert x.dtype == np.float32, x.dtype
    return (np.exp(x / np.float32(2 ** 16 / loss)) - np.float32(preadd)) / np.float32(2 ** prescale)


class LnQ16_VQL():
    """
    * optimized for decoder speed
    * good for storing prices
    * suitable for positive values down to 1e-12
    * uses lz4 compression
    * targets 120ppm rtol (1.2 basis point) over whole dynamic range


        # price will compress incredibly well as compared to `qty`, so the slightly worse compression
        # of lz4 as compared to zlib does not matter
    """

    def __init__(self, comp = 'lz4', loq_loss=25):
        self._signed = False
        self.delta_order = 1
        self.loq_loss = loq_loss
        self.loq_prescale = 37  # 2**-37 => 1e-12, generally good
        self.loq_preadd = .0001

        assert comp in {'lz4', 'zlib'}
        self.comp = comp

        self.rtol_reg = 1e-10  # regularization constant when computing rtol
        self.rtol_max = 200e-6  # upper limit of rtol

    def encode(self, arr):
        # used for px, also supports unsigned values but not encouraged
        # loq+diff+vql
        signed = self._signed
        if not signed:
            assert arr.min() >= 0
        i = ln_q16(np.abs(arr) if signed else arr, self.loq_loss, prescale=self.loq_prescale, preadd=self.loq_preadd)
        if signed:
            assert i.min() >= 0, (i.min())
            i *= np.sign(arr)
        i = i.astype(np.uint64)
        for _ in range(self.delta_order):
            i = np.diff(i, prepend=0)
        zigzag_encode_inplace(i)
        assert i.min() >= 0, i.min()
        buf = nparray_varint_encode(i.astype(np.uint64))
        buf = lz4_compressHC(buf) if self.comp == 'lz4' else zlib.compress(buf)  # using lz4 here for fast retrieval

        return buf

    def decode(self, buf):
        buf = lz4_decompress(buf) if self.comp == 'lz4' else zlib.decompress(buf)
        i = nparray_varint_decode(buf)
        zigzag_decode_inplace(i)
        for _ in range(self.delta_order):
            i = np.cumsum(i)
        f = e_q16(i.astype(np.float32), self.loq_loss, self.loq_prescale, preadd=self.loq_preadd)
        assert f.dtype == np.float32
        return f


class LnQ16_15_VQL_lz4(LnQ16_VQL):
    def __init__(self):
        super().__init__(comp='lz4', loq_loss=15)

class LnQ16_15_VQL_zlib(LnQ16_VQL):
    def __init__(self):
        super().__init__(comp='zlib', loq_loss=15)


class LnQ16_zlib:
    """
    * signed inputs
    * noisy inputs with no auto-correlation
    * int32 tobytes + zlib
    * e.g. volume data
    """

    def __init__(self, loq_loss=15):
        self.loq_loss = loq_loss
        self.loq_prescale = 37  # 2**-37 => 1e-12, generally good
        self.loq_preadd = .0001

        self.rtol_reg = 1e-10  # regularization constant when computing rtol
        self.rtol_max = 200e-6  # upper limit of rtol

    def encode(self, arr):
        i = ln_q16(np.abs(arr), self.loq_loss, self.loq_prescale, preadd=self.loq_preadd)
        assert i.min() >= 0
        i *= np.sign(arr)
        buf = i.astype(np.int32).tobytes()
        buf = zlib.compress(buf)
        return buf

    def decode(self, buf: bytes):
        buf = zlib.decompress(buf)
        i = np.frombuffer(buf, dtype=np.int32)
        f = np.abs(i).astype(np.float32)
        f = e_q16(f, self.loq_loss, prescale=self.loq_prescale, preadd=self.loq_preadd) * np.sign(i)
        return f
