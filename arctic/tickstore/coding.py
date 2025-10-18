import io
import lzma
import math
import zlib

import brotli
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
    ft = np.float32
    assert x.dtype == ft, x.dtype
    s = ft(2 ** prescale)
    o = ft(preadd)
    if s == 1 and o == 0:
        f = np.round(np.log(x) * ft(2 ** 16 / loss))
    else:
        f = np.round(np.log(x * s + o) * ft(2 ** 16 / loss))
    assert f.dtype == ft
    return f


def e_q16(x, loss=15, prescale=LOQ_PRESCALE, preadd=LOQ_PREADD):
    # e^x is faster than 10^x (and 2^x on some machines/older numpy?)!
    assert x.dtype == np.float32, x.dtype
    if prescale == 0 and preadd == 0:
        return np.exp(x / np.float32(2 ** 16 / loss))
    else:
        return (np.exp(x / np.float32(2 ** 16 / loss)) - np.float32(preadd)) / np.float32(2 ** prescale)


binary_compressors = dict(
    n=lambda d: d,
    lz4=lz4_compressHC,  # fast retrieval
    gz=zlib.compress,  # partial(zlib.compress, wbits=15),
    # sgz_=partial(zlib.compress, wbits=15),
    lzma=lambda d: lzma.compress(d, format=lzma.FORMAT_ALONE, preset=lzma.PRESET_EXTREME),
    br_9=lambda d: brotli.compress(d, quality=9, mode=brotli.MODE_FONT), # no-gil
    br_10=lambda d: brotli.compress(d, quality=10),
    br=lambda d: brotli.compress(d, quality=11),  # default q=11
)

binary_decompressors = dict(
    n=lambda d: d,
    lz4=lz4_decompress,  # fast retrieval
    gz=zlib.decompress,  # partial(zlib.compress, wbits=15),
    lzma=lambda d: lzma.decompress(d, format=lzma.FORMAT_ALONE),
    br_9=lambda d: brotli.decompress(d),
    br_10=lambda d: brotli.decompress(d),
    br=lambda d: brotli.decompress(d),
)

try:
    # py3.14
    from compression import zstd

    binary_compressors['zstd'] = lambda b: zstd.compress(b, level=20)  # 3 is default
    binary_decompressors['zstd'] = lambda b: zstd.decompress(b)
except ImportError:
    pass

try:
    # pypi version (wraps lib by Y?)
    import zstd

    binary_compressors['zstdY'] = lambda b: zstd.compress(b, 3)  # 3 is default
    binary_decompressors['zstdY'] = lambda b: zstd.decompress(b)
except ImportError:
    pass

try:
    import gorillacompression as gc

    binary_compressors['gc32'] = lambda a: gc.ValuesEncoder.encode_all(a, 'f32')
    binary_decompressors['gc32'] = lambda b: gc.ValuesDecoder.decode_all(b)
except ImportError:
    pass


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

    def __repr__(self):
        return f'LnQ16_VQL({repr(self.comp)}, {self.loq_loss}, {self.loq_prescale}, {self.loq_preadd})'

    def __init__(self, comp='lz4', loq_loss=25, log_prescale=37, loq_preadd=.0001):
        """

        Parameters
        ----------
        comp
        loq_loss how much loss TODO
        log_prescale: tune the dynamic range to avoid arithmetic overflow when using float32 math.
                        set this to the highest number so that the code doesn't overflow.
        loq_preadd : tune the precision for small numbers. decrease to increase precision.
                    TODO why?
        """
        self._signed = False
        self.delta_order = 1
        self.loq_loss = loq_loss
        self.loq_prescale = log_prescale  # 2**-37 => 1e-12, generally good, overflow at 2.2e+27
        self.loq_preadd = loq_preadd

        assert comp in binary_decompressors and comp in binary_compressors, comp
        self.comp = comp

        self.rtol_reg = 1e-10  # regularization constant when computing rtol
        self.rtol_max = 200e-6  # upper limit of rtol

    def encode(self, arr):
        # used for px, also supports unsigned values but not encouraged
        # loq+diff+vql
        signed = self._signed
        if not signed:
            assert arr.min() >= 0, arr.min()
        i = ln_q16(np.abs(arr) if signed else arr, self.loq_loss, prescale=self.loq_prescale, preadd=self.loq_preadd)
        assert math.isfinite(i.max()), ("code overflow", arr[np.argmax(i)])
        if signed:
            assert i.min() >= 0, (i.min())
            i *= np.sign(arr)
        i = i.astype(np.int64)
        for _ in range(self.delta_order):
            i = np.diff(i, prepend=np.int64(0))
        zigzag_encode_inplace(i)
        assert i.min() >= 0, (np.argmin(i), i.min(), arr[np.argmin(i) - 1:np.argmin(i) + 1])
        buf = nparray_varint_encode(i.astype(np.uint64))
        buf = binary_compressors.get(self.comp)(buf)
        return buf

    def decode(self, buf):
        buf = binary_decompressors.get(self.comp)(buf)
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

    def __repr__(self):
        return f'LnQ16_zlib({self.loq_loss},{self.loq_prescale},{self.loq_preadd})'

    def __init__(self, loq_loss=15, loq_prescale=37, loq_preadd=.0001):
        self.loq_loss = loq_loss
        self.loq_prescale = loq_prescale  # 2**-37 => 1e-12, generally good
        self.loq_preadd = loq_preadd

        self.rtol_reg = 1e-10  # regularization constant when computing rtol
        self.rtol_max = 200e-6  # upper limit of rtol

    def encode(self, arr):
        i = ln_q16(np.abs(arr), self.loq_loss, self.loq_prescale, preadd=self.loq_preadd)
        if i.min() < 0:
            imin = np.argmin(i)
            raise ValueError('input[%s] %s maps to %s<0!' % (imin, arr[imin], i[imin]))
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


class LnQ16:
    """
    * signed inputs
    * noisy inputs with no auto-correlation
    * int32 tobytes + zlib
    * e.g. volume data
    """

    def __repr__(self):
        return f'LnQ16({repr(self.comp)},{self.loq_loss},{self.loq_prescale},{self.loq_preadd})'

    def __init__(self, comp, loq_loss=15, loq_prescale=37, loq_preadd=.0001):
        self.comp = comp
        self.loq_loss = loq_loss
        self.loq_prescale = loq_prescale  # 2**-37 => 1e-12, generally good
        self.loq_preadd = loq_preadd

        self.rtol_reg = 1e-10  # regularization constant when computing rtol
        self.rtol_max = 200e-6  # upper limit of rtol

    def encode(self, arr):
        i = ln_q16(np.abs(arr), self.loq_loss, self.loq_prescale, preadd=self.loq_preadd)
        if i.min() < 0:
            imin = np.argmin(i)
            raise ValueError('input[%s] %s maps to %s<0!' % (imin, arr[imin], i[imin]))
        i *= np.sign(arr)
        buf = i.astype(np.int32).tobytes()
        buf = binary_compressors[self.comp](buf)
        return buf

    def decode(self, buf: bytes):
        buf = binary_decompressors[self.comp](buf)
        i = np.frombuffer(buf, dtype=np.int32)
        f = np.abs(i).astype(np.float32)
        f = e_q16(f, self.loq_loss, prescale=self.loq_prescale, preadd=self.loq_preadd) * np.sign(i)
        return f
