import io

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


def log_q16_10(x, loss=10):
    f = np.round(np.log10(x * (2 ** 16) + 1) * (2 ** 16 / loss))
    return numpy_fill(f).astype('int')


def exp_q16_10(x, loss=10):
    return (np.power(10, (x / (2 ** 16 / loss))) - 1) / (2 ** 16)


def encode_logQ16_10_dzv(arr, delta_order=1):
    i = log_q16_10(arr)
    for _ in range(delta_order):
        i = np.diff(i, prepend=0)
    zigzag_encode_inplace(i)
    assert i.min() >= 0
    buf = nparray_varint_encode(i.astype(np.uint64))
    buf = lz4_compressHC(buf)
    return buf


def decode_logQ16_10_dzv(buf: bytes, delta_order=1):
    buf = lz4_decompress(buf)
    i = nparray_varint_decode(buf)
    zigzag_decode_inplace(i)
    for _ in range(delta_order):
        i = np.cumsum(i)
    f = exp_q16_10(i)
    return f
