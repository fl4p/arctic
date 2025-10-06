cimport cython

from libc.math cimport fabs

import numpy as np
import math

cimport numpy as np
cimport numpy as cnp

DEF  BUF_SIZ = 256*8
DEF VARINT_MAX_BITS =64


@cython.boundscheck(False)
def varint_encode(arr, write):
    cdef np.ndarray[long] val = arr
    cdef Py_ssize_t i
    cdef int value, bits
    cdef cython.char[:] buf = np.zeros([BUF_SIZ], dtype=np.byte)
    cdef Py_ssize_t pos = 0


    for i in range(0, len(val)):
        value = val[i]
        bits = value & 0x7f
        value >>= 7
        while value:
            buf[pos] = 0x80 | bits
            pos += 1
            bits = value & 0x7f
            value >>= 7
        buf[pos] = bits
        pos += 1
        if pos >= (BUF_SIZ/2):
            assert pos <= BUF_SIZ
            write(buf[:pos]) # memoryview, no-copy!
            pos = 0

    if pos > 0:
        write(buf[:pos]) # memoryview, no-copy!


@cython.boundscheck(False)
def varint_decode_unsigned(buffer, mask, result_type):
    cdef const unsigned char[:] buf = buffer
    cdef Py_ssize_t bufLen = len(buffer)

    cdef unsigned long result, mask_ = mask
    cdef unsigned char b, shift
    cdef Py_ssize_t pos = 0, resPos = 0

    arr = np.zeros([bufLen], dtype=np.uint64)
    cdef unsigned long[:] res = arr

    while pos < bufLen:
        result = 0
        shift = 0
        while 1:
            b = buf[pos]
            pos += 1
            result |= ((b & 0x7f) << shift)
            if not (b & 0x80):
                result &= mask_
                res[resPos] = result
                resPos += 1
                break
            shift += 7
            assert shift < VARINT_MAX_BITS, 'Too many bytes when decoding varint.'

    return arr[:resPos]
