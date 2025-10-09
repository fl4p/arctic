# Arctic

This is a fork of the old [Arctic](README-arctic.md) time series store with mongodb backend.

`TickStore` has been modified to be compatible with newer pandas and numpy.
It comes with a couple of compression techniques to lower disk usage.

* Pandas 2.x compatibility
* `float32` support (use: `ts.write(..., to_dtype=np.float32)`) for reduced storage size to 60~80%)
* Added lossy codec for floating point numbers to further improve compression. Expect 30~40% compression ratio with a max relative
  tolerance of 0.02 % (200ppm) for numbers in the range `[1e-9, 2e31]` (use: `ts.write(..., codec='loq16_10_dzv1')`)
* Uses `zlib` instead of `lz4`
* Added index varint compression


