# TickStore lossy column codecs

`TickStore.write(symbol, data, codec=...)` compresses each numeric column with a **lossy
log-quantization codec** instead of the default lossless lz4. On financial L2 / tick data this is
typically **3–4× smaller than lossless** at a controlled relative error (~hundreds of ppm), because
prices and sizes only need constant *relative* precision, not full float32 mantissa.

`codec` may be a single codec name (applied to every column) or a `{column: name}` dict.

```python
ts.write('XBTUSD@bitmex', df, codec='LnQ20VQLgz')          # all columns
ts.write('XBTUSD@bitmex', df, codec={'vwapBid': 'LnQ15VQLlz4',
                                     'volAt40': 'LnQ15gz'})  # per column
```

Reads are transparent: the codec name is stored per column in the bucket and the matching decoder is
looked up from the registry — no codec argument is needed on `read()`.

## The pipeline

A codec maps each float through a logarithm, quantizes to an integer, then entropy-compresses the
integer stream. Two structural variants exist, and **which one is best depends on the compressor**:

```
VQL variant (LnQ16_VQL):  ln_q16 → int64 → Δ (delta, order 1) → zigzag → varint → compress
int32 variant (LnQ16):    ln_q16 → int32 bytes →                                  compress
```

- `ln_q16(x) = round(ln(x·2^prescale + preadd) · 2^16/loss)` — log-quantize. `prescale`/`preadd`
  only shift the representable range to avoid float32 overflow / gain small-number precision; they do
  **not** affect compressed size.
- The **VQL** variant (delta + zigzag + varint) suits autocorrelated data and the byte-oriented
  compressors (`gz`, `brotli`, `zstd`).
- The **int32** variant (raw fixed-width) suits `lzma`, whose context model reads the aligned bytes
  better than a varint stream — feeding lzma a varint/delta stream is *worse*.

## Error model

Quantization step is uniform in log-space, so the error is a near-constant **relative** error:

```
max error ≈ loss × 7.7 ppm
```

| loss | ~max error |
|-----:|-----------:|
| 15   | 115 ppm    |
| 20   | 154 ppm    |
| 25   | 194 ppm    |
| 30   | 232 ppm    |

Higher `loss` = coarser = smaller output. The working budget on this data is **< 250 ppm**, so loss
up to ~32 is usable. Each codec declares `rtol_max`; `write(..., verify_codec=True)` (the default)
decodes every column and raises `ArcticException` if the observed error exceeds `rtol_max` — so a
codec whose `loss` implies >`rtol_max` error must set `rtol_max` accordingly (see `LnQ30brW10q10`).

## Registered codecs

| name | class / params | ~error | compressor | use |
|------|----------------|-------:|------------|-----|
| `LnQ15VQLlz4`  | `LnQ16_VQL(loss=15, lz4)` | 115 ppm | lz4 | prices, fast decode |
| `LnQ20VQLgz`   | `LnQ16_VQL(loss=20, gz, prescale=20, preadd=1e-12)` | 154 ppm | gz | **general L2**, range [1e-15, 2e+32] |
| `LnQ25VQLgz`   | `LnQ16_VQL(loss=25, gz)` | 194 ppm | gz | general L2, range [1e-13, 2e+27] |
| `LnQ185VQLgz`  | `LnQ16_VQL(loss=1.85, gz, prescale=0, preadd=0)` | 16 ppm | gz | **deprecated for writes** → use `LnQ185pco` (still decodes) |
| `LnQ185pco`    | `LnQ16_pco(loss=1.85, prescale=0, preadd=0)` | ~18 ppm | pcodec (`pco` crate) | **prices**, full float32 range (~1e-45…3.4e38); pco twin of `LnQ185VQLgz` |
| `LnQ15gz`      | `LnQ16_zlib(loss=15)` (int32) | 115 ppm | gz | signed qty / volume (no autocorr) |
| `LnQ15br9`     | `LnQ16(br_9, loss=15)` (int32) | 115 ppm | brotli q9 | qty, abs(x) ≥ 0.73e-11 |
| `LnQ15pcoS`    | `LnQ16_pco(loss=15, prescale=37, preadd=1e-4, signed=True)` | 115 ppm | pcodec | **signed qty**, ~14% smaller than `LnQ15br9` |
| `LnQ30brW10q10`| `LnQ16_VQL(loss=30, br_w10q10, prescale=20, preadd=1e-12)`, `rtol_max=250e-6` | 232 ppm | brotli q10/lgwin10 | smallest general L2 (slow encode) |
| `LnQ30pco`     | `LnQ16_pco(loss=30, prescale=20, preadd=1e-12)`, `rtol_max=250e-6` | 232 ppm | pcodec (`pco` crate) | **general L2**, ~same size as brW10q10, ~37× cheaper encode |

### `LnQ30brW10q10` — the size-optimal general codec

Found by a sweep over the structural knobs on real L2 data (`find_coders.py` in the jnb repo):

- **~24–30% smaller** than `LnQ20VQLgz` (combining a stronger compressor with the full 250 ppm budget).
- Brotli **quality 10** (≈ q11 size, ~3× faster encode) with the **smallest window** (`lgwin=10`, 1 KB)
  — the delta+varint stream is only short-range correlated, so a large window adds modeling overhead
  for no gain (~1.4% smaller than default brotli).
- **Read cost:** the *column-data* decode is ~1.9× gz (brotli decompress is slower); the index and
  rowmask are unaffected (always lz4, ~0.45 ms/bucket). So budget reads at ~1.6–1.9× `LnQ20VQLgz`.
- Encode is much slower than gz — best for write-once / read-many archival.

Tradeoff at loss=20, real L2 data: `lzma` int32 (79%, slow decode) ≈ `brotli` VQL (81%, fast decode)
< `zstd` VQL (89%) < `gz` (100%). Brotli was the best size **and** decode point; zstd never reached
brotli's size; lzma matched it but decodes ~2.5× slower.

### `LnQ30pco` — same error, far cheaper than brotli

`LnQ16_pco` keeps the **exact** `ln_q16(loss=30)` quantization grid (so its error is identical to
`LnQ30brW10q10`, 232 ppm) but stores the integer stream with **pcodec** (the `pco` crate) instead of
delta+zigzag+varint+brotli. pcodec does its own delta detection and adaptive bin-packing, so:

- **Size:** ~equal to `LnQ30brW10q10` on L2 (within a couple %), ~18–32% smaller than `LnQ20VQLgz`.
- **Speed:** encodes ~**37× faster** than brotli q10 and decodes ~**3× faster** — the better
  write-once/read-many *and* re-encode-heavy choice. (Brotli q10 spends most of its CPU on a large
  context model that the short-range delta stream barely uses.)
- pcodec also rediscovers the log structure on its own: feeding it *mantissa-rounded floats* (no
  `ln_q16` at all) gets within a few % of the same size.

`pcodec` is an **optional dependency** (`pip install pcodec`). The codec registers without it; only an
actual encode/decode imports it, and raises a clear `ImportError` if it is missing. So a deployment
that never selects `LnQ30pco` needs nothing extra. The on-disk blob is pco-format (like every other
TickStore column blob, it is only readable by TickStore — not a regression).

> **Lossless option.** pcodec also stores raw float32 losslessly at ~1.5 B/value (vs lz4's ~3.7),
> which none of the `LnQ*` codecs offer. Not registered as a TickStore codec (the column path is
> inherently lossy-or-lz4), but it is the strongest lossless float option if ever needed — see
> `find_coders.py --pco`.

### `LnQ185pco` — wide-range price codec (replaces `LnQ185VQLgz`)

Prices need a much wider dynamic range than the general L2 codecs: `LnQ30pco`/`LnQ20VQLgz`
(`prescale=20`) compute `log(x·2²⁰ + …)`, which **overflows float32 above ~3.2e32** and is too coarse
(232/154 ppm) for price precision. `LnQ185pco` keeps the exact grid the old `LnQ185VQLgz` used —
`loss=1.85, prescale=0, preadd=0`, so `ln_q16` takes `log(x)` directly (no `2^prescale` factor to
overflow) — giving **~18 ppm over the full positive float32 range (~1e-45 … 3.4e38)**. It stores the
integer stream with **pcodec** instead of delta+varint+gz.

Because the grid is identical, error and range match `LnQ185VQLgz` exactly; only size and speed change.
On real price data (per-symbol, encoded in 4M-row buckets — the trade `tickstore_chunk_size`):

| symbol | n | `LnQ185VQLgz` | `LnQ185pco` | ratio |
|--------|--:|--------------:|------------:|------:|
| BTCUSDT@binance_futures | 918 M | 0.076 B/val | 0.070 B/val | **92.2 %** |
| XBTUSD@bitmex | 63 M | 0.358 B/val | 0.301 B/val | **84.1 %** |
| FX_BTC_JPY@bitflyer | 8 M | 0.516 B/val | 0.450 B/val | **87.2 %** |
| BCH-USDC@coinbasepro | 8 M | 0.581 B/val | 0.545 B/val | **93.8 %** |

So **~6–16 % smaller than `LnQ185VQLgz`** at identical precision, plus pcodec's ~30–40× cheaper encode
and ~3× cheaper decode. `LnQ185VQLgz` is therefore **deprecated for writes** (it stays registered so
existing buckets decode); new price writes should use `LnQ185pco`. (For reference, `LnQ30pco` is ~17–27 %
of `LnQ185VQLgz` here — far smaller, but that is loss=30/232 ppm, ~13× coarser, and overflow-prone on
wide ranges; it is the *general L2* codec, not a price codec.)

## Index compression (`index_compressor`)

`codec` compresses the column *values*; the timestamp **index** is a separate stream with its own
codec, selected by `write(..., index_compressor=...)`:

```python
ts.write('XBTUSD@bitmex', df, codec='LnQ30pco', index_compressor='pco')
```

- `'lz4'` (default) — the legacy layout: `lz4(first_stamp + int64-or-varint deltas)`. Unchanged; all
  existing buckets read back identically.
- `'pco'` — store the uint64 delta array straight through **pcodec** (lossless, integer-exact). On a
  realistic irregular tick index it is ~**0.9 B/val vs ~1.7** for lz4 (~48% smaller on a large bucket)
  and on a regular grid it is essentially free (pco detects the constant stride). Encodes ~20× faster
  and decodes ~3× faster than the lz4(diff) path. Requires the optional `pcodec` package.

The codec is recorded per bucket in the `INDEX_COMPRESSION` field, so reads pick it up automatically —
no argument needed on `read()`. A bucket with no field defaults to lz4. The index codec is independent
of the column `codec`: any combination works (e.g. `pco` index + `LnQ30pco` columns). The per-column
**rowmask** bitmaps are always lz4 regardless (pco is a numeric-array codec, not a byte compressor).

> Byte compressors (`gz`, `lzma`, …) are also accepted for the index and now round-trip correctly
> (the read path was previously hardcoded to lz4); pco is the recommended non-lz4 choice.

## Constraints

- **float32 only.** Codecs `assert` float32 input. A non-float32 column, or a `to_dtype` gain-downcast
  (e.g. float16), combined with a codec is **rejected** (`ArcticException`) — two stacked lossy stages
  the codec can't verify. Cast to float32 or drop the codec.
- **Positive values** for the VQL codecs (prices/sizes); the `LnQ16_zlib`/`LnQ16` (int32) codecs carry
  a sign and handle signed data (e.g. signed trade qty).
- **No NaN/inf** reaches a codec: `_to_bucket_pandas` masks NaN gaps via the per-column rowmask before
  encoding. If non-finite *did* reach a quantizer it now raises `ValueError` (fail-fast) rather than
  silently casting to `INT_MIN` and corrupting data.

## Adding a codec

1. (If a new compressor) add a `name → fn` pair to `binary_compressors` **and** `binary_decompressors`
   in `coding.py`.
2. Register in `tickstore.py`:
   ```python
   c = LnQ16_VQL('br_w10q10', loq_loss=30, log_prescale=20, loq_preadd=1e-12)
   c.rtol_max = 250e-6                     # only if loss implies >200 ppm (the class default)
   register_codec('LnQ30brW10q10', c)      # name < 16 chars, unique
   ```
3. Old data keeps decoding regardless — deprecating a codec only blocks *writes*. Two mechanisms:
   the three `*dzv*` codecs are blocked by name in `_encode`; a **registered** codec is deprecated by
   setting `coder.deprecated = '<replacement>'` (e.g. `LnQ185VQLgz`), which `_encode` rejects on write
   while it stays in `codec_registry` so existing buckets still decode via `_decode_bucket`.

Benchmark harness: `test/coding/find_coders.py` (jnb repo) — `--raw` (un-quantized influx data),
`--frontier` (loss/size/error sweep), `--brotli` (quality×lgwin×mode grid), `--pco` (pcodec lossless
+ lossy vs the LnQ baselines), `--pickle <path>` (load a dumped dataset, e.g. to run zstd/pco under a
specific venv). Regression tests: `tests/unit/tickstore/test_codec_lossy.py`.
