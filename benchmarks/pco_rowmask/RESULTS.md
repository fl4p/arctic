# pcodec for TickStore rowmasks — benchmark results

Run: `/Users/fab/dev/venvs/jnb314/bin/python3 benchmarks/pco_rowmask/bench_rowmask_pco.py`
(pcodec 1.0.2, level 8; numpy 2.3.3). All representations are round-trip verified.

## Question

TickStore stores each column's ROWMASK (which ticks carry that column) as
`lz4_compressHC(np.packbits(mask).tobytes())`. The code comment in
`tickstore.py` claims *"pco is a numeric-array codec that cannot compress the
packed-bit bytes anyway."* This benchmark tests whether pco — fed a sensible
representation of the mask — beats the lz4 baseline.

## What was compared (all on N=100,000 = default chunk_size)

| repr | what it stores | notes |
|---|---|---|
| `lz4HC(packbits)` | **baseline** (current) | |
| `lz4(packbits)` | lz4 fast level | cheap encode, bigger |
| `pco(packbits u8)` | the packed bytes → pco (`enable_8_bit`) | smallest code change |
| `pco(bits u8)` | raw 0/1 byte per row → pco | |
| `pco(gaps u32)` | gaps between set-bit indices → pco | |
| `pco(runlen u32)` | alternating run lengths → pco | |
| `brotli11(packbits)` | reference (size ceiling) | |

Note: pco **refuses 8-bit dtypes** unless `ChunkConfig(enable_8_bit=True)` — it
warns this "is often a mistake". So the packed bytes need that flag; gaps/runs
use uint32 and don't.

## Headline numbers (aggregate over a synthetic depth-10 L2 frame, 44 columns)

| repr | total bytes | vs base | enc µs | dec µs |
|---|---:|---:|---:|---:|
| lz4HC(packbits) **[base]** | 414,436 | 100.0% | 7,585 | 415 |
| lz4(packbits) | 433,713 | 104.7% | **396** | 403 |
| **pco(packbits u8)** | 345,116 | **83.3%** | 9,134 | 1,093 |
| pco(bits u8) | 345,518 | 83.4% | 15,427 | 3,506 |
| **pco(gaps u32)** | 339,438 | **81.9%** | 19,913 | 9,672 |
| pco(runlen u32) | 375,028 | 90.5% | 24,111 | 364,233 |
| brotli11(packbits) | 341,915 | 82.5% | 607,145 | 1,953 |

Synthetic-distribution aggregate (9 masks) tells the same story: pco(gaps)
66.7%, pco(packbits) 73.6%, baseline 100%.

## Findings

1. **The comment is wrong: pco beats lz4 on rowmasks.** Feeding the *packed
   bytes* straight to pco (`enable_8_bit=True`) is **~17% smaller** than the
   lz4-HC baseline on the L2 frame, at encode cost comparable to lz4-HC. So the
   "can't compress packed bits" claim doesn't hold — pco's entropy coder does
   find structure lz4 misses.

2. **`pco(gaps)` is the size winner**, especially on **sparse and bursty**
   columns — the ones that actually matter:
   - trade fields (density ~1.6%): **−50%** vs baseline (e.g. 812→403 bytes).
   - bursty ~5%/long-run: **−51%** (230→112 bytes).
   - deep book levels (1–12% density): −37…−48%.
   It needs the original length on decode (already known: `doc_length`) and an
   empty-mask special case.

3. **Mid-density random masks (~0.3–0.55) are incompressible** — everything,
   including pco and brotli, sits at ~1.0 bit/row (the Shannon limit for a coin
   flip). My L2 model makes level-presence an *independent* Bernoulli per row,
   which is the worst case. **Real book updates are autocorrelated** (a level
   present this tick is likely present next tick) → real masks look like the
   *bursty* scenarios, where pco(gaps)/pco(runlen) win big (−50%). So the L2
   aggregate here **understates** the realistic gain.

4. **All-ones masks (the common dense column) are already free** — 63 B
   (baseline) vs 21 B (pco). Irrelevant in absolute terms either way.

5. **Decode stays cheap** for the viable options: pco(packbits) ~25–30 µs/mask
   (~3× lz4's ~10 µs, but negligible next to column-data decode). pco(gaps) is
   25–450 µs (cost scales with set-bit count — slow on dense masks).
   `pco(runlen)` decode is catastrophic here (up to 15 ms) only because the
   reference impl is a pure-Python run loop; it could be vectorized with
   `np.repeat`, but it doesn't win on size anyway, so skip it.

6. **brotli matches pco on size but is ~60–100× slower to encode** (607 ms vs
   9 ms total) — not viable on the write-once path. pco gives ~brotli size at
   ~lz4 encode cost.

7. **`pco(bits u8)` is risky** — on one bursty/low-density mask it *expanded*
   to +1224% (pco's mode detection mis-fired on the 0/1 byte stream). Avoid;
   `pco(packbits)` and `pco(gaps)` were robust across every scenario.

## REAL DATA — bitmex XBTUSD L2 book (the decisive test)

`bench_rowmask_real.py` on a disk-cached arctic read: **10.2M rows × 33 book
columns**, first 5M rows = **50 buckets × 33 cols = 1,650 real masks**, present
= `~isnan` (the real `_to_bucket_pandas` rule). All round-trips verified.

In *this* slice every column is sparse-ish — densities **0.03 → 0.46**, no
all-ones column — which is why pco(gaps) wins below. **NOTE:** a wider analysis
(full range / aggregated depth-band volumes) finds these masks are actually
**dense / ≈all-ones**, where gaps *regresses*. The −31% gaps number below is
specific to this sparse-ish slice; read the regime caveat before trusting it.
`pco(packbits)` is the only recommendation robust to both — see the end.

| representation | total bytes | vs base | enc µs/mask | dec µs/mask |
|---|---:|---:|---:|---:|
| lz4HC(packbits) **[base]** | 16,358,157 | 100.0% | 210.7 | 11.4 |
| lz4(packbits) | 17,807,646 | 108.9% | 13.5 | 10.7 |
| **pco(packbits u8)** | 12,583,705 | **76.9%** | 215.8 | 30.1 |
| pco(bits u8) | 15,244,137 | 93.2% | 375.3 | 85.1 |
| **pco(gaps u32)** | 11,252,118 | **68.8%** | 332.5 | 124.5 |
| pco(runlen u32) | 14,489,459 | 88.6% | 656.4 | 12,755 |
| brotli11(packbits) | 11,998,227 | 73.3% | 15,334 | 94.5 |

By density band (total bytes vs baseline):

| band | n | lz4HC | pco(packbits) | pco(gaps) | brotli11 |
|---|---:|---:|---:|---:|---:|
| [0.00,0.02) | 2 | 100% | 59% | **55%** | 63% |
| [0.02,0.20) | 663 | 100% | 69% | **59%** | 65% |
| [0.20,0.80) | 985 | 100% | 81% | **73%** | 77% |

**This confirms and exceeds the synthetic prediction:**
- **`pco(gaps)` is 31% smaller** than the lz4 baseline overall; **`pco(packbits)`
  23% smaller** for a one-line codec swap.
- The synthetic model said the 0.2–0.8 mid-density band was ~incompressible
  (~100%, random coin-flip). On **real** data pco(gaps) still gets it to **73%**
  — because real book masks are **autocorrelated/bursty**, not random. Exactly
  the effect flagged in finding #3; real data makes pco win where Bernoulli
  said it couldn't.
- brotli ≈ pco(gaps) on size but **~46–70× slower to encode** (15.3 ms vs 0.33 ms
  per mask) — pco gives brotli-class size at lz4-class cost.
- Decode stays cheap for the viable options: pco(gaps) 124 µs/mask (11× lz4 but
  negligible absolute); pco(packbits) 30 µs. `pco(runlen)` decode is still
  unusable (12.8 ms, the pure-Python run loop) and doesn't win on size anyway.
- Peak RSS 1.5 GB; whole run 297 s incl. the 1.4 GB pickle load.

## Recommendation

- **The one to wire: `pco(packbits u8)`** — keep `np.packbits`, swap lz4 for pco.
  Robust across every regime (−23% to −38%, never regresses), encode ≈ lz4-HC,
  decode still ~µs, smallest possible change.
- `pco(gaps u32)` looks best *on the synthetic frame* and on my sparse-ish real
  slice, **but regresses +12% on dense real masks** — it is sparse-only and NOT
  a safe default. See the regime caveat at the end before considering it.

### Is it worth wiring in?

Rowmasks are a *fraction* of a chunk (column DATA + the timestamp INDEX
dominate — check `TickStore.last_write_bytes = (bytes_i, bytes_d, bytes_m)`;
`bytes_m` is the rowmask share). A 17% rowmask reduction → low-single-digit %
of total chunk size. The catch: unlike the index (one codec marker per bucket
via `INDEX_COMPRESSION`), rowmask is **per column**, and `_decode_bucket`
currently `lz4_decompress`es it unconditionally (twice — once for the union
mask, once per column). Adopting pco needs a stored codec marker the reader
dispatches on (per-bucket flag, or reuse the column `CODEC`/a new field) and a
read-path branch — a real format change, bigger than the index swap was.

### IMPORTANT — gaps is regime-dependent; do NOT default to it

Two real-data measurements disagree on the *mask density*, and that flips the
gaps verdict:

- This script's run (first 5M rows of `cc_l2_ext_XBTUSD_bitmex`, raw `read()`
  frame, `.notna()`): densities **0.03–0.46**, no dense/all-ones column. In that
  regime pco(gaps) won every band → −31% overall (the table above).
- fab's wider analysis (dense / ≈all-ones aggregated depth-band volume masks —
  `volBt/volAt`, present on nearly every tick; includes zstd): pco(gaps)
  **regresses ~+12%** on the dominant dense masks (gaps is a *sparse* repr —
  one int per set bit — so it bloats as density → 1), while pco(packbits) is
  −35% dense / −38% mid. There, 395 dense masks total ~17 KB (~43 B each).

These are consistent: same codec, different input. gaps wins only on genuinely
sparse masks (< ~0.1 density). The real cache here does **not** contain such
columns (per-level book depth / bursty trade-presence), so the synthetic
gaps −50% sparse result is **neither confirmed nor refuted** by real data.

**Verdict:** the comment's claim is false (pco does compress rowmasks), but the
payoff is small and the bytes are a rounding error next to column DATA + INDEX
(check `last_write_bytes = (bytes_i, bytes_d, bytes_m)` — `bytes_m` is tiny).
**Low priority.** If you wire anything:

1. **Use `pco(packbits u8)`** — the robust universal win: −23% to −38% across
   every regime tested, *never regresses*, smallest change (swap lz4→pco on the
   same packed bytes; remember `ChunkConfig(enable_8_bit=True)`). `zstd19(packbits)`
   is a clean alternative (−56% dense, fast) if zstd is already a dependency.
2. **Do NOT default to `pco(gaps u32)`** — it regresses +12% on the dense masks
   that dominate real data. Only worth it behind a `density < threshold` switch,
   and only once genuinely-sparse real columns are confirmed to exist.
3. Avoid `pco(bits u8)` (−7%, can expand) and `pco(runlen)` (best pco *size* on
   dense/clustered, but the fragmented-mask decode landmine — 12 ms — isn't
   worth the margin over packbits).

Adoption cost is the format change: rowmask is per-column and `_decode_bucket`
lz4-decompresses it unconditionally (twice), so it needs a stored codec marker
the reader dispatches on + a read branch — bigger than the index swap was.
