<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Invariant-divisor division prototype

This branch holds the measurements behind the epic about dividing by loop-invariant divisors with a precomputed reciprocal. It is a measurement prototype, not the proposed API. The branch starts from apache/arrow-rs@6b34163d5.

## Contents

- `divproto/` is a standalone crate with its own workspace, so it stays out of the arrow-rs build. It has Granlund-Montgomery divisors for `u64`, `u128` and `u256`, Lemire divisors for `u32` and `u64`, signed wrappers that match `wrapping_div` and `wrapping_rem`, and variants of Comet's `pmod`. Its tests compare every method with `/`, `%`, `wrapping_div` and `wrapping_rem`, and its criterion benchmarks produce the microbenchmark tables in the epic.
- The `arrow-cast` change on this branch adds `div_rem_by`, `div_rem_narrow_by` and `all_narrow` to `DecimalCast`. For `i128` it uses Granlund-Montgomery, and the decimal downscale uses the hardware 64-bit divide instead when a scan of the batch shows every value fits in 64 bits. `arrow-cast/benches/decimal_downscale.rs` measures it.
- `divproto/variants/` has patches for the other columns of the epic's tables. The three `arrow-cast-*.patch` files apply to this branch and switch the downscale to `strength_reduce`, to `strength_reduce` with a 64-bit path chosen on every row, or to Granlund-Montgomery without the 64-bit path. `comet-wide-decimal.patch` applies to apache/datafusion-comet@8d0bb01a5 and adds the `WideDecimalBinaryExpr` variants and two benchmark cases.

## Running the measurements

The microbenchmarks:

```shell
cd divproto
cargo test --release
cargo bench --bench div
```

The arrow-cast downscale, from the repository root. To measure another column, apply a patch first and remove it with `git checkout -- arrow-cast` afterwards:

```shell
cargo bench -p arrow-cast --bench decimal_downscale -- decimal128
git apply divproto/variants/arrow-cast-gm-only.patch
cargo bench -p arrow-cast --bench decimal_downscale -- decimal128
git checkout -- arrow-cast
```

The Comet kernel, in a checkout of apache/datafusion-comet@8d0bb01a5 with the patch applied. `COMET_DIV_VARIANT` selects `current`, `one_knuth`, `gm` or `gm_scan`:

```shell
cd native
COMET_DIV_VARIANT=gm_scan cargo bench -p datafusion-comet-spark-expr --bench wide_decimal -- "fused/multiply"
```

The epic's numbers came from an Apple M5 Max. `divproto` and Comet used rustc 1.98.1. Inside this repository `divproto` picks up the 1.99.0 toolchain from `rust-toolchain.toml`.
