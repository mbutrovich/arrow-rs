// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::hint::black_box;

use arrow_buffer::i256;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use divproto::*;
use rand::{Rng, SeedableRng, rngs::StdRng};
use strength_reduce::{StrengthReducedU64, StrengthReducedU128};

const LEN: usize = 8192;

fn random_digits(rng: &mut StdRng, digits: u32) -> i128 {
    let mut v: i128 = rng.gen_range(1..10);
    for _ in 1..digits {
        v = v * 10 + rng.gen_range(0..10);
    }
    if rng.r#gen() { -v } else { v }
}

fn u64_rem(c: &mut Criterion) {
    let mut g = c.benchmark_group("u64_rem");
    g.throughput(Throughput::Elements(LEN as u64));
    let mut rng = StdRng::seed_from_u64(7);
    let hashes: Vec<u64> = (0..LEN).map(|_| rng.r#gen()).collect();
    let mut out = vec![0u64; LEN];
    for n in [13u64, 16, 200] {
        let d = black_box(n);
        g.bench_function(BenchmarkId::new("hardware", n), |b| {
            b.iter(|| {
                for (o, h) in out.iter_mut().zip(&hashes) {
                    *o = h % d;
                }
                black_box(&out);
            })
        });
        let lemire = LemireU64::new(d);
        g.bench_function(BenchmarkId::new("lemire_datafusion", n), |b| {
            b.iter(|| {
                // Dispatch once per batch, as DataFusion's partition_indices does
                match lemire {
                    LemireU64::PowerOfTwo { mask } => {
                        for (o, h) in out.iter_mut().zip(&hashes) {
                            *o = h & mask;
                        }
                    }
                    LemireU64::Reciprocal {
                        divisor,
                        reciprocal,
                    } => {
                        for (o, h) in out.iter_mut().zip(&hashes) {
                            *o = h - LemireU64::quotient(*h, reciprocal) * divisor;
                        }
                    }
                }
                black_box(&out);
            })
        });
        let gm = GmU64::new(d).unwrap();
        g.bench_function(BenchmarkId::new("gm", n), |b| {
            b.iter(|| {
                for (o, h) in out.iter_mut().zip(&hashes) {
                    *o = gm.div_rem(*h).1;
                }
                black_box(&out);
            })
        });
        let sr = StrengthReducedU64::new(d);
        g.bench_function(BenchmarkId::new("strength_reduce", n), |b| {
            b.iter(|| {
                for (o, h) in out.iter_mut().zip(&hashes) {
                    *o = *h % sr;
                }
                black_box(&out);
            })
        });
    }
    g.finish();
}

fn sr_i128_div_rem(sr: StrengthReducedU128, neg: bool, x: i128) -> (i128, i128) {
    let (q, r) = StrengthReducedU128::div_rem(x.unsigned_abs(), sr);
    let q = if x.is_negative() != neg {
        (q as i128).wrapping_neg()
    } else {
        q as i128
    };
    let r = if x.is_negative() {
        (r as i128).wrapping_neg()
    } else {
        r as i128
    };
    (q, r)
}

fn i128_div_rem(c: &mut Criterion) {
    let mut g = c.benchmark_group("i128_div_rem");
    g.throughput(Throughput::Elements(LEN as u64));
    let mut rng = StdRng::seed_from_u64(8);
    let mut out = vec![(0i128, 0i128); LEN];
    for digits in [18u32, 38] {
        let xs: Vec<i128> = (0..LEN).map(|_| random_digits(&mut rng, digits)).collect();
        for k in [4u32, 18, 20, 30] {
            let id = format!("{digits}d/10^{k}");
            let d = black_box(10i128.pow(k));
            g.bench_function(BenchmarkId::new("hardware", &id), |b| {
                b.iter(|| {
                    for (o, x) in out.iter_mut().zip(&xs) {
                        *o = (x / d, x % d);
                    }
                    black_box(&out);
                })
            });
            let sr = StrengthReducedU128::new(d.unsigned_abs());
            g.bench_function(BenchmarkId::new("strength_reduce", &id), |b| {
                b.iter(|| {
                    for (o, x) in out.iter_mut().zip(&xs) {
                        *o = sr_i128_div_rem(sr, false, *x);
                    }
                    black_box(&out);
                })
            });
            let gm = GmI128::<false>::new(d).unwrap();
            g.bench_function(BenchmarkId::new("gm", &id), |b| {
                b.iter(|| {
                    for (o, x) in out.iter_mut().zip(&xs) {
                        *o = gm.div_rem(*x);
                    }
                    black_box(&out);
                })
            });
            let gm = GmI128::<true>::new(d).unwrap();
            g.bench_function(BenchmarkId::new("gm_fast64", &id), |b| {
                b.iter(|| {
                    for (o, x) in out.iter_mut().zip(&xs) {
                        *o = gm.div_rem(*x);
                    }
                    black_box(&out);
                })
            });
        }
    }
    g.finish();
}

fn i256_div_rem(c: &mut Criterion) {
    let mut g = c.benchmark_group("i256_div_rem");
    g.throughput(Throughput::Elements(LEN as u64));
    let mut rng = StdRng::seed_from_u64(9);
    let mut out = vec![(i256::ZERO, i256::ZERO); LEN];
    // Products of two decimals, as in Comet's WideDecimalBinaryExpr multiply
    for digits in [15u32, 19, 38] {
        let xs: Vec<i256> = (0..LEN)
            .map(|_| {
                i256::from_i128(random_digits(&mut rng, digits))
                    .wrapping_mul(i256::from_i128(random_digits(&mut rng, digits)))
            })
            .collect();
        for k in [6u32, 14, 20, 38] {
            let id = format!("{digits}dx{digits}d/10^{k}");
            let d = black_box(i256::from_i128(10).wrapping_pow(k));
            g.bench_function(BenchmarkId::new("arrow_div_and_rem", &id), |b| {
                b.iter(|| {
                    for (o, x) in out.iter_mut().zip(&xs) {
                        *o = (*x / d, *x % d);
                    }
                    black_box(&out);
                })
            });
            g.bench_function(BenchmarkId::new("arrow_div_only", &id), |b| {
                b.iter(|| {
                    for (o, x) in out.iter_mut().zip(&xs) {
                        *o = (*x / d, i256::ZERO);
                    }
                    black_box(&out);
                })
            });
            let gm = GmI256::new(d).unwrap();
            g.bench_function(BenchmarkId::new("gm", &id), |b| {
                b.iter(|| {
                    for (o, x) in out.iter_mut().zip(&xs) {
                        *o = gm.div_rem(*x);
                    }
                    black_box(&out);
                })
            });
        }
    }
    g.finish();
}

fn construct(c: &mut Criterion) {
    let mut g = c.benchmark_group("construct");
    g.bench_function("gm_u64(200)", |b| b.iter(|| GmU64::new(black_box(200))));
    g.bench_function("lemire_u64(200)", |b| {
        b.iter(|| LemireU64::new(black_box(200)))
    });
    g.bench_function("strength_reduce_u64(200)", |b| {
        b.iter(|| StrengthReducedU64::new(black_box(200)))
    });
    g.bench_function("gm_i128(10^18)", |b| {
        b.iter(|| GmI128::<true>::new(black_box(10i128.pow(18))))
    });
    g.bench_function("strength_reduce_u128(10^18)", |b| {
        b.iter(|| StrengthReducedU128::new(black_box(10u128.pow(18))))
    });
    let d = i256::from_i128(10).wrapping_pow(14);
    g.bench_function("gm_i256(10^14)", |b| b.iter(|| GmI256::new(black_box(d))));
    g.finish();
}

fn u32_pmod(c: &mut Criterion) {
    let mut g = c.benchmark_group("u32_pmod");
    g.throughput(Throughput::Elements(LEN as u64));
    let mut rng = StdRng::seed_from_u64(10);
    let hashes: Vec<u32> = (0..LEN).map(|_| rng.r#gen()).collect();
    let mut out = vec![0usize; LEN];
    for n in [13u32, 16, 200] {
        let nn = black_box(n as usize);
        g.bench_function(BenchmarkId::new("comet_today", n), |b| {
            b.iter(|| {
                for (o, h) in out.iter_mut().zip(&hashes) {
                    *o = comet_pmod(*h, nn);
                }
                black_box(&out);
            })
        });
        g.bench_function(BenchmarkId::new("one_div", n), |b| {
            b.iter(|| {
                for (o, h) in out.iter_mut().zip(&hashes) {
                    *o = comet_pmod_one_div(*h, nn);
                }
                black_box(&out);
            })
        });
        let n = black_box(n);
        let d = LemireU32::new(n).unwrap();
        g.bench_function(BenchmarkId::new("lemire", n), |b| {
            b.iter(|| {
                // Dispatch once per batch on the power-of-two case
                match d {
                    LemireU32::PowerOfTwo { mask, .. } => {
                        for (o, h) in out.iter_mut().zip(&hashes) {
                            let h = *h as i32;
                            let r = h.unsigned_abs() & mask;
                            *o = (if h < 0 && r != 0 { n - r } else { r }) as usize;
                        }
                    }
                    LemireU32::Reciprocal { d, m } => {
                        for (o, h) in out.iter_mut().zip(&hashes) {
                            let h = *h as i32;
                            let low = m.wrapping_mul(h.unsigned_abs() as u64);
                            let r = ((low as u128 * d as u128) >> 64) as u32;
                            *o = (if h < 0 && r != 0 { n - r } else { r }) as usize;
                        }
                    }
                }
                black_box(&out);
            })
        });
        let biased = BiasedPmod::new(n).unwrap();
        g.bench_function(BenchmarkId::new("lemire_biased", n), |b| {
            b.iter(|| {
                if n.is_power_of_two() {
                    // Euclidean remainder by a power of two is a mask of the two's complement bits
                    let mask = n - 1;
                    for (o, h) in out.iter_mut().zip(&hashes) {
                        *o = (h & mask) as usize;
                    }
                } else {
                    for (o, h) in out.iter_mut().zip(&hashes) {
                        *o = biased.pmod(*h) as usize;
                    }
                }
                black_box(&out);
            })
        });
    }
    g.finish();
}

fn u32_rem(c: &mut Criterion) {
    let mut g = c.benchmark_group("u32_rem");
    g.throughput(Throughput::Elements(LEN as u64));
    let mut rng = StdRng::seed_from_u64(11);
    let xs: Vec<u32> = (0..LEN).map(|_| rng.r#gen()).collect();
    let mut out = vec![0u32; LEN];
    for n in [13u32, 16, 200] {
        let d = black_box(n);
        g.bench_function(BenchmarkId::new("hardware", n), |b| {
            b.iter(|| {
                for (o, x) in out.iter_mut().zip(&xs) {
                    *o = x % d;
                }
                black_box(&out);
            })
        });
        let l = LemireU32::new(d).unwrap();
        g.bench_function(BenchmarkId::new("lemire", n), |b| {
            b.iter(|| {
                match l {
                    LemireU32::PowerOfTwo { mask, .. } => {
                        for (o, x) in out.iter_mut().zip(&xs) {
                            *o = x & mask;
                        }
                    }
                    LemireU32::Reciprocal { .. } => {
                        for (o, x) in out.iter_mut().zip(&xs) {
                            *o = l.rem(*x);
                        }
                    }
                }
                black_box(&out);
            })
        });
    }
    g.finish();
}

fn i128_narrow(c: &mut Criterion) {
    let mut g = c.benchmark_group("i128_narrow");
    g.throughput(Throughput::Elements(LEN as u64));
    let mut rng = StdRng::seed_from_u64(12);
    let mut out = vec![(0i128, 0i128); LEN];
    let xs: Vec<i128> = (0..LEN).map(|_| random_digits(&mut rng, 18)).collect();
    for k in [4u32, 9, 18] {
        let id = format!("18d/10^{k}");
        let d = black_box(10i128.pow(k));
        g.bench_function(BenchmarkId::new("hardware_i128", &id), |b| {
            b.iter(|| {
                for (o, x) in out.iter_mut().zip(&xs) {
                    *o = (x / d, x % d);
                }
                black_box(&out);
            })
        });
        let per_row = GmI128::<true>::new(d).unwrap();
        g.bench_function(BenchmarkId::new("per_row_fast64", &id), |b| {
            b.iter(|| {
                for (o, x) in out.iter_mut().zip(&xs) {
                    *o = per_row.div_rem(*x);
                }
                black_box(&out);
            })
        });
        let d64 = d as u64;
        g.bench_function(BenchmarkId::new("batch_hw_u64", &id), |b| {
            b.iter(|| {
                for (o, x) in out.iter_mut().zip(&xs) {
                    *o = narrow_hw_div_rem(*x, d64, false);
                }
                black_box(&out);
            })
        });
        let l = LemireDivU64::new(d64).unwrap();
        g.bench_function(BenchmarkId::new("batch_lemire_u64", &id), |b| {
            b.iter(|| {
                match l {
                    LemireDivU64::PowerOfTwo { .. } | LemireDivU64::Reciprocal { .. } => {
                        for (o, x) in out.iter_mut().zip(&xs) {
                            *o = narrow_lemire_div_rem(*x, l, false);
                        }
                    }
                }
                black_box(&out);
            })
        });
        let gm = GmU64::new(d64).unwrap();
        g.bench_function(BenchmarkId::new("batch_gm_u64", &id), |b| {
            b.iter(|| {
                for (o, x) in out.iter_mut().zip(&xs) {
                    *o = narrow_gm_div_rem(*x, gm, false);
                }
                black_box(&out);
            })
        });
    }
    g.finish();
}

fn u32_vector(c: &mut Criterion) {
    let mut g = c.benchmark_group("u32_vector");
    g.throughput(Throughput::Elements(LEN as u64));
    let mut rng = StdRng::seed_from_u64(13);
    let xs: Vec<u32> = (0..LEN).map(|_| rng.r#gen()).collect();
    let signed: Vec<i32> = xs.iter().map(|x| *x as i32).collect();
    let mut out = vec![0u32; LEN];
    let mut out_i = vec![0i32; LEN];
    let mut out_us = vec![0usize; LEN];
    for n in [13u32, 200, 1000] {
        let d = black_box(n);
        g.bench_function(BenchmarkId::new("rem_hardware", n), |b| {
            b.iter(|| {
                for (o, x) in out.iter_mut().zip(&xs) {
                    *o = x % d;
                }
                black_box(&out);
            })
        });
        let m = u64::MAX / d as u64 + 1;
        g.bench_function(BenchmarkId::new("rem_lemire", n), |b| {
            b.iter(|| {
                for (o, x) in out.iter_mut().zip(&xs) {
                    let low = m.wrapping_mul(*x as u64);
                    *o = ((low as u128 * d as u128) >> 64) as u32;
                }
                black_box(&out);
            })
        });
        let gm = GmU32::new(d).unwrap();
        g.bench_function(BenchmarkId::new("rem_gm32", n), |b| {
            b.iter(|| {
                for (o, x) in out.iter_mut().zip(&xs) {
                    *o = gm.rem(*x);
                }
                black_box(&out);
            })
        });
        let sd = black_box(n as i32);
        g.bench_function(BenchmarkId::new("div_i32_hardware", n), |b| {
            b.iter(|| {
                for (o, x) in out_i.iter_mut().zip(&signed) {
                    *o = x.wrapping_div(sd);
                }
                black_box(&out_i);
            })
        });
        let gmi = GmI32::new(sd).unwrap();
        g.bench_function(BenchmarkId::new("div_i32_gm32", n), |b| {
            b.iter(|| {
                for (o, x) in out_i.iter_mut().zip(&signed) {
                    *o = gmi.div(*x);
                }
                black_box(&out_i);
            })
        });
        let nn = black_box(n as usize);
        g.bench_function(BenchmarkId::new("pmod_one_div", n), |b| {
            b.iter(|| {
                for (o, x) in out_us.iter_mut().zip(&xs) {
                    *o = comet_pmod_one_div(*x, nn);
                }
                black_box(&out_us);
            })
        });
        let bl = BiasedPmod::new(d).unwrap();
        g.bench_function(BenchmarkId::new("pmod_biased_lemire", n), |b| {
            b.iter(|| {
                for (o, x) in out.iter_mut().zip(&xs) {
                    *o = bl.pmod(*x);
                }
                black_box(&out);
            })
        });
        let bg = BiasedGmPmod::new(d).unwrap();
        g.bench_function(BenchmarkId::new("pmod_biased_gm32", n), |b| {
            b.iter(|| {
                for (o, x) in out.iter_mut().zip(&xs) {
                    *o = bg.pmod(*x);
                }
                black_box(&out);
            })
        });
    }
    g.finish();
}

fn batches(c: &mut Criterion) {
    use arrow_arith::numeric::{div, rem};
    use arrow_array::types::{Decimal128Type, Int32Type, UInt32Type, UInt64Type};
    use arrow_array::{Decimal128Array, Int32Array, Scalar, UInt32Array, UInt64Array};

    let mut g = c.benchmark_group("batches");
    let mut rng = StdRng::seed_from_u64(14);
    let u32s = UInt32Array::from_iter_values((0..LEN).map(|_| rng.r#gen::<u32>()));
    let i32s = Int32Array::from_iter_values((0..LEN).map(|_| rng.r#gen::<i32>()));
    let u64s = UInt64Array::from_iter_values((0..LEN).map(|_| rng.r#gen::<u64>()));

    for n in [13u32, 16, 200] {
        let scalar = Scalar::new(UInt32Array::from(vec![n]));
        g.bench_function(BenchmarkId::new("u32_rem/arrow_rem", n), |b| {
            b.iter(|| black_box(rem(&u32s, &scalar).unwrap()))
        });
        if n.is_power_of_two() {
            let mask = n - 1;
            g.bench_function(BenchmarkId::new("u32_rem/mask", n), |b| {
                b.iter(|| black_box(u32s.unary::<_, UInt32Type>(|x| x & mask)))
            });
        } else {
            let d = black_box(n);
            let m = u64::MAX / d as u64 + 1;
            g.bench_function(BenchmarkId::new("u32_rem/lemire", n), |b| {
                b.iter(|| {
                    black_box(u32s.unary::<_, UInt32Type>(|x| {
                        let low = m.wrapping_mul(x as u64);
                        ((low as u128 * d as u128) >> 64) as u32
                    }))
                })
            });
            let gm = GmU32::new(d).unwrap();
            g.bench_function(BenchmarkId::new("u32_rem/gm32", n), |b| {
                b.iter(|| black_box(u32s.unary::<_, UInt32Type>(|x| gm.rem(x))))
            });
        }

        let scalar = Scalar::new(Int32Array::from(vec![n as i32]));
        g.bench_function(BenchmarkId::new("i32_div/arrow_div", n), |b| {
            b.iter(|| black_box(div(&i32s, &scalar).unwrap()))
        });
        let gmi = GmI32::new(black_box(n as i32)).unwrap();
        g.bench_function(BenchmarkId::new("i32_div/gm32", n), |b| {
            b.iter(|| black_box(i32s.unary::<_, Int32Type>(|x| gmi.div(x))))
        });

        let scalar = Scalar::new(UInt64Array::from(vec![n as u64]));
        g.bench_function(BenchmarkId::new("u64_rem/arrow_rem", n), |b| {
            b.iter(|| black_box(rem(&u64s, &scalar).unwrap()))
        });
        match LemireU64::new(black_box(n as u64)) {
            LemireU64::PowerOfTwo { mask } => {
                g.bench_function(BenchmarkId::new("u64_rem/mask", n), |b| {
                    b.iter(|| black_box(u64s.unary::<_, UInt64Type>(|x| x & mask)))
                });
            }
            LemireU64::Reciprocal {
                divisor,
                reciprocal,
            } => {
                g.bench_function(BenchmarkId::new("u64_rem/lemire", n), |b| {
                    b.iter(|| {
                        black_box(u64s.unary::<_, UInt64Type>(|x| {
                            x - LemireU64::quotient(x, reciprocal) * divisor
                        }))
                    })
                });
                let gm = GmU64::new(divisor).unwrap();
                g.bench_function(BenchmarkId::new("u64_rem/gm64", n), |b| {
                    b.iter(|| black_box(u64s.unary::<_, UInt64Type>(|x| gm.div_rem(x).1)))
                });
            }
        }

        let nn = black_box(n);
        g.bench_function(BenchmarkId::new("pmod/comet_today", n), |b| {
            b.iter(|| black_box(u32s.unary::<_, UInt32Type>(|h| comet_pmod(h, nn as usize) as u32)))
        });
        g.bench_function(BenchmarkId::new("pmod/one_div", n), |b| {
            b.iter(|| {
                black_box(
                    u32s.unary::<_, UInt32Type>(|h| comet_pmod_one_div(h, nn as usize) as u32),
                )
            })
        });
        if n.is_power_of_two() {
            let mask = n - 1;
            g.bench_function(BenchmarkId::new("pmod/mask", n), |b| {
                b.iter(|| black_box(u32s.unary::<_, UInt32Type>(|h| h & mask)))
            });
        } else {
            let bl = BiasedPmod::new(nn).unwrap();
            g.bench_function(BenchmarkId::new("pmod/biased_lemire", n), |b| {
                b.iter(|| black_box(u32s.unary::<_, UInt32Type>(|h| bl.pmod(h))))
            });
            let bg = BiasedGmPmod::new(nn).unwrap();
            g.bench_function(BenchmarkId::new("pmod/biased_gm32", n), |b| {
                b.iter(|| black_box(u32s.unary::<_, UInt32Type>(|h| bg.pmod(h))))
            });
        }
    }

    // Half-up rounded quotient, as in the decimal downscale
    for digits in [18u32, 38] {
        let values: Decimal128Array = (0..LEN)
            .map(|_| Some(random_digits(&mut rng, digits)))
            .collect();
        let values = values.with_precision_and_scale(38, 0).unwrap();
        for k in [4u32, 18] {
            let id = format!("{digits}d/10^{k}");
            let d = black_box(10i128.pow(k));
            let half = d / 2;
            let round = move |(q, r): (i128, i128)| {
                if r >= half {
                    q + 1
                } else if r <= -half {
                    q - 1
                } else {
                    q
                }
            };
            g.bench_function(BenchmarkId::new("i128_round/hardware_i128", &id), |b| {
                b.iter(|| black_box(values.unary::<_, Decimal128Type>(|x| round((x / d, x % d)))))
            });
            let gm = GmI128::<false>::new(d).unwrap();
            g.bench_function(BenchmarkId::new("i128_round/gm128", &id), |b| {
                b.iter(|| black_box(values.unary::<_, Decimal128Type>(|x| round(gm.div_rem(x)))))
            });
            if digits <= 18 {
                let d64 = d as u64;
                g.bench_function(BenchmarkId::new("i128_round/narrow_hw_u64", &id), |b| {
                    b.iter(|| {
                        black_box(values.unary::<_, Decimal128Type>(|x| {
                            round(narrow_hw_div_rem(x, d64, false))
                        }))
                    })
                });
                let gm64 = GmU64::new(d64).unwrap();
                g.bench_function(BenchmarkId::new("i128_round/narrow_gm64", &id), |b| {
                    b.iter(|| {
                        black_box(values.unary::<_, Decimal128Type>(|x| {
                            round(narrow_gm_div_rem(x, gm64, false))
                        }))
                    })
                });
                let l = LemireDivU64::new(d64).unwrap();
                g.bench_function(BenchmarkId::new("i128_round/narrow_lemire", &id), |b| {
                    b.iter(|| {
                        black_box(values.unary::<_, Decimal128Type>(|x| {
                            round(narrow_lemire_div_rem(x, l, false))
                        }))
                    })
                });
            }
        }
    }
    g.finish();
}

criterion_group!(benches, u64_rem, i128_div_rem, i256_div_rem, construct);
criterion_group!(more, u32_pmod, u32_rem, i128_narrow, u32_vector);
criterion_group!(arrow_batches, batches);
criterion_main!(benches, more, arrow_batches);
