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

//! Prototype divisors for dividing many values by the same divisor.
//!
//! `Gm*` types use the Granlund-Montgomery "round-up" method (Hacker's Delight 2nd ed.,
//! chapter 10; Granlund & Montgomery 1994, figure 4.1): an N-bit magic number `m` and
//! `q = (t + ((n - t) >> sh1)) >> sh2` where `t = mulhi(m, n)`. It is exact for every
//! divisor >= 1 and every N-bit dividend, without a per-divisor branch.

use arrow_buffer::i256;

#[inline(always)]
fn mulhi_u64(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

#[inline(always)]
pub fn mulhi_u128(a: u128, b: u128) -> u128 {
    let (a0, a1) = (a as u64 as u128, a >> 64);
    let (b0, b1) = (b as u64 as u128, b >> 64);
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 64) + (p01 as u64 as u128) + (p10 as u64 as u128);
    p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64)
}

/// Returns `(sh1, sh2)` for divisor bit length `l = ceil(log2(d))`.
fn shifts(l: u32) -> (u32, u32) {
    (l.min(1), l.saturating_sub(1))
}

#[derive(Debug, Clone, Copy)]
pub struct GmU64 {
    d: u64,
    m: u64,
    sh1: u32,
    sh2: u32,
}

impl GmU64 {
    pub fn new(d: u64) -> Option<Self> {
        if d == 0 {
            return None;
        }
        let l = 64 - (d - 1).leading_zeros();
        let hi = ((1u128 << l) - d as u128) as u64;
        let m = ((((hi as u128) << 64) / d as u128) + 1) as u64;
        let (sh1, sh2) = shifts(l);
        Some(Self { d, m, sh1, sh2 })
    }

    #[inline(always)]
    pub fn div(&self, n: u64) -> u64 {
        let t = mulhi_u64(self.m, n);
        (t + ((n - t) >> self.sh1)) >> self.sh2
    }

    #[inline(always)]
    pub fn div_rem(&self, n: u64) -> (u64, u64) {
        let q = self.div(n);
        (q, n - q * self.d)
    }
}

/// DataFusion's `StrengthReducedU64` (datafusion/physical-plan/src/repartition/mod.rs),
/// copied for comparison.
#[derive(Debug, Clone, Copy)]
pub enum LemireU64 {
    PowerOfTwo { mask: u64 },
    Reciprocal { divisor: u64, reciprocal: u128 },
}

impl LemireU64 {
    pub fn new(divisor: u64) -> Self {
        if divisor.is_power_of_two() {
            Self::PowerOfTwo { mask: divisor - 1 }
        } else {
            Self::Reciprocal {
                divisor,
                reciprocal: u128::MAX / u128::from(divisor) + 1,
            }
        }
    }

    #[inline(always)]
    pub fn quotient(value: u64, reciprocal: u128) -> u64 {
        let reciprocal_low = reciprocal as u64;
        let reciprocal_high = (reciprocal >> 64) as u64;
        let low_product = u128::from(value) * u128::from(reciprocal_low);
        let high_product = u128::from(value) * u128::from(reciprocal_high);
        let carry = ((high_product & u128::from(u64::MAX)) + (low_product >> 64)) >> 64;
        ((high_product >> 64) + carry) as u64
    }

    #[inline(always)]
    pub fn remainder(self, value: u64) -> u64 {
        match self {
            Self::PowerOfTwo { mask } => value & mask,
            Self::Reciprocal {
                divisor,
                reciprocal,
            } => value - Self::quotient(value, reciprocal) * divisor,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GmU128 {
    d: u128,
    m: u128,
    sh1: u32,
    sh2: u32,
}

impl GmU128 {
    pub fn new(d: u128) -> Option<Self> {
        if d == 0 {
            return None;
        }
        let l = 128 - (d - 1).leading_zeros();
        // 2^l - d < 2^127, so it is a non-negative high limb of an i256
        let hi = if l == 128 {
            0u128.wrapping_sub(d)
        } else {
            (1u128 << l) - d
        };
        let num = i256::from_parts(0, hi as i128);
        let q = num.wrapping_div(i256::from_parts(d, 0));
        let m = q.to_parts().0.wrapping_add(1);
        let (sh1, sh2) = shifts(l);
        Some(Self { d, m, sh1, sh2 })
    }

    #[inline(always)]
    pub fn div(&self, n: u128) -> u128 {
        let t = mulhi_u128(self.m, n);
        (t + ((n - t) >> self.sh1)) >> self.sh2
    }

    #[inline(always)]
    pub fn div_rem(&self, n: u128) -> (u128, u128) {
        let q = self.div(n);
        (q, n - q * self.d)
    }
}

/// Truncating signed division of `i128` (same results as `wrapping_div` / `wrapping_rem`).
/// With `FAST64`, magnitudes that fit in 64 bits use a hardware 64-bit divide.
#[derive(Debug, Clone, Copy)]
pub struct GmI128<const FAST64: bool> {
    gm: GmU128,
    d64: u64,
    neg: bool,
}

impl<const FAST64: bool> GmI128<FAST64> {
    pub fn new(d: i128) -> Option<Self> {
        let gm = GmU128::new(d.unsigned_abs())?;
        let d64 = u64::try_from(d.unsigned_abs()).unwrap_or(0);
        Some(Self {
            gm,
            d64,
            neg: d < 0,
        })
    }

    #[inline(always)]
    pub fn div_rem(&self, x: i128) -> (i128, i128) {
        let n = x.unsigned_abs();
        let (q, r) = if FAST64 && (n >> 64) == 0 {
            if self.d64 != 0 {
                let (n, d) = (n as u64, self.d64);
                ((n / d) as u128, (n % d) as u128)
            } else {
                (0, n)
            }
        } else {
            self.gm.div_rem(n)
        };
        let q = if x.is_negative() != self.neg {
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
}

/// Little-endian 256-bit unsigned integer.
pub type U256 = [u64; 4];

fn u256_from_i256_magnitude(x: i256) -> U256 {
    let (lo, hi) = x.wrapping_abs().to_parts();
    let hi = hi as u128;
    [lo as u64, (lo >> 64) as u64, hi as u64, (hi >> 64) as u64]
}

fn u256_to_i256(x: U256) -> i256 {
    let lo = x[0] as u128 | (x[1] as u128) << 64;
    let hi = x[2] as u128 | (x[3] as u128) << 64;
    i256::from_parts(lo, hi as i128)
}

#[inline(always)]
fn u256_sub(a: U256, b: U256) -> U256 {
    let mut out = [0; 4];
    let mut borrow = false;
    for i in 0..4 {
        let (v, b1) = a[i].overflowing_sub(b[i]);
        let (v, b2) = v.overflowing_sub(borrow as u64);
        out[i] = v;
        borrow = b1 | b2;
    }
    out
}

#[inline(always)]
fn u256_add(a: U256, b: U256) -> U256 {
    let mut out = [0; 4];
    let mut carry = false;
    for i in 0..4 {
        let (v, c1) = a[i].overflowing_add(b[i]);
        let (v, c2) = v.overflowing_add(carry as u64);
        out[i] = v;
        carry = c1 | c2;
    }
    out
}

#[inline(always)]
fn u256_shr(a: U256, sh: u32) -> U256 {
    let words = (sh / 64) as usize;
    let bits = sh % 64;
    let mut out = [0; 4];
    for i in 0..4 - words {
        let lo = a[i + words] >> bits;
        let hi = if bits != 0 && i + words + 1 < 4 {
            a[i + words + 1] << (64 - bits)
        } else {
            0
        };
        out[i] = lo | hi;
    }
    out
}

/// Low 256 bits of a 256x256 product.
#[inline(always)]
fn u256_mul_lo(a: U256, b: U256) -> U256 {
    let mut out = [0u64; 4];
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 - i {
            let t = a[i] as u128 * b[j] as u128 + out[i + j] as u128 + carry;
            out[i + j] = t as u64;
            carry = t >> 64;
        }
    }
    out
}

/// High 256 bits of a 256x256 product.
#[inline(always)]
pub fn u256_mulhi(a: U256, b: U256) -> U256 {
    let mut p = [0u64; 8];
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let t = a[i] as u128 * b[j] as u128 + p[i + j] as u128 + carry;
            p[i + j] = t as u64;
            carry = t >> 64;
        }
        p[i + 4] = carry as u64;
    }
    [p[4], p[5], p[6], p[7]]
}

fn u256_leading_zeros(a: U256) -> u32 {
    for i in (0..4).rev() {
        if a[i] != 0 {
            return (3 - i as u32) * 64 + a[i].leading_zeros();
        }
    }
    256
}

fn u256_ge(a: U256, b: U256) -> bool {
    for i in (0..4).rev() {
        if a[i] != b[i] {
            return a[i] > b[i];
        }
    }
    true
}

/// floor((hi * 2^256) / d) for hi < d, by restoring division.
fn u256_div_shifted(mut r: U256, d: U256) -> U256 {
    let mut q = [0u64; 4];
    for _ in 0..256 {
        let carry = r[3] >> 63;
        r = [
            r[0] << 1,
            r[1] << 1 | r[0] >> 63,
            r[2] << 1 | r[1] >> 63,
            r[3] << 1 | r[2] >> 63,
        ];
        q = [
            q[0] << 1,
            q[1] << 1 | q[0] >> 63,
            q[2] << 1 | q[1] >> 63,
            q[3] << 1 | q[2] >> 63,
        ];
        if carry != 0 || u256_ge(r, d) {
            r = u256_sub(r, d);
            q[0] |= 1;
        }
    }
    q
}

#[derive(Debug, Clone, Copy)]
pub struct GmU256 {
    d: U256,
    m: U256,
    sh1: u32,
    sh2: u32,
}

impl GmU256 {
    pub fn new(d: U256) -> Option<Self> {
        if d == [0; 4] {
            return None;
        }
        let l = 256 - u256_leading_zeros(u256_sub(d, [1, 0, 0, 0]));
        // 2^l - d, computed mod 2^256
        let pow = if l == 256 {
            [0; 4]
        } else {
            let mut p = [0u64; 4];
            p[(l / 64) as usize] = 1 << (l % 64);
            p
        };
        let hi = u256_sub(pow, d);
        let m = u256_add(u256_div_shifted(hi, d), [1, 0, 0, 0]);
        let (sh1, sh2) = shifts(l);
        Some(Self { d, m, sh1, sh2 })
    }

    #[inline(always)]
    pub fn div_rem(&self, n: U256) -> (U256, U256) {
        let t = u256_mulhi(self.m, n);
        let q = u256_shr(u256_add(t, u256_shr(u256_sub(n, t), self.sh1)), self.sh2);
        (q, u256_sub(n, u256_mul_lo(q, self.d)))
    }
}

/// Truncating signed division of `i256` (same results as `wrapping_div` / `wrapping_rem`).
#[derive(Debug, Clone, Copy)]
pub struct GmI256 {
    gm: GmU256,
    neg: bool,
}

impl GmI256 {
    pub fn new(d: i256) -> Option<Self> {
        let gm = GmU256::new(u256_from_i256_magnitude(d))?;
        Some(Self {
            gm,
            neg: d.is_negative(),
        })
    }

    #[inline(always)]
    pub fn div_rem(&self, x: i256) -> (i256, i256) {
        let (q, r) = self.gm.div_rem(u256_from_i256_magnitude(x));
        let (q, r) = (u256_to_i256(q), u256_to_i256(r));
        let q = if x.is_negative() != self.neg {
            q.wrapping_neg()
        } else {
            q
        };
        let r = if x.is_negative() { r.wrapping_neg() } else { r };
        (q, r)
    }
}

/// Lemire, Kaser and Kurz (2019) for 32-bit dividends: a 64-bit reciprocal
/// `ceil(2^64 / d)`. Powers of two (including 1) use a shift and mask instead.
#[derive(Debug, Clone, Copy)]
pub enum LemireU32 {
    PowerOfTwo { shift: u32, mask: u32 },
    Reciprocal { d: u32, m: u64 },
}

impl LemireU32 {
    pub fn new(d: u32) -> Option<Self> {
        match d {
            0 => None,
            d if d.is_power_of_two() => Some(Self::PowerOfTwo {
                shift: d.trailing_zeros(),
                mask: d - 1,
            }),
            d => Some(Self::Reciprocal {
                d,
                m: u64::MAX / d as u64 + 1,
            }),
        }
    }

    #[inline(always)]
    pub fn rem(self, n: u32) -> u32 {
        match self {
            Self::PowerOfTwo { mask, .. } => n & mask,
            Self::Reciprocal { d, m } => {
                let low = m.wrapping_mul(n as u64);
                ((low as u128 * d as u128) >> 64) as u32
            }
        }
    }

    #[inline(always)]
    pub fn div_rem(self, n: u32) -> (u32, u32) {
        match self {
            Self::PowerOfTwo { shift, mask } => (n >> shift, n & mask),
            Self::Reciprocal { d, m } => {
                let q = ((m as u128 * n as u128) >> 64) as u32;
                (q, n - q * d)
            }
        }
    }
}

/// Lemire's quotient for 64-bit dividends (the `StrengthReducedU64::quotient` method),
/// returning both the quotient and the remainder.
#[derive(Debug, Clone, Copy)]
pub enum LemireDivU64 {
    PowerOfTwo { shift: u32, mask: u64 },
    Reciprocal { d: u64, m: u128 },
}

impl LemireDivU64 {
    pub fn new(d: u64) -> Option<Self> {
        match d {
            0 => None,
            d if d.is_power_of_two() => Some(Self::PowerOfTwo {
                shift: d.trailing_zeros(),
                mask: d - 1,
            }),
            d => Some(Self::Reciprocal {
                d,
                m: u128::MAX / d as u128 + 1,
            }),
        }
    }

    #[inline(always)]
    pub fn div_rem(self, n: u64) -> (u64, u64) {
        match self {
            Self::PowerOfTwo { shift, mask } => (n >> shift, n & mask),
            Self::Reciprocal { d, m } => {
                let q = LemireU64::quotient(n, m);
                (q, n - q * d)
            }
        }
    }
}

/// Spark's `pmod(hash, n)` as Comet computes it today.
#[inline(always)]
pub fn comet_pmod(hash: u32, n: usize) -> usize {
    let hash = hash as i32;
    let n = n as i32;
    let r = hash % n;
    let result = if r < 0 { (r + n) % n } else { r };
    result as usize
}

/// `comet_pmod` without the second division: `|r| < n`, so `r + n` is already in range.
#[inline(always)]
pub fn comet_pmod_one_div(hash: u32, n: usize) -> usize {
    let n = n as i32;
    let r = hash as i32 % n;
    (if r < 0 { r + n } else { r }) as usize
}

/// `comet_pmod` with a precomputed divisor for the positive partition count `n`.
#[inline(always)]
pub fn lemire_pmod(hash: u32, n: u32, d: LemireU32) -> usize {
    let h = hash as i32;
    let r = d.rem(h.unsigned_abs());
    (if h < 0 && r != 0 { n - r } else { r }) as usize
}

/// Truncating `i128` division for a batch where every `|x| < 2^64`, using the hardware
/// 64-bit divide. `d` must fit in 64 bits.
#[inline(always)]
pub fn narrow_hw_div_rem(x: i128, d: u64, neg: bool) -> (i128, i128) {
    let n = x.unsigned_abs() as u64;
    sign_fix(x, neg, (n / d) as u128, (n % d) as u128)
}

/// As `narrow_hw_div_rem`, with Lemire's 64-bit reciprocal.
#[inline(always)]
pub fn narrow_lemire_div_rem(x: i128, d: LemireDivU64, neg: bool) -> (i128, i128) {
    let (q, r) = d.div_rem(x.unsigned_abs() as u64);
    sign_fix(x, neg, q as u128, r as u128)
}

/// As `narrow_hw_div_rem`, with a Granlund-Montgomery 64-bit magic number.
#[inline(always)]
pub fn narrow_gm_div_rem(x: i128, d: GmU64, neg: bool) -> (i128, i128) {
    let (q, r) = d.div_rem(x.unsigned_abs() as u64);
    sign_fix(x, neg, q as u128, r as u128)
}

#[inline(always)]
fn sign_fix(x: i128, neg: bool, q: u128, r: u128) -> (i128, i128) {
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

/// `comet_pmod` without signed arithmetic. With `u = hash ^ 2^31` (the hash as i32 plus 2^31),
/// `pmod(h, n) = (u mod n - 2^31 mod n) mod n`, so one unsigned remainder and one conditional
/// subtract give the result.
#[derive(Debug, Clone, Copy)]
pub struct BiasedPmod {
    n: u32,
    d: LemireU32,
    /// `n - (2^31 mod n)`, added instead of subtracting `2^31 mod n`
    offset: u32,
}

impl BiasedPmod {
    pub fn new(n: u32) -> Option<Self> {
        let d = LemireU32::new(n)?;
        let offset = n - ((1u32 << 31) % n);
        Some(Self { n, d, offset })
    }

    #[inline(always)]
    pub fn pmod(&self, hash: u32) -> u32 {
        let r = self.d.rem(hash ^ (1 << 31)) + self.offset;
        if r >= self.n { r - self.n } else { r }
    }
}

/// Granlund-Montgomery round-up method for 32-bit dividends. Its high multiply is
/// 32x32 to 64 bits, which has vector forms (NEON `umull`, x86 `vpmuludq`), so loops
/// over it can vectorize.
#[derive(Debug, Clone, Copy)]
pub struct GmU32 {
    d: u32,
    m: u32,
    sh1: u32,
    sh2: u32,
}

impl GmU32 {
    pub fn new(d: u32) -> Option<Self> {
        if d == 0 {
            return None;
        }
        let l = 32 - (d - 1).leading_zeros();
        let hi = ((1u64 << l) - d as u64) as u32;
        let m = ((((hi as u64) << 32) / d as u64) + 1) as u32;
        let (sh1, sh2) = shifts(l);
        Some(Self { d, m, sh1, sh2 })
    }

    #[inline(always)]
    pub fn div(&self, n: u32) -> u32 {
        let t = ((self.m as u64 * n as u64) >> 32) as u32;
        (t + ((n - t) >> self.sh1)) >> self.sh2
    }

    #[inline(always)]
    pub fn rem(&self, n: u32) -> u32 {
        n - self.div(n) * self.d
    }
}

/// Truncating `i32` division with the same results as `wrapping_div`, built on [`GmU32`].
#[derive(Debug, Clone, Copy)]
pub struct GmI32 {
    gm: GmU32,
    neg: bool,
}

impl GmI32 {
    pub fn new(d: i32) -> Option<Self> {
        Some(Self {
            gm: GmU32::new(d.unsigned_abs())?,
            neg: d < 0,
        })
    }

    #[inline(always)]
    pub fn div(&self, x: i32) -> i32 {
        let q = self.gm.div(x.unsigned_abs()) as i32;
        if (x < 0) != self.neg {
            q.wrapping_neg()
        } else {
            q
        }
    }
}

/// `comet_pmod` with [`GmU32`], in the biased form of [`BiasedPmod`].
#[derive(Debug, Clone, Copy)]
pub struct BiasedGmPmod {
    n: u32,
    gm: GmU32,
    offset: u32,
}

impl BiasedGmPmod {
    pub fn new(n: u32) -> Option<Self> {
        let gm = GmU32::new(n)?;
        Some(Self {
            n,
            gm,
            offset: n - ((1u32 << 31) % n),
        })
    }

    #[inline(always)]
    pub fn pmod(&self, hash: u32) -> u32 {
        let r = self.gm.rem(hash ^ (1 << 31)) + self.offset;
        if r >= self.n { r - self.n } else { r }
    }
}

/// Out-of-line loops for inspecting the generated code with `cargo asm`.
pub mod asm_probe {
    use super::*;

    #[inline(never)]
    pub fn rem_hardware(xs: &[u32], out: &mut [u32], d: u32) {
        for (o, x) in out.iter_mut().zip(xs) {
            *o = x % d;
        }
    }

    #[inline(never)]
    pub fn rem_lemire(xs: &[u32], out: &mut [u32], d: u32, m: u64) {
        for (o, x) in out.iter_mut().zip(xs) {
            let low = m.wrapping_mul(*x as u64);
            *o = ((low as u128 * d as u128) >> 64) as u32;
        }
    }

    #[inline(never)]
    pub fn rem_gm32(xs: &[u32], out: &mut [u32], d: &GmU32) {
        for (o, x) in out.iter_mut().zip(xs) {
            *o = d.rem(*x);
        }
    }

    #[inline(never)]
    pub fn div_i32_gm32(xs: &[i32], out: &mut [i32], d: &GmI32) {
        for (o, x) in out.iter_mut().zip(xs) {
            *o = d.div(*x);
        }
    }

    #[inline(never)]
    pub fn pmod_gm32(xs: &[u32], out: &mut [u32], p: &BiasedGmPmod) {
        for (o, x) in out.iter_mut().zip(xs) {
            *o = p.pmod(*x);
        }
    }
}

impl LemireU32 {
    /// The reciprocal method for any divisor of at least 2, including powers of two,
    /// for measuring it against the mask.
    pub fn reciprocal(d: u32) -> Option<Self> {
        (d >= 2).then(|| Self::Reciprocal {
            d,
            m: u64::MAX / d as u64 + 1,
        })
    }
}

impl LemireDivU64 {
    /// The reciprocal method for any divisor of at least 2, including powers of two.
    pub fn reciprocal(d: u64) -> Option<Self> {
        (d >= 2).then(|| Self::Reciprocal {
            d,
            m: u128::MAX / d as u128 + 1,
        })
    }
}

/// Truncating `i32` division with the same results as `wrapping_div`, using Lemire's
/// quotient on the magnitude. It holds the reciprocal directly, so the per-row path has
/// no branch on the divisor.
#[derive(Debug, Clone, Copy)]
pub struct LemireI32 {
    m: u64,
    neg: bool,
}

impl LemireI32 {
    /// Returns `None` unless `|d| >= 2`.
    pub fn new(d: i32) -> Option<Self> {
        let abs = d.unsigned_abs();
        (abs >= 2).then(|| Self {
            m: u64::MAX / abs as u64 + 1,
            neg: d < 0,
        })
    }

    #[inline(always)]
    pub fn div(&self, x: i32) -> i32 {
        let q = ((self.m as u128 * x.unsigned_abs() as u128) >> 64) as i32;
        if (x < 0) != self.neg {
            q.wrapping_neg()
        } else {
            q
        }
    }
}

/// `comet_pmod` with Lemire's remainder in the biased form of [`BiasedPmod`]. It holds the
/// reciprocal directly, so the per-row path has no branch on the divisor.
#[derive(Debug, Clone, Copy)]
pub struct BiasedLemirePmod {
    n: u32,
    m: u64,
    offset: u32,
}

impl BiasedLemirePmod {
    /// Returns `None` unless `n >= 2`.
    pub fn new(n: u32) -> Option<Self> {
        (n >= 2).then(|| Self {
            n,
            m: u64::MAX / n as u64 + 1,
            offset: n - ((1u32 << 31) % n),
        })
    }

    #[inline(always)]
    pub fn pmod(&self, hash: u32) -> u32 {
        let low = self.m.wrapping_mul((hash ^ (1 << 31)) as u64);
        let r = ((low as u128 * self.n as u128) >> 64) as u32 + self.offset;
        if r >= self.n { r - self.n } else { r }
    }
}
