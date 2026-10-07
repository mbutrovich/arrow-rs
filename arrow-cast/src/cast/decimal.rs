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

use crate::cast::*;
use crate::parse::{DecimalParseError, parse_decimal_checked};

/// A utility trait that provides checked conversions between
/// decimal types inspired by [`NumCast`]
pub trait DecimalCast: Sized {
    /// Convert the decimal to an i32
    fn to_i32(self) -> Option<i32>;

    /// Convert the decimal to an i64
    fn to_i64(self) -> Option<i64>;

    /// Convert the decimal to an i128
    fn to_i128(self) -> Option<i128>;

    /// Convert the decimal to an i256
    fn to_i256(self) -> Option<i256>;

    /// Convert a decimal from a decimal
    fn from_decimal<T: DecimalCast>(n: T) -> Option<Self>;

    /// Convert a decimal from a f64
    fn from_f64(n: f64) -> Option<Self>;

    /// Returns a function that computes the truncated quotient and remainder of its argument
    /// divided by `divisor`, with the same results as `div_wrapping` and `mod_wrapping`.
    ///
    /// Use it to divide many values by the same divisor: implementations may precompute
    /// a form of `divisor` that is cheaper to divide by than a hardware or software division.
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is zero
    ///
    /// # Example
    ///
    /// ```
    /// # use arrow_cast::DecimalCast;
    /// let div_rem = i128::div_rem_by(1_000);
    /// assert_eq!(div_rem(-12_345), (-12, -345));
    /// ```
    fn div_rem_by(divisor: Self) -> impl Fn(Self) -> (Self, Self) + Copy
    where
        Self: ArrowNativeTypeOp,
    {
        assert!(!divisor.is_zero(), "division by zero");
        move |x| (x.div_wrapping(divisor), x.mod_wrapping(divisor))
    }

    /// As [`Self::div_rem_by`], for callers that have checked with [`Self::all_narrow`]
    /// that every dividend fits in 64 bits.
    fn div_rem_narrow_by(divisor: Self) -> impl Fn(Self) -> (Self, Self) + Copy
    where
        Self: ArrowNativeTypeOp,
    {
        Self::div_rem_by(divisor)
    }

    /// Returns true if every value in `values` has a magnitude below 2^64 and
    /// [`Self::div_rem_narrow_by`] is cheaper than [`Self::div_rem_by`] for it.
    fn all_narrow(_values: &[Self]) -> bool {
        false
    }
}

/// High 128 bits of a 128x128-bit product.
#[inline(always)]
fn mulhi_u128(a: u128, b: u128) -> u128 {
    let (a0, a1) = (a as u64 as u128, a >> 64);
    let (b0, b1) = (b as u64 as u128, b >> 64);
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let mid = ((a0 * b0) >> 64) + (p01 as u64 as u128) + (p10 as u64 as u128);
    a1 * b1 + (p01 >> 64) + (p10 >> 64) + (mid >> 64)
}

/// Unsigned division by an invariant divisor with the Granlund-Montgomery round-up method
/// (Hacker's Delight 2nd ed., chapter 10): one 128-bit magic number, no per-divisor branch.
#[derive(Clone, Copy)]
struct GmU128 {
    d: u128,
    m: u128,
    sh1: u32,
    sh2: u32,
}

impl GmU128 {
    fn new(d: u128) -> Self {
        assert_ne!(d, 0, "division by zero");
        let l = 128 - (d - 1).leading_zeros();
        // 2^l - d < 2^127, so it is a non-negative high limb of an i256
        let hi = if l == 128 {
            0u128.wrapping_sub(d)
        } else {
            (1u128 << l) - d
        };
        let q = i256::from_parts(0, hi as i128).wrapping_div(i256::from_parts(d, 0));
        Self {
            d,
            m: q.to_parts().0.wrapping_add(1),
            sh1: l.min(1),
            sh2: l.saturating_sub(1),
        }
    }

    #[inline(always)]
    fn div_rem(&self, n: u128) -> (u128, u128) {
        let t = mulhi_u128(self.m, n);
        let q = (t + ((n - t) >> self.sh1)) >> self.sh2;
        (q, n - q * self.d)
    }
}

/// [`GmU128`] with a 64-bit magic number, for dividends that fit in 64 bits.
#[derive(Clone, Copy)]
struct GmU64 {
    d: u64,
    m: u64,
    sh1: u32,
    sh2: u32,
}

impl GmU64 {
    fn new(d: u64) -> Self {
        assert_ne!(d, 0, "division by zero");
        let l = 64 - (d - 1).leading_zeros();
        let hi = ((1u128 << l) - d as u128) as u64;
        Self {
            d,
            m: ((((hi as u128) << 64) / d as u128) + 1) as u64,
            sh1: l.min(1),
            sh2: l.saturating_sub(1),
        }
    }

    #[inline(always)]
    fn div_rem(&self, n: u64) -> (u64, u64) {
        let t = ((self.m as u128 * n as u128) >> 64) as u64;
        let q = (t + ((n - t) >> self.sh1)) >> self.sh2;
        (q, n - q * self.d)
    }
}

/// Applies truncating-division signs to the quotient and remainder of `|x| / |divisor|`.
/// `as` wraps `i128::MIN / -1` like `div_wrapping` does.
#[inline(always)]
fn signed_div_rem(x: i128, divisor: i128, (q, r): (u128, u128)) -> (i128, i128) {
    let q = if x.is_negative() != divisor.is_negative() {
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

impl DecimalCast for i32 {
    fn to_i32(self) -> Option<i32> {
        Some(self)
    }

    fn to_i64(self) -> Option<i64> {
        Some(self as i64)
    }

    fn to_i128(self) -> Option<i128> {
        Some(self as i128)
    }

    fn to_i256(self) -> Option<i256> {
        Some(i256::from_i128(self as i128))
    }

    fn from_decimal<T: DecimalCast>(n: T) -> Option<Self> {
        n.to_i32()
    }

    fn from_f64(n: f64) -> Option<Self> {
        n.to_i32()
    }
}

impl DecimalCast for i64 {
    fn to_i32(self) -> Option<i32> {
        i32::try_from(self).ok()
    }

    fn to_i64(self) -> Option<i64> {
        Some(self)
    }

    fn to_i128(self) -> Option<i128> {
        Some(self as i128)
    }

    fn to_i256(self) -> Option<i256> {
        Some(i256::from_i128(self as i128))
    }

    fn from_decimal<T: DecimalCast>(n: T) -> Option<Self> {
        n.to_i64()
    }

    fn from_f64(n: f64) -> Option<Self> {
        // Call implementation explicitly otherwise this resolves to `to_i64`
        // in arrow-buffer that behaves differently.
        num_traits::ToPrimitive::to_i64(&n)
    }
}

impl DecimalCast for i128 {
    fn to_i32(self) -> Option<i32> {
        i32::try_from(self).ok()
    }

    fn to_i64(self) -> Option<i64> {
        i64::try_from(self).ok()
    }

    fn to_i128(self) -> Option<i128> {
        Some(self)
    }

    fn to_i256(self) -> Option<i256> {
        Some(i256::from_i128(self))
    }

    fn from_decimal<T: DecimalCast>(n: T) -> Option<Self> {
        n.to_i128()
    }

    fn from_f64(n: f64) -> Option<Self> {
        n.to_i128()
    }

    fn div_rem_by(divisor: Self) -> impl Fn(Self) -> (Self, Self) + Copy {
        let gm = GmU128::new(divisor.unsigned_abs());
        move |x| signed_div_rem(x, divisor, gm.div_rem(x.unsigned_abs()))
    }

    fn div_rem_narrow_by(divisor: Self) -> impl Fn(Self) -> (Self, Self) + Copy {
        assert_ne!(divisor, 0, "division by zero");
        let gm = u64::try_from(divisor.unsigned_abs()).ok().map(GmU64::new);
        move |x| {
            debug_assert_eq!(x.unsigned_abs() >> 64, 0);
            let n = x.unsigned_abs() as u64;
            let qr = match gm {
                Some(gm) => {
                    let (q, r) = gm.div_rem(n);
                    (q as u128, r as u128)
                }
                // The divisor exceeds every 64-bit dividend
                None => (0, n as u128),
            };
            signed_div_rem(x, divisor, qr)
        }
    }

    fn all_narrow(values: &[Self]) -> bool {
        values
            .iter()
            .fold(0, |acc, x| acc | (x.unsigned_abs() >> 64))
            == 0
    }
}

impl DecimalCast for i256 {
    fn to_i32(self) -> Option<i32> {
        self.to_i128().map(|x| i32::try_from(x).ok())?
    }

    fn to_i64(self) -> Option<i64> {
        self.to_i128().map(|x| i64::try_from(x).ok())?
    }

    fn to_i128(self) -> Option<i128> {
        self.to_i128()
    }

    fn to_i256(self) -> Option<i256> {
        Some(self)
    }

    fn from_decimal<T: DecimalCast>(n: T) -> Option<Self> {
        n.to_i256()
    }

    fn from_f64(n: f64) -> Option<Self> {
        i256::from_f64(n)
    }
}

/// Construct closures to upscale decimals from `(input_precision, input_scale)` to
/// `(output_precision, output_scale)`.
///
/// Returns `(f_fallible, f_infallible)` where:
/// * `f_fallible` yields `None` when the requested cast would overflow
/// * `f_infallible` is present only when every input is guaranteed to succeed; otherwise it is `None`
///   and callers must fall back to `f_fallible`
///
/// Returns `None` if the required scale increase `delta_scale = output_scale - input_scale`
/// exceeds the supported precomputed precision table `O::MAX_FOR_EACH_PRECISION`.
/// In that case, the caller should treat this as an overflow for the output scale
/// and handle it accordingly (e.g., return a cast error).
#[expect(clippy::type_complexity)]
fn make_upscaler<I: DecimalType, O: DecimalType>(
    input_precision: u8,
    input_scale: i8,
    output_precision: u8,
    output_scale: i8,
) -> Option<(
    impl Fn(I::Native) -> Option<O::Native>,
    Option<impl Fn(I::Native) -> O::Native>,
)>
where
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
{
    let delta_scale = output_scale as i16 - input_scale as i16;

    // O::MAX_FOR_EACH_PRECISION[k] stores 10^k - 1 (e.g., 9, 99, 999, ...).
    // Adding 1 yields exactly 10^k without computing a power at runtime.
    // Using the precomputed table avoids pow(10, k) and its checked/overflow
    // handling, which is faster and simpler for scaling by 10^delta_scale.
    let max = O::MAX_FOR_EACH_PRECISION.get(delta_scale as usize)?;
    let mul = max.add_wrapping(O::Native::ONE);
    let f_fallible = move |x| O::Native::from_decimal(x)?.mul_checked(mul).ok();

    // if the gain in precision (digits) is greater than the multiplication due to scaling
    // every number will fit into the output type
    // Example: If we are starting with any number of precision 5 [xxxxx],
    // then an increase of scale by 3 will have the following effect on the representation:
    // [xxxxx] -> [xxxxx000], so for the cast to be infallible, the output type
    // needs to provide at least 8 digits precision
    let is_infallible_cast = (input_precision as i16) + delta_scale <= (output_precision as i16);
    let f_infallible = is_infallible_cast
        .then_some(move |x| O::Native::from_decimal(x).unwrap().mul_wrapping(mul));
    Some((f_fallible, f_infallible))
}

/// Construct closures to downscale decimals from `(input_precision, input_scale)` to
/// `(output_precision, output_scale)`.
///
/// Returns `(f_fallible, f_infallible)` where:
/// * `f_fallible` yields `None` when the requested cast would overflow
/// * `f_infallible` is present only when every input is guaranteed to succeed; otherwise it is `None`
///   and callers must fall back to `f_fallible`
///
/// Returns `None` if the required scale reduction `delta_scale = input_scale - output_scale`
/// exceeds the supported precomputed precision table `I::MAX_FOR_EACH_PRECISION`.
/// In this scenario, any value would round to zero (e.g., dividing by 10^k where k exceeds the
/// available precision). Callers should therefore produce zero values (preserving nulls) rather
/// than returning an error.
#[expect(clippy::type_complexity)]
fn make_downscaler<I: DecimalType, O: DecimalType, F, D>(
    input_precision: u8,
    input_scale: i8,
    output_precision: u8,
    output_scale: i8,
    make_div_rem: F,
) -> Option<(
    impl Fn(I::Native) -> Option<O::Native>,
    Option<impl Fn(I::Native) -> O::Native>,
)>
where
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
    F: FnOnce(I::Native) -> D,
    D: Fn(I::Native) -> (I::Native, I::Native) + Copy,
{
    let delta_scale = input_scale as i16 - output_scale as i16;

    // delta_scale is guaranteed to be > 0, but may also be larger than I::MAX_PRECISION. If so, the
    // scale change divides out more digits than the input has precision and the result of the cast
    // is always zero. For example, if we try to apply delta_scale=10 a decimal32 value, the largest
    // possible result is 999999999/10000000000 = 0.0999999999, which rounds to zero. Smaller values
    // (e.g. 1/10000000000) or larger delta_scale (e.g. 999999999/10000000000000) produce even
    // smaller results, which also round to zero. In that case, just return an array of zeros.
    let max = I::MAX_FOR_EACH_PRECISION.get(delta_scale as usize)?;

    let div = max.add_wrapping(I::Native::ONE);
    let half = div.div_wrapping(I::Native::ONE.add_wrapping(I::Native::ONE));
    let half_neg = half.neg_wrapping();
    let div_rem = make_div_rem(div);

    let f_fallible = move |x: I::Native| {
        // div is >= 10 and so this cannot overflow
        let (d, r) = div_rem(x);

        // Round result
        let adjusted = match x >= I::Native::ZERO {
            true if r >= half => d.add_wrapping(I::Native::ONE),
            false if r <= half_neg => d.sub_wrapping(I::Native::ONE),
            _ => d,
        };
        O::Native::from_decimal(adjusted)
    };

    // if the reduction of the input number through scaling (dividing) is greater
    // than a possible precision loss (plus potential increase via rounding)
    // every input number will fit into the output type
    // Example: If we are starting with any number of precision 5 [xxxxx],
    // then and decrease the scale by 3 will have the following effect on the representation:
    // [xxxxx] -> [xx] (+ 1 possibly, due to rounding).
    // The rounding may add a digit, so the cast to be infallible,
    // the output type needs to have at least 3 digits of precision.
    // e.g. Decimal(5, 3) 99.999 to Decimal(3, 0) will result in 100:
    // [99999] -> [99] + 1 = [100], a cast to Decimal(2, 0) would not be possible
    let is_infallible_cast = (input_precision as i16) - delta_scale < (output_precision as i16);
    let f_infallible = is_infallible_cast.then_some(move |x| f_fallible(x).unwrap());
    Some((f_fallible, f_infallible))
}

/// Apply the rescaler function to the value.
/// If the rescaler is infallible, use the infallible function.
/// Otherwise, use the fallible function and validate the precision.
fn apply_rescaler<I: DecimalType, O: DecimalType>(
    value: I::Native,
    output_precision: u8,
    f: impl Fn(I::Native) -> Option<O::Native>,
    f_infallible: Option<impl Fn(I::Native) -> O::Native>,
) -> Option<O::Native>
where
    I::Native: DecimalCast,
    O::Native: DecimalCast,
{
    if let Some(f_infallible) = f_infallible {
        Some(f_infallible(value))
    } else {
        f(value).filter(|v| O::is_valid_decimal_precision(*v, output_precision))
    }
}

/// Rescales a decimal value from `(input_precision, input_scale)` to
/// `(output_precision, output_scale)` and returns the converted number when it fits
/// within the output precision.
///
/// The function first validates that the requested precision and scale are supported for
/// both the source and destination decimal types. It then either upscales (multiplying
/// by an appropriate power of ten) or downscales (dividing with rounding) the input value.
/// When the scaling factor exceeds the precision table of the destination type, the value
/// is treated as an overflow for upscaling, or rounded to zero for downscaling (as any
/// possible result would be zero at the requested scale).
///
/// This mirrors the column-oriented helpers of decimal casting but operates on a single value
/// (row-level) instead of an entire array.
///
/// Returns `None` if the value cannot be represented with the requested precision.
pub fn rescale_decimal<I: DecimalType, O: DecimalType>(
    value: I::Native,
    input_precision: u8,
    input_scale: i8,
    output_precision: u8,
    output_scale: i8,
) -> Option<O::Native>
where
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
{
    validate_decimal_precision_and_scale::<I>(input_precision, input_scale).ok()?;
    validate_decimal_precision_and_scale::<O>(output_precision, output_scale).ok()?;

    if input_scale <= output_scale {
        let (f, f_infallible) =
            make_upscaler::<I, O>(input_precision, input_scale, output_precision, output_scale)?;
        apply_rescaler::<I, O>(value, output_precision, f, f_infallible)
    } else {
        let Some((f, f_infallible)) = make_downscaler::<I, O, _, _>(
            input_precision,
            input_scale,
            output_precision,
            output_scale,
            I::Native::div_rem_by,
        ) else {
            // Scale reduction exceeds supported precision; result mathematically rounds to zero
            return Some(O::Native::ZERO);
        };
        apply_rescaler::<I, O>(value, output_precision, f, f_infallible)
    }
}

fn cast_decimal_to_decimal_error<I, O>(
    output_precision: u8,
    output_scale: i8,
) -> impl Fn(<I as ArrowPrimitiveType>::Native) -> ArrowError
where
    I: DecimalType,
    O: DecimalType,
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
{
    move |x: I::Native| {
        ArrowError::CastError(format!(
            "Cannot cast to {}({}, {}). Overflowing on {:?}",
            O::PREFIX,
            output_precision,
            output_scale,
            x
        ))
    }
}

fn apply_decimal_cast<I: DecimalType, O: DecimalType>(
    array: &PrimitiveArray<I>,
    output_precision: u8,
    output_scale: i8,
    f_fallible: impl Fn(I::Native) -> Option<O::Native>,
    f_infallible: Option<impl Fn(I::Native) -> O::Native>,
    cast_options: &CastOptions,
) -> Result<PrimitiveArray<O>, ArrowError>
where
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
{
    let array = if let Some(f_infallible) = f_infallible {
        array.unary(f_infallible)
    } else if cast_options.safe {
        array.unary_opt(|x| {
            f_fallible(x).filter(|v| O::is_valid_decimal_precision(*v, output_precision))
        })
    } else {
        let error = cast_decimal_to_decimal_error::<I, O>(output_precision, output_scale);
        array.try_unary(|x| {
            let v = f_fallible(x).ok_or_else(|| error(x))?;
            O::validate_decimal_precision(v, output_precision, output_scale).map(|()| v)
        })?
    };
    Ok(array)
}

fn convert_to_smaller_scale_decimal<I, O>(
    array: &PrimitiveArray<I>,
    input_precision: u8,
    input_scale: i8,
    output_precision: u8,
    output_scale: i8,
    cast_options: &CastOptions,
) -> Result<PrimitiveArray<O>, ArrowError>
where
    I: DecimalType,
    O: DecimalType,
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
{
    // Choose the division method once per batch. `unary` also evaluates null slots, so the
    // check covers every value, not only the valid ones.
    if I::Native::all_narrow(array.values()) {
        downscale_with::<I, O, _, _>(
            array,
            input_precision,
            input_scale,
            output_precision,
            output_scale,
            cast_options,
            I::Native::div_rem_narrow_by,
        )
    } else {
        downscale_with::<I, O, _, _>(
            array,
            input_precision,
            input_scale,
            output_precision,
            output_scale,
            cast_options,
            I::Native::div_rem_by,
        )
    }
}

fn downscale_with<I, O, F, D>(
    array: &PrimitiveArray<I>,
    input_precision: u8,
    input_scale: i8,
    output_precision: u8,
    output_scale: i8,
    cast_options: &CastOptions,
    make_div_rem: F,
) -> Result<PrimitiveArray<O>, ArrowError>
where
    I: DecimalType,
    O: DecimalType,
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
    F: FnOnce(I::Native) -> D,
    D: Fn(I::Native) -> (I::Native, I::Native) + Copy,
{
    if let Some((f_fallible, f_infallible)) = make_downscaler::<I, O, _, _>(
        input_precision,
        input_scale,
        output_precision,
        output_scale,
        make_div_rem,
    ) {
        apply_decimal_cast(
            array,
            output_precision,
            output_scale,
            f_fallible,
            f_infallible,
            cast_options,
        )
    } else {
        // Scale reduction exceeds supported precision; result mathematically rounds to zero
        let zeros = vec![O::Native::ZERO; array.len()];
        Ok(PrimitiveArray::new(zeros.into(), array.nulls().cloned()))
    }
}

fn convert_to_bigger_or_equal_scale_decimal<I, O>(
    array: &PrimitiveArray<I>,
    input_precision: u8,
    input_scale: i8,
    output_precision: u8,
    output_scale: i8,
    cast_options: &CastOptions,
) -> Result<PrimitiveArray<O>, ArrowError>
where
    I: DecimalType,
    O: DecimalType,
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
{
    if let Some((f, f_infallible)) =
        make_upscaler::<I, O>(input_precision, input_scale, output_precision, output_scale)
    {
        apply_decimal_cast(
            array,
            output_precision,
            output_scale,
            f,
            f_infallible,
            cast_options,
        )
    } else {
        // Scale increase exceeds supported precision; return overflow error
        Err(ArrowError::CastError(format!(
            "Cannot cast to {}({}, {}). Value overflows for output scale",
            O::PREFIX,
            output_precision,
            output_scale
        )))
    }
}

// Only support one type of decimal cast operations
pub(crate) fn cast_decimal_to_decimal_same_type<T>(
    array: &PrimitiveArray<T>,
    input_precision: u8,
    input_scale: i8,
    output_precision: u8,
    output_scale: i8,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError>
where
    T: DecimalType,
    T::Native: DecimalCast + ArrowNativeTypeOp,
{
    let array: PrimitiveArray<T> =
        if input_scale == output_scale && input_precision <= output_precision {
            array.clone()
        } else if input_scale <= output_scale {
            convert_to_bigger_or_equal_scale_decimal::<T, T>(
                array,
                input_precision,
                input_scale,
                output_precision,
                output_scale,
                cast_options,
            )?
        } else {
            // input_scale > output_scale
            convert_to_smaller_scale_decimal::<T, T>(
                array,
                input_precision,
                input_scale,
                output_precision,
                output_scale,
                cast_options,
            )?
        };

    Ok(Arc::new(array.with_precision_and_scale(
        output_precision,
        output_scale,
    )?))
}

// Support two different types of decimal cast operations
pub(crate) fn cast_decimal_to_decimal<I, O>(
    array: &PrimitiveArray<I>,
    input_precision: u8,
    input_scale: i8,
    output_precision: u8,
    output_scale: i8,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError>
where
    I: DecimalType,
    O: DecimalType,
    I::Native: DecimalCast + ArrowNativeTypeOp,
    O::Native: DecimalCast + ArrowNativeTypeOp,
{
    let array: PrimitiveArray<O> = if input_scale > output_scale {
        convert_to_smaller_scale_decimal::<I, O>(
            array,
            input_precision,
            input_scale,
            output_precision,
            output_scale,
            cast_options,
        )?
    } else {
        convert_to_bigger_or_equal_scale_decimal::<I, O>(
            array,
            input_precision,
            input_scale,
            output_precision,
            output_scale,
            cast_options,
        )?
    };

    Ok(Arc::new(array.with_precision_and_scale(
        output_precision,
        output_scale,
    )?))
}

/// Parses the given string as a decimal with the given scale, returning the
/// unscaled representation in the decimal type's native integer (e.g. `i32`
/// for `Decimal32Type`, `i256` for `Decimal256Type`).
///
/// Returns an error if the input is not a valid decimal string, or if the
/// scaled and rounded value does not fit the maximum precision of the decimal
/// type. The caller is responsible for validating the result against any
/// smaller target precision.
#[deprecated(
    since = "60.0.0",
    note = "Use `arrow_cast::parse::parse_decimal` instead"
)]
pub fn parse_string_to_decimal_native<T: DecimalType>(
    value_str: &str,
    scale: usize,
) -> Result<T::Native, ArrowError> {
    let overflow = || {
        ArrowError::InvalidArgumentError(format!(
            "Cannot convert {value_str} to {}: Overflow",
            T::PREFIX
        ))
    };
    let scale = i8::try_from(scale).map_err(|_| overflow())?;
    parse_decimal_checked::<T>(value_str, T::MAX_PRECISION, scale).map_err(|e| match e {
        DecimalParseError::InvalidFormat => {
            ArrowError::InvalidArgumentError(format!("Invalid decimal format: {value_str:?}"))
        }
        DecimalParseError::Overflow => overflow(),
    })
}

pub(crate) fn generic_string_to_decimal_cast<'a, T, S>(
    from: &'a S,
    precision: u8,
    scale: i8,
    cast_options: &CastOptions,
) -> Result<PrimitiveArray<T>, ArrowError>
where
    T: DecimalType,
    &'a S: StringArrayType<'a>,
{
    if cast_options.safe {
        let iter = from
            .iter()
            .map(|v| parse_decimal_checked::<T>(v?, precision, scale).ok());
        // Benefit:
        //     15-19% faster than appending to a PrimitiveBuilder (measured
        //     with the cast_kernels string-to-decimal benchmarks)
        // Soundness:
        //     The iterator is trustedLen because it comes from a `StringArray`.
        Ok(unsafe {
            PrimitiveArray::<T>::from_trusted_len_iter(iter)
                .with_precision_and_scale(precision, scale)?
        })
    } else {
        let mut builder = PrimitiveBuilder::<T>::with_capacity(from.len());
        for v in from.iter() {
            match v {
                Some(v) => {
                    let v = parse_decimal_checked::<T>(v, precision, scale).map_err(|e| {
                        let reason = match e {
                            DecimalParseError::InvalidFormat => "invalid decimal format",
                            DecimalParseError::Overflow => "value does not fit",
                        };
                        ArrowError::CastError(format!(
                            "Cannot cast string '{v}' to value of {}({precision}, {scale}) type: {reason}",
                            T::PREFIX,
                        ))
                    })?;
                    builder.append_value(v);
                }
                None => builder.append_null(),
            }
        }
        builder.finish().with_precision_and_scale(precision, scale)
    }
}

pub(crate) fn string_to_decimal_cast<T: DecimalType, Offset: OffsetSizeTrait>(
    from: &GenericStringArray<Offset>,
    precision: u8,
    scale: i8,
    cast_options: &CastOptions,
) -> Result<PrimitiveArray<T>, ArrowError> {
    generic_string_to_decimal_cast::<T, GenericStringArray<Offset>>(
        from,
        precision,
        scale,
        cast_options,
    )
}

pub(crate) fn string_view_to_decimal_cast<T: DecimalType>(
    from: &StringViewArray,
    precision: u8,
    scale: i8,
    cast_options: &CastOptions,
) -> Result<PrimitiveArray<T>, ArrowError> {
    generic_string_to_decimal_cast::<T, StringViewArray>(from, precision, scale, cast_options)
}

/// Cast Utf8 to decimal
pub(crate) fn cast_string_to_decimal<T: DecimalType, Offset: OffsetSizeTrait>(
    from: &dyn Array,
    precision: u8,
    scale: i8,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    validate_decimal_precision_and_scale::<T>(precision, scale)?;

    let result = match from.data_type() {
        DataType::Utf8View => string_view_to_decimal_cast::<T>(
            from.as_any().downcast_ref::<StringViewArray>().unwrap(),
            precision,
            scale,
            cast_options,
        )?,
        DataType::Utf8 | DataType::LargeUtf8 => string_to_decimal_cast::<T, Offset>(
            from.as_any()
                .downcast_ref::<GenericStringArray<Offset>>()
                .unwrap(),
            precision,
            scale,
            cast_options,
        )?,
        other => {
            return Err(ArrowError::ComputeError(format!(
                "Cannot cast {other:?} to decimal",
            )));
        }
    };

    Ok(Arc::new(result))
}

pub(crate) fn cast_floating_point_to_decimal<T: ArrowPrimitiveType, D>(
    array: &PrimitiveArray<T>,
    precision: u8,
    scale: i8,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    D: DecimalType + ArrowPrimitiveType,
    <D as ArrowPrimitiveType>::Native: DecimalCast,
{
    let mul = 10_f64.powi(scale as i32);

    if cast_options.safe {
        array
            .unary_opt::<_, D>(|v| {
                single_float_to_decimal::<D>(v.as_(), mul)
                    .filter(|v| D::is_valid_decimal_precision(*v, precision))
            })
            .with_precision_and_scale(precision, scale)
            .map(|a| Arc::new(a) as ArrayRef)
    } else {
        array
            .try_unary::<_, D, _>(|v| {
                let v = single_float_to_decimal::<D>(v.as_(), mul).ok_or_else(|| {
                    ArrowError::CastError(format!(
                        "Cannot cast to {}({}, {}). Overflowing on {:?}",
                        D::PREFIX,
                        precision,
                        scale,
                        v
                    ))
                })?;
                D::validate_decimal_precision(v, precision, scale).map(|()| v)
            })?
            .with_precision_and_scale(precision, scale)
            .map(|a| Arc::new(a) as ArrayRef)
    }
}

/// Cast a single floating point value to a decimal native with the given multiple.
/// Returns `None` if the value cannot be represented with the requested precision.
#[inline(always)]
pub fn single_float_to_decimal<D>(input: f64, mul: f64) -> Option<D::Native>
where
    D: DecimalType + ArrowPrimitiveType,
    <D as ArrowPrimitiveType>::Native: DecimalCast,
{
    D::Native::from_f64((mul * input).round())
}

pub(crate) fn cast_decimal_to_integer<D, T>(
    array: &dyn Array,
    base: D::Native,
    scale: i8,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError>
where
    T: ArrowPrimitiveType,
    <T as ArrowPrimitiveType>::Native: NumCast,
    D: DecimalType + ArrowPrimitiveType,
    <D as ArrowPrimitiveType>::Native: ToPrimitive,
{
    let array = array.as_primitive::<D>();

    let div: D::Native = base.pow_checked(scale.unsigned_abs() as u32).map_err(|_| {
        ArrowError::CastError(format!(
            "Cannot cast to {:?}. The scale {} causes overflow.",
            D::PREFIX,
            scale,
        ))
    })?;

    let mut value_builder = PrimitiveBuilder::<T>::with_capacity(array.len());

    if scale < 0 {
        match cast_options.safe {
            true => {
                for i in 0..array.len() {
                    if array.is_null(i) {
                        value_builder.append_null();
                    } else {
                        let v = array
                            .value(i)
                            .mul_checked(div)
                            .ok()
                            .and_then(<T::Native as NumCast>::from::<D::Native>);
                        value_builder.append_option(v);
                    }
                }
            }
            false => {
                for i in 0..array.len() {
                    if array.is_null(i) {
                        value_builder.append_null();
                    } else {
                        let v = array.value(i).mul_checked(div)?;

                        let value =
                            <T::Native as NumCast>::from::<D::Native>(v).ok_or_else(|| {
                                ArrowError::CastError(format!(
                                    "value of {:?} is out of range {}",
                                    v,
                                    T::DATA_TYPE
                                ))
                            })?;

                        value_builder.append_value(value);
                    }
                }
            }
        }
    } else {
        match cast_options.safe {
            true => {
                for i in 0..array.len() {
                    if array.is_null(i) {
                        value_builder.append_null();
                    } else {
                        let v = array
                            .value(i)
                            .div_checked(div)
                            .ok()
                            .and_then(<T::Native as NumCast>::from::<D::Native>);
                        value_builder.append_option(v);
                    }
                }
            }
            false => {
                for i in 0..array.len() {
                    if array.is_null(i) {
                        value_builder.append_null();
                    } else {
                        let v = array.value(i).div_checked(div)?;

                        let value =
                            <T::Native as NumCast>::from::<D::Native>(v).ok_or_else(|| {
                                ArrowError::CastError(format!(
                                    "value of {:?} is out of range {}",
                                    v,
                                    T::DATA_TYPE
                                ))
                            })?;

                        value_builder.append_value(value);
                    }
                }
            }
        }
    }
    Ok(Arc::new(value_builder.finish()))
}

/// Cast a decimal array to a floating point array.
///
/// Conversion is lossy and follows standard floating point semantics. Values
/// that exceed the representable range become `INFINITY` or `-INFINITY` without
/// returning an error.
pub(crate) fn cast_decimal_to_float<D: DecimalType, T: ArrowPrimitiveType, F>(
    array: &dyn Array,
    op: F,
) -> Result<ArrayRef, ArrowError>
where
    F: Fn(D::Native) -> T::Native,
{
    let array = array.as_primitive::<D>();
    let array = array.unary::<_, T>(op);
    Ok(Arc::new(array))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(deprecated)]
    fn test_parse_string_to_decimal_native() {
        assert_eq!(
            parse_string_to_decimal_native::<Decimal128Type>("123.456", 2).unwrap(),
            12346
        );
        // The value is checked against the maximum precision of the type, not
        // against any target precision
        assert_eq!(
            parse_string_to_decimal_native::<Decimal128Type>(&"9".repeat(38), 0).unwrap(),
            10_i128.pow(38) - 1
        );
        assert!(
            parse_string_to_decimal_native::<Decimal128Type>(&i128::MAX.to_string(), 0).is_err()
        );
        assert_eq!(
            parse_string_to_decimal_native::<Decimal128Type>("abc", 2)
                .unwrap_err()
                .to_string(),
            "Invalid argument error: Invalid decimal format: \"abc\""
        );
        assert_eq!(
            parse_string_to_decimal_native::<Decimal32Type>("1", 10)
                .unwrap_err()
                .to_string(),
            "Invalid argument error: Cannot convert 1 to Decimal32: Overflow"
        );
    }

    #[test]
    fn test_rescale_decimal_upscale_within_precision() {
        let result = rescale_decimal::<Decimal128Type, Decimal128Type>(
            12_345_i128, // 123.45 with scale 2
            5,
            2,
            8,
            5,
        );
        assert_eq!(result, Some(12_345_000_i128));
    }

    #[test]
    fn test_rescale_decimal_downscale_rounds_half_away_from_zero() {
        let positive = rescale_decimal::<Decimal128Type, Decimal128Type>(
            1_050_i128, // 1.050 with scale 3
            5, 3, 5, 1,
        );
        assert_eq!(positive, Some(11_i128)); // 1.1 with scale 1

        let negative = rescale_decimal::<Decimal128Type, Decimal128Type>(
            -1_050_i128, // -1.050 with scale 3
            5,
            3,
            5,
            1,
        );
        assert_eq!(negative, Some(-11_i128)); // -1.1 with scale 1
    }

    #[test]
    fn test_rescale_decimal128_downscale_rounding_every_delta() {
        let max = 10_i128.pow(38) - 1;
        for delta in 1..=38 {
            let half = 5 * 10_i128.pow(delta - 1);
            let rescale = |value| {
                rescale_decimal::<Decimal128Type, Decimal128Type>(value, 38, delta as i8, 38, 0)
            };
            assert_eq!(rescale(0), Some(0), "delta {delta}");
            assert_eq!(rescale(half), Some(1), "delta {delta}");
            assert_eq!(rescale(-half), Some(-1), "delta {delta}");
            assert_eq!(rescale(half - 1), Some(0), "delta {delta}");
            assert_eq!(rescale(1 - half), Some(0), "delta {delta}");
            let rounded_max = 10_i128.pow(38 - delta);
            assert_eq!(rescale(max), Some(rounded_max), "delta {delta}");
            assert_eq!(rescale(-max), Some(-rounded_max), "delta {delta}");
        }
    }

    #[test]
    fn test_div_rem_narrow_by_matches_wrapping_div_rem() {
        let mut divisors: Vec<i128> = (0..=38).map(|k| 10_i128.pow(k)).collect();
        divisors.extend(divisors.clone().iter().map(|d| -d));
        divisors.extend([
            2,
            3,
            7,
            u64::MAX as i128,
            1 << 64,
            (1 << 64) + 1,
            i128::MAX,
            i128::MIN,
        ]);
        let max = u64::MAX as i128;
        let values = [
            0,
            1,
            -1,
            9,
            -9,
            10,
            -10,
            10_i128.pow(18) - 1,
            1 - 10_i128.pow(18),
            max,
            -max,
            max - 1,
        ];
        for divisor in divisors {
            let div_rem = i128::div_rem_narrow_by(divisor);
            for value in values {
                let expected = (value.wrapping_div(divisor), value.wrapping_rem(divisor));
                assert_eq!(div_rem(value), expected, "{value} / {divisor}");
            }
        }
    }

    #[test]
    fn test_all_narrow() {
        let max = u64::MAX as i128;
        assert!(i128::all_narrow(&[]));
        assert!(i128::all_narrow(&[0, max, -max]));
        assert!(!i128::all_narrow(&[0, max + 1]));
        assert!(!i128::all_narrow(&[-max - 1]));
        assert!(!i128::all_narrow(&[i128::MIN]));
        assert!(!i32::all_narrow(&[0]));
        assert!(!i256::all_narrow(&[i256::ZERO]));
    }

    #[test]
    fn test_downscale_values_beyond_declared_precision() {
        // Declared precision does not bound the stored values, so the per-batch
        // check has to look at the values themselves
        let big = 10_i128.pow(30) + 5;
        let array = Decimal128Array::from(vec![Some(big), Some(-big), Some(123_456), None])
            .with_precision_and_scale(10, 4)
            .unwrap();
        let options = CastOptions {
            safe: true,
            ..Default::default()
        };
        let result = cast_with_options(&array, &DataType::Decimal128(38, 1), &options).unwrap();
        let result = result.as_primitive::<Decimal128Type>();
        assert_eq!(result.value(0), 10_i128.pow(27));
        assert_eq!(result.value(1), -10_i128.pow(27));
        assert_eq!(result.value(2), 123);
        assert!(result.is_null(3));
    }

    #[test]
    fn test_div_rem_by_matches_wrapping_div_rem() {
        let mut divisors: Vec<i128> = (0..=38).map(|k| 10_i128.pow(k)).collect();
        divisors.extend(divisors.clone().iter().map(|d| -d));
        divisors.extend([
            2,
            3,
            7,
            1 << 64,
            (1 << 64) + 1,
            i128::MAX,
            i128::MIN,
            i128::MIN + 1,
        ]);
        let values = [
            0,
            1,
            -1,
            9,
            -9,
            10,
            -10,
            11,
            -11,
            123_456_789_012_345_678_901_234_567_890,
            -123_456_789_012_345_678_901_234_567_890,
            10_i128.pow(38) - 1,
            1 - 10_i128.pow(38),
            i128::MAX,
            i128::MIN,
            i128::MAX - 1,
            i128::MIN + 1,
        ];
        for divisor in divisors {
            let div_rem = i128::div_rem_by(divisor);
            let div_rem_i256 = i256::div_rem_by(i256::from_i128(divisor));
            for value in values {
                let expected = (value.wrapping_div(divisor), value.wrapping_rem(divisor));
                assert_eq!(div_rem(value), expected, "{value} / {divisor}");
                assert_eq!(
                    div_rem_i256(i256::from_i128(value)),
                    (
                        i256::from_i128(value).wrapping_div(i256::from_i128(divisor)),
                        i256::from_i128(value).wrapping_rem(i256::from_i128(divisor))
                    ),
                    "{value} / {divisor}"
                );
            }
        }
    }

    #[test]
    fn test_rescale_decimal_downscale_large_delta_returns_zero() {
        let result = rescale_decimal::<Decimal32Type, Decimal32Type>(12_345_i32, 9, 9, 9, 4);
        assert_eq!(result, Some(0_i32));
    }

    #[test]
    fn test_rescale_decimal_upscale_overflow_returns_none() {
        let result = rescale_decimal::<Decimal32Type, Decimal32Type>(9_999_i32, 4, 0, 5, 2);
        assert_eq!(result, None);
    }

    #[test]
    fn test_rescale_decimal256_extreme_scales() {
        assert_eq!(
            rescale_decimal::<Decimal256Type, Decimal256Type>(i256::ONE, 76, -76, 76, 0),
            None
        );
        assert_eq!(
            rescale_decimal::<Decimal256Type, Decimal256Type>(i256::ZERO, 76, -76, 76, 0),
            Some(i256::ZERO)
        );
        assert_eq!(
            rescale_decimal::<Decimal256Type, Decimal256Type>(i256::ONE, 76, i8::MIN, 76, 76),
            None
        );
        assert_eq!(
            rescale_decimal::<Decimal256Type, Decimal256Type>(i256::ONE, 76, 76, 76, i8::MIN),
            Some(i256::ZERO)
        );
    }

    #[test]
    fn test_rescale_decimal_invalid_input_precision_scale_returns_none() {
        let result = rescale_decimal::<Decimal128Type, Decimal128Type>(123_i128, 39, 39, 38, 38);
        assert_eq!(result, None);
    }

    #[test]
    fn test_rescale_decimal_invalid_output_precision_scale_returns_none() {
        let result = rescale_decimal::<Decimal128Type, Decimal128Type>(123_i128, 38, 38, 39, 39);
        assert_eq!(result, None);
    }
}
