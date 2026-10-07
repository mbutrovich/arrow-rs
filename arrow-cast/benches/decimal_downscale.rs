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
use std::sync::Arc;

use arrow_array::ArrowNativeTypeOp;
use arrow_array::types::{
    Decimal32Type, Decimal64Type, Decimal128Type, Decimal256Type, DecimalType,
};
use arrow_array::{ArrayRef, PrimitiveArray};
use arrow_buffer::ArrowNativeType;
use arrow_cast::cast;
use arrow_schema::DataType;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const ARRAY_LEN: usize = 8192;

/// Creates an array of [`ARRAY_LEN`] random values with `digits` significant
/// digits (no leading zero, half of them negative) for the given `precision`
/// and `scale`
fn decimal_array<T: DecimalType>(digits: u32, precision: u8, scale: i8) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(42);
    let ten = T::Native::usize_as(10);
    let values = (0..ARRAY_LEN).map(|_| {
        let mut value = T::Native::usize_as(rng.random_range(1..10));
        for _ in 1..digits {
            value = value
                .mul_wrapping(ten)
                .add_wrapping(T::Native::usize_as(rng.random_range(0..10)));
        }
        if rng.random::<bool>() {
            value.neg_wrapping()
        } else {
            value
        }
    });
    let array = PrimitiveArray::<T>::from_iter_values(values)
        .with_precision_and_scale(precision, scale)
        .unwrap();
    Arc::new(array)
}

fn downscale(c: &mut Criterion) {
    let mut group = c.benchmark_group("decimal_downscale");
    group.throughput(Throughput::Elements(ARRAY_LEN as u64));

    for (id, array, to_type) in [
        (
            BenchmarkId::new("decimal32", "(9, 4) -> (9, 2) 9 digits"),
            decimal_array::<Decimal32Type>(9, 9, 4),
            DataType::Decimal32(9, 2),
        ),
        (
            BenchmarkId::new("decimal64", "(18, 6) -> (18, 2) 18 digits"),
            decimal_array::<Decimal64Type>(18, 18, 6),
            DataType::Decimal64(18, 2),
        ),
        (
            BenchmarkId::new("decimal128", "(18, 6) -> (18, 2) 18 digits"),
            decimal_array::<Decimal128Type>(18, 18, 6),
            DataType::Decimal128(18, 2),
        ),
        (
            BenchmarkId::new("decimal128", "(38, 10) -> (38, 2) 38 digits"),
            decimal_array::<Decimal128Type>(38, 38, 10),
            DataType::Decimal128(38, 2),
        ),
        (
            BenchmarkId::new("decimal128", "(38, 20) -> (38, 2) 38 digits"),
            decimal_array::<Decimal128Type>(38, 38, 20),
            DataType::Decimal128(38, 2),
        ),
        (
            BenchmarkId::new("decimal128", "(38, 10) -> (38, 2) 15 digits"),
            decimal_array::<Decimal128Type>(15, 38, 10),
            DataType::Decimal128(38, 2),
        ),
        (
            BenchmarkId::new("decimal128", "(38, 10) -> (30, 2) 30 digits fallible"),
            decimal_array::<Decimal128Type>(30, 38, 10),
            DataType::Decimal128(30, 2),
        ),
        (
            BenchmarkId::new("decimal256", "(76, 10) -> (76, 2) 38 digits"),
            decimal_array::<Decimal256Type>(38, 76, 10),
            DataType::Decimal256(76, 2),
        ),
    ] {
        group.bench_function(id, |b| {
            b.iter(|| black_box(cast(&array, &to_type).unwrap()))
        });
    }

    group.finish();
}

criterion_group!(benches, downscale);
criterion_main!(benches);
