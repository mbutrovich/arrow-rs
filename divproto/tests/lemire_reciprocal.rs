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

use divproto::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn divisors_u32(rng: &mut StdRng) -> Vec<u32> {
    let mut d: Vec<u32> = (2..=5000).collect();
    d.extend((1..32).map(|k| 1u32 << k));
    d.extend([u32::MAX, u32::MAX - 1, (1 << 31) + 1, i32::MAX as u32]);
    d.extend(
        (0..2000)
            .map(|_| rng.gen_range(2..=u32::MAX) >> rng.gen_range(0..31))
            .filter(|d| *d >= 2),
    );
    d
}

fn values_u32(rng: &mut StdRng) -> Vec<u32> {
    let mut v: Vec<u32> = vec![
        0,
        1,
        2,
        u32::MAX,
        u32::MAX - 1,
        1 << 31,
        (1 << 31) - 1,
        (1 << 31) + 1,
    ];
    v.extend((0..3000).map(|_| rng.r#gen::<u32>() >> rng.gen_range(0..32)));
    v
}

#[test]
fn lemire_u32_reciprocal_matches_hardware_including_powers_of_two() {
    let mut rng = StdRng::seed_from_u64(10);
    let values = values_u32(&mut rng);
    for d in divisors_u32(&mut rng) {
        let l = LemireU32::reciprocal(d).unwrap();
        assert!(matches!(l, LemireU32::Reciprocal { .. }));
        for &n in values.iter().chain(&[d, d - 1, d.wrapping_add(1)]) {
            assert_eq!(l.rem(n), n % d, "{n} % {d}");
            assert_eq!(l.div_rem(n), (n / d, n % d), "{n} / {d}");
        }
    }
    assert!(LemireU32::reciprocal(0).is_none());
    assert!(LemireU32::reciprocal(1).is_none());
}

#[test]
fn lemire_u64_reciprocal_matches_hardware_including_powers_of_two() {
    let mut rng = StdRng::seed_from_u64(11);
    let mut divisors: Vec<u64> = (2..=3000).collect();
    divisors.extend((1..64).map(|k| 1u64 << k));
    divisors.extend([u64::MAX, u64::MAX - 1, (1 << 63) + 1]);
    divisors.extend(
        (0..1000)
            .map(|_| rng.gen_range(2..=u64::MAX) >> rng.gen_range(0..63))
            .filter(|d| *d >= 2),
    );
    let mut values: Vec<u64> = vec![0, 1, u64::MAX, u64::MAX - 1, 1 << 63, (1 << 63) - 1];
    values.extend((0..2000).map(|_| rng.r#gen::<u64>() >> rng.gen_range(0..64)));
    for d in divisors {
        let l = LemireDivU64::reciprocal(d).unwrap();
        for &n in &values {
            assert_eq!(l.div_rem(n), (n / d, n % d), "{n} / {d}");
        }
    }
    assert!(LemireDivU64::reciprocal(1).is_none());
}

#[test]
fn lemire_i32_matches_wrapping_div() {
    let mut rng = StdRng::seed_from_u64(12);
    let values = values_u32(&mut rng);
    let mut signed: Vec<i32> = divisors_u32(&mut rng)
        .iter()
        .map(|d| *d as i32)
        .filter(|d| d.unsigned_abs() >= 2)
        .collect();
    signed.extend(signed.clone().iter().map(|d| d.wrapping_neg()));
    signed.extend([i32::MIN, i32::MAX, i32::MIN + 1, 2, -2]);
    for d in signed {
        let l = LemireI32::new(d).unwrap();
        for &v in &values {
            for x in [v as i32, (v as i32).wrapping_neg(), i32::MIN, i32::MAX] {
                assert_eq!(l.div(x), x.wrapping_div(d), "{x} / {d}");
            }
        }
    }
    assert!(LemireI32::new(0).is_none());
    assert!(LemireI32::new(1).is_none());
    assert!(LemireI32::new(-1).is_none());
}

#[test]
fn biased_lemire_pmod_matches_comet() {
    let mut rng = StdRng::seed_from_u64(13);
    let hashes = values_u32(&mut rng);
    let mut partitions: Vec<u32> = (2..=1000).collect();
    partitions.push(i32::MAX as u32);
    for n in partitions {
        let p = BiasedLemirePmod::new(n).unwrap();
        for &h in &hashes {
            assert_eq!(
                p.pmod(h) as usize,
                comet_pmod(h, n as usize),
                "pmod({h}, {n})"
            );
        }
    }
    assert!(BiasedLemirePmod::new(1).is_none());
}
