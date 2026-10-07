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

#[test]
fn lemire_u32_matches_hardware() {
    let mut rng = StdRng::seed_from_u64(4);
    let mut divisors: Vec<u32> = (1..=5000).collect();
    divisors.extend([
        u32::MAX,
        u32::MAX - 1,
        1 << 31,
        (1 << 31) + 1,
        i32::MAX as u32,
    ]);
    divisors.extend((0..2000).map(|_| rng.gen_range(1..=u32::MAX)));
    let mut values: Vec<u32> = vec![0, 1, u32::MAX, u32::MAX - 1, 1 << 31, (1 << 31) - 1];
    values.extend((0..3000).map(|_| rng.r#gen::<u32>()));
    for &d in &divisors {
        let l = LemireU32::new(d).unwrap();
        for &n in values.iter().chain(&[d, d - 1, d.wrapping_add(1)]) {
            assert_eq!(l.rem(n), n % d, "{n} % {d}");
            assert_eq!(l.div_rem(n), (n / d, n % d), "{n} / {d}");
        }
    }
    assert!(LemireU32::new(0).is_none());
}

#[test]
fn pmod_variants_match_comet() {
    let mut rng = StdRng::seed_from_u64(5);
    let mut partitions: Vec<u32> = (1..=1000).collect();
    partitions.push(i32::MAX as u32);
    let mut hashes: Vec<u32> = vec![0, 1, u32::MAX, 1 << 31, (1 << 31) - 1, (1 << 31) + 1];
    hashes.extend((0..3000).map(|_| rng.r#gen::<u32>()));
    for &n in &partitions {
        let d = LemireU32::new(n).unwrap();
        for &h in &hashes {
            let expected = comet_pmod(h, n as usize);
            assert_eq!(
                comet_pmod_one_div(h, n as usize),
                expected,
                "pmod({h}, {n})"
            );
            assert_eq!(lemire_pmod(h, n, d), expected, "pmod({h}, {n})");
            assert_eq!(
                BiasedPmod::new(n).unwrap().pmod(h) as usize,
                expected,
                "biased pmod({h}, {n})"
            );
        }
    }
}

#[test]
fn lemire_div_u64_and_narrow_paths_match() {
    let mut rng = StdRng::seed_from_u64(6);
    let mut divisors: Vec<u64> = (1..=2000).collect();
    divisors.extend((1..=19).map(|k| 10u64.pow(k)));
    divisors.extend([u64::MAX, u64::MAX - 1, 1 << 63, (1 << 63) + 1]);
    divisors.extend(
        (0..1000)
            .map(|_| rng.gen_range(1..=u64::MAX) >> rng.gen_range(0..64))
            .filter(|d| *d != 0),
    );
    let mut values: Vec<u64> = vec![0, 1, u64::MAX, u64::MAX - 1, 1 << 63, (1 << 63) - 1];
    values.extend((0..2000).map(|_| rng.r#gen::<u64>() >> rng.gen_range(0..64)));
    for &d in &divisors {
        let l = LemireDivU64::new(d).unwrap();
        let gm = GmU64::new(d).unwrap();
        for &n in &values {
            assert_eq!(l.div_rem(n), (n / d, n % d), "{n} / {d}");
        }
        for neg in [false, true] {
            let sd = if neg { -(d as i128) } else { d as i128 };
            for &n in &values {
                for x in [n as i128, -(n as i128)] {
                    let expected = (x / sd, x % sd);
                    assert_eq!(narrow_hw_div_rem(x, d, neg), expected, "{x} / {sd}");
                    assert_eq!(narrow_lemire_div_rem(x, l, neg), expected, "{x} / {sd}");
                    assert_eq!(narrow_gm_div_rem(x, gm, neg), expected, "{x} / {sd}");
                }
            }
        }
    }
}
