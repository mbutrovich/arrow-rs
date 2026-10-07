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

use arrow_buffer::i256;
use divproto::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn edge_u128() -> Vec<u128> {
    let mut v = vec![
        0,
        1,
        2,
        3,
        5,
        7,
        9,
        10,
        11,
        u64::MAX as u128,
        u64::MAX as u128 + 1,
        u128::MAX,
        u128::MAX - 1,
        1 << 127,
        (1 << 127) - 1,
        (1 << 127) + 1,
    ];
    for k in 0..=38 {
        let p = 10u128.pow(k);
        v.extend([p, p - 1, p + 1, p / 2]);
    }
    v
}

#[test]
fn gm_u64_and_lemire_match_hardware() {
    let mut rng = StdRng::seed_from_u64(1);
    let mut divisors: Vec<u64> = (1..=1000).collect();
    divisors.extend([
        u64::MAX,
        u64::MAX - 1,
        1 << 63,
        (1 << 63) + 1,
        (1 << 32) + 1,
        10_000_000_000_000_000_000,
    ]);
    divisors.extend(
        (0..2000)
            .map(|_| rng.gen_range(1..u64::MAX) >> rng.gen_range(0..64))
            .filter(|d| *d != 0),
    );
    let mut values: Vec<u64> = vec![0, 1, u64::MAX, u64::MAX - 1, 1 << 63];
    values.extend((0..2000).map(|_| rng.r#gen::<u64>() >> rng.gen_range(0..64)));
    for &d in &divisors {
        let gm = GmU64::new(d).unwrap();
        let lemire = LemireU64::new(d);
        for &n in values
            .iter()
            .chain(&[d, d - 1, d.wrapping_add(1), d.wrapping_mul(7)])
        {
            assert_eq!(gm.div_rem(n), (n / d, n % d), "{n} / {d}");
            assert_eq!(lemire.remainder(n), n % d, "{n} % {d}");
        }
    }
    assert!(GmU64::new(0).is_none());
}

#[test]
fn gm_u128_and_i128_match_hardware() {
    let mut rng = StdRng::seed_from_u64(2);
    let mut divisors = edge_u128();
    divisors.retain(|d| *d != 0);
    divisors.extend(
        (0..3000)
            .map(|_| rng.r#gen::<u128>() >> rng.gen_range(0..128))
            .filter(|d| *d != 0),
    );
    let mut values = edge_u128();
    values.extend((0..3000).map(|_| rng.r#gen::<u128>() >> rng.gen_range(0..128)));
    for &d in &divisors {
        let gm = GmU128::new(d).unwrap();
        for &n in &values {
            assert_eq!(gm.div_rem(n), (n / d, n % d), "{n} / {d}");
        }
        for sd in [d as i128, (d as i128).wrapping_neg()] {
            if sd == 0 {
                continue;
            }
            let a = GmI128::<false>::new(sd).unwrap();
            let b = GmI128::<true>::new(sd).unwrap();
            for &n in &values {
                for x in [n as i128, (n as i128).wrapping_neg()] {
                    let expected = (x.wrapping_div(sd), x.wrapping_rem(sd));
                    assert_eq!(a.div_rem(x), expected, "{x} / {sd}");
                    assert_eq!(b.div_rem(x), expected, "{x} / {sd} fast64");
                }
            }
        }
    }
    assert!(GmU128::new(0).is_none());
}

#[test]
fn gm_i256_matches_arrow() {
    let mut rng = StdRng::seed_from_u64(3);
    let rand_i256 = |rng: &mut StdRng| {
        let v = i256::from_parts(rng.r#gen(), rng.r#gen());
        let shift = rng.gen_range(0..255);
        v >> shift as u8
    };
    let mut divisors: Vec<i256> = vec![
        i256::ONE,
        i256::MINUS_ONE,
        i256::MAX,
        i256::MIN,
        i256::from_i128(2),
        i256::from_i128(3),
    ];
    let mut p = i256::ONE;
    for _ in 0..=76 {
        divisors.extend([
            p,
            p.wrapping_neg(),
            p.wrapping_add(i256::ONE),
            p.wrapping_sub(i256::ONE),
        ]);
        p = p.wrapping_mul(i256::from_i128(10));
    }
    divisors.extend((0..300).map(|_| rand_i256(&mut rng)));
    divisors.retain(|d| *d != i256::ZERO);
    let mut values: Vec<i256> = vec![
        i256::ZERO,
        i256::ONE,
        i256::MINUS_ONE,
        i256::MAX,
        i256::MIN,
        i256::MAX.wrapping_sub(i256::ONE),
        i256::MIN.wrapping_add(i256::ONE),
    ];
    values.extend((0..300).map(|_| rand_i256(&mut rng)));
    for &d in &divisors {
        let gm = GmI256::new(d).unwrap();
        for &x in values
            .iter()
            .chain(&[d, d.wrapping_neg(), d.wrapping_mul(i256::from_i128(7))])
        {
            let expected = (x.wrapping_div(d), x.wrapping_rem(d));
            assert_eq!(gm.div_rem(x), expected, "{x} / {d}");
        }
    }
    assert!(GmI256::new(i256::ZERO).is_none());
}
