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

fn divisors(rng: &mut StdRng) -> Vec<u32> {
    let mut d: Vec<u32> = (1..=5000).collect();
    d.extend([
        u32::MAX,
        u32::MAX - 1,
        1 << 31,
        (1 << 31) + 1,
        i32::MAX as u32,
    ]);
    d.extend((1..=9).map(|k| 10u32.pow(k)));
    d.extend(
        (0..2000)
            .map(|_| rng.gen_range(1..=u32::MAX) >> rng.gen_range(0..32))
            .filter(|d| *d != 0),
    );
    d
}

fn values(rng: &mut StdRng) -> Vec<u32> {
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
fn gm_u32_matches_hardware() {
    let mut rng = StdRng::seed_from_u64(7);
    let values = values(&mut rng);
    for d in divisors(&mut rng) {
        let gm = GmU32::new(d).unwrap();
        for &n in values.iter().chain(&[d, d - 1, d.wrapping_add(1)]) {
            assert_eq!(gm.div(n), n / d, "{n} / {d}");
            assert_eq!(gm.rem(n), n % d, "{n} % {d}");
        }
    }
    assert!(GmU32::new(0).is_none());
}

#[test]
fn gm_i32_matches_wrapping_div() {
    let mut rng = StdRng::seed_from_u64(8);
    let values = values(&mut rng);
    let mut signed: Vec<i32> = divisors(&mut rng)
        .iter()
        .map(|d| *d as i32)
        .filter(|d| *d != 0)
        .collect();
    signed.extend(signed.clone().iter().map(|d| d.wrapping_neg()));
    signed.extend([i32::MIN, i32::MAX, -1, 1]);
    for d in signed {
        let gm = GmI32::new(d).unwrap();
        for &v in &values {
            for x in [v as i32, (v as i32).wrapping_neg(), i32::MIN, i32::MAX] {
                assert_eq!(gm.div(x), x.wrapping_div(d), "{x} / {d}");
            }
        }
    }
    assert!(GmI32::new(0).is_none());
}

#[test]
fn biased_gm_pmod_matches_comet() {
    let mut rng = StdRng::seed_from_u64(9);
    let hashes = values(&mut rng);
    let mut partitions: Vec<u32> = (1..=1000).collect();
    partitions.push(i32::MAX as u32);
    for n in partitions {
        let p = BiasedGmPmod::new(n).unwrap();
        for &h in &hashes {
            assert_eq!(
                p.pmod(h) as usize,
                comet_pmod(h, n as usize),
                "pmod({h}, {n})"
            );
        }
    }
}
