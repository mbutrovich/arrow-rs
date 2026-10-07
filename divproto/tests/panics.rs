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
use divproto::{GmI128, GmI256};
use std::hint::black_box;
use std::panic::catch_unwind;

fn outcome<T: std::fmt::Debug>(f: impl FnOnce() -> T + std::panic::UnwindSafe) -> String {
    match catch_unwind(f) {
        Ok(v) => format!("Ok({v:?})"),
        Err(e) => format!(
            "panic({})",
            e.downcast_ref::<String>()
                .cloned()
                .or(e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default()
        ),
    }
}

#[test]
fn panic_semantics() {
    std::panic::set_hook(Box::new(|_| {}));
    let mode = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let (min, m1, z) = (black_box(i128::MIN), black_box(-1i128), black_box(0i128));
    println!(
        "[{mode}] i128::MIN / -1      -> {}",
        outcome(move || min / m1)
    );
    println!(
        "[{mode}] i128::MIN % -1      -> {}",
        outcome(move || min % m1)
    );
    println!("[{mode}] 1i128 / 0           -> {}", outcome(move || 1 / z));
    println!(
        "[{mode}] i32::MIN % -1       -> {}",
        outcome(|| black_box(i32::MIN) % black_box(-1i32))
    );
    let (imin, im1, iz) = (
        black_box(i256::MIN),
        black_box(i256::MINUS_ONE),
        black_box(i256::ZERO),
    );
    println!(
        "[{mode}] i256::MIN / -1 (op) -> {}",
        outcome(move || imin / im1)
    );
    println!(
        "[{mode}] i256::MIN % -1 (op) -> {}",
        outcome(move || imin % im1)
    );
    println!(
        "[{mode}] i256 1 / 0 (op)     -> {}",
        outcome(move || i256::ONE / iz)
    );
    println!(
        "[{mode}] gm i128 MIN / -1    -> {}",
        outcome(move || GmI128::<false>::new(-1).unwrap().div_rem(min))
    );
    println!(
        "[{mode}] gm i128 new(0)      -> {}",
        outcome(|| GmI128::<false>::new(0).is_none())
    );
    println!(
        "[{mode}] gm i256 MIN / -1    -> {}",
        outcome(move || GmI256::new(i256::MINUS_ONE).unwrap().div_rem(imin))
    );
}
