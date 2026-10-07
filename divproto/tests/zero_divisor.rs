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

use arrow_arith::numeric::{div, rem};
use arrow_array::{Int64Array, Scalar};

#[test]
fn scalar_zero_divisor_on_main() {
    let zero = Scalar::new(Int64Array::from(vec![0]));
    let all_null = Int64Array::from(vec![None, None]);
    let empty = Int64Array::from(Vec::<i64>::new());
    let valid = Int64Array::from(vec![Some(1), None]);
    println!(
        "all-null / 0 -> {:?}",
        div(&all_null, &zero).map(|a| a.len())
    );
    println!("empty / 0    -> {:?}", div(&empty, &zero).map(|a| a.len()));
    println!(
        "all-null % 0 -> {:?}",
        rem(&all_null, &zero).map(|a| a.len())
    );
    println!("[1,null] / 0 -> {:?}", div(&valid, &zero).map(|a| a.len()));
}
