// RUN: mlir-opt --split-input-file %s \
// RUN:   -transform-preload-library='transform-library-paths=%p/td/unroll-contract.mlir' \
// RUN:   -transform-interpreter=entry-point=unroll_contract | FileCheck %s

//===----------------------------------------------------------------------===//
// Test PureReductionContractToMultiReduction
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @pure_reduction_1d_dot_product(
// CHECK-SAME: %[[LHS:.+]]: vector<8xf32>, %[[RHS:.+]]: vector<8xf32>, %[[ACC:.+]]: f32
func.func @pure_reduction_1d_dot_product(
    %lhs: vector<8xf32>, %rhs: vector<8xf32>, %acc: f32) -> f32 {

  // 1-D dot product: single reduction iterator
  // vector<8xf32> * vector<8xf32> -> f32

  // CHECK: %[[PROD:.+]] = arith.mulf %[[LHS]], %[[RHS]] : vector<8xf32>
  // CHECK: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[PROD]], %[[ACC]] [0]
  // CHECK-SAME: : vector<8xf32> to f32

  %result = vector.contract {
      indexing_maps = [
          affine_map<(k) -> (k)>,
          affine_map<(k) -> (k)>,
          affine_map<(k) -> ()>
      ],
      iterator_types = ["reduction"]
  } %lhs, %rhs, %acc : vector<8xf32>, vector<8xf32> into f32

  // CHECK: return %[[RESULT]]
  return %result : f32
}

// -----

// CHECK-LABEL: func @pure_reduction_2d(
// CHECK-SAME: %[[LHS:.+]]: vector<4x8xf32>, %[[RHS:.+]]: vector<4x8xf32>, %[[ACC:.+]]: f32
func.func @pure_reduction_2d(
    %lhs: vector<4x8xf32>, %rhs: vector<4x8xf32>, %acc: f32) -> f32 {

  // 2-D reduction: two reduction iterators
  // vector<4x8xf32> * vector<4x8xf32> -> f32

  // CHECK: %[[PROD:.+]] = arith.mulf %[[LHS]], %[[RHS]] : vector<4x8xf32>
  // CHECK: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[PROD]], %[[ACC]] [0, 1]
  // CHECK-SAME: : vector<4x8xf32> to f32

  %result = vector.contract {
      indexing_maps = [
          affine_map<(k0, k1) -> (k0, k1)>,
          affine_map<(k0, k1) -> (k0, k1)>,
          affine_map<(k0, k1) -> ()>
      ],
      iterator_types = ["reduction", "reduction"]
  } %lhs, %rhs, %acc : vector<4x8xf32>, vector<4x8xf32> into f32

  // CHECK: return %[[RESULT]]
  return %result : f32
}

// -----

// CHECK-LABEL: func @pure_reduction_3d(
// CHECK-SAME: %[[LHS:.+]]: vector<2x4x8xf32>, %[[RHS:.+]]: vector<2x4x8xf32>, %[[ACC:.+]]: f32
func.func @pure_reduction_3d(
    %lhs: vector<2x4x8xf32>, %rhs: vector<2x4x8xf32>, %acc: f32) -> f32 {

  // 3-D reduction: three reduction iterators
  // vector<2x4x8xf32> * vector<2x4x8xf32> -> f32

  // CHECK: %[[PROD:.+]] = arith.mulf %[[LHS]], %[[RHS]] : vector<2x4x8xf32>
  // CHECK: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[PROD]], %[[ACC]] [0, 1, 2]
  // CHECK-SAME: : vector<2x4x8xf32> to f32

  %result = vector.contract {
      indexing_maps = [
          affine_map<(k0, k1, k2) -> (k0, k1, k2)>,
          affine_map<(k0, k1, k2) -> (k0, k1, k2)>,
          affine_map<(k0, k1, k2) -> ()>
      ],
      iterator_types = ["reduction", "reduction", "reduction"]
  } %lhs, %rhs, %acc : vector<2x4x8xf32>, vector<2x4x8xf32> into f32

  // CHECK: return %[[RESULT]]
  return %result : f32
}

// -----

// CHECK-LABEL: func @pure_reduction_integer(
// CHECK-SAME: %[[LHS:.+]]: vector<8xi32>, %[[RHS:.+]]: vector<8xi32>, %[[ACC:.+]]: i32
func.func @pure_reduction_integer(
    %lhs: vector<8xi32>, %rhs: vector<8xi32>, %acc: i32) -> i32 {

  // Integer dot product

  // CHECK: %[[PROD:.+]] = arith.muli %[[LHS]], %[[RHS]] : vector<8xi32>
  // CHECK: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[PROD]], %[[ACC]] [0]
  // CHECK-SAME: : vector<8xi32> to i32

  %result = vector.contract {
      indexing_maps = [
          affine_map<(k) -> (k)>,
          affine_map<(k) -> (k)>,
          affine_map<(k) -> ()>
      ],
      iterator_types = ["reduction"]
  } %lhs, %rhs, %acc : vector<8xi32>, vector<8xi32> into i32

  // CHECK: return %[[RESULT]]
  return %result : i32
}

// -----

// CHECK-LABEL: func @pure_reduction_2d_integer(
// CHECK-SAME: %[[LHS:.+]]: vector<4x8xi32>, %[[RHS:.+]]: vector<4x8xi32>, %[[ACC:.+]]: i32
func.func @pure_reduction_2d_integer(
    %lhs: vector<4x8xi32>, %rhs: vector<4x8xi32>, %acc: i32) -> i32 {

  // 2-D integer reduction

  // CHECK: %[[PROD:.+]] = arith.muli %[[LHS]], %[[RHS]] : vector<4x8xi32>
  // CHECK: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[PROD]], %[[ACC]] [0, 1]
  // CHECK-SAME: : vector<4x8xi32> to i32

  %result = vector.contract {
      indexing_maps = [
          affine_map<(k0, k1) -> (k0, k1)>,
          affine_map<(k0, k1) -> (k0, k1)>,
          affine_map<(k0, k1) -> ()>
      ],
      iterator_types = ["reduction", "reduction"]
  } %lhs, %rhs, %acc : vector<4x8xi32>, vector<4x8xi32> into i32

  // CHECK: return %[[RESULT]]
  return %result : i32
}

// -----

// CHECK-LABEL: func @pure_reduction_zero_acc(
// CHECK-SAME: %[[LHS:.+]]: vector<8xf32>, %[[RHS:.+]]: vector<8xf32>
func.func @pure_reduction_zero_acc(
    %lhs: vector<8xf32>, %rhs: vector<8xf32>) -> f32 {

  // Dot product with zero accumulator (common case)

  // CHECK: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[PROD:.+]] = arith.mulf %[[LHS]], %[[RHS]] : vector<8xf32>
  // CHECK: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[PROD]], %[[ZERO]] [0]
  // CHECK-SAME: : vector<8xf32> to f32

  %zero = arith.constant 0.0 : f32
  %result = vector.contract {
      indexing_maps = [
          affine_map<(k) -> (k)>,
          affine_map<(k) -> (k)>,
          affine_map<(k) -> ()>
      ],
      iterator_types = ["reduction"]
  } %lhs, %rhs, %zero : vector<8xf32>, vector<8xf32> into f32

  // CHECK: return %[[RESULT]]
  return %result : f32
}

// -----

// CHECK-LABEL: func @pure_reduction_size_one(
// CHECK-SAME: %[[LHS:.+]]: vector<1xf32>, %[[RHS:.+]]: vector<1xf32>, %[[ACC:.+]]: f32
func.func @pure_reduction_size_one(
    %lhs: vector<1xf32>, %rhs: vector<1xf32>, %acc: f32) -> f32 {

  // Edge case: single element reduction

  // CHECK: %[[PROD:.+]] = arith.mulf %[[LHS]], %[[RHS]] : vector<1xf32>
  // CHECK: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[PROD]], %[[ACC]] [0]
  // CHECK-SAME: : vector<1xf32> to f32

  %result = vector.contract {
      indexing_maps = [
          affine_map<(k) -> (k)>,
          affine_map<(k) -> (k)>,
          affine_map<(k) -> ()>
      ],
      iterator_types = ["reduction"]
  } %lhs, %rhs, %acc : vector<1xf32>, vector<1xf32> into f32

  // CHECK: return %[[RESULT]]
  return %result : f32
}

// -----

// CHECK-LABEL: func @pure_reduction_masked(
// CHECK-SAME: %[[LHS:.+]]: vector<8xf32>, %[[RHS:.+]]: vector<8xf32>, %[[ACC:.+]]: f32, %[[MASK:.+]]: vector<8xi1>
func.func @pure_reduction_masked(
    %lhs: vector<8xf32>, %rhs: vector<8xf32>, %acc: f32, %mask: vector<8xi1>) -> f32 {

  // Masked dot product: only reduce where mask is true

  // CHECK-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<8xf32>
  // CHECK: %[[PROD:.+]] = arith.mulf %[[LHS]], %[[RHS]] : vector<8xf32>
  // CHECK: %[[MASKED_PROD:.+]] = arith.select %[[MASK]], %[[PROD]], %[[ZERO]]
  // CHECK: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[MASKED_PROD]], %[[ACC]] [0]
  // CHECK-SAME: : vector<8xf32> to f32

  %result = vector.mask %mask {
    vector.contract {
        indexing_maps = [
            affine_map<(k) -> (k)>,
            affine_map<(k) -> (k)>,
            affine_map<(k) -> ()>
        ],
        iterator_types = ["reduction"]
    } %lhs, %rhs, %acc : vector<8xf32>, vector<8xf32> into f32
  } : vector<8xi1> -> f32

  // CHECK: return %[[RESULT]]
  return %result : f32
}

// -----

// CHECK-LABEL: func @pure_reduction_2d_masked(
// CHECK-SAME: %[[LHS:.+]]: vector<4x8xf32>, %[[RHS:.+]]: vector<4x8xf32>, %[[ACC:.+]]: f32, %[[MASK:.+]]: vector<4x8xi1>
func.func @pure_reduction_2d_masked(
    %lhs: vector<4x8xf32>, %rhs: vector<4x8xf32>, %acc: f32, %mask: vector<4x8xi1>) -> f32 {

  // Masked 2-D reduction

  // CHECK-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<4x8xf32>
  // CHECK: %[[PROD:.+]] = arith.mulf %[[LHS]], %[[RHS]] : vector<4x8xf32>
  // CHECK: %[[MASKED_PROD:.+]] = arith.select %[[MASK]], %[[PROD]], %[[ZERO]]
  // CHECK: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[MASKED_PROD]], %[[ACC]] [0, 1]
  // CHECK-SAME: : vector<4x8xf32> to f32

  %result = vector.mask %mask {
    vector.contract {
        indexing_maps = [
            affine_map<(k0, k1) -> (k0, k1)>,
            affine_map<(k0, k1) -> (k0, k1)>,
            affine_map<(k0, k1) -> ()>
        ],
        iterator_types = ["reduction", "reduction"]
    } %lhs, %rhs, %acc : vector<4x8xf32>, vector<4x8xf32> into f32
  } : vector<4x8xi1> -> f32

  // CHECK: return %[[RESULT]]
  return %result : f32
}

// -----

// Edge case: f16 element type
// CHECK-LABEL: func @pure_reduction_f16(
// CHECK-SAME: %[[LHS:.+]]: vector<8xf16>, %[[RHS:.+]]: vector<8xf16>, %[[ACC:.+]]: f16
func.func @pure_reduction_f16(
    %lhs: vector<8xf16>, %rhs: vector<8xf16>, %acc: f16) -> f16 {

  // CHECK: %[[PROD:.+]] = arith.mulf %[[LHS]], %[[RHS]] : vector<8xf16>
  // CHECK: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[PROD]], %[[ACC]] [0]
  // CHECK-SAME: : vector<8xf16> to f16

  %result = vector.contract {
      indexing_maps = [
          affine_map<(k) -> (k)>,
          affine_map<(k) -> (k)>,
          affine_map<(k) -> ()>
      ],
      iterator_types = ["reduction"]
  } %lhs, %rhs, %acc : vector<8xf16>, vector<8xf16> into f16

  // CHECK: return %[[RESULT]]
  return %result : f16
}

// -----

// Edge case: bf16 element type
// CHECK-LABEL: func @pure_reduction_bf16(
// CHECK-SAME: %[[LHS:.+]]: vector<8xbf16>, %[[RHS:.+]]: vector<8xbf16>, %[[ACC:.+]]: bf16
func.func @pure_reduction_bf16(
    %lhs: vector<8xbf16>, %rhs: vector<8xbf16>, %acc: bf16) -> bf16 {

  // CHECK: %[[PROD:.+]] = arith.mulf %[[LHS]], %[[RHS]] : vector<8xbf16>
  // CHECK: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[PROD]], %[[ACC]] [0]
  // CHECK-SAME: : vector<8xbf16> to bf16

  %result = vector.contract {
      indexing_maps = [
          affine_map<(k) -> (k)>,
          affine_map<(k) -> (k)>,
          affine_map<(k) -> ()>
      ],
      iterator_types = ["reduction"]
  } %lhs, %rhs, %acc : vector<8xbf16>, vector<8xbf16> into bf16

  // CHECK: return %[[RESULT]]
  return %result : bf16
}

// -----

// Edge case: i8 element type
// CHECK-LABEL: func @pure_reduction_i8(
// CHECK-SAME: %[[LHS:.+]]: vector<8xi8>, %[[RHS:.+]]: vector<8xi8>, %[[ACC:.+]]: i8
func.func @pure_reduction_i8(
    %lhs: vector<8xi8>, %rhs: vector<8xi8>, %acc: i8) -> i8 {

  // CHECK: %[[PROD:.+]] = arith.muli %[[LHS]], %[[RHS]] : vector<8xi8>
  // CHECK: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[PROD]], %[[ACC]] [0]
  // CHECK-SAME: : vector<8xi8> to i8

  %result = vector.contract {
      indexing_maps = [
          affine_map<(k) -> (k)>,
          affine_map<(k) -> (k)>,
          affine_map<(k) -> ()>
      ],
      iterator_types = ["reduction"]
  } %lhs, %rhs, %acc : vector<8xi8>, vector<8xi8> into i8

  // CHECK: return %[[RESULT]]
  return %result : i8
}
