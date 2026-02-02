// RUN: mlir-opt --split-input-file %s \
// RUN:   -transform-preload-library='transform-library-paths=%p/td/unroll-contract.mlir' \
// RUN:   -transform-interpreter=entry-point=unroll_contract | FileCheck %s

//===----------------------------------------------------------------------===//
// Test UnrollContractAlongRhsFreeDim
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @unroll_contract_rhs_free_basic
// CHECK-SAME: %[[A:.+]]: vector<4x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<6x8xf32>,
// CHECK-SAME: %[[C:.+]]: vector<6x4xf32>
func.func @unroll_contract_rhs_free_basic(
    %A: vector<4x8xf32>,
    %B: vector<6x8xf32>,
    %C: vector<6x4xf32>) -> vector<6x4xf32> {

  // Patterns are applied iteratively:
  // 1. UnrollContractAlongRhsFreeDim unrolls along n (size 6)
  // 2. UnrollContractAlongLhsFreeDim unrolls along m (size 4) within each
  // Final result: 6*4 = 24 dot products (scalar contracts)

  // First, B is extracted along n
  // CHECK-DAG: %[[B0:.+]] = vector.extract %[[B]][0] : vector<8xf32> from vector<6x8xf32>
  // CHECK-DAG: %[[B1:.+]] = vector.extract %[[B]][1] : vector<8xf32> from vector<6x8xf32>
  // CHECK-DAG: %[[B2:.+]] = vector.extract %[[B]][2] : vector<8xf32> from vector<6x8xf32>
  // CHECK-DAG: %[[B3:.+]] = vector.extract %[[B]][3] : vector<8xf32> from vector<6x8xf32>
  // CHECK-DAG: %[[B4:.+]] = vector.extract %[[B]][4] : vector<8xf32> from vector<6x8xf32>
  // CHECK-DAG: %[[B5:.+]] = vector.extract %[[B]][5] : vector<8xf32> from vector<6x8xf32>

  // Then A is extracted along m (for each n slice)
  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0] : vector<8xf32> from vector<4x8xf32>
  // CHECK-DAG: %[[A1:.+]] = vector.extract %[[A]][1] : vector<8xf32> from vector<4x8xf32>
  // CHECK-DAG: %[[A2:.+]] = vector.extract %[[A]][2] : vector<8xf32> from vector<4x8xf32>
  // CHECK-DAG: %[[A3:.+]] = vector.extract %[[A]][3] : vector<8xf32> from vector<4x8xf32>

  // C is extracted along both n and m, producing scalars
  // We'll check a few representative ones
  // CHECK-DAG: vector.extract %[[C]][0, 0] : f32 from vector<6x4xf32>
  // CHECK-DAG: vector.extract %[[C]][0, 1] : f32 from vector<6x4xf32>

  // Final contracts are dot products: vector<8xf32> dot vector<8xf32> -> f32
  // Now lowered to multi_reduction
  // CHECK: arith.mulf
  // CHECK: vector.multi_reduction <add>
  // CHECK-SAME: [0] : vector<8xf32> to f32

  %result = vector.contract {
      indexing_maps = [
          affine_map<(n, m, k) -> (m, k)>,
          affine_map<(n, m, k) -> (n, k)>,
          affine_map<(n, m, k) -> (n, m)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<4x8xf32>, vector<6x8xf32> into vector<6x4xf32>

  return %result : vector<6x4xf32>
}

// -----

// CHECK-LABEL: func @unroll_contract_rhs_free_vecmat
// CHECK-SAME: %[[A:.+]]: vector<8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<3x8xf32>,
// CHECK-SAME: %[[C:.+]]: vector<3xf32>
func.func @unroll_contract_rhs_free_vecmat(
    %A: vector<8xf32>,
    %B: vector<3x8xf32>,
    %C: vector<3xf32>) -> vector<3xf32> {

  // Vector-matrix multiply: 1x8 * 8x3 -> 1x3
  // Unrolls along n (size 3), producing 3 dot products

  // CHECK-DAG: %[[B0:.+]] = vector.extract %[[B]][0] : vector<8xf32> from vector<3x8xf32>
  // CHECK-DAG: %[[B1:.+]] = vector.extract %[[B]][1] : vector<8xf32> from vector<3x8xf32>
  // CHECK-DAG: %[[B2:.+]] = vector.extract %[[B]][2] : vector<8xf32> from vector<3x8xf32>

  // CHECK-DAG: %[[C0:.+]] = vector.extract %[[C]][0] : f32 from vector<3xf32>
  // CHECK-DAG: %[[C1:.+]] = vector.extract %[[C]][1] : f32 from vector<3xf32>
  // CHECK-DAG: %[[C2:.+]] = vector.extract %[[C]][2] : f32 from vector<3xf32>

  // Final contracts are dot products
  // Now lowered to multi_reduction
  // CHECK: arith.mulf %[[A]], %[[B0]] : vector<8xf32>
  // CHECK: vector.multi_reduction <add>
  // CHECK-SAME: [0] : vector<8xf32> to f32

  // CHECK: arith.mulf %[[A]], %[[B1]] : vector<8xf32>
  // CHECK: vector.multi_reduction <add>
  // CHECK-SAME: [0] : vector<8xf32> to f32

  // CHECK: arith.mulf %[[A]], %[[B2]] : vector<8xf32>
  // CHECK: vector.multi_reduction <add>
  // CHECK-SAME: [0] : vector<8xf32> to f32

  %result = vector.contract {
      indexing_maps = [
          affine_map<(n, k) -> (k)>,
          affine_map<(n, k) -> (n, k)>,
          affine_map<(n, k) -> (n)>
      ],
      iterator_types = ["parallel", "reduction"]
  } %A, %B, %C : vector<8xf32>, vector<3x8xf32> into vector<3xf32>

  return %result : vector<3xf32>
}

// -----

// Smaller unroll factor for readability
// CHECK-LABEL: func @unroll_contract_rhs_free_small
// CHECK-SAME: %[[A:.+]]: vector<4x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<2x8xf32>,
// CHECK-SAME: %[[C:.+]]: vector<2x4xf32>
func.func @unroll_contract_rhs_free_small(
    %A: vector<4x8xf32>,
    %B: vector<2x8xf32>,
    %C: vector<2x4xf32>) -> vector<2x4xf32> {

  // Patterns are applied iteratively:
  // 1. UnrollContractAlongRhsFreeDim unrolls along n (size 2)
  // 2. UnrollContractAlongLhsFreeDim unrolls along m (size 4)
  // Final: 2*4 = 8 dot products

  // CHECK-DAG: %[[B0:.+]] = vector.extract %[[B]][0] : vector<8xf32> from vector<2x8xf32>
  // CHECK-DAG: %[[B1:.+]] = vector.extract %[[B]][1] : vector<8xf32> from vector<2x8xf32>

  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0] : vector<8xf32> from vector<4x8xf32>
  // CHECK-DAG: %[[A1:.+]] = vector.extract %[[A]][1] : vector<8xf32> from vector<4x8xf32>

  // CHECK: arith.mulf
  // CHECK: vector.multi_reduction <add>
  // CHECK-SAME: [0] : vector<8xf32> to f32

  %result = vector.contract {
      indexing_maps = [
          affine_map<(n, m, k) -> (m, k)>,
          affine_map<(n, m, k) -> (n, k)>,
          affine_map<(n, m, k) -> (n, m)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<4x8xf32>, vector<2x8xf32> into vector<2x4xf32>

  return %result : vector<2x4xf32>
}

// -----

// Negative test: free RHS dim not at outermost position in ACC
// CHECK-LABEL: func @unroll_contract_rhs_free_not_outermost_acc
func.func @unroll_contract_rhs_free_not_outermost_acc(
    %A: vector<4x8xf32>,
    %B: vector<6x8xf32>,
    %C: vector<4x6xf32>) -> vector<4x6xf32> {

  // 'n' is at position 0 in RHS (n, k), but at position 1 in ACC (m, n).
  // Pattern should NOT match because n is not outermost in ACC.
  // The free LHS pattern might match for m instead, leading to multi_reduction.

  // CHECK: arith.mulf
  // CHECK: vector.multi_reduction <add>

  %result = vector.contract {
      indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (n, k)>,
          affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<4x8xf32>, vector<6x8xf32> into vector<4x6xf32>

  return %result : vector<4x6xf32>
}

// -----

// CHECK-LABEL: func @unroll_contract_rhs_free_masked
// CHECK-SAME: %[[A:.+]]: vector<4x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<3x8xf32>,
// CHECK-SAME: %[[C:.+]]: vector<3x4xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<3x4x8xi1>
func.func @unroll_contract_rhs_free_masked(
    %A: vector<4x8xf32>,
    %B: vector<3x8xf32>,
    %C: vector<3x4xf32>,
    %mask: vector<3x4x8xi1>) -> vector<3x4xf32> {

  // Patterns are applied iteratively with masking:
  // 1. UnrollContractAlongRhsFreeDim unrolls along n (size 3)
  // 2. UnrollContractAlongLhsFreeDim unrolls along m (size 4)
  // Final: 3*4 = 12 masked dot products

  // CHECK-DAG: %[[B0:.+]] = vector.extract %[[B]][0]
  // CHECK-DAG: %[[B1:.+]] = vector.extract %[[B]][1]
  // CHECK-DAG: %[[B2:.+]] = vector.extract %[[B]][2]

  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0]
  // CHECK-DAG: %[[A1:.+]] = vector.extract %[[A]][1]

  // Masks are extracted too
  // CHECK-DAG: vector.extract %[[MASK]][0, 0]

  // Final masked dot products
  // The mask is applied via arith.select on the product, then reduced
  // CHECK: arith.mulf
  // CHECK: arith.select
  // CHECK: vector.multi_reduction <add>
  // CHECK-SAME: [0] : vector<8xf32> to f32

  %result = vector.mask %mask {
    vector.contract {
        indexing_maps = [
            affine_map<(n, m, k) -> (m, k)>,
            affine_map<(n, m, k) -> (n, k)>,
            affine_map<(n, m, k) -> (n, m)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } %A, %B, %C : vector<4x8xf32>, vector<3x8xf32> into vector<3x4xf32>
  } : vector<3x4x8xi1> -> vector<3x4xf32>

  return %result : vector<3x4xf32>
}

// -----

// Edge case: unrolling produces contracts with single reduction iterator (dot products)
// CHECK-LABEL: func @unroll_contract_rhs_free_to_dot
// CHECK-SAME: %[[A:.+]]: vector<4xf32>,
// CHECK-SAME: %[[B:.+]]: vector<2x4xf32>,
// CHECK-SAME: %[[C:.+]]: vector<2xf32>
func.func @unroll_contract_rhs_free_to_dot(
    %A: vector<4xf32>,
    %B: vector<2x4xf32>,
    %C: vector<2xf32>) -> vector<2xf32> {

  // Unroll along n (size 2), each iteration is already a dot product
  // No further unrolling occurs

  // CHECK-DAG: %[[B0:.+]] = vector.extract %[[B]][0] : vector<4xf32> from vector<2x4xf32>
  // CHECK-DAG: %[[B1:.+]] = vector.extract %[[B]][1] : vector<4xf32> from vector<2x4xf32>

  // CHECK-DAG: %[[C0:.+]] = vector.extract %[[C]][0] : f32 from vector<2xf32>
  // CHECK-DAG: %[[C1:.+]] = vector.extract %[[C]][1] : f32 from vector<2xf32>

  // CHECK: %[[A0M:.+]] = arith.mulf %[[A]], %[[B0]] : vector<4xf32>
  // CHECK: %[[R0:.+]] = vector.multi_reduction <add>, %[[A0M]], %[[C0]] [0]
  // CHECK-SAME: : vector<4xf32> to f32

  // CHECK: %[[A1M:.+]] = arith.mulf %[[A]], %[[B1]] : vector<4xf32>
  // CHECK: %[[R1:.+]] = vector.multi_reduction <add>, %[[A1M]], %[[C1]] [0]
  // CHECK-SAME: : vector<4xf32> to f32

  %result = vector.contract {
      indexing_maps = [
          affine_map<(n, k) -> (k)>,
          affine_map<(n, k) -> (n, k)>,
          affine_map<(n, k) -> (n)>
      ],
      iterator_types = ["parallel", "reduction"]
  } %A, %B, %C : vector<4xf32>, vector<2x4xf32> into vector<2xf32>

  return %result : vector<2xf32>
}

// -----

// Edge case: unrolling with size 1 (degenerate case)
// CHECK-LABEL: func @unroll_contract_rhs_free_size_one
// CHECK-SAME: %[[A:.+]]: vector<4x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<1x8xf32>,
// CHECK-SAME: %[[C:.+]]: vector<1x4xf32>
func.func @unroll_contract_rhs_free_size_one(
    %A: vector<4x8xf32>,
    %B: vector<1x8xf32>,
    %C: vector<1x4xf32>) -> vector<1x4xf32> {

  // Unroll along n (size 1), then along m (size 4)
  // Final: 1*4 = 4 dot products

  // CHECK-DAG: %[[B0:.+]] = vector.extract %[[B]][0] : vector<8xf32> from vector<1x8xf32>

  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0] : vector<8xf32> from vector<4x8xf32>

  // CHECK: arith.mulf
  // CHECK: vector.multi_reduction <add>
  // CHECK-SAME: [0] : vector<8xf32> to f32

  %result = vector.contract {
      indexing_maps = [
          affine_map<(n, m, k) -> (m, k)>,
          affine_map<(n, m, k) -> (n, k)>,
          affine_map<(n, m, k) -> (n, m)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<4x8xf32>, vector<1x8xf32> into vector<1x4xf32>

  return %result : vector<1x4xf32>
}
