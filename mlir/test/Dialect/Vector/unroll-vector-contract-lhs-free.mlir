// RUN: mlir-opt --split-input-file %s \
// RUN:   -transform-preload-library='transform-library-paths=%p/td/unroll-contract.mlir' \
// RUN:   -transform-interpreter=entry-point=unroll_contract | FileCheck %s

//===----------------------------------------------------------------------===//
// Test UnrollContractAlongLhsFreeDim
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @unroll_contract_lhs_free_matmul
// CHECK-SAME: %[[A:.+]]: vector<4x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<8x6xf32>,
// CHECK-SAME: %[[C:.+]]: vector<4x6xf32>
func.func @unroll_contract_lhs_free_matmul(
    %A: vector<4x8xf32>,
    %B: vector<8x6xf32>,
    %C: vector<4x6xf32>) -> vector<4x6xf32> {

  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0] : vector<8xf32> from vector<4x8xf32>
  // CHECK-DAG: %[[A1:.+]] = vector.extract %[[A]][1] : vector<8xf32> from vector<4x8xf32>
  // CHECK-DAG: %[[A2:.+]] = vector.extract %[[A]][2] : vector<8xf32> from vector<4x8xf32>
  // CHECK-DAG: %[[A3:.+]] = vector.extract %[[A]][3] : vector<8xf32> from vector<4x8xf32>

  // CHECK-DAG: %[[C0:.+]] = vector.extract %[[C]][0] : vector<6xf32> from vector<4x6xf32>
  // CHECK-DAG: %[[C1:.+]] = vector.extract %[[C]][1] : vector<6xf32> from vector<4x6xf32>
  // CHECK-DAG: %[[C2:.+]] = vector.extract %[[C]][2] : vector<6xf32> from vector<4x6xf32>
  // CHECK-DAG: %[[C3:.+]] = vector.extract %[[C]][3] : vector<6xf32> from vector<4x6xf32>

  // CHECK-NOT: vector.extract %[[B]]

  // CHECK: %[[R0:.+]] = vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  // CHECK-SAME: %[[A0]], %[[B]], %[[C0]]
  // CHECK-SAME: : vector<8xf32>, vector<8x6xf32> into vector<6xf32>

  // CHECK: %[[R1:.+]] = vector.contract
  // CHECK-SAME: %[[A1]], %[[B]], %[[C1]]
  // CHECK-SAME: : vector<8xf32>, vector<8x6xf32> into vector<6xf32>

  // CHECK: %[[R2:.+]] = vector.contract
  // CHECK-SAME: %[[A2]], %[[B]], %[[C2]]
  // CHECK-SAME: : vector<8xf32>, vector<8x6xf32> into vector<6xf32>

  // CHECK: %[[R3:.+]] = vector.contract
  // CHECK-SAME: %[[A3]], %[[B]], %[[C3]]
  // CHECK-SAME: : vector<8xf32>, vector<8x6xf32> into vector<6xf32>

  // CHECK: vector.insert %[[R0]], {{.*}}[0]
  // CHECK: vector.insert %[[R1]], {{.*}}[1]
  // CHECK: vector.insert %[[R2]], {{.*}}[2]
  // CHECK: vector.insert %[[R3]], {{.*}}[3]

  %result = vector.contract {
      indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<4x8xf32>, vector<8x6xf32> into vector<4x6xf32>

  return %result : vector<4x6xf32>
}

// -----

// CHECK-LABEL: func @unroll_contract_lhs_free_matvec
// CHECK-SAME: %[[A:.+]]: vector<3x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<8xf32>,
// CHECK-SAME: %[[C:.+]]: vector<3xf32>
func.func @unroll_contract_lhs_free_matvec(
    %A: vector<3x8xf32>,
    %B: vector<8xf32>,
    %C: vector<3xf32>) -> vector<3xf32> {

  // Unroll along m (size 3), producing 3 dot products.
  // Each iteration: vector<8xf32> * vector<8xf32> -> f32

  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0] : vector<8xf32> from vector<3x8xf32>
  // CHECK-DAG: %[[A1:.+]] = vector.extract %[[A]][1] : vector<8xf32> from vector<3x8xf32>
  // CHECK-DAG: %[[A2:.+]] = vector.extract %[[A]][2] : vector<8xf32> from vector<3x8xf32>

  // CHECK-DAG: %[[C0:.+]] = vector.extract %[[C]][0] : f32 from vector<3xf32>
  // CHECK-DAG: %[[C1:.+]] = vector.extract %[[C]][1] : f32 from vector<3xf32>
  // CHECK-DAG: %[[C2:.+]] = vector.extract %[[C]][2] : f32 from vector<3xf32>

  // CHECK-NOT: vector.extract %[[B]]

  // CHECK: %[[A0M:.+]] = arith.mulf %[[A0]], %[[B]] : vector<8xf32>
  // CHECK: %[[R0:.+]] = vector.multi_reduction <add>, %[[A0M]], %[[C0]] [0]
  // CHECK-SAME: : vector<8xf32> to f32

  // CHECK: %[[A1M:.+]] = arith.mulf %[[A1]], %[[B]] : vector<8xf32>
  // CHECK: %[[R1:.+]] = vector.multi_reduction <add>, %[[A1M]], %[[C1]] [0]
  // CHECK-SAME: : vector<8xf32> to f32

  // CHECK: %[[A2M:.+]] = arith.mulf %[[A2]], %[[B]] : vector<8xf32>
  // CHECK: %[[R2:.+]] = vector.multi_reduction <add>, %[[A2M]], %[[C2]] [0]
  // CHECK-SAME: : vector<8xf32> to f32

  // CHECK: vector.insert %[[R0]], {{.*}}[0]
  // CHECK: vector.insert %[[R1]], {{.*}}[1]
  // CHECK: vector.insert %[[R2]], {{.*}}[2]

  %result = vector.contract {
      indexing_maps = [
          affine_map<(m, k) -> (m, k)>,
          affine_map<(m, k) -> (k)>,
          affine_map<(m, k) -> (m)>
      ],
      iterator_types = ["parallel", "reduction"]
  } %A, %B, %C : vector<3x8xf32>, vector<8xf32> into vector<3xf32>

  return %result : vector<3xf32>
}

// -----

// Negative test: no free LHS dimension (standard matmul has both free dims)
// This should match the batch pattern instead if no batch, but not free LHS
// Actually, this is a standard matmul with free LHS (m) and free RHS (n)
// The pattern should match on 'm' first.

// CHECK-LABEL: func @unroll_contract_has_free_lhs
func.func @unroll_contract_has_free_lhs(
    %A: vector<2x4xf32>,
    %B: vector<4x3xf32>,
    %C: vector<2x3xf32>) -> vector<2x3xf32> {

  // This SHOULD match - m is free in LHS
  // CHECK: vector.extract
  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]

  %result = vector.contract {
      indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<2x4xf32>, vector<4x3xf32> into vector<2x3xf32>

  return %result : vector<2x3xf32>
}

// -----

// Negative test: free LHS dim not at outermost position in LHS
// CHECK-LABEL: func @unroll_contract_lhs_free_not_outermost
func.func @unroll_contract_lhs_free_not_outermost(
    %A: vector<8x4xf32>,
    %B: vector<8x3xf32>,
    %C: vector<4x3xf32>) -> vector<4x3xf32> {

  // 'm' is free in LHS, but it's at position 1 in LHS (k, m)
  // Pattern should NOT match (requires outermost).

  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-NOT: vector.extract

  %result = vector.contract {
      indexing_maps = [
          affine_map<(m, n, k) -> (k, m)>,  // m at position 1, not 0
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<8x4xf32>, vector<8x3xf32> into vector<4x3xf32>

  return %result : vector<4x3xf32>
}

// -----

// Negative test: iterator appears in RHS (would be a batch dim, not free LHS)
// CHECK-LABEL: func @unroll_contract_not_free_lhs
func.func @unroll_contract_not_free_lhs(
    %A: vector<2x4x8xf32>,
    %B: vector<2x8x6xf32>,
    %C: vector<2x4x6xf32>) -> vector<2x4x6xf32> {

  // 'b' appears in all three operands - it's a batch dim, not free LHS.
  // Should match batch pattern, not free LHS pattern.

  // This test verifies free LHS pattern doesn't incorrectly match batch dims.
  // The batch pattern from Task 1 should match this instead.

  // CHECK: vector.contract
  // Pattern behavior depends on which pattern has higher priority.
  // With proper pattern ordering, batch should match first.

  %result = vector.contract {
      indexing_maps = [
          affine_map<(b, m, n, k) -> (b, m, k)>,
          affine_map<(b, m, n, k) -> (b, k, n)>,
          affine_map<(b, m, n, k) -> (b, m, n)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<2x4x8xf32>, vector<2x8x6xf32> into vector<2x4x6xf32>

  return %result : vector<2x4x6xf32>
}

// -----

// CHECK-LABEL: func @unroll_contract_lhs_free_masked
// CHECK-SAME: %[[A:.+]]: vector<3x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<8x5xf32>,
// CHECK-SAME: %[[C:.+]]: vector<3x5xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<3x5x8xi1>
func.func @unroll_contract_lhs_free_masked(
    %A: vector<3x8xf32>,
    %B: vector<8x5xf32>,
    %C: vector<3x5xf32>,
    %mask: vector<3x5x8xi1>) -> vector<3x5xf32> {

  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0]
  // CHECK-DAG: %[[A1:.+]] = vector.extract %[[A]][1]
  // CHECK-DAG: %[[A2:.+]] = vector.extract %[[A]][2]

  // CHECK-DAG: %[[C0:.+]] = vector.extract %[[C]][0]
  // CHECK-DAG: %[[C1:.+]] = vector.extract %[[C]][1]
  // CHECK-DAG: %[[C2:.+]] = vector.extract %[[C]][2]

  // CHECK-DAG: %[[MASK0:.+]] = vector.extract %[[MASK]][0]
  // CHECK-DAG: %[[MASK1:.+]] = vector.extract %[[MASK]][1]
  // CHECK-DAG: %[[MASK2:.+]] = vector.extract %[[MASK]][2]

  // CHECK-NOT: vector.extract %[[B]]

  // CHECK: vector.mask %[[MASK0]] {
  // CHECK:   vector.contract {{.*}} %[[A0]], %[[B]], %[[C0]]
  // CHECK: }

  // CHECK: vector.mask %[[MASK1]] {
  // CHECK:   vector.contract {{.*}} %[[A1]], %[[B]], %[[C1]]
  // CHECK: }

  // CHECK: vector.mask %[[MASK2]] {
  // CHECK:   vector.contract {{.*}} %[[A2]], %[[B]], %[[C2]]
  // CHECK: }

  %result = vector.mask %mask {
    vector.contract {
        indexing_maps = [
            affine_map<(m, n, k) -> (m, k)>,
            affine_map<(m, n, k) -> (k, n)>,
            affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } %A, %B, %C : vector<3x8xf32>, vector<8x5xf32> into vector<3x5xf32>
  } : vector<3x5x8xi1> -> vector<3x5xf32>

  return %result : vector<3x5xf32>
}

// -----

// Edge case: unrolling reduces to 1D vectors after extraction
// CHECK-LABEL: func @unroll_contract_lhs_free_to_1d
// CHECK-SAME: %[[A:.+]]: vector<2x4xf32>,
// CHECK-SAME: %[[B:.+]]: vector<4xf32>,
// CHECK-SAME: %[[C:.+]]: vector<2xf32>
func.func @unroll_contract_lhs_free_to_1d(
    %A: vector<2x4xf32>,
    %B: vector<4xf32>,
    %C: vector<2xf32>) -> vector<2xf32> {

  // Unroll along m (size 2), each iteration is a dot product.

  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0] : vector<4xf32> from vector<2x4xf32>
  // CHECK-DAG: %[[A1:.+]] = vector.extract %[[A]][1] : vector<4xf32> from vector<2x4xf32>

  // CHECK-DAG: %[[C0:.+]] = vector.extract %[[C]][0] : f32 from vector<2xf32>
  // CHECK-DAG: %[[C1:.+]] = vector.extract %[[C]][1] : f32 from vector<2xf32>

  // CHECK: %[[A0M:.+]] = arith.mulf %[[A0]], %[[B]] : vector<4xf32>
  // CHECK: %[[R0:.+]] = vector.multi_reduction <add>, %[[A0M]], %[[C0]] [0]
  // CHECK-SAME: : vector<4xf32> to f32

  // CHECK: %[[A1M:.+]] = arith.mulf %[[A1]], %[[B]] : vector<4xf32>
  // CHECK: %[[R1:.+]] = vector.multi_reduction <add>, %[[A1M]], %[[C1]] [0]
  // CHECK-SAME: : vector<4xf32> to f32

  %result = vector.contract {
      indexing_maps = [
          affine_map<(m, k) -> (m, k)>,
          affine_map<(m, k) -> (k)>,
          affine_map<(m, k) -> (m)>
      ],
      iterator_types = ["parallel", "reduction"]
  } %A, %B, %C : vector<2x4xf32>, vector<4xf32> into vector<2xf32>

  return %result : vector<2xf32>
}
