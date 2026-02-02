// RUN: mlir-opt --split-input-file %s \
// RUN:   -transform-preload-library='transform-library-paths=%p/td/unroll-contract.mlir' \
// RUN:   -transform-interpreter=entry-point=unroll_contract | FileCheck %s

//===----------------------------------------------------------------------===//
// Test UnrollContractAlongBatchDim + UnrollContractAlongLhsFreeDim
//===----------------------------------------------------------------------===//

// This test verifies that a batched matmul is fully unrolled:
// 1. First, UnrollContractAlongBatchDim unrolls along the batch dimension (b=2)
// 2. Then, UnrollContractAlongLhsFreeDim unrolls along the free LHS dimension (m=4)
// The final contracts have iterator_types = ["parallel", "reduction"]

// CHECK-LABEL: func @unroll_contract_batch_matmul
// CHECK-SAME: %[[A:.+]]: vector<2x4x3xf32>,
// CHECK-SAME: %[[B:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[C:.+]]: vector<2x4x5xf32>
func.func @unroll_contract_batch_matmul(
    %A: vector<2x4x3xf32>,
    %B: vector<2x3x5xf32>,
    %C: vector<2x4x5xf32>) -> vector<2x4x5xf32> {

  // After batch unrolling, B is extracted per batch.
  // After LHS free dim unrolling, B slices are reused across m iterations.
  // CHECK-DAG: %[[B0:.+]] = vector.extract %[[B]][0] : vector<3x5xf32> from vector<2x3x5xf32>
  // CHECK-DAG: %[[B1:.+]] = vector.extract %[[B]][1] : vector<3x5xf32> from vector<2x3x5xf32>

  // Extracts for batch 0, m=0,1,2,3
  // CHECK-DAG: %[[A00:.+]] = vector.extract %[[A]][0, 0] : vector<3xf32> from vector<2x4x3xf32>
  // CHECK-DAG: %[[C00:.+]] = vector.extract %[[C]][0, 0] : vector<5xf32> from vector<2x4x5xf32>
  // CHECK-DAG: %[[A01:.+]] = vector.extract %[[A]][0, 1] : vector<3xf32> from vector<2x4x3xf32>
  // CHECK-DAG: %[[C01:.+]] = vector.extract %[[C]][0, 1] : vector<5xf32> from vector<2x4x5xf32>
  // CHECK-DAG: %[[A02:.+]] = vector.extract %[[A]][0, 2] : vector<3xf32> from vector<2x4x3xf32>
  // CHECK-DAG: %[[C02:.+]] = vector.extract %[[C]][0, 2] : vector<5xf32> from vector<2x4x5xf32>
  // CHECK-DAG: %[[A03:.+]] = vector.extract %[[A]][0, 3] : vector<3xf32> from vector<2x4x3xf32>
  // CHECK-DAG: %[[C03:.+]] = vector.extract %[[C]][0, 3] : vector<5xf32> from vector<2x4x5xf32>

  // Final contracts: vector<3xf32>, vector<3x5xf32> -> vector<5xf32>
  // These have iterator_types = ["parallel", "reduction"]
  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  // CHECK-SAME: %[[A00]], %[[B0]], %[[C00]]
  // CHECK-SAME: : vector<3xf32>, vector<3x5xf32> into vector<5xf32>

  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  // CHECK-SAME: %[[A01]], %[[B0]], %[[C01]]

  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  // CHECK-SAME: %[[A02]], %[[B0]], %[[C02]]

  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  // CHECK-SAME: %[[A03]], %[[B0]], %[[C03]]

  %result = vector.contract {
      indexing_maps = [
          affine_map<(b, m, n, k) -> (b, m, k)>,
          affine_map<(b, m, n, k) -> (b, k, n)>,
          affine_map<(b, m, n, k) -> (b, m, n)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<2x4x3xf32>, vector<2x3x5xf32> into vector<2x4x5xf32>

  return %result : vector<2x4x5xf32>
}

// -----

// Test UnrollContractAlongLhsFreeDim on standard matmul (no batch dimension).
// The pattern unrolls along the free LHS dimension 'm' (size 4).
// Note: This was previously a "negative test" for batch unrolling, but now
// UnrollContractAlongLhsFreeDim matches and transforms it.

// CHECK-LABEL: func @unroll_contract_lhs_free_dim
// CHECK-SAME: %[[A:.+]]: vector<4x3xf32>,
// CHECK-SAME: %[[B:.+]]: vector<3x5xf32>,
// CHECK-SAME: %[[C:.+]]: vector<4x5xf32>
func.func @unroll_contract_lhs_free_dim(
    %A: vector<4x3xf32>,
    %B: vector<3x5xf32>,
    %C: vector<4x5xf32>) -> vector<4x5xf32> {

  // B is reused (not extracted) since 'm' doesn't appear in B's indexing map.
  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0] : vector<3xf32> from vector<4x3xf32>
  // CHECK-DAG: %[[C0:.+]] = vector.extract %[[C]][0] : vector<5xf32> from vector<4x5xf32>
  // CHECK-DAG: %[[A1:.+]] = vector.extract %[[A]][1] : vector<3xf32> from vector<4x3xf32>
  // CHECK-DAG: %[[C1:.+]] = vector.extract %[[C]][1] : vector<5xf32> from vector<4x5xf32>
  // CHECK-DAG: %[[A2:.+]] = vector.extract %[[A]][2] : vector<3xf32> from vector<4x3xf32>
  // CHECK-DAG: %[[C2:.+]] = vector.extract %[[C]][2] : vector<5xf32> from vector<4x5xf32>
  // CHECK-DAG: %[[A3:.+]] = vector.extract %[[A]][3] : vector<3xf32> from vector<4x3xf32>
  // CHECK-DAG: %[[C3:.+]] = vector.extract %[[C]][3] : vector<5xf32> from vector<4x5xf32>

  // Contracts with B reused (not sliced):
  // CHECK: vector.contract {{.*}} %[[A0]], %[[B]], %[[C0]] : vector<3xf32>, vector<3x5xf32> into vector<5xf32>
  // CHECK: vector.contract {{.*}} %[[A1]], %[[B]], %[[C1]] : vector<3xf32>, vector<3x5xf32> into vector<5xf32>
  // CHECK: vector.contract {{.*}} %[[A2]], %[[B]], %[[C2]] : vector<3xf32>, vector<3x5xf32> into vector<5xf32>
  // CHECK: vector.contract {{.*}} %[[A3]], %[[B]], %[[C3]] : vector<3xf32>, vector<3x5xf32> into vector<5xf32>

  %result = vector.contract {
      indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<4x3xf32>, vector<3x5xf32> into vector<4x5xf32>

  return %result : vector<4x5xf32>
}

// -----

// Test: Batch dim not at outermost position, but LHS free dim IS outermost.
// UnrollContractAlongBatchDim does NOT match (batch 'b' is at position 1).
// UnrollContractAlongLhsFreeDim DOES match ('m' is at position 0 in LHS/ACC,
// and 'm' does not appear in RHS).

// CHECK-LABEL: func @unroll_contract_lhs_free_not_batch
// CHECK-SAME: %[[A:.+]]: vector<4x2x3xf32>,
// CHECK-SAME: %[[B:.+]]: vector<3x2x5xf32>,
// CHECK-SAME: %[[C:.+]]: vector<4x2x5xf32>
func.func @unroll_contract_lhs_free_not_batch(
    %A: vector<4x2x3xf32>,
    %B: vector<3x2x5xf32>,
    %C: vector<4x2x5xf32>) -> vector<4x2x5xf32> {

  // 'm' is unrolled (size 4). B is reused since 'm' doesn't appear in its map.
  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0] : vector<2x3xf32> from vector<4x2x3xf32>
  // CHECK-DAG: %[[C0:.+]] = vector.extract %[[C]][0] : vector<2x5xf32> from vector<4x2x5xf32>
  // CHECK-DAG: %[[A1:.+]] = vector.extract %[[A]][1] : vector<2x3xf32> from vector<4x2x3xf32>
  // CHECK-DAG: %[[C1:.+]] = vector.extract %[[C]][1] : vector<2x5xf32> from vector<4x2x5xf32>
  // CHECK-DAG: %[[A2:.+]] = vector.extract %[[A]][2] : vector<2x3xf32> from vector<4x2x3xf32>
  // CHECK-DAG: %[[C2:.+]] = vector.extract %[[C]][2] : vector<2x5xf32> from vector<4x2x5xf32>
  // CHECK-DAG: %[[A3:.+]] = vector.extract %[[A]][3] : vector<2x3xf32> from vector<4x2x3xf32>
  // CHECK-DAG: %[[C3:.+]] = vector.extract %[[C]][3] : vector<2x5xf32> from vector<4x2x5xf32>

  // The resulting contracts still have the batch dim 'b' (now at iterator position 0
  // after 'm' was removed), but it's not at outermost in the operands.
  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-SAME: %[[A0]], %[[B]], %[[C0]]
  // CHECK-SAME: : vector<2x3xf32>, vector<3x2x5xf32> into vector<2x5xf32>

  // CHECK: vector.contract {{.*}} %[[A1]], %[[B]], %[[C1]]
  // CHECK: vector.contract {{.*}} %[[A2]], %[[B]], %[[C2]]
  // CHECK: vector.contract {{.*}} %[[A3]], %[[B]], %[[C3]]

  %result = vector.contract {
      indexing_maps = [
          affine_map<(m, b, n, k) -> (m, b, k)>,
          affine_map<(m, b, n, k) -> (k, b, n)>,
          affine_map<(m, b, n, k) -> (m, b, n)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<4x2x3xf32>, vector<3x2x5xf32> into vector<4x2x5xf32>

  return %result : vector<4x2x5xf32>
}

// -----

// Test masked batch matmul with full unrolling (batch + LHS free dim).

// CHECK-LABEL: func @unroll_contract_batch_masked
// CHECK-SAME: %[[A:.+]]: vector<2x4x3xf32>,
// CHECK-SAME: %[[B:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[C:.+]]: vector<2x4x5xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<2x4x5x3xi1>
func.func @unroll_contract_batch_masked(
    %A: vector<2x4x3xf32>,
    %B: vector<2x3x5xf32>,
    %C: vector<2x4x5xf32>,
    %mask: vector<2x4x5x3xi1>) -> vector<2x4x5xf32> {

  // B extracted per batch (batch unrolling)
  // CHECK-DAG: %[[B0:.+]] = vector.extract %[[B]][0] : vector<3x5xf32> from vector<2x3x5xf32>

  // Extracts for batch 0, with mask slices (LHS free dim unrolling)
  // CHECK-DAG: %[[A00:.+]] = vector.extract %[[A]][0, 0] : vector<3xf32> from vector<2x4x3xf32>
  // CHECK-DAG: %[[C00:.+]] = vector.extract %[[C]][0, 0] : vector<5xf32> from vector<2x4x5xf32>
  // CHECK-DAG: %[[MASK00:.+]] = vector.extract %[[MASK]][0, 0] : vector<5x3xi1> from vector<2x4x5x3xi1>

  // Masked contracts with B0 reused
  // CHECK: vector.mask %[[MASK00]] {
  // CHECK:   vector.contract {{.*}} %[[A00]], %[[B0]], %[[C00]]
  // CHECK:   : vector<3xf32>, vector<3x5xf32> into vector<5xf32>
  // CHECK: } : vector<5x3xi1> -> vector<5xf32>

  %result = vector.mask %mask {
    vector.contract {
        indexing_maps = [
            affine_map<(b, m, n, k) -> (b, m, k)>,
            affine_map<(b, m, n, k) -> (b, k, n)>,
            affine_map<(b, m, n, k) -> (b, m, n)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } %A, %B, %C : vector<2x4x3xf32>, vector<2x3x5xf32> into vector<2x4x5xf32>
  } : vector<2x4x5x3xi1> -> vector<2x4x5xf32>

  return %result : vector<2x4x5xf32>
}
