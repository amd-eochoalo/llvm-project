// RUN: mlir-opt --split-input-file %s \
// RUN:   -transform-preload-library='transform-library-paths=%p/td/unroll-contract.mlir' \
// RUN:   -transform-interpreter=entry-point=unroll_contract | FileCheck %s

//===----------------------------------------------------------------------===//
// Test UnrollContractAlongBatchDim
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @unroll_contract_batch_matmul
// CHECK-SAME: %[[A:.+]]: vector<2x4x3xf32>,
// CHECK-SAME: %[[B:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[C:.+]]: vector<2x4x5xf32>
func.func @unroll_contract_batch_matmul(
    %A: vector<2x4x3xf32>,
    %B: vector<2x3x5xf32>,
    %C: vector<2x4x5xf32>) -> vector<2x4x5xf32> {

  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0] : vector<4x3xf32> from vector<2x4x3xf32>
  // CHECK-DAG: %[[A1:.+]] = vector.extract %[[A]][1] : vector<4x3xf32> from vector<2x4x3xf32>
  // CHECK-DAG: %[[B0:.+]] = vector.extract %[[B]][0] : vector<3x5xf32> from vector<2x3x5xf32>
  // CHECK-DAG: %[[B1:.+]] = vector.extract %[[B]][1] : vector<3x5xf32> from vector<2x3x5xf32>
  // CHECK-DAG: %[[C0:.+]] = vector.extract %[[C]][0] : vector<4x5xf32> from vector<2x4x5xf32>
  // CHECK-DAG: %[[C1:.+]] = vector.extract %[[C]][1] : vector<4x5xf32> from vector<2x4x5xf32>

  // CHECK: %[[R0:.+]] = vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-SAME: %[[A0]], %[[B0]], %[[C0]]
  // CHECK-SAME: : vector<4x3xf32>, vector<3x5xf32> into vector<4x5xf32>

  // CHECK: %[[R1:.+]] = vector.contract
  // CHECK-SAME: %[[A1]], %[[B1]], %[[C1]]
  // CHECK-SAME: : vector<4x3xf32>, vector<3x5xf32> into vector<4x5xf32>

  // CHECK: %[[INSERT0:.+]] = vector.insert %[[R0]], %{{.*}} [0] : vector<4x5xf32> into vector<2x4x5xf32>
  // CHECK: %[[INSERT1:.+]] = vector.insert %[[R1]], %[[INSERT0]] [1] : vector<4x5xf32> into vector<2x4x5xf32>

  %result = vector.contract {
      indexing_maps = [
          affine_map<(b, m, n, k) -> (b, m, k)>,
          affine_map<(b, m, n, k) -> (b, k, n)>,
          affine_map<(b, m, n, k) -> (b, m, n)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<2x4x3xf32>, vector<2x3x5xf32> into vector<2x4x5xf32>

  // CHECK: return %[[INSERT1]]
  return %result : vector<2x4x5xf32>
}

// -----

// Negative test: no batch dimension (standard matmul should not match)
// CHECK-LABEL: func @unroll_contract_no_batch
func.func @unroll_contract_no_batch(
    %A: vector<4x3xf32>,
    %B: vector<3x5xf32>,
    %C: vector<4x5xf32>) -> vector<4x5xf32> {

  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-NOT: vector.extract

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

// Negative test: batch dim not at outermost position
// CHECK-LABEL: func @unroll_contract_batch_not_outermost
func.func @unroll_contract_batch_not_outermost(
    %A: vector<4x2x3xf32>,
    %B: vector<3x2x5xf32>,
    %C: vector<4x2x5xf32>) -> vector<4x2x5xf32> {

  // Batch dim 'b' is at position 1 in lhs (m, b, k), not outermost.
  // Pattern should not match.

  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-NOT: vector.extract

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

  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0]
  // CHECK-DAG: %[[B0:.+]] = vector.extract %[[B]][0]
  // CHECK-DAG: %[[C0:.+]] = vector.extract %[[C]][0]
  // CHECK-DAG: %[[MASK0:.+]] = vector.extract %[[MASK]][0]

  // CHECK: vector.mask %[[MASK0]] {
  // CHECK:   vector.contract {{.*}} %[[A0]], %[[B0]], %[[C0]]
  // CHECK: }

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
