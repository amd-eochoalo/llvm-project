// Transform dialect library for testing vector.contract unrolling patterns.

module attributes {transform.with_named_sequence} {

  // Entry point for unrolling contract operations.
  transform.named_sequence @unroll_contract(
      %module_op: !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.unroll_contract
    } : !transform.any_op
    transform.yield
  }

}
