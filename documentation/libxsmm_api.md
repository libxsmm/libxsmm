## Application Programming Interface

### Overview

LIBXSMM splits each call into `dispatch`, which JIT compiles the kernel by selecting the optimal placement of instructions for a particular set of matrices and returns a function pointer, and `invoke`, which calls that function pointer with the matrices' pointers as arguments.

The basic pattern is:
```
// Get the shape of the operands to build the right op
op_shape = libxsmm_create_<op>_shape(matrix dimensions, leading dimensions, data types);

// Define a new kernel
libxsmm_xsmmfunction kernel = {NULL};

// Dispatch (JIT compile or retrieve the previously compiled function)
kernel.<op> = libxsmm_dispatch_<op>_v2(op_shape, FLAGS);

// Prepare the execution parameters
libxsmm_<op>_param op_param;
op_param.?.? = op parameters

// Call the kernel
kernel.<op>(&op_param);
```

The `<op>` pattern is different, depending on the operation group you want to call.

There are four operation groups:
 * `meltw_unary`: For element-wise operations with a single input and a single output (ex. ReLU).
 * `melw_binary`: For element-wise operations with two inputs and a single output (ex. Add, Sub).
 * `meltw_ternary`: For element-wise operations with three inputs and a single output (ex. ??).
 * `gemm` or `brgemm`: For matrix multiplication operations with three inputs and a single output (ex. GEMM, BRGEMM).

In any of those operations:
 * The output can alias with one of the inputs (ex. accumulation, in-place operations).
 * If one input has a lower rank than the other, and the dimensions are compatible, a broadcast is performed before the operation.
 * If one input has a smaller type (of the same family) than the other, a (safe) type promotion is performed before the operation.

### Unary Operations


```
 LIBXSMM_API libxsmm_meltw_unary_shape libxsmm_create_meltw_unary_shape( const libxsmm_blasint m, const libxsmm_blasint n,
                                                                          const libxsmm_blasint ldi, const libxsmm_blasint ldo,
                                                                          const libxsmm_datatype in0_type, const libxsmm_datatype out_type, const libxsmm_datatype comp_type );

  LIBXSMM_API libxsmm_meltwfunction_unary libxsmm_dispatch_meltw_unary_v2( const libxsmm_meltw_unary_type unary_type, const libxsmm_meltw_unary_shape unary_shape, const libxsmm_bitfield unary_flags );
```

### Binary Operations

```
  LIBXSMM_API libxsmm_meltw_binary_shape libxsmm_create_meltw_binary_shape( const libxsmm_blasint m, const libxsmm_blasint n,
                                                                            const libxsmm_blasint ldi, const libxsmm_blasint ldi2, const libxsmm_blasint ldo,
                                                                            const libxsmm_datatype in0_type, const libxsmm_datatype in1_type, const libxsmm_datatype out_type, const libxsmm_datatype comp_type );
  LIBXSMM_API libxsmm_meltwfunction_binary libxsmm_dispatch_meltw_binary_v2( const libxsmm_meltw_binary_type binary_type, const libxsmm_meltw_binary_shape binary_shape, const libxsmm_bitfield binary_flags );
```

### Ternary Operations

```
  LIBXSMM_API libxsmm_meltw_ternary_shape libxsmm_create_meltw_ternary_shape( const libxsmm_blasint m, const libxsmm_blasint n,
                                                                              const libxsmm_blasint ldi, const libxsmm_blasint ldi2, const libxsmm_blasint ldi3, const libxsmm_blasint ldo,
                                                                              const libxsmm_datatype in0_type, const libxsmm_datatype in1_type, const libxsmm_datatype in2_type, const libxsmm_datatype out_type, const libxsmm_datatype comp_  type );

  LIBXSMM_API libxsmm_meltwfunction_ternary libxsmm_dispatch_meltw_ternary_v2( const libxsmm_meltw_ternary_type ternary_type, const libxsmm_meltw_ternary_shape ternary_shape, const libxsmm_bitfield ternary_flags );
```

### GEMM Operations

```
LIBXSMM_API libxsmm_gemm_shape libxsmm_create_gemm_shape( const libxsmm_blasint m, const libxsmm_blasint n, const libxsmm_blasint k,
                                                            const libxsmm_blasint lda, const libxsmm_blasint ldb, const libxsmm_blasint ldc,
                                                            const libxsmm_datatype a_in_type, const libxsmm_datatype b_in_type, const libxsmm_datatype out_type, const libxsmm_datatype comp_type );

  /** Query or JIT-generate SMM-kernel general mixed precision options and batch reduce; returns NULL if it does not exist or if JIT is not supported */
  LIBXSMM_API libxsmm_gemmfunction libxsmm_dispatch_gemm_v2( const libxsmm_gemm_shape gemm_shape, const libxsmm_bitfield gemm_flags,
                                                          const libxsmm_bitfield prefetch_flags );
  /** Query or JIT-generate BRGEMM-kernel general mixed precision options and batch reduce; returns NULL if it does not exist or if JIT is not supported */
  LIBXSMM_API libxsmm_gemmfunction libxsmm_dispatch_brgemm_v2( const libxsmm_gemm_shape gemm_shape, const libxsmm_bitfield gemm_flags,
                                                            const libxsmm_bitfield prefetch_flags, const libxsmm_gemm_batch_reduce_config brgemm_config );
  /** Query or JIT-generate BRGEMM-kernel with fusion, general mixed precision options and batch reduce; returns NULL if it does not exist or if JIT is not supported */
  LIBXSMM_API libxsmm_gemmfunction_ext libxsmm_dispatch_brgemm_ext_v2( const libxsmm_gemm_shape gemm_shape, const libxsmm_bitfield gemm_flags,
                                                                    const libxsmm_bitfield prefetch_flags, const libxsmm_gemm_batch_reduce_config brgemm_config,
                                                                    const libxsmm_gemm_ext_unary_argops unary_argops, const libxsmm_gemm_ext_binary_postops binary_postops );
```
