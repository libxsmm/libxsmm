/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_H
#define LIBXSMM_H

#include "libxsmm_config.h"

#if !defined(LIBXSMM_DESCRIPTION)
# define LIBXSMM_DESCRIPTION \
    "Library for specialized dense and sparse matrix " \
    "operations, and deep learning primitives."
#endif

/**
 * Strings to denote the version of LIBXSMM (libxsmm_config.h).
 * LIBXSMM_VERSION: Name of the version (stringized version numbers).
 * LIBXSMM_BRANCH:  Name of the branch this version is derived from.
 */
#define LIBXSMM_VERSION LIBXSMM_CONFIG_VERSION
#define LIBXSMM_BRANCH  LIBXSMM_CONFIG_BRANCH

/**
 * Semantic version according to https://semver.org/ (see also libxsmm_config.h).
 * LIBXSMM_VERSION_MAJOR:  Major version derived from the most recent RCS-tag.
 * LIBXSMM_VERSION_MINOR:  Minor version derived from the most recent RCS-tag.
 * LIBXSMM_VERSION_UPDATE: Update number derived from the most recent RCS-tag.
 * LIBXSMM_VERSION_PATCH:  Patch number based on distance to most recent RCS-tag.
 */
#define LIBXSMM_VERSION_MAJOR  LIBXSMM_CONFIG_VERSION_MAJOR
#define LIBXSMM_VERSION_MINOR  LIBXSMM_CONFIG_VERSION_MINOR
#define LIBXSMM_VERSION_UPDATE LIBXSMM_CONFIG_VERSION_UPDATE
#define LIBXSMM_VERSION_PATCH  LIBXSMM_CONFIG_VERSION_PATCH

/**
 * The utilities (libxsmm_utils.h) shall be explicitly
 * included, i.e., separate from libxsmm.h.
*/
#include "libxsmm_generator.h"
#include "libxsmm_fsspmdm.h"
#include "libxsmm_memory.h"
#include "libxsmm_malloc.h"
#include "libxsmm_cpuid.h"
#include "libxsmm_math.h"
#include "libxsmm_sync.h"

#if (defined(LIBXSMM_INIT) || defined(LIBXSMM_CTOR))
# undef LIBXSMM_INIT
# define LIBXSMM_INIT LIBXSMM_ASSERT_MSG(1 < libxsmm_ninit, "LIBXSMM is not initialized");
# define LIBXSMM_INIT_COMPLETED
#else
# define LIBXSMM_INIT if (2 > libxsmm_ninit) libxsmm_init();
#endif


/** Initialize the library; pay for setup cost at a specific point. */
LIBXSMM_API void libxsmm_init(void);
/** De-initialize the library and free internal memory (optional). */
LIBXSMM_API void libxsmm_finalize(void);

/**
 * Returns the architecture and instruction set extension as determined by the CPUID flags, as set
 * by the libxsmm_get_target_arch* functions, or as set by the LIBXSMM_TARGET environment variable.
 */
LIBXSMM_API int libxsmm_get_target_archid(void);
/** Set target architecture (id: see libxsmm_typedefs.h) for subsequent code generation (JIT). */
LIBXSMM_API void libxsmm_set_target_archid(int id);

/** Returns the type-name of data-type (can be also libxsmm_datatype). */
LIBXSMM_API const char* libxsmm_get_typename(libxsmm_datatype datatype);

/**
 * Returns the name of the target architecture as determined by the CPUID flags, as set by the
 * libxsmm_get_target_arch* functions, or as set by the LIBXSMM_TARGET environment variable.
 */
LIBXSMM_API const char* libxsmm_get_target_arch(void);
/** Set target architecture (arch="0|sse|snb|hsw|skx|clx|cpx|spr", NULL/"0": CPUID). */
LIBXSMM_API void libxsmm_set_target_arch(const char* arch);

/** Get the level of verbosity. */
LIBXSMM_API int libxsmm_get_verbosity(void);
/**
 * Set the level of verbosity (0: off, positive value: verbosity level,
 * negative value: maximum verbosity, which also dumps JIT-code)
 */
LIBXSMM_API void libxsmm_set_verbosity(int level);

/** Get information about the matrix multiplication kernel. */
LIBXSMM_API int libxsmm_get_mmkernel_info(libxsmm_xmmfunction kernel, libxsmm_mmkernel_info* info);
/** Get information about the matrix eltwise kernel. */
LIBXSMM_API int libxsmm_get_meltwkernel_info(libxsmm_xmeltwfunction kernel, libxsmm_meltwkernel_info* info);

/** Receive information about JIT-generated code (kernel or registry entry). */
LIBXSMM_API int libxsmm_get_kernel_info(const void* kernel, libxsmm_kernel_info* info);

/** Get information about the code registry. */
LIBXSMM_API int libxsmm_get_registry_info(libxsmm_registry_info* info);
/** Enumerate registry by kind (e.g., LIBXSMM_KERNEL_KIND_USER); can be NULL (no such kind). */
LIBXSMM_API void* libxsmm_get_registry_begin(libxsmm_kernel_kind kind, const void** key);
/** Receive next (or NULL) based on given entry (see libxsmm_get_registry_begin). */
LIBXSMM_API void* libxsmm_get_registry_next(const void* regentry, const void** key);

/**
 * Register user-defined key-value; value can be queried (libxsmm_xdispatch).
 * Since the key-type is unknown to LIBXSMM, the key must be binary reproducible,
 * i.e., a structured type (can be padded) must be initialized like a binary blob
 * (memset) followed by an element-wise initialization. The size of the
 * key is limited (see documentation). The given value is copied by LIBXSMM and
 * can be initialized prior to registration or whenever queried. Registered data
 * is released when the program terminates but can be also released if needed
 * (libxsmm_xrelease), .e.g., in case of a larger value reusing the same key.
 */
LIBXSMM_API void* libxsmm_xregister(const void* key, size_t key_size,
  size_t value_size, const void* value_init);
/** Query user-defined value from LIBXSMM's code registry. */
LIBXSMM_API void* libxsmm_xdispatch(const void* key, size_t key_size);
/** Remove key-value pair from code registry and release memory. */
LIBXSMM_API void libxsmm_xrelease(const void* key, size_t key_size);

/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (descriptor form). */
LIBXSMM_API libxsmm_xmmfunction libxsmm_xmmdispatch(const libxsmm_gemm_descriptor* descriptor);
/** Query or JIT-generate SMM-kernel general mixed precision options and batch reduce; returns NULL if it does not exist or if JIT is not supported */
LIBXSMM_API libxsmm_gemmfunction libxsmm_dispatch_gemm( const libxsmm_gemm_shape gemm_shape, const libxsmm_bitfield gemm_flags,
                                                        const libxsmm_bitfield prefetch_flags );
/** Query or JIT-generate BRGEMM-kernel general mixed precision options and batch reduce; returns NULL if it does not exist or if JIT is not supported */
LIBXSMM_API libxsmm_gemmfunction libxsmm_dispatch_brgemm( const libxsmm_gemm_shape gemm_shape, const libxsmm_bitfield gemm_flags,
                                                          const libxsmm_bitfield prefetch_flags, const libxsmm_gemm_batch_reduce_config brgemm_config );
/** Query or JIT-generate BRGEMM-kernel with fusion, general mixed precision options and batch reduce; returns NULL if it does not exist or if JIT is not supported */
LIBXSMM_API libxsmm_gemmfunction_ext libxsmm_dispatch_brgemm_ext( const libxsmm_gemm_shape gemm_shape, const libxsmm_bitfield gemm_flags,
                                                                  const libxsmm_bitfield prefetch_flags, const libxsmm_gemm_batch_reduce_config brgemm_config,
                                                                  const libxsmm_gemm_ext_unary_argops unary_argops, const libxsmm_gemm_ext_binary_postops binary_postops );
/** Query or JIT-generate Tileconfig kernles, if the machine doesn't support Intel AMX, the kernel can be still called */
LIBXSMM_API libxsmm_tilecfgfunction libxsmm_dispatch_tilecfg_gemm( const libxsmm_gemm_shape gemm_shape, const libxsmm_bitfield gemm_flags );

/** Code generation routine for matrix-eltwise using a descriptor. */
LIBXSMM_API libxsmm_xmeltwfunction libxsmm_dispatch_meltw( const libxsmm_meltw_descriptor* descriptor );
LIBXSMM_API libxsmm_meltwfunction_unary libxsmm_dispatch_meltw_unary( const libxsmm_meltw_unary_type unary_type, const libxsmm_meltw_unary_shape unary_shape, const libxsmm_bitfield unary_flags );
LIBXSMM_API libxsmm_meltwfunction_binary libxsmm_dispatch_meltw_binary( const libxsmm_meltw_binary_type binary_type, const libxsmm_meltw_binary_shape binary_shape, const libxsmm_bitfield binary_flags );
LIBXSMM_API libxsmm_meltwfunction_ternary libxsmm_dispatch_meltw_ternary( const libxsmm_meltw_ternary_type ternary_type, const libxsmm_meltw_ternary_shape ternary_shape, const libxsmm_bitfield ternary_flags );

/** matrix equation interface */
LIBXSMM_API libxsmm_blasint libxsmm_meqn_create(void);
LIBXSMM_API libxsmm_meqn_arg_shape libxsmm_create_meqn_arg_shape( const libxsmm_blasint m, const libxsmm_blasint n, const libxsmm_blasint ld, const libxsmm_datatype type );
LIBXSMM_API libxsmm_matrix_arg_attributes libxsmm_create_matrix_arg_attributes( const libxsmm_matrix_arg_type type, const libxsmm_matrix_arg_set_type set_type, const libxsmm_blasint set_cardinality_hint, const libxsmm_blasint set_stride_hint );
LIBXSMM_API libxsmm_meqn_arg_metadata libxsmm_create_meqn_arg_metadata( const libxsmm_blasint eqn_idx, const libxsmm_blasint in_arg_pos );
LIBXSMM_API libxsmm_meqn_op_metadata libxsmm_create_meqn_op_metadata( const libxsmm_blasint eqn_idx, const libxsmm_blasint op_arg_pos );
LIBXSMM_API int libxsmm_meqn_push_back_arg( const libxsmm_meqn_arg_metadata arg_metadata, const libxsmm_meqn_arg_shape arg_shape, libxsmm_matrix_arg_attributes arg_attr);
LIBXSMM_API int libxsmm_meqn_push_back_unary_op( const libxsmm_meqn_op_metadata op_metadata, const libxsmm_meltw_unary_type type, const libxsmm_datatype dtype, const libxsmm_bitfield flags);
LIBXSMM_API int libxsmm_meqn_push_back_binary_op( const libxsmm_meqn_op_metadata op_metadata, const libxsmm_meltw_binary_type type, const libxsmm_datatype dtype, const libxsmm_bitfield flags);
LIBXSMM_API int libxsmm_meqn_push_back_ternary_op( const libxsmm_meqn_op_metadata op_metadata, const libxsmm_meltw_ternary_type type, const libxsmm_datatype dtype, const libxsmm_bitfield flags);

LIBXSMM_API void libxsmm_meqn_tree_print( const libxsmm_blasint idx );
LIBXSMM_API void libxsmm_meqn_rpn_print( const libxsmm_blasint idx );
LIBXSMM_API libxsmm_meqn_function libxsmm_dispatch_meqn_desc( const libxsmm_meqn_descriptor* descriptor );
LIBXSMM_API libxsmm_meqn_function libxsmm_dispatch_meqn( const libxsmm_blasint idx, const libxsmm_meqn_arg_shape out_shape );

/**
 * Code generation routine for the CSR format which multiplies a dense SOA matrix (each element holds a SIMD-width
 * wide vector) and a sparse matrix or a sparse matrix with a dense SOA matrix.
 * The result is always a SOA matrix. There is no code cache, and user code has to manage the code pointers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_gemmfunction libxsmm_create_packed_spgemm_csr(
  const libxsmm_gemm_shape gemm_shape, const libxsmm_bitfield gemm_flags, const libxsmm_bitfield prefetch_flags, const libxsmm_blasint packed_width,
  const unsigned int* row_ptr, const unsigned int* column_idx, const void* values);

/**
 * Code generation routine for the CSC format which multiplies a dense SOA matrix (each element holds a SIMD-width
 * wide vector) and a sparse matrix or a sparse matrix with a dense SOA matrix.
 * The result is always a SOA matrix. There is no code cache, and user code has to manage the code pointers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_gemmfunction libxsmm_create_packed_spgemm_csc(
  const libxsmm_gemm_shape gemm_shape, const libxsmm_bitfield gemm_flags, const libxsmm_bitfield prefetch_flags, const libxsmm_blasint packed_width,
  const unsigned int* column_ptr, const unsigned int* row_idx, const void* values);

LIBXSMM_API libxsmm_gemmfunction libxsmm_create_packed_spgemm_bcsc(
  const libxsmm_gemm_shape gemm_shape, const libxsmm_bitfield gemm_flags, const libxsmm_bitfield prefetch_flags, const libxsmm_spgemm_config spgemm_config);

LIBXSMM_API libxsmm_tilecfgfunction libxsmm_create_tilecfg_packed_spgemm_bcsc(
  const libxsmm_gemm_shape gemm_shape, const libxsmm_bitfield gemm_flags, const libxsmm_spgemm_config spgemm_config );

/**
 * Code generation routine for packed GEMM. In this case A is [K][M][packed], B is [K][N][packed] and C is [N][M][packed],
 * that mans the  memory layout of the matricis is in SOA [row][col][packed].
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_gemmfunction libxsmm_create_packed_gemm( const libxsmm_gemm_shape gemm_shape,
  const libxsmm_bitfield gemm_flags, const libxsmm_bitfield prefetch_flags, const libxsmm_blasint packed_width );

/**
 * Code generation routine for row-major format B matrix which is multiplied by a dense packed matrix (each element holds a SIMD-width
 * wide vector) and the result is another packed matrix. The memory layout of the SOA matrix is [row][col][packed].
 * here is no code cache, and user code has to manage the code pointers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_gemmfunction libxsmm_create_packed_gemm_ac_rm( const libxsmm_gemm_shape gemm_shape,
  const libxsmm_bitfield gemm_flags, const libxsmm_bitfield prefetch_flags, const libxsmm_blasint packed_width );

/**
 * Code generation routine for row-major format A matrix which is multiplied by a dense packed matrix (each element holds a SIMD-width
 * wide vector) and the result is another packed matrix. The memory layout of the packed matrix is [row][col][packed].
 * here is no code cache, and user code has to manage the code pointers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_gemmfunction libxsmm_create_packed_gemm_bc_rm( const libxsmm_gemm_shape gemm_shape,
  const libxsmm_bitfield gemm_flags, const libxsmm_bitfield prefetch_flags, const libxsmm_blasint packed_width );

/**
 * Code generation routine for the CSR format which multiplies a dense matrix "b" into a dense matrix "c".
 * The sparse matrix "a" is kept in registers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_gemmfunction libxsmm_create_spgemm_csr_areg( const libxsmm_gemm_shape gemm_shape,
  const libxsmm_bitfield gemm_flags, const libxsmm_bitfield prefetch_flags,
  const libxsmm_blasint max_N, const unsigned int* row_ptr, const unsigned int* column_idx, const double* values );

/**
 * Deallocates the JIT'ted code as returned by libxsmm_create_* functions,
 * unregisters and releases code from the code registry.
 */
LIBXSMM_API void libxsmm_release_kernel(const void* kernel);

/** Dispatched general dense matrix multiplication (double-precision). */
LIBXSMM_API void libxsmm_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc);
/** Dispatched general dense matrix multiplication (single-precision). */
LIBXSMM_API void libxsmm_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc);

#if !defined(LIBXSMM_DEFAULT_CONFIG) && (!defined(LIBXSMM_SOURCE_H) || defined(LIBXSMM_CONFIGURED))

#endif /*!defined(LIBXSMM_DEFAULT_CONFIG)*/

#if defined(__cplusplus)

/** Map built-in type to libxsmm_datatype (libxsmm_datatype_enum). */
template<typename T> struct libxsmm_datatype_enum         { static const libxsmm_datatype value = static_cast<libxsmm_datatype>(LIBXSMM_DATATYPE_UNSUPPORTED); };
template<> struct libxsmm_datatype_enum<double>           { static const libxsmm_datatype value = LIBXSMM_DATATYPE_F64; };
template<> struct libxsmm_datatype_enum<float>            { static const libxsmm_datatype value = LIBXSMM_DATATYPE_F32; };

/** Determine default output type based on the input-type. */
template<typename INP_TYPE> struct libxsmm_gemm_default_output  { typedef INP_TYPE type; };

/** Default-initialize libxsmm_gemm_param structure for the given prefetch-strategy. */
template<int PREFETCH> inline/*superfluous*/ void libxsmm_mmfunction_prefetch(
  const libxsmm_gemmfunction& function, libxsmm_gemm_param& args)
{
  libxsmm_mmkernel_info info;
  libxsmm_xmmfunction xmm;
  xmm.gemm = function;
  LIBXSMM_ASSERT(LIBXSMM_GEMM_PREFETCH_NONE != PREFETCH);
  if (0/*EXIT_SUCCESS*/ == libxsmm_get_mmkernel_info(xmm, &info) && LIBXSMM_GEMM_PREFETCH_NONE != info.prefetch) {
    const size_t itypesize = LIBXSMM_TYPESIZE(info.iprecision), otypesize = LIBXSMM_TYPESIZE(info.oprecision);
    args.a.senary = static_cast<char*>(args.a.primary) + itypesize * info.m * info.k;
    args.b.senary = static_cast<char*>(args.b.primary) + itypesize * info.k * info.n;
    args.c.senary = static_cast<char*>(args.c.primary) + otypesize * info.m * info.n;
  }
}
template<> inline/*superfluous*/ void libxsmm_mmfunction_prefetch<LIBXSMM_GEMM_PREFETCH_NONE>(
  const libxsmm_gemmfunction& function, libxsmm_gemm_param& args)
{
  LIBXSMM_UNUSED(function);
#if defined(NDEBUG)
  LIBXSMM_UNUSED(args);
#else
  args.a.senary = args.b.senary = args.c.senary = NULL;
#endif
}

/** Construct and execute a specialized function. */
template<typename INP_TYPE, typename OUT_TYPE = typename libxsmm_gemm_default_output<INP_TYPE>::type,
  int PREFETCH_DEFAULT = LIBXSMM_PREFETCH/*LIBXSMM_PREFETCH_AUTO*/>
class libxsmm_mmfunction {
  /*retargetable*/ libxsmm_gemmfunction m_function;
public:
  typedef INP_TYPE itype;
  typedef OUT_TYPE otype;
public:
  libxsmm_mmfunction() { m_function = NULL; }
  libxsmm_mmfunction(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k) {
    const libxsmm_blasint lda = m, ldb = k, ldc = m;
    const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(m, n, k, lda, ldb, ldc,
      libxsmm_datatype_enum<itype>::value, libxsmm_datatype_enum<itype>::value,
      libxsmm_datatype_enum<otype>::value, libxsmm_datatype_enum<otype>::value);
    m_function = libxsmm_dispatch_gemm(gemm_shape, 0/*flags*/,
      static_cast<libxsmm_bitfield>(PREFETCH_DEFAULT));
  }
  libxsmm_mmfunction(int flags, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, int prefetch = PREFETCH_DEFAULT) {
    const libxsmm_blasint lda = m, ldb = k, ldc = m;
    const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(m, n, k, lda, ldb, ldc,
      libxsmm_datatype_enum<itype>::value, libxsmm_datatype_enum<itype>::value,
      libxsmm_datatype_enum<otype>::value, libxsmm_datatype_enum<otype>::value);
    m_function = libxsmm_dispatch_gemm(gemm_shape,
      static_cast<libxsmm_bitfield>(flags),
      static_cast<libxsmm_bitfield>(prefetch));
  }
  libxsmm_mmfunction(int flags, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, otype alpha, otype beta, int prefetch = PREFETCH_DEFAULT) {
    const libxsmm_blasint lda = m, ldb = k, ldc = m;
    const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(m, n, k, lda, ldb, ldc,
      libxsmm_datatype_enum<itype>::value, libxsmm_datatype_enum<itype>::value,
      libxsmm_datatype_enum<otype>::value, libxsmm_datatype_enum<otype>::value);
    m_function = (LIBXSMM_GEMM_NO_BYPASS(flags, alpha, beta) ? libxsmm_dispatch_gemm(gemm_shape,
      static_cast<libxsmm_bitfield>(flags | (LIBXSMM_NEQ(0, beta) ? 0 : LIBXSMM_GEMM_FLAG_BETA_0)),
      static_cast<libxsmm_bitfield>(prefetch)) : NULL);
  }
  libxsmm_mmfunction(int flags, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
    libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, int prefetch = PREFETCH_DEFAULT)
  {
    const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(m, n, k, lda, ldb, ldc,
      libxsmm_datatype_enum<itype>::value, libxsmm_datatype_enum<itype>::value,
      libxsmm_datatype_enum<otype>::value, libxsmm_datatype_enum<otype>::value);
    m_function = libxsmm_dispatch_gemm(gemm_shape,
      static_cast<libxsmm_bitfield>(flags),
      static_cast<libxsmm_bitfield>(prefetch));
  }
  libxsmm_mmfunction(int flags, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
    libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, otype alpha, otype beta,
    int prefetch = PREFETCH_DEFAULT)
  {
    const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(m, n, k, lda, ldb, ldc,
      libxsmm_datatype_enum<itype>::value, libxsmm_datatype_enum<itype>::value,
      libxsmm_datatype_enum<otype>::value, libxsmm_datatype_enum<otype>::value);
    m_function = (LIBXSMM_GEMM_NO_BYPASS(flags, alpha, beta) ? libxsmm_dispatch_gemm(gemm_shape,
      static_cast<libxsmm_bitfield>(flags | (LIBXSMM_NEQ(0, beta) ? 0 : LIBXSMM_GEMM_FLAG_BETA_0)),
      static_cast<libxsmm_bitfield>(prefetch)) : NULL);
  }
public:
  const libxsmm_gemmfunction& kernel() const {
    return m_function;
  }
  operator const void*() const {
    return NULL != m_function ? this : NULL;
  }
  void operator()(const itype* a, const itype* b, otype* c) const {
    libxsmm_gemm_param args;
    args.a.primary = const_cast<itype*>(a);
    args.b.primary = const_cast<itype*>(b);
    args.c.primary = c;
    libxsmm_mmfunction_prefetch<PREFETCH_DEFAULT>(m_function, args);
    LIBXSMM_ASSERT(NULL != m_function);
    m_function(&args);
  }
  void operator()(const itype* a, const itype* b, otype* c, const itype* pa, const itype* pb, const otype* pc) const {
    libxsmm_gemm_param args;
    args.a.primary = const_cast<itype*>(a);
    args.b.primary = const_cast<itype*>(b);
    args.c.primary = c;
    args.a.senary = const_cast<itype*>(pa);
    args.b.senary = const_cast<itype*>(pb);
    args.c.senary = const_cast<otype*>(pc);
    LIBXSMM_ASSERT(NULL != m_function);
    m_function(&args);
  }
};

/** Dispatched general dense matrix multiplication (double-precision). */
inline void libxsmm_gemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
                       const double* b, const libxsmm_blasint* ldb,
  const double* beta,        double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void libxsmm_gemm(const char* transa, const char* transb,
  /* by-value */ libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
                       const double* b, const libxsmm_blasint* ldb,
  const double* beta,        double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (single-precision). */
inline void libxsmm_gemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
                      const float* b, const libxsmm_blasint* ldb,
  const float* beta,        float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void libxsmm_gemm(const char* transa, const char* transb,
  /* by-value */ libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
                      const float* b, const libxsmm_blasint* ldb,
  const float* beta,        float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif /*__cplusplus*/

#endif /*LIBXSMM_H*/
