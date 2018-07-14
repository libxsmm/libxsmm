/******************************************************************************
** Copyright (c) 2014-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_H
#define LIBXSMM_H

#include "libxsmm_config.h"

/**
 * Strings to denote the version of LIBXSMM (libxsmm_config.h).
 * LIBXSMM_VERSION: Name of the version (stringized version numbers).
 * LIBXSMM_BRANCH:  Name of the branch this version is derived from.
 */
#define LIBXSMM_VERSION LIBXSMM_CONFIG_VERSION
#define LIBXSMM_BRANCH  LIBXSMM_CONFIG_BRANCH

/**
 * Numbers to denote the version of LIBXSMM (libxsmm_config.h).
 * LIBXSMM_VERSION_MAJOR:  Major version derived from the most recent RCS-tag.
 * LIBXSMM_VERSION_MINOR:  Minor version derived from the most recent RCS-tag.
 * LIBXSMM_VERSION_UPDATE: Update number derived from the most recent RCS-tag.
 * LIBXSMM_VERSION_PATCH:  Patch number based on distance to most recent RCS-tag.
 */
#define LIBXSMM_VERSION_MAJOR  LIBXSMM_CONFIG_VERSION_MAJOR
#define LIBXSMM_VERSION_MINOR  LIBXSMM_CONFIG_VERSION_MINOR
#define LIBXSMM_VERSION_UPDATE LIBXSMM_CONFIG_VERSION_UPDATE
#define LIBXSMM_VERSION_PATCH  LIBXSMM_CONFIG_VERSION_PATCH

#include "libxsmm_frontend.h"
#include "libxsmm_generator.h"
#include "libxsmm_fsspmdm.h"
#include "libxsmm_malloc.h"
#include "libxsmm_bgemm.h"
#include "libxsmm_spmdm.h"
#include "libxsmm_cpuid.h"
#include "libxsmm_timer.h"
#include "libxsmm_math.h"
#include "libxsmm_sync.h"
#include "libxsmm_dnn.h"
#include "libxsmm_dnn_rnncell.h"
#include "libxsmm_dnn_lstmcell.h"


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

/**
 * Returns the name of the target architecture as determined by the CPUID flags, as set by the
 * libxsmm_get_target_arch* functions, or as set by the LIBXSMM_TARGET environment variable.
 */
LIBXSMM_API const char* libxsmm_get_target_arch(void);
/** Set target architecture (arch="0|sse|snb|hsw|knl|knm|skx|icl", NULL/"0": CPUID) for subsequent code generation (JIT). */
LIBXSMM_API void libxsmm_set_target_arch(const char* arch);

/** Get the level of verbosity. */
LIBXSMM_API int libxsmm_get_verbosity(void);
/**
 * Set the level of verbosity (0: off, positive value: verbosity level,
 * negative value: maximum verbosity, which also dumps JIT-code)
 */
LIBXSMM_API void libxsmm_set_verbosity(int level);

/** Query the try-lock property of the code registry. */
LIBXSMM_API int libxsmm_get_dispatch_trylock(void);
/** Set the try-lock property of the code registry. */
LIBXSMM_API void libxsmm_set_dispatch_trylock(int trylock);

/** Get the default prefetch strategy. */
LIBXSMM_API libxsmm_gemm_prefetch_type libxsmm_get_gemm_auto_prefetch(void);
/** Set the default prefetch strategy. */
LIBXSMM_API void libxsmm_set_gemm_auto_prefetch(libxsmm_gemm_prefetch_type strategy);

/** Determine the kind of JIT-kernel. */
LIBXSMM_API int libxsmm_get_kernel_kind(const void* kernel, libxsmm_kernel_kind* kind);

/** Get information about the matrix multiplication kernel. */
LIBXSMM_API int libxsmm_get_mmkernel_info(libxsmm_xmmfunction kernel, libxsmm_mmkernel_info* info, size_t* code_size);

/** Get information about the matrix transpose kernel. */
LIBXSMM_API int libxsmm_get_transkernel_info(libxsmm_xtransfunction kernel, libxsmm_transkernel_info* info, size_t* code_size);

/** Get information about the matrix copy kernel. */
LIBXSMM_API int libxsmm_get_mcopykernel_info(libxsmm_xmcopyfunction kernel, libxsmm_mcopykernel_info* info, size_t* code_size);

/** Get information about the code registry. */
LIBXSMM_API int libxsmm_get_registry_info(libxsmm_registry_info* info);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (descriptor form). */
LIBXSMM_API libxsmm_xmmfunction libxsmm_xmmdispatch(const libxsmm_gemm_descriptor* descriptor);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (double-precision). */
LIBXSMM_API libxsmm_dmmfunction libxsmm_dmmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (single-precision). */
LIBXSMM_API libxsmm_smmfunction libxsmm_smmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (low/short-precision), int-accumulate */
LIBXSMM_API libxsmm_wimmfunction libxsmm_wimmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (low/short-precision), fp-accumulate */
LIBXSMM_API libxsmm_wsmmfunction libxsmm_wsmmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch);

/** Process a series of matrix multiplications (batch). */
LIBXSMM_API int libxsmm_mmbatch(
  /** Kernel (matches precision, transa, transb, beta, etc.). */
  libxsmm_xmmfunction kernel,
  /** Determines index-base (usually 0, 1 for one-based indexes); uses the same unit as the strides. */
  libxsmm_blasint index_base,
  /**
   * Stride used to walk stride_a, stride_b, and stride_c; zero turns stride_* into scalar values.
   * The index_stride is always measured in Bytes (sizeof(libxsmm_blasint) determines packed indexes).
   */
  libxsmm_blasint index_stride,
  /**
   * index_stride==0: a single value (measured in Bytes) for stride_a, stride_b, and stride_c is expected,
   * index_stride!=0: stride_a, stride_b, and stride_c are arrays of indexes (measured in elements);
   *                  array size equals batchsize, and indexes are discovered using the index_stride
   *                  (measured in Bytes). The typical value of index_stride is sizeof(libxsmm_blasint),
   *                  which determines a packed array of indexes.
   * A stride of zero (NULL pointer, or zero-index) does not advance the corresponding matrix-operand.
   * Note: accesses to the same C-matrix are internally synchronized.
   */
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  /**
   * Arrays of matrix operands (a, b, c). Depending on index_stride, the arrays are:
   * index_stride==0: pointers to pointers of elements e.g., double** for the C matrices.
   * index_stride!=0: pointer to elements e.g., const double* for the A and B matrices.
   */
  const void* a, const void* b, void* c,
  /**
   * Number of matrix multiplications. If the size is given as a negative value,
   * then internal synchronization is omitted.
   */
  libxsmm_blasint batchsize,
  /** Thread-ID (TID), and number of threads. */
  /*unsigned*/int tid, /*unsigned*/int nthreads);

/** Process a series of matrix multiplications (batch) similar to libxsmm_mmbatch; MT via libxsmmext. */
LIBXSMM_APIEXT int libxsmm_mmbatch_omp(libxsmm_xmmfunction kernel, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, libxsmm_blasint batchsize);

/** Process a series of matrix multiplications (batch); sequential. See also libxsmm_mmbatch. */
LIBXSMM_API void libxsmm_gemm_batch2(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, const char* transa, const char* transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
                     const void* b, const libxsmm_blasint* ldb,
   const void* beta,       void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize);
LIBXSMM_API void libxsmm_gemm_batch(libxsmm_gemm_precision precision, const char* transa, const char* transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize);

/** Process a series of matrix multiplications (batch); MT via libxsmmext. See also libxsmm_mmbatch. */
LIBXSMM_APIEXT void libxsmm_gemm_batch2_omp(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, const char* transa, const char* transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
                     const void* b, const libxsmm_blasint* ldb,
   const void* beta,       void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize);
LIBXSMM_APIEXT void libxsmm_gemm_batch_omp(libxsmm_gemm_precision precision, const char* transa, const char* transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
                     const void* b, const libxsmm_blasint* ldb,
   const void* beta,       void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize);

/**
 * This function is a no-op unless LIBXSMM is built to intercept GEMM calls.
 * Pointer arguments are used to filter intercepted GEMM calls such that
 * non-NULL values match. Otherwise (NULL) the respective argument is
 * considered a "free value" i.e., every value can match; libxsmmext required.
 */
LIBXSMM_APIEXT void libxsmm_mmbatch_begin2(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, const int* flags,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta);
LIBXSMM_APIEXT void libxsmm_mmbatch_begin(libxsmm_gemm_precision precision, const int* flags,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta);

/** Processes the batch of previously recorded matrix multiplications (libxsmm_mmbatch_begin); libxsmmext required. */
LIBXSMM_APIEXT void libxsmm_mmbatch_end(void);

/** Code generation routine for matrix-copy using a descriptor. */
LIBXSMM_API libxsmm_xmcopyfunction libxsmm_dispatch_mcopy(const libxsmm_mcopy_descriptor* descriptor);

/** Code generation routine for transposes using a descriptor */
LIBXSMM_API libxsmm_xtransfunction libxsmm_dispatch_trans(const libxsmm_trans_descriptor* descriptor);

/** Code generation routine for TRSM using a descriptor */
LIBXSMM_API libxsmm_xtrsmfunction libxsmm_dispatch_trsm(const libxsmm_trsm_descriptor* descriptor);

/**
 * Code generation routine for the CSR format which multiplies a dense SOA matrix (each element holds a SIMD-width
 * wide vector) and a sparse matrix or a sparse matrix with a dense SOA matrix.
 * The result is always a SOA matrix. There is no code cache, and user code has to manage the code pointers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_xmmfunction libxsmm_create_xcsr_soa(const libxsmm_gemm_descriptor* descriptor,
   const unsigned int* row_ptr, const unsigned int* column_idx, const void* values);

/**
 * Code generation routine for the CSC format which multiplies a dense SOA matrix (each element holds a SIMD-width
 * wide vector) and a sparse matrix or a sparse matrix with a dense SOA matrix.
 * The result is always a SOA matrix. There is no code cache, and user code has to manage the code pointers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_xmmfunction libxsmm_create_xcsc_soa(const libxsmm_gemm_descriptor* descriptor,
   const unsigned int* column_ptr, const unsigned int* row_idx, const void* values);

/**
 * Code generation routine for row-major format B matrix which is multiplied by a dense SOA matrix (each element holds a SIMD-width
 * wide vector) and the result is another SOA matrix. The memory layout of the SOA matrix is [row][col][soa].
 * here is no code cache, and user code has to manage the code pointers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_xmmfunction libxsmm_create_rm_ac_soa(const libxsmm_gemm_descriptor* descriptor);

/**
 * Code generation routine for row-major format A matrix which is multiplied by a dense SOA matrix (each element holds a SIMD-width
 * wide vector) and the result is another SOA matrix. The memory layout of the SOA matrix is [row][col][soa].
 * here is no code cache, and user code has to manage the code pointers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_xmmfunction libxsmm_create_rm_bc_soa(const libxsmm_gemm_descriptor* descriptor);

/**
 * Code generation routine for the CSR format which multiplies a dense matrix B into a dense matrix C.
 * The sparse matrix a is kept in registers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_dmmfunction libxsmm_create_dcsr_reg(const libxsmm_gemm_descriptor* descriptor,
   const unsigned int* row_ptr, const unsigned int* column_idx, const double* values);

/**
 * Code generation routine for the CSR format which multiplies a dense matrix B into a dense matrix C.
 * The sparse matrix a is kept in registers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_smmfunction libxsmm_create_scsr_reg(const libxsmm_gemm_descriptor* descriptor,
   const unsigned int* row_ptr, const unsigned int* column_idx, const float* values);

/** Deallocates the JIT'ted code as returned by libxsmm_create_* function. TODO: this is a no-op at the moment. */
LIBXSMM_API void libxsmm_release_kernel(const void* jit_code);

/** Matrix copy function ("in" can be NULL to zero the destination). */
LIBXSMM_API void libxsmm_matcopy(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch);

/** Matrix copy function ("in" can be NULL to zero the destination, per-thread form). */
LIBXSMM_API void libxsmm_matcopy_thread(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch, int tid, int nthreads);

/** Matrix copy function ("in" can be NULL to zero the destination); MT via libxsmmext. */
LIBXSMM_APIEXT void libxsmm_matcopy_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch);

/** Matrix transposition (out-of-place form). */
LIBXSMM_API void libxsmm_otrans(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo);

/** Matrix transposition (out-of-place form, per-thread form). */
LIBXSMM_API void libxsmm_otrans_thread(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  int tid, int nthreads);

/** Matrix transposition; MT via libxsmmext (out-of-place form). */
LIBXSMM_APIEXT void libxsmm_otrans_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo);

/** Matrix transposition (in-place form). */
LIBXSMM_API void libxsmm_itrans(void* inout, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld);

/** General dense matrix multiplication; MT via libxsmmext (double-precision). */
LIBXSMM_APIEXT void libxsmm_dgemm_omp(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc);
/** General dense matrix multiplication; MT via libxsmmext (single-precision). */
LIBXSMM_APIEXT void libxsmm_sgemm_omp(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc);

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
/** Dispatched general dense matrix multiplication (I16 input, I32 result). */
LIBXSMM_API void libxsmm_wigemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const int* alpha, const short* a, const libxsmm_blasint* lda,
  const short* b, const libxsmm_blasint* ldb,
  const int* beta, int* c, const libxsmm_blasint* ldc);
/** Dispatched general dense matrix multiplication (I16 input, F32 result). */
LIBXSMM_API void libxsmm_wsgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const short* a, const libxsmm_blasint* lda,
  const short* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc);
$MNK_INTERFACE_LIST
#if defined(__cplusplus)

template<typename T> struct libxsmm_gemm_precision_enum {};     /** Map a built-in type to libxsmm_gemm_precision. */
template<> struct libxsmm_gemm_precision_enum<double>           { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_F64; };
template<> struct libxsmm_gemm_precision_enum<float>            { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_F32; };
template<> struct libxsmm_gemm_precision_enum<int>              { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_I32; };
template<> struct libxsmm_gemm_precision_enum</*signed*/short>  { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_I16; };
template<> struct libxsmm_gemm_precision_enum<unsigned short>   { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_I16; };
template<> struct libxsmm_gemm_precision_enum<signed char>      { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_I8; };
template<> struct libxsmm_gemm_precision_enum<unsigned char>    { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_I8; };
template<> struct libxsmm_gemm_precision_enum<char>             { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_I8; };

template<typename INP_TYPE> struct libxsmm_gemm_default_output  { typedef INP_TYPE type; };
template<> struct libxsmm_gemm_default_output</*signed*/short>  { typedef int type; };
template<> struct libxsmm_gemm_default_output<unsigned short>   { typedef int type; };

/** Construct and execute a specialized function. */
template<typename INP_TYPE, typename OUT_TYPE = typename libxsmm_gemm_default_output<INP_TYPE>::type>
class LIBXSMM_RETARGETABLE libxsmm_mmfunction {
  mutable/*retargetable*/ libxsmm_xmmfunction m_function;
public:
  typedef INP_TYPE itype;
  typedef OUT_TYPE otype;
public:
  libxsmm_mmfunction() { m_function.xmm = 0; }
  libxsmm_mmfunction(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, int flags = LIBXSMM_FLAGS) {
    libxsmm_descriptor_blob blob;
    const libxsmm_gemm_descriptor *const desc = libxsmm_gemm_descriptor_init2(&blob,
      libxsmm_gemm_precision_enum<itype>::value, libxsmm_gemm_precision_enum<otype>::value,
      m, n, k, m, k, m, NULL/*alpha*/, NULL/*beta*/, flags, libxsmm_get_gemm_xprefetch(NULL));
    m_function.xmm = (0 != desc ? libxsmm_xmmdispatch(desc).xmm : 0);
  }
  libxsmm_mmfunction(int flags, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, int prefetch) {
    libxsmm_descriptor_blob blob;
    const libxsmm_gemm_descriptor *const desc = libxsmm_gemm_descriptor_init2(&blob,
      libxsmm_gemm_precision_enum<itype>::value, libxsmm_gemm_precision_enum<otype>::value,
      m, n, k, m, k, m, NULL/*alpha*/, NULL/*beta*/, flags, libxsmm_get_gemm_prefetch(prefetch));
    m_function.xmm = (0 != desc ? libxsmm_xmmdispatch(desc).xmm : 0);
  }
  libxsmm_mmfunction(int flags, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, otype alpha, otype beta) {
    libxsmm_descriptor_blob blob;
    const libxsmm_gemm_descriptor *const desc = libxsmm_gemm_descriptor_init2(&blob,
      libxsmm_gemm_precision_enum<itype>::value, libxsmm_gemm_precision_enum<otype>::value,
      m, n, k, m, k, m, &alpha, &beta, flags, libxsmm_get_gemm_xprefetch(NULL));
    m_function.xmm = (0 != desc ? libxsmm_xmmdispatch(desc).xmm : 0);
  }
  libxsmm_mmfunction(int flags, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, otype alpha, otype beta, int prefetch) {
    libxsmm_descriptor_blob blob;
    const libxsmm_gemm_descriptor *const desc = libxsmm_gemm_descriptor_init2(&blob,
      libxsmm_gemm_precision_enum<itype>::value, libxsmm_gemm_precision_enum<otype>::value,
      m, n, k, m, k, m, &alpha, &beta, flags, libxsmm_get_gemm_prefetch(prefetch));
    m_function.xmm = (0 != desc ? libxsmm_xmmdispatch(desc).xmm : 0);
  }
  libxsmm_mmfunction(int flags, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
    libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, int prefetch)
  {
    libxsmm_descriptor_blob blob;
    const libxsmm_gemm_descriptor *const desc = libxsmm_gemm_descriptor_init2(&blob,
      libxsmm_gemm_precision_enum<itype>::value, libxsmm_gemm_precision_enum<otype>::value,
      m, n, k, lda, ldb, ldc, NULL/*alpha*/, NULL/*beta*/, flags, libxsmm_get_gemm_prefetch(prefetch));
    m_function.xmm = (0 != desc ? libxsmm_xmmdispatch(desc).xmm : 0);
  }
  libxsmm_mmfunction(int flags, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
    libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, otype alpha, otype beta)
  {
    libxsmm_descriptor_blob blob;
    const libxsmm_gemm_descriptor *const desc = libxsmm_gemm_descriptor_init2(&blob,
      libxsmm_gemm_precision_enum<itype>::value, libxsmm_gemm_precision_enum<otype>::value,
      m, n, k, lda, ldb, ldc, &alpha, &beta, flags, libxsmm_get_gemm_xprefetch(NULL));
    m_function.xmm = (0 != desc ? libxsmm_xmmdispatch(desc).xmm : 0);
  }
  libxsmm_mmfunction(int flags, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
    libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, otype alpha, otype beta, int prefetch)
  {
    libxsmm_descriptor_blob blob;
    const libxsmm_gemm_descriptor *const desc = libxsmm_gemm_descriptor_init2(&blob,
      libxsmm_gemm_precision_enum<itype>::value, libxsmm_gemm_precision_enum<otype>::value,
      m, n, k, lda, ldb, ldc, &alpha, &beta, flags, libxsmm_get_gemm_prefetch(prefetch));
    m_function.xmm = (0 != desc ? libxsmm_xmmdispatch(desc).xmm : 0);
  }
public:
  const libxsmm_xmmfunction& kernel() const {
    return m_function;
  }
  operator const void*() const {
    return 0 != m_function.xmm ? this : 0;
  }
  void operator()(const itype* a, const itype* b, otype* c) const {
    LIBXSMM_MMCALL_ABC(m_function.xmm, a, b, c);
  }
  void operator()(const itype* a, const itype* b, otype* c, const itype* pa, const itype* pb, const otype* pc) const {
    LIBXSMM_MMCALL_PRF(m_function.xmm, a, b, c, pa, pb, pc);
  }
};

/** Matrix copy function ("in" can be NULL to zero the destination). */
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy(T* out, const T* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  return libxsmm_matcopy(out, in, sizeof(T), m, n, ldi, ldo);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy(T* out, const T* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi)
{
  return libxsmm_matcopy(out, in, m, n, ldi, ldi);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy(T* out, const T* in,
  libxsmm_blasint m, libxsmm_blasint n)
{
  return libxsmm_matcopy(out, in, m, n, m);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy(T* out, const T* in,
  libxsmm_blasint n)
{
  return libxsmm_matcopy(out, in, n, n);
}

/** Matrix copy function ("in" can be NULL to zero the destination); MT via libxsmmext. */
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy_omp(T* out, const T* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  return libxsmm_matcopy_omp(out, in, sizeof(T), m, n, ldi, ldo);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy_omp(T* out, const T* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi)
{
  return libxsmm_matcopy_omp(out, in, m, n, ldi, ldi);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy_omp(T* out, const T* in,
  libxsmm_blasint m, libxsmm_blasint n)
{
  return libxsmm_matcopy_omp(out, in, m, n, m);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy_omp(T* out, const T* in,
  libxsmm_blasint n)
{
  return libxsmm_matcopy_omp(out, in, n, n);
}

/** Matrix transposition (out-of-place form). */
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans(T* out, const T* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  return libxsmm_otrans(out, in, sizeof(T), m, n, ldi, ldo);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans(T* out, const T* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi)
{
  return libxsmm_trans(out, in, m, n, ldi, ldi);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans(T* out, const T* in,
  libxsmm_blasint m, libxsmm_blasint n)
{
  return libxsmm_trans(out, in, m, n, m);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans(T* out, const T* in,
  libxsmm_blasint n)
{
  return libxsmm_trans(out, in, n, n);
}

/** Matrix transposition; MT via libxsmmext (out-of-place form). */
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans_omp(T* out, const T* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  return libxsmm_otrans_omp(out, in, sizeof(T), m, n, ldi, ldo);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans_omp(T* out, const T* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi)
{
  return libxsmm_trans_omp(out, in, m, n, ldi, ldi);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans_omp(T* out, const T* in,
  libxsmm_blasint m, libxsmm_blasint n)
{
  return libxsmm_trans_omp(out, in, m, n, m);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans_omp(T* out, const T* in,
  libxsmm_blasint n)
{
  return libxsmm_trans_omp(out, in, n, n);
}

/** Matrix transposition (in-place form). */
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans(T* inout,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi)
{
  return libxsmm_itrans(inout, sizeof(T), m, n, ldi);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans(T* inout,
  libxsmm_blasint m, libxsmm_blasint n)
{
  return libxsmm_trans(inout, m, n, m);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans(T* inout,
  libxsmm_blasint n)
{
  return libxsmm_trans(inout, n, n);
}

/** Dispatched general dense matrix multiplication (double-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_gemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
                       const double* b, const libxsmm_blasint* ldb,
   const double* beta,       double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline LIBXSMM_RETARGETABLE void libxsmm_gemm(const char* transa, const char* transb,
  /* by-value */ libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
                       const double* b, const libxsmm_blasint* ldb,
   const double* beta,       double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (single-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_gemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
                      const float* b, const libxsmm_blasint* ldb,
   const float* beta,       float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline LIBXSMM_RETARGETABLE void libxsmm_gemm(const char* transa, const char* transb,
  /* by-value */ libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
                      const float* b, const libxsmm_blasint* ldb,
   const float* beta,       float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (low-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_gemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const int* alpha, const short* a, const libxsmm_blasint* lda,
                    const short* b, const libxsmm_blasint* ldb,
   const int* beta,         int* c, const libxsmm_blasint* ldc)
{
  libxsmm_wigemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline LIBXSMM_RETARGETABLE void libxsmm_gemm(const char* transa, const char* transb,
  /* by-value */ libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const int* alpha, const short* a, const libxsmm_blasint* lda,
                    const short* b, const libxsmm_blasint* ldb,
   const int* beta,         int* c, const libxsmm_blasint* ldc)
{
  libxsmm_wigemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (low-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_gemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const short* a, const libxsmm_blasint* lda,
                      const short* b, const libxsmm_blasint* ldb,
   const float* beta,       float* c, const libxsmm_blasint* ldc)
{
  libxsmm_wsgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline LIBXSMM_RETARGETABLE void libxsmm_gemm(const char* transa, const char* transb,
  /* by-value */ libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const short* a, const libxsmm_blasint* lda,
                      const short* b, const libxsmm_blasint* ldb,
   const float* beta,       float* c, const libxsmm_blasint* ldc)
{
  libxsmm_wsgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (double-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_blas_gemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
                       const double* b, const libxsmm_blasint* ldb,
   const double* beta,       double* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline LIBXSMM_RETARGETABLE void libxsmm_blas_gemm(const char* transa, const char* transb,
  /* by-value */ libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
                       const double* b, const libxsmm_blasint* ldb,
   const double* beta,       double* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_dgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (single-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_blas_gemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
                      const float* b, const libxsmm_blasint* ldb,
   const float* beta,       float* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline LIBXSMM_RETARGETABLE void libxsmm_blas_gemm(const char* transa, const char* transb,
  /* by-value */ libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
                      const float* b, const libxsmm_blasint* ldb,
   const float* beta,       float* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_sgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif /*__cplusplus*/
#endif /*LIBXSMM_H*/

