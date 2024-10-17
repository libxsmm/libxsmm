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
#ifndef LIBXSMM_MACROS_H
#define LIBXSMM_MACROS_H

#include "libxsmm_config.h"

/** Parameters the library was built for. */
#define LIBXSMM_CACHELINE LIBXSMM_CONFIG_CACHELINE
#define LIBXSMM_ALIGNMENT LIBXSMM_CONFIG_ALIGNMENT
#define LIBXSMM_MALLOC LIBXSMM_CONFIG_MALLOC
#define LIBXSMM_ILP64 LIBXSMM_CONFIG_ILP64
#define LIBXSMM_SYNC LIBXSMM_CONFIG_SYNC
#define LIBXSMM_JIT LIBXSMM_CONFIG_JIT

/** Parameters of GEMM domain (static kernels, etc). */
#define LIBXSMM_PREFETCH LIBXSMM_CONFIG_PREFETCH
#define LIBXSMM_MAX_MNK LIBXSMM_CONFIG_MAX_MNK
#define LIBXSMM_MAX_DIM LIBXSMM_CONFIG_MAX_DIM
#define LIBXSMM_MAX_M LIBXSMM_CONFIG_MAX_M
#define LIBXSMM_MAX_N LIBXSMM_CONFIG_MAX_N
#define LIBXSMM_MAX_K LIBXSMM_CONFIG_MAX_K
#define LIBXSMM_FLAGS LIBXSMM_CONFIG_FLAGS
#define LIBXSMM_ALPHA LIBXSMM_CONFIG_ALPHA
#define LIBXSMM_BETA LIBXSMM_CONFIG_BETA

/* Parameters of matrix equations */
#define LIBXSMM_MAX_EQN_COUNT LIBXSMM_CONFIG_MAX_EQN_COUNT

/**
 * Use "make PLATFORM=1" to disable platform checks.
 * The platform check is to bail-out with an error
 * message for an attempt to build an upstream package
 * and subsequently to list LIBXSMM as "broken" on
 * that platform.
 * Note: successful compilation on an unsupported
 * platform is desired, but only fallback code is
 * present at best.
 */
#if !defined(LIBXSMM_PLATFORM_FORCE) && 0
# define LIBXSMM_PLATFORM_FORCE
#endif

#if !defined(LIBXSMM_PLATFORM_X86) && ( \
    (defined(__x86_64__) && 0 != (__x86_64__)) || \
    (defined(__amd64__) && 0 != (__amd64__)) || \
    (defined(_M_X64) || defined(_M_AMD64)) || \
    (defined(__i386__) && 0 != (__i386__)) || \
    (defined(_M_IX86)))
# define LIBXSMM_PLATFORM_X86
#endif
#if !defined(LIBXSMM_PLATFORM_AARCH64) && \
    (defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64))
# define LIBXSMM_PLATFORM_AARCH64
#endif
#if !defined(LIBXSMM_PLATFORM_SUPPORTED)
# if defined(LIBXSMM_PLATFORM_X86) || defined(LIBXSMM_PLATFORM_AARCH64)
#   define LIBXSMM_PLATFORM_SUPPORTED
# elif !defined(LIBXSMM_PLATFORM_FORCE)
#   error LIBXSMM requires X86_64, AArch64, or compatible CPUs!
# endif
#endif
#if !defined(LIBXSMM_BITS)
# if  (defined(__SIZEOF_PTRDIFF_T__) && 4 < (__SIZEOF_PTRDIFF_T__)) || \
      (defined(__SIZE_MAX__) && (4294967295U < (__SIZE_MAX__))) || \
      (defined(__aarch64__) || defined(__arm64__)) || \
      (defined(__x86_64__) && 0 != (__x86_64__)) || \
      (defined(__amd64__) && 0 != (__amd64__)) || \
      (defined(_M_X64) || defined(_M_AMD64)) || \
      (defined(_WIN64) || defined(_M_ARM64)) || \
      (defined(__powerpc64))
#   define LIBXSMM_UNLIMITED 0xFFFFFFFFFFFFFFFF
#   define LIBXSMM_BITS 64
# elif !defined(LIBXSMM_PLATFORM_FORCE) && defined(NDEBUG)
#   error LIBXSMM is only supported on 64-bit platforms!
# else /* JIT-generated code (among other issues) is not supported! */
#   define LIBXSMM_UNLIMITED 0xFFFFFFFF
#   define LIBXSMM_BITS 32
# endif
#endif

#define LIBXSMM_STRINGIFY_AUX(SYMBOL) #SYMBOL
#define LIBXSMM_STRINGIFY(SYMBOL) LIBXSMM_STRINGIFY_AUX(SYMBOL)
#define LIBXSMM_TOSTRING(SYMBOL) LIBXSMM_STRINGIFY(SYMBOL)
#define LIBXSMM_CONCATENATE2(A, B) A##B
#define LIBXSMM_CONCATENATE3(A, B, C) LIBXSMM_CONCATENATE(LIBXSMM_CONCATENATE(A, B), C)
#define LIBXSMM_CONCATENATE4(A, B, C, D) LIBXSMM_CONCATENATE(LIBXSMM_CONCATENATE3(A, B, C), D)
#define LIBXSMM_CONCATENATE(A, B) LIBXSMM_CONCATENATE2(A, B)
#define LIBXSMM_FSYMBOL(SYMBOL) LIBXSMM_CONCATENATE(SYMBOL, _)
#define LIBXSMM_UNIQUE(NAME) LIBXSMM_CONCATENATE(NAME, __LINE__)
#define LIBXSMM_EXPAND(...) __VA_ARGS__
#define LIBXSMM_ELIDE(...)

/** MKL_DIRECT_CALL requires to include the MKL interface. */
#if (defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL) || \
    (defined(__MKL) && !defined(LIBXSMM_BUILD) && \
    (!defined(__BLAS) || (0 != __BLAS)))) && \
    (defined(LIBXSMM_PLATFORM_X86))
# if (0 != LIBXSMM_ILP64 && !defined(MKL_ILP64))
#   error "Inconsistent ILP64 configuration detected!"
# endif
# include <mkl.h>
#endif
/** __INTEL_MKL__ is needed later to fix some NOTHROW issue. */
#if defined(__MKL) && !defined(__INTEL_MKL__) && defined(LIBXSMM_PLATFORM_X86) && \
    defined(NOTHROW)
# include <mkl_version.h>
#endif

/** Use LIBXSMM_VERSION2 instead of LIBXSMM_VERSION3, e.g., if __GNUC_PATCHLEVEL__ or __clang_patchlevel__ is zero (0). */
#define LIBXSMM_VERSION2(MAJOR, MINOR) ((MAJOR) * 10000 + (MINOR) * 100)
#define LIBXSMM_VERSION3(MAJOR, MINOR, UPDATE) (LIBXSMM_VERSION2(MAJOR, MINOR) + (UPDATE))
#define LIBXSMM_VERSION4(MAJOR, MINOR, UPDATE, PATCH) \
  (((0x7F & (MAJOR)) << 24) | ((0x1F & (MINOR)) << 19) | ((0x1F & (UPDATE)) << 14) | (0x3FFF & (PATCH)))
#define LIBXSMM_VERSION41(VERSION) (((VERSION) >> 24))
#define LIBXSMM_VERSION42(VERSION) (((VERSION) >> 19) & 0x1F)
#define LIBXSMM_VERSION43(VERSION) (((VERSION) >> 14) & 0x1F)
#define LIBXSMM_VERSION44(VERSION) (((VERSION)) & 0x3FFF)

#if !defined(LIBXSMM_VERSION_NUMBER)
# define LIBXSMM_VERSION_NUMBER LIBXSMM_VERSION4(LIBXSMM_VERSION_MAJOR, \
    LIBXSMM_VERSION_MINOR, LIBXSMM_VERSION_UPDATE, LIBXSMM_VERSION_PATCH)
#endif

#define LIBXSMM_VERSION_CHECK(COMP, MAJOR, MINOR, UPDATE, PATCH) \
  (LIBXSMM_VERSION_NUMBER COMP LIBXSMM_VERSION4(MAJOR, MINOR, UPDATE, PATCH))

/**
 * Macro to check minimum version requirements in code, for example:
 * #if LIBXSMM_VERSION_GE(1, 17, 0, 0)
 * // code requiring version 1.17 or later
 * #else
 * // fallback code
 * #endif
*/
#define LIBXSMM_VERSION_GE(MAJOR, MINOR, UPDATE, PATCH) \
  LIBXSMM_VERSION_CHECK(>=, MAJOR, MINOR, UPDATE, PATCH)

/** Calculation of INTEL_MKL_VERSION has no continuous history. */
#if defined(__INTEL_MKL__) && defined(__INTEL_MKL_MINOR__) && defined(__INTEL_MKL_UPDATE__)
# define LIBXSMM_MKL_VERSION3 LIBXSMM_VERSION3(__INTEL_MKL__, __INTEL_MKL_MINOR__, __INTEL_MKL_UPDATE__)
# define LIBXSMM_MKL_VERSION2 LIBXSMM_VERSION2(__INTEL_MKL__, __INTEL_MKL_MINOR__)
#endif

/** Evaluates to true if the value falls into the interval [LO, HI]. */
#define LIBXSMM_IS_INTEGER(TYPE, VALUE, LO, HI) ( \
  ((LO) == (TYPE)(VALUE) || (LO) < (TYPE)(VALUE)) && LIBXSMM_MIN(1ULL*(VALUE),HI) <= (HI) && \
  ((0 <= (double)(VALUE) || (0 > (LO) && 0 < (HI)))))
/** LIBXSMM_IS_TYPE: check value against type-range of TYPE. */
#define LIBXSMM_IS_ULLONG(VALUE) LIBXSMM_IS_INTEGER(unsigned long long, VALUE, 0, ULLONG_MAX)
#define LIBXSMM_IS_LLONG(VALUE) LIBXSMM_IS_INTEGER(/*signed*/long long, VALUE, LLONG_MIN, LLONG_MAX)
#define LIBXSMM_IS_ULONG(VALUE) LIBXSMM_IS_INTEGER(unsigned long, VALUE, 0, ULONG_MAX)
#define LIBXSMM_IS_LONG(VALUE) LIBXSMM_IS_INTEGER(/*signed*/long, VALUE, LONG_MIN, LONG_MAX)
#define LIBXSMM_IS_USHORT(VALUE) LIBXSMM_IS_INTEGER(unsigned short, VALUE, 0, USHRT_MAX)
#define LIBXSMM_IS_SHORT(VALUE) LIBXSMM_IS_INTEGER(/*signed*/short, VALUE, SHRT_MIN, SHRT_MAX)
#define LIBXSMM_IS_UCHAR(VALUE) LIBXSMM_IS_INTEGER(unsigned char, VALUE, 0, UCHAR_MAX)
#define LIBXSMM_IS_ICHAR(VALUE) LIBXSMM_IS_INTEGER(signed char, VALUE, SCHAR_MIN, SCHAR_MAX)
#define LIBXSMM_IS_CHAR(VALUE) LIBXSMM_IS_INTEGER(char, VALUE, CHAR_MIN, CHAR_MAX)
#define LIBXSMM_IS_UINT(VALUE) LIBXSMM_IS_INTEGER(unsigned int, VALUE, 0, UINT_MAX)
#define LIBXSMM_IS_INT(VALUE) LIBXSMM_IS_INTEGER(/*signed*/int, VALUE, INT_MIN, INT_MAX)

#if !defined(LIBXSMM_CAST_CHECK)
# if !defined(__COVERITY__) && 1
#   define LIBXSMM_CAST_CHECK(EXPR, MSG) LIBXSMM_ASSERT_MSG(EXPR, MSG)
# else
#   define LIBXSMM_CAST_CHECK(EXPR, MSG) LIBXSMM_ASSERT_MSG(1, MSG)
# endif
#endif

/**
 * LIBXSMM_CAST: Perform type-cast with following two advantages:
 *               (1) Make it easy to locate/find the type-cast.
 *               (2) Range-check to ensure fitting into type.
 */
#define LIBXSMM_CAST_ULLONG(VALUE) ((unsigned long long)(LIBXSMM_CAST_CHECK(LIBXSMM_IS_ULLONG(VALUE), "Value cannot be represented as ULLONG"), VALUE))
#define LIBXSMM_CAST_LLONG(VALUE) ((/*signed*/long long)(LIBXSMM_CAST_CHECK(LIBXSMM_IS_LLONG(VALUE), "Value cannot be represented as LLONG"), VALUE))
#define LIBXSMM_CAST_ULONG(VALUE) ((unsigned long)(LIBXSMM_CAST_CHECK(LIBXSMM_IS_ULONG(VALUE), "Value cannot be represented as ULONG"), VALUE))
#define LIBXSMM_CAST_LONG(VALUE) ((/*signed*/long)(LIBXSMM_CAST_CHECK(LIBXSMM_IS_LONG(VALUE), "Value cannot be represented as LONG"), VALUE))
#define LIBXSMM_CAST_USHORT(VALUE) ((unsigned short)(LIBXSMM_CAST_CHECK(LIBXSMM_IS_USHORT(VALUE), "Value cannot be represented as USHORT"), VALUE))
#define LIBXSMM_CAST_SHORT(VALUE) ((/*signed*/short)(LIBXSMM_CAST_CHECK(LIBXSMM_IS_SHORT(VALUE), "Value cannot be represented as SHORT"), VALUE))
#define LIBXSMM_CAST_UCHAR(VALUE) ((unsigned char)(LIBXSMM_CAST_CHECK(LIBXSMM_IS_UCHAR(VALUE), "Value cannot be represented as UCHAR"), VALUE))
#define LIBXSMM_CAST_ICHAR(VALUE) ((signed char)(LIBXSMM_CAST_CHECK(LIBXSMM_IS_ICHAR(VALUE), "Value cannot be represented as ICHAR"), VALUE))
#define LIBXSMM_CAST_CHAR(VALUE) ((char)(LIBXSMM_CAST_CHECK(LIBXSMM_IS_CHAR(VALUE), "Value cannot be represented as CHAR"), VALUE))
#define LIBXSMM_CAST_UINT(VALUE) ((unsigned int)(LIBXSMM_CAST_CHECK(LIBXSMM_IS_UINT(VALUE), "Value cannot be represented as UINT"), VALUE))
#define LIBXSMM_CAST_INT(VALUE) ((/*signed*/int)(LIBXSMM_CAST_CHECK(LIBXSMM_IS_INT(VALUE), "Value cannot be represented as INT"), VALUE))

#if (0 != LIBXSMM_ILP64)
# define LIBXSMM_IS_BLASINT(VALUE) LIBXSMM_IS_LLONG(VALUE)
# define LIBXSMM_CAST_BLASINT(VALUE) LIBXSMM_CAST_LLONG(VALUE)
#else /* LP64 */
# define LIBXSMM_IS_BLASINT(VALUE) LIBXSMM_IS_INT(VALUE)
# define LIBXSMM_CAST_BLASINT(VALUE) LIBXSMM_CAST_INT(VALUE)
#endif

/* Const-qualify BLAS functions */
#if defined(LIBXSMM_BLAS_CONST)
# undef LIBXSMM_BLAS_CONST
# define LIBXSMM_BLAS_CONST const
#elif defined(OPENBLAS_CONST)
# define LIBXSMM_BLAS_CONST OPENBLAS_CONST
#elif (defined(LIBXSMM_BLAS_NONCONST) || defined(__OPENBLAS) || defined(__OPENBLAS77)) \
   && !defined(LIBXSMM_BUILD)
# define LIBXSMM_BLAS_CONST
#else
# define LIBXSMM_BLAS_CONST const
#endif

/* Control BLAS dependency */
#if !defined(LIBXSMM_NO_BLAS)
# if (!defined(__BLAS) || (0 != __BLAS)) && \
      !(defined(LIBXSMM_PLATFORM_AARCH64) && defined(_WIN32))
#   define LIBXSMM_NO_BLAS 0
#   define LIBXSMM_BLAS 1
# else
#   define LIBXSMM_NO_BLAS 1
#   define LIBXSMM_BLAS 0
# endif
#endif

#if defined(LIBXSMM_BUILD)
# if defined(LIBXSMM_BUILD_EXT) && defined(_WINDLL) && \
    (defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__))
#   define LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_APIEXT
# elif defined(LIBXSMM_NO_BLAS) && (1 == LIBXSMM_NO_BLAS)
#   define LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_API
# endif
#endif
#if !defined(LIBXSMM_BLAS_SYMBOL_VISIBILITY)
# define LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_EXTERN LIBXSMM_VISIBILITY_IMPORT
#endif

#if defined(NOTHROW)
# define LIBXSMM_BLAS_NOEXCEPT_AUX NOTHROW
#else
# define LIBXSMM_BLAS_NOEXCEPT_AUX LIBXSMM_NOEXCEPT
#endif
#define LIBXSMM_BLAS_NOEXCEPT(KIND) LIBXSMM_CONCATENATE(LIBXSMM_BLAS_NOEXCEPT_, KIND)
#if defined(LIBXSMM_MKL_VERSION3) && (LIBXSMM_VERSION3(2020, 0, 2) <= LIBXSMM_MKL_VERSION3)
# define LIBXSMM_BLAS_NOEXCEPT_gemm_batch_strided LIBXSMM_BLAS_NOEXCEPT_AUX
# define LIBXSMM_BLAS_NOEXCEPT_gemm_batch LIBXSMM_BLAS_NOEXCEPT_AUX
#else
# define LIBXSMM_BLAS_NOEXCEPT_gemm_batch_strided
# define LIBXSMM_BLAS_NOEXCEPT_gemm_batch
#endif
#define LIBXSMM_BLAS_NOEXCEPT_gemm LIBXSMM_BLAS_NOEXCEPT_AUX
#define LIBXSMM_BLAS_NOEXCEPT_gemv LIBXSMM_BLAS_NOEXCEPT_AUX

/** Construct BLAS-style prefixes. */
#define LIBXSMM_TPREFIX_NAME(TYPE) LIBXSMM_CONCATENATE(LIBXSMM_TPREFIX_, TYPE)
#define LIBXSMM_TPREFIX(TYPE, FUNCTION) LIBXSMM_CONCATENATE(LIBXSMM_TPREFIX_NAME(TYPE), FUNCTION)
#define LIBXSMM_TPREFIX_doubledouble d
#define LIBXSMM_TPREFIX_floatfloat s
/** Defaults if only the input type is specified. */
#define LIBXSMM_TPREFIX_double LIBXSMM_TPREFIX_doubledouble
#define LIBXSMM_TPREFIX_float LIBXSMM_TPREFIX_floatfloat

/* Construct prefix names, function type or dispatch function from given input and output types. */
#define LIBXSMM_TPREFIX2(ITYPE, OTYPE, FUNCTION) LIBXSMM_TPREFIX(LIBXSMM_CONCATENATE(ITYPE, OTYPE), FUNCTION)

/** Construct symbol name from a given real type name (float, double, etc.). */
#define LIBXSMM_BLAS_SYMBOL(TYPE, KIND) LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(TYPE, KIND))
#define LIBXSMM_CBLAS_SYMBOL LIBXSMM_TPREFIX

#define LIBXSMM_BLAS_SYMBOL_SIGNATURE_gemm_batch_strided(CONST_STAR, STAR, TYPE) char CONST_STAR /*transa*/, char CONST_STAR /*transb*/, \
  libxsmm_blasint CONST_STAR /*m*/, libxsmm_blasint CONST_STAR /*n*/, libxsmm_blasint CONST_STAR /*k*/, \
  TYPE CONST_STAR /*alpha*/, TYPE CONST_STAR /*a*/, libxsmm_blasint CONST_STAR /*lda*/, libxsmm_blasint CONST_STAR /*stride_a*/, \
                             TYPE CONST_STAR /*b*/, libxsmm_blasint CONST_STAR /*ldb*/, libxsmm_blasint CONST_STAR /*stride_b*/, \
  TYPE CONST_STAR /*beta*/,  TYPE       STAR /*c*/, libxsmm_blasint CONST_STAR /*ldc*/, libxsmm_blasint CONST_STAR /*stride_c*/, \
  libxsmm_blasint CONST_STAR /*batchsize*/
#define LIBXSMM_BLAS_SYMBOL_SIGNATURE_gemm_batch(CONST_STAR, STAR, TYPE) char CONST_STAR /*transa*/, char CONST_STAR /*transb*/, \
  libxsmm_blasint CONST_STAR, libxsmm_blasint CONST_STAR, libxsmm_blasint CONST_STAR, \
  TYPE CONST_STAR, TYPE CONST_STAR STAR, libxsmm_blasint CONST_STAR, TYPE CONST_STAR STAR, libxsmm_blasint CONST_STAR, \
  TYPE CONST_STAR, TYPE STAR STAR, libxsmm_blasint CONST_STAR, libxsmm_blasint CONST_STAR, libxsmm_blasint CONST_STAR
#define LIBXSMM_BLAS_SYMBOL_SIGNATURE_gemm(CONST_STAR, STAR, TYPE) char CONST_STAR /*transa*/, char CONST_STAR /*transb*/, \
  libxsmm_blasint CONST_STAR, libxsmm_blasint CONST_STAR, libxsmm_blasint CONST_STAR, TYPE CONST_STAR, TYPE CONST_STAR, libxsmm_blasint CONST_STAR, \
  TYPE CONST_STAR, libxsmm_blasint CONST_STAR, TYPE CONST_STAR, TYPE STAR, libxsmm_blasint CONST_STAR
#define LIBXSMM_BLAS_SYMBOL_SIGNATURE_gemv(CONST_STAR, STAR, TYPE) char CONST_STAR, libxsmm_blasint CONST_STAR, libxsmm_blasint CONST_STAR, \
  TYPE CONST_STAR, TYPE CONST_STAR, libxsmm_blasint CONST_STAR, TYPE CONST_STAR, libxsmm_blasint CONST_STAR, \
  TYPE CONST_STAR, TYPE STAR, libxsmm_blasint CONST_STAR
#define LIBXSMM_BLAS_SYMBOL_SIGNATURE(CONST_STAR, STAR, TYPE, KIND) LIBXSMM_CONCATENATE(LIBXSMM_BLAS_SYMBOL_SIGNATURE_, KIND)(CONST_STAR, STAR, TYPE)
#define LIBXSMM_BLAS_SYMBOL_FDECL(CONST_STAR, STAR, TYPE, KIND) LIBXSMM_BLAS_SYMBOL_VISIBILITY \
  void LIBXSMM_BLAS_SYMBOL(TYPE, KIND)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(CONST_STAR, STAR, TYPE, KIND)) LIBXSMM_BLAS_NOEXCEPT(KIND)
#define LIBXSMM_BLAS_SYMBOL_CDECL(CONST_STAR, STAR, TYPE, KIND) LIBXSMM_BLAS_SYMBOL_VISIBILITY \
  void LIBXSMM_CBLAS_SYMBOL(TYPE, KIND)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(CONST_STAR, STAR, TYPE, KIND)) LIBXSMM_BLAS_NOEXCEPT(KIND)

#define LIBXSMM_BLAS_DECL(TYPE, KIND, DECL) LIBXSMM_CONCATENATE(LIBXSMM_BLAS_, LIBXSMM_TPREFIX(TYPE, KIND))(DECL)
#if !defined(MKL_DIRECT_CALL_SEQ) && !defined(MKL_DIRECT_CALL)
# define LIBXSMM_BLAS_dgemm(DECL) DECL;
# define LIBXSMM_BLAS_sgemm(DECL) DECL;
# define LIBXSMM_BLAS_dgemv(DECL) DECL;
# define LIBXSMM_BLAS_sgemv(DECL) DECL;
#else
# define LIBXSMM_BLAS_dgemm
# define LIBXSMM_BLAS_sgemm
# define LIBXSMM_BLAS_dgemv
# define LIBXSMM_BLAS_sgemv
#endif

#if (0 != LIBXSMM_BLAS) /* BLAS available */
# define LIBXSMM_BLAS_SYMBOL_DECL(TYPE, KIND) LIBXSMM_BLAS_DECL(TYPE, KIND, LIBXSMM_BLAS_SYMBOL_FDECL(LIBXSMM_BLAS_CONST*, *, TYPE, KIND))
#else
# define LIBXSMM_BLAS_SYMBOL_DECL(TYPE, KIND)
#endif

/** Map to appropriate BLAS function (or fallback). */
#define LIBXSMM_BLAS_FUNCTION(ITYPE, OTYPE, FUNCTION) LIBXSMM_CONCATENATE(LIBXSMM_BLAS_FUNCTION_, LIBXSMM_TPREFIX2(ITYPE, OTYPE, FUNCTION))
#if (0 != LIBXSMM_BLAS)
# if defined(LIBXSMM_INIT_COMPLETED)
#   define LIBXSMM_BLAS_FUNCTION_dgemm_batch_strided libxsmm_original_dgemm_batch_strided_function
#   define LIBXSMM_BLAS_FUNCTION_sgemm_batch_strided libxsmm_original_sgemm_batch_strided_function
#   define LIBXSMM_BLAS_FUNCTION_dgemm_batch libxsmm_original_dgemm_batch_function
#   define LIBXSMM_BLAS_FUNCTION_sgemm_batch libxsmm_original_sgemm_batch_function
#   define LIBXSMM_BLAS_FUNCTION_dgemm libxsmm_original_dgemm_function
#   define LIBXSMM_BLAS_FUNCTION_sgemm libxsmm_original_sgemm_function
#   define LIBXSMM_BLAS_FUNCTION_dgemv libxsmm_original_dgemv_function
#   define LIBXSMM_BLAS_FUNCTION_sgemv libxsmm_original_sgemv_function
# else
#   define LIBXSMM_BLAS_FUNCTION_dgemm_batch_strided libxsmm_original_dgemm_batch_strided()
#   define LIBXSMM_BLAS_FUNCTION_sgemm_batch_strided libxsmm_original_sgemm_batch_strided()
#   define LIBXSMM_BLAS_FUNCTION_dgemm_batch libxsmm_original_dgemm_batch()
#   define LIBXSMM_BLAS_FUNCTION_sgemm_batch libxsmm_original_sgemm_batch()
#   define LIBXSMM_BLAS_FUNCTION_dgemm libxsmm_original_dgemm()
#   define LIBXSMM_BLAS_FUNCTION_sgemm libxsmm_original_sgemm()
#   define LIBXSMM_BLAS_FUNCTION_dgemv libxsmm_original_dgemv()
#   define LIBXSMM_BLAS_FUNCTION_sgemv libxsmm_original_sgemv()
# endif
#else /* no BLAS */
# define LIBXSMM_BLAS_FUNCTION_dgemm_batch_strided libxsmm_blas_error("dgemm_batch_strided")
# define LIBXSMM_BLAS_FUNCTION_sgemm_batch_strided libxsmm_blas_error("sgemm_batch_strided")
# define LIBXSMM_BLAS_FUNCTION_dgemm_batch libxsmm_blas_error("dgemm_batch")
# define LIBXSMM_BLAS_FUNCTION_sgemm_batch libxsmm_blas_error("sgemm_batch")
# define LIBXSMM_BLAS_FUNCTION_dgemm libxsmm_blas_error("dgemm")
# define LIBXSMM_BLAS_FUNCTION_sgemm libxsmm_blas_error("sgemm")
# define LIBXSMM_BLAS_FUNCTION_dgemv libxsmm_blas_error("dgemv")
# define LIBXSMM_BLAS_FUNCTION_sgemv libxsmm_blas_error("sgemv")
#endif

/** Short-cut macros to construct desired BLAS function symbol. */
#define LIBXSMM_GEMM_BATCH_STRIDED_SYMBOL(TYPE) LIBXSMM_BLAS_FUNCTION(TYPE, TYPE, gemm_batch_strided)
#define LIBXSMM_GEMM_BATCH_SYMBOL(TYPE) LIBXSMM_BLAS_FUNCTION(TYPE, TYPE, gemm_batch)
#define LIBXSMM_GEMM_SYMBOL(TYPE) LIBXSMM_BLAS_FUNCTION(TYPE, TYPE, gemm)
#define LIBXSMM_GEMV_SYMBOL(TYPE) LIBXSMM_BLAS_FUNCTION(TYPE, TYPE, gemv)

/** Consolidate BLAS-transpose into a set of flags. */
#define LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB) /* check for N/n rather than T/t since C/c is also valid! */ \
   ((('n' == (TRANSA) || *"N" == (TRANSA)) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_A) \
  | (('n' == (TRANSB) || *"N" == (TRANSB)) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_B))

/** Allow NULL-requests (transposes) and map to some default. */
#define LIBXSMM_GEMM_PFLAGS(TRANSA, TRANSB, DEFAULT) LIBXSMM_GEMM_FLAGS( \
  NULL != ((const void*)(TRANSA)) ? (*(const char*)(TRANSA)) : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & (DEFAULT)) ? 'n' : 't'), \
  NULL != ((const void*)(TRANSB)) ? (*(const char*)(TRANSB)) : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & (DEFAULT)) ? 'n' : 't')) \
  | (~(LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B) & (DEFAULT))

/** Consolidate CBLAS transpose requests into a set of flags. */
#define LIBXSMM_GEMM_CFLAGS(TRANSA, TRANSB) /* check for N/n rather than T/t since C/c is also valid! */ \
   ((CblasNoTrans == (TRANSA) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_A) \
  | (CblasNoTrans == (TRANSB) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_B))

/** Consolidate transpose requests into a set of flags. */
#define LIBXSMM_GEMM_VNNI_FLAGS(TRANSA, TRANSB, VNNIA, VNNIB) /* check for N/n rather than T/t since C/c is also valid! */ \
   ((('n' == (TRANSA) || *"N" == (TRANSA)) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_A) \
  | (('n' == (TRANSB) || *"N" == (TRANSB)) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_B) \
  | (('n' == (VNNIA) || *"N" == (VNNIA)) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_VNNI_A) \
  | (('n' == (VNNIB) || *"N" == (VNNIB)) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_VNNI_B))

/** Calculate problem size from M, N, and K using the correct integer type in order to cover the general case. */
#define LIBXSMM_MNK_SIZE(M, N, K) (((size_t)(M)) * ((size_t)(N)) * ((size_t)(K)))
/** Calculate total number of matrix-elements; matrices A, B, C are given per M, N, K, and emphasize (S) the C-size. */
#define LIBXSMM_SIZE(M, N, K, S) \
    (((size_t)(M) * (size_t)(K)) + ((size_t)(K) * (size_t)(N)) + \
    (((size_t)(S) * (size_t)(M) * (size_t)(N))))
/** Condition based on arithmetic intensity (AI) */
#define LIBXSMM_SMM_AI(M, N, K, S, TYPESIZE) \
    ((LIBXSMM_MNK_SIZE(M, N, K) * 2) <= ((size_t)(TYPESIZE) * 4/*AI*/ * LIBXSMM_SIZE(M, N, K, S)))
/** Determine whether an SMM is suitable, i.e., small enough. */
#if !defined(LIBXSMM_THRESHOLD_AI) /* traditional MNK-threshold */
# define LIBXSMM_SMM(M, N, K, S, TYPESIZE) (LIBXSMM_MNK_SIZE(M, N, K) <= (LIBXSMM_MAX_MNK))
#else /* threshold based on arithmetic intensity */
# define LIBXSMM_SMM LIBXSMM_SMM_AI
#endif

#if !defined(LIBXSMM_UNPACKED) && (defined(_CRAYC) || \
  (0 == LIBXSMM_SYNC)/*Windows: missing pack(pop) error*/)
# define LIBXSMM_UNPACKED
#endif
#if defined(_WIN32) && !defined(__GNUC__) && !defined(__clang__)
# define LIBXSMM_ATTRIBUTE(A) __declspec(A)
# if defined(__cplusplus)
#   define LIBXSMM_INLINE_ALWAYS __forceinline
# else
#   define LIBXSMM_INLINE_ALWAYS static __forceinline
# endif
# define LIBXSMM_ALIGNED(DECL, N) LIBXSMM_ATTRIBUTE(align(N)) DECL
# if !defined(LIBXSMM_UNPACKED)
#   define LIBXSMM_PACKED(TYPE) LIBXSMM_PRAGMA(pack(1)) TYPE
# endif
# define LIBXSMM_CDECL __cdecl
#elif (defined(__GNUC__) || defined(__clang__) || defined(__PGI))
# define LIBXSMM_ATTRIBUTE(A) __attribute__((A))
# define LIBXSMM_INLINE_ALWAYS LIBXSMM_ATTRIBUTE(always_inline) LIBXSMM_INLINE
# define LIBXSMM_ALIGNED(DECL, N) LIBXSMM_ATTRIBUTE(aligned(N)) DECL
# if !defined(LIBXSMM_UNPACKED)
#   define LIBXSMM_PACKED(TYPE) TYPE LIBXSMM_ATTRIBUTE(__packed__)
# endif
# define LIBXSMM_CDECL LIBXSMM_ATTRIBUTE(cdecl)
#else
# define LIBXSMM_ATTRIBUTE(A)
# define LIBXSMM_INLINE_ALWAYS LIBXSMM_INLINE
# define LIBXSMM_ALIGNED(DECL, N) DECL
# define LIBXSMM_CDECL
#endif
#if !defined(LIBXSMM_PACKED)
# define LIBXSMM_PACKED(TYPE) TYPE
# if !defined(LIBXSMM_UNPACKED)
#   define LIBXSMM_UNPACKED
# endif
#endif
#if !defined(LIBXSMM_UNPACKED) && 0
/* no braces around EXPR */
# define LIBXSMM_PAD(EXPR) EXPR;
#endif
#if !defined(LIBXSMM_PAD)
# define LIBXSMM_PAD(EXPR)
#endif

#if defined(__INTEL_COMPILER)
# if !defined(__INTEL_COMPILER_UPDATE)
#   define LIBXSMM_INTEL_COMPILER __INTEL_COMPILER
# else
#   define LIBXSMM_INTEL_COMPILER (__INTEL_COMPILER + __INTEL_COMPILER_UPDATE)
# endif
#elif defined(__INTEL_COMPILER_BUILD_DATE)
# define LIBXSMM_INTEL_COMPILER ((__INTEL_COMPILER_BUILD_DATE / 10000 - 2000) * 100)
#endif

/* LIBXSMM_ATTRIBUTE_USED: mark library functions as used to avoid warning */
#if defined(__GNUC__) || defined(__clang__) || (defined(__INTEL_COMPILER) && !defined(_WIN32))
# if !defined(__cplusplus) || !defined(__clang__)
#   define LIBXSMM_ATTRIBUTE_COMMON LIBXSMM_ATTRIBUTE(common)
# else
#   define LIBXSMM_ATTRIBUTE_COMMON
# endif
# define LIBXSMM_ATTRIBUTE_MALLOC LIBXSMM_ATTRIBUTE(malloc)
# define LIBXSMM_ATTRIBUTE_UNUSED LIBXSMM_ATTRIBUTE(unused)
# define LIBXSMM_ATTRIBUTE_USED LIBXSMM_ATTRIBUTE(used)
#else
# if defined(_WIN32) && !defined(LIBXSMM_PLATFORM_AARCH64)
#   define LIBXSMM_ATTRIBUTE_COMMON LIBXSMM_ATTRIBUTE(selectany)
# else
#   define LIBXSMM_ATTRIBUTE_COMMON
# endif
# define LIBXSMM_ATTRIBUTE_MALLOC
# define LIBXSMM_ATTRIBUTE_UNUSED
# define LIBXSMM_ATTRIBUTE_USED
#endif
#if !defined(__INTEL_COMPILER) && (defined(__clang__) /*|| defined(__PGLLVM__)*/)
# define LIBXSMM_ATTRIBUTE_NO_SANITIZE(KIND) LIBXSMM_ATTRIBUTE(no_sanitize(LIBXSMM_STRINGIFY(KIND)))
#elif defined(__GNUC__) && LIBXSMM_VERSION2(4, 8) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__) \
  && !defined(__INTEL_COMPILER) && !defined(__PGLLVM__)
# define LIBXSMM_ATTRIBUTE_NO_SANITIZE(KIND) LIBXSMM_ATTRIBUTE(LIBXSMM_CONCATENATE(no_sanitize_, KIND))
#else
# define LIBXSMM_ATTRIBUTE_NO_SANITIZE(KIND)
#endif

#if defined(__cplusplus)
# define LIBXSMM_VARIADIC ...
# define LIBXSMM_EXTERN extern "C"
# define LIBXSMM_EXTERN_C extern "C"
# define LIBXSMM_INLINE_KEYWORD inline
# define LIBXSMM_INLINE LIBXSMM_INLINE_KEYWORD
# if defined(__GNUC__) || defined(_CRAYC)
#   define LIBXSMM_CALLER __PRETTY_FUNCTION__
# elif defined(_MSC_VER)
#   define LIBXSMM_CALLER __FUNCDNAME__
#   define LIBXSMM_FUNCNAME __FUNCTION__
# else
#   define LIBXSMM_CALLER __FUNCNAME__
# endif
/** Function argument with default value (C++). */
# define LIBXSMM_ARGDEF(ARG, DEF) ARG = DEF
#else /* C */
# define LIBXSMM_VARIADIC
# define LIBXSMM_EXTERN extern
# define LIBXSMM_EXTERN_C
# if defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__) /*C99*/
#   define LIBXSMM_PRAGMA(DIRECTIVE) _Pragma(LIBXSMM_STRINGIFY(DIRECTIVE))
#   define LIBXSMM_CALLER __func__
#   define LIBXSMM_RESTRICT restrict
#   define LIBXSMM_INLINE_KEYWORD inline
# elif defined(_MSC_VER)
#   define LIBXSMM_CALLER __FUNCDNAME__
#   define LIBXSMM_FUNCNAME __FUNCTION__
#   define LIBXSMM_INLINE_KEYWORD __inline
#   define LIBXSMM_INLINE_FIXUP
# elif defined(__GNUC__) && !defined(__STRICT_ANSI__)
#   define LIBXSMM_CALLER __PRETTY_FUNCTION__
# endif
# if !defined(LIBXSMM_INLINE_KEYWORD)
#   define LIBXSMM_INLINE_KEYWORD
#   define LIBXSMM_INLINE_FIXUP
# endif
/** Function argument with default value (C++). */
# define LIBXSMM_ARGDEF(ARG, DEF) ARG
/* LIBXSMM_ATTRIBUTE_USED: increases compile-time of header-only by a large factor */
# define LIBXSMM_INLINE static LIBXSMM_INLINE_KEYWORD LIBXSMM_ATTRIBUTE_UNUSED
#endif /*__cplusplus*/
#if !defined(LIBXSMM_FUNCNAME)
# if defined(LIBXSMM_CALLER)
#   define LIBXSMM_FUNCNAME LIBXSMM_CALLER
# else
#   define LIBXSMM_CALLER NULL
#   define LIBXSMM_FUNCNAME ""
# endif
#endif
#if !defined(LIBXSMM_CALLER_ID)
# if defined(__GNUC__) || 1
#   define LIBXSMM_CALLER_ID ((const void*)((uintptr_t)libxsmm_hash_string(LIBXSMM_CALLER)))
# else /* assume no string-pooling (perhaps unsafe) */
#   define LIBXSMM_CALLER_ID LIBXSMM_CALLER
# endif
#endif

/* may include Clang and other compatible compilers */
#if defined(__GNUC__) && !defined(_WIN32) && !defined(__CYGWIN__) && !defined(__MINGW32__)
# define LIBXSMM_VISIBILITY_INTERNAL LIBXSMM_ATTRIBUTE(visibility("internal"))
# define LIBXSMM_VISIBILITY_HIDDEN LIBXSMM_ATTRIBUTE(visibility("hidden"))
# define LIBXSMM_VISIBILITY_PUBLIC LIBXSMM_ATTRIBUTE(visibility("default"))
#endif
#if !defined(LIBXSMM_VISIBILITY_INTERNAL)
# define LIBXSMM_VISIBILITY_INTERNAL
#endif
#if !defined(LIBXSMM_VISIBILITY_HIDDEN)
# define LIBXSMM_VISIBILITY_HIDDEN
#endif
#if !defined(LIBXSMM_VISIBILITY_PUBLIC)
# define LIBXSMM_VISIBILITY_PUBLIC
#endif
#if !defined(LIBXSMM_VISIBILITY_PRIVATE)
# define LIBXSMM_VISIBILITY_PRIVATE LIBXSMM_VISIBILITY_HIDDEN
#endif

/* Windows Dynamic Link Library (DLL) */
#if defined(_WINDLL) && (defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__))
# define LIBXSMM_VISIBILITY_EXPORT LIBXSMM_ATTRIBUTE(dllexport)
# define LIBXSMM_VISIBILITY_IMPORT LIBXSMM_ATTRIBUTE(dllimport)
#endif
#if !defined(LIBXSMM_VISIBILITY_EXPORT)
# define LIBXSMM_VISIBILITY_EXPORT LIBXSMM_VISIBILITY_PUBLIC
#endif
#if !defined(LIBXSMM_VISIBILITY_IMPORT)
# define LIBXSMM_VISIBILITY_IMPORT LIBXSMM_VISIBILITY_PUBLIC
#endif

#if defined(LIBXSMM_SOURCE_H) /* header-only mode */
# define LIBXSMM_API_VISIBILITY_EXPORT
# define LIBXSMM_API_VISIBILITY_IMPORT
# define LIBXSMM_API_VISIBILITY_INTERN
# define LIBXSMM_API_COMMON LIBXSMM_ATTRIBUTE_COMMON
# define LIBXSMM_API_TARGET LIBXSMM_API_INLINE
# define LIBXSMM_API_EXTERN LIBXSMM_EXTERN_C
#else /* classic ABI */
# if defined(LIBXSMM_BUILD_EXT)
#   define LIBXSMM_API_VISIBILITY_EXPORT LIBXSMM_VISIBILITY_IMPORT
#   define LIBXSMM_API_VISIBILITY_IMPORT LIBXSMM_VISIBILITY_EXPORT
#   define LIBXSMM_API_VISIBILITY_INTERN LIBXSMM_VISIBILITY_PRIVATE
# elif defined(LIBXSMM_BUILD)
#   define LIBXSMM_API_VISIBILITY_EXPORT LIBXSMM_VISIBILITY_EXPORT
#   define LIBXSMM_API_VISIBILITY_IMPORT LIBXSMM_VISIBILITY_IMPORT
#   define LIBXSMM_API_VISIBILITY_INTERN LIBXSMM_VISIBILITY_PRIVATE
# else /* import */
#   define LIBXSMM_API_VISIBILITY_EXPORT LIBXSMM_VISIBILITY_IMPORT
#   define LIBXSMM_API_VISIBILITY_IMPORT LIBXSMM_VISIBILITY_IMPORT
#   define LIBXSMM_API_VISIBILITY_INTERN
# endif
# define LIBXSMM_API_EXTERN LIBXSMM_EXTERN
# define LIBXSMM_API_COMMON
# define LIBXSMM_API_TARGET
#endif

#define LIBXSMM_API_VISIBILITY(VISIBILITY) LIBXSMM_CONCATENATE(LIBXSMM_API_VISIBILITY_, VISIBILITY)
#define LIBXSMM_APIVAR(DECL, VISIBILITY, EXTERN) EXTERN LIBXSMM_API_COMMON LIBXSMM_API_VISIBILITY(VISIBILITY) DECL
#define LIBXSMM_API_INLINE LIBXSMM_INLINE
#define LIBXSMM_API_DEF

#if (!defined(__INTEL_COMPILER) || !defined(_WIN32))
#define LIBXSMM_APIVAR_ALIGNED(DECL, VISIBILITY) LIBXSMM_ALIGNED(LIBXSMM_APIVAR(DECL, VISIBILITY, LIBXSMM_API_DEF), LIBXSMM_CONFIG_CACHELINE)
#else
#define LIBXSMM_APIVAR_ALIGNED(DECL, VISIBILITY) LIBXSMM_APIVAR(DECL, VISIBILITY, LIBXSMM_API_DEF)
#endif

/** Public variable declaration (without definition) located in header file. */
#define LIBXSMM_APIVAR_PUBLIC(DECL) LIBXSMM_APIVAR(DECL, EXPORT, LIBXSMM_API_EXTERN)
/** Public variable definition (complements declaration) located in source file. */
#define LIBXSMM_APIVAR_PUBLIC_DEF(DECL) LIBXSMM_APIVAR_ALIGNED(DECL, EXPORT)
/** Private variable declaration (without definition) located in header file. */
#define LIBXSMM_APIVAR_PRIVATE(DECL) LIBXSMM_APIVAR(DECL, INTERN, LIBXSMM_API_EXTERN)
/** Private variable definition (complements declaration) located in source file. */
#define LIBXSMM_APIVAR_PRIVATE_DEF(DECL) LIBXSMM_APIVAR_ALIGNED(DECL, INTERN)
/** Private variable (declaration and definition) located in source file. */
#define LIBXSMM_APIVAR_DEFINE(DECL) LIBXSMM_APIVAR_PRIVATE(DECL); LIBXSMM_APIVAR_PRIVATE_DEF(DECL)
/** Function decoration used for private functions. */
#define LIBXSMM_API_INTERN LIBXSMM_API_EXTERN LIBXSMM_API_TARGET LIBXSMM_API_VISIBILITY(INTERN)
/** Function decoration used for public functions of LIBXSMMext library. */
#define LIBXSMM_APIEXT LIBXSMM_API_EXTERN LIBXSMM_API_TARGET LIBXSMM_API_VISIBILITY(IMPORT)
/** Function decoration used for public functions of LIBXSMM library. */
#define LIBXSMM_API LIBXSMM_API_EXTERN LIBXSMM_API_TARGET LIBXSMM_API_VISIBILITY(EXPORT)

#if !defined(LIBXSMM_RESTRICT)
# if ((defined(__GNUC__) && !defined(__CYGWIN32__)) || defined(LIBXSMM_INTEL_COMPILER)) && !defined(_WIN32)
#   define LIBXSMM_RESTRICT __restrict__
# elif defined(_MSC_VER) || defined(LIBXSMM_INTEL_COMPILER)
#   define LIBXSMM_RESTRICT __restrict
# else
#   define LIBXSMM_RESTRICT
# endif
#endif /*LIBXSMM_RESTRICT*/

#if !defined(LIBXSMM_PRAGMA)
# if defined(LIBXSMM_INTEL_COMPILER) || defined(_MSC_VER)
#   define LIBXSMM_PRAGMA(DIRECTIVE) __pragma(LIBXSMM_EXPAND(DIRECTIVE))
# else
#   define LIBXSMM_PRAGMA(DIRECTIVE)
# endif
#endif

#if !defined(LIBXSMM_OPENMP_SIMD)
# if defined(LIBXSMM_INTEL_COMPILER) && (1500 <= LIBXSMM_INTEL_COMPILER)
#   define LIBXSMM_OPENMP_SIMD
# elif defined(_OPENMP) && (201307/*v4.0*/ <= _OPENMP)
#   define LIBXSMM_OPENMP_SIMD
# endif
#endif

#if !defined(LIBXSMM_INTEL_COMPILER) || (LIBXSMM_INTEL_COMPILER < 9900)
# if defined(LIBXSMM_OPENMP_SIMD)
#   define LIBXSMM_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXSMM_PRAGMA(omp simd reduction(EXPRESSION))
#   define LIBXSMM_PRAGMA_SIMD_COLLAPSE(N) LIBXSMM_PRAGMA(omp simd collapse(N))
#   define LIBXSMM_PRAGMA_SIMD_PRIVATE(...) LIBXSMM_PRAGMA(omp simd private(__VA_ARGS__))
#   define LIBXSMM_PRAGMA_SIMD LIBXSMM_PRAGMA(omp simd)
# elif defined(__INTEL_COMPILER)
#   define LIBXSMM_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXSMM_PRAGMA(simd reduction(EXPRESSION))
#   define LIBXSMM_PRAGMA_SIMD_COLLAPSE(N) LIBXSMM_PRAGMA(simd collapse(N))
#   define LIBXSMM_PRAGMA_SIMD_PRIVATE(...) LIBXSMM_PRAGMA(simd private(__VA_ARGS__))
#   define LIBXSMM_PRAGMA_SIMD LIBXSMM_PRAGMA(simd)
# endif
#endif
#if !defined(LIBXSMM_PRAGMA_SIMD)
# define LIBXSMM_PRAGMA_SIMD_REDUCTION(EXPRESSION)
# define LIBXSMM_PRAGMA_SIMD_COLLAPSE(N)
# define LIBXSMM_PRAGMA_SIMD_PRIVATE(...)
# define LIBXSMM_PRAGMA_SIMD
#endif

#if defined(__INTEL_COMPILER)
# define LIBXSMM_PRAGMA_NONTEMPORAL(...) LIBXSMM_PRAGMA(vector nontemporal(__VA_ARGS__))
# define LIBXSMM_PRAGMA_VALIGNED LIBXSMM_PRAGMA(vector aligned)
# define LIBXSMM_PRAGMA_NOVECTOR LIBXSMM_PRAGMA(novector)
# define LIBXSMM_PRAGMA_FORCEINLINE LIBXSMM_PRAGMA(forceinline)
# define LIBXSMM_PRAGMA_LOOP_COUNT(MIN, MAX, AVG) LIBXSMM_PRAGMA(loop_count min=MIN max=MAX avg=AVG)
# define LIBXSMM_PRAGMA_UNROLL_AND_JAM(N) LIBXSMM_PRAGMA(unroll_and_jam(N))
# define LIBXSMM_PRAGMA_UNROLL_N(N) LIBXSMM_PRAGMA(unroll(N))
# define LIBXSMM_PRAGMA_UNROLL LIBXSMM_PRAGMA(unroll)
# define LIBXSMM_PRAGMA_VALIGNED_VAR(A) LIBXSMM_ASSUME_ALIGNED(A, LIBXSMM_ALIGNMENT);
/*# define LIBXSMM_UNUSED(VARIABLE) LIBXSMM_PRAGMA(unused(VARIABLE))*/
#else
# if defined(LIBXSMM_OPENMP_SIMD) && (201811/*v5.0*/ <= _OPENMP) && !defined(__PGLLVM__)
#   define LIBXSMM_PRAGMA_NONTEMPORAL(...) LIBXSMM_PRAGMA(omp simd nontemporal(__VA_ARGS__))
# else
#   define LIBXSMM_PRAGMA_NONTEMPORAL(...)
# endif
# if defined(__clang__)
#   define LIBXSMM_PRAGMA_VALIGNED_VAR(A)
#   define LIBXSMM_PRAGMA_VALIGNED
#   define LIBXSMM_PRAGMA_NOVECTOR LIBXSMM_PRAGMA(clang loop vectorize(disable))
#   define LIBXSMM_PRAGMA_FORCEINLINE
#   define LIBXSMM_PRAGMA_LOOP_COUNT(MIN, MAX, AVG) LIBXSMM_PRAGMA(unroll(AVG))
#   define LIBXSMM_PRAGMA_UNROLL_AND_JAM(N) LIBXSMM_PRAGMA(unroll(N))
#   define LIBXSMM_PRAGMA_UNROLL_N(N) LIBXSMM_PRAGMA(unroll(N))
#   define LIBXSMM_PRAGMA_UNROLL LIBXSMM_PRAGMA_UNROLL_N(4)
# else
#   define LIBXSMM_PRAGMA_VALIGNED_VAR(A)
#   define LIBXSMM_PRAGMA_VALIGNED
#   define LIBXSMM_PRAGMA_NOVECTOR
#   define LIBXSMM_PRAGMA_FORCEINLINE
#   define LIBXSMM_PRAGMA_LOOP_COUNT(MIN, MAX, AVG)
#   define LIBXSMM_PRAGMA_UNROLL_AND_JAM(N)
#   define LIBXSMM_PRAGMA_UNROLL
# endif
#endif
#if !defined(LIBXSMM_PRAGMA_UNROLL_N)
# if defined(__GNUC__) && (LIBXSMM_VERSION2(8, 3) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__))
#   define LIBXSMM_PRAGMA_UNROLL_N(N) LIBXSMM_PRAGMA(GCC unroll N)
# else
#   define LIBXSMM_PRAGMA_UNROLL_N(N)
# endif
#endif

#if !defined(__INTEL_COMPILER)
# if defined(__clang__)
#   define LIBXSMM_PRAGMA_DIAG_PUSH()     LIBXSMM_PRAGMA(clang diagnostic push)
#   define LIBXSMM_PRAGMA_DIAG_POP()      LIBXSMM_PRAGMA(clang diagnostic pop)
#   define LIBXSMM_PRAGMA_DIAG_OFF(DIAG)  LIBXSMM_PRAGMA(clang diagnostic ignored DIAG)
#   define LIBXSMM_PRAGMA_DIAG
# elif defined(__GNUC__) && LIBXSMM_VERSION2(4, 6) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__)
#   define LIBXSMM_PRAGMA_DIAG_PUSH()     LIBXSMM_PRAGMA(GCC diagnostic push)
#   define LIBXSMM_PRAGMA_DIAG_POP()      LIBXSMM_PRAGMA(GCC diagnostic pop)
#   define LIBXSMM_PRAGMA_DIAG_OFF(DIAG)  LIBXSMM_PRAGMA(GCC diagnostic ignored DIAG)
#   define LIBXSMM_PRAGMA_DIAG
# endif
#endif
#if !defined(LIBXSMM_PRAGMA_DIAG)
# define LIBXSMM_PRAGMA_DIAG_PUSH()
# define LIBXSMM_PRAGMA_DIAG_POP()
# define LIBXSMM_PRAGMA_DIAG_OFF(DIAG)
#endif

#if defined(LIBXSMM_INTEL_COMPILER)
# define LIBXSMM_PRAGMA_OPTIMIZE_OFF LIBXSMM_PRAGMA(optimize("", off))
# define LIBXSMM_PRAGMA_OPTIMIZE_ON  LIBXSMM_PRAGMA(optimize("", on))
#elif defined(__clang__)
# define LIBXSMM_PRAGMA_OPTIMIZE_OFF LIBXSMM_PRAGMA(clang optimize off)
# define LIBXSMM_PRAGMA_OPTIMIZE_ON  LIBXSMM_PRAGMA(clang optimize on)
#elif defined(__GNUC__)
# define LIBXSMM_PRAGMA_OPTIMIZE_OFF LIBXSMM_PRAGMA(GCC push_options) LIBXSMM_PRAGMA(GCC optimize("O0"))
# define LIBXSMM_PRAGMA_OPTIMIZE_ON  LIBXSMM_PRAGMA(GCC pop_options)
#else
# define LIBXSMM_PRAGMA_OPTIMIZE_OFF
# define LIBXSMM_PRAGMA_OPTIMIZE_ON
#endif

#if defined(_OPENMP) && (200805/*v3.0*/ <= _OPENMP) \
 && defined(NDEBUG) /* CCE complains for debug builds */
# define LIBXSMM_OPENMP_COLLAPSE(N) collapse(N)
#else
# define LIBXSMM_OPENMP_COLLAPSE(N)
#endif

/** LIBXSMM_UP2POT rounds up to the next power of two (POT). */
#define LIBXSMM_UP2POT_01(N) ((N) | ((N) >> 1))
#define LIBXSMM_UP2POT_02(N) (LIBXSMM_UP2POT_01(N) | (LIBXSMM_UP2POT_01(N) >> 2))
#define LIBXSMM_UP2POT_04(N) (LIBXSMM_UP2POT_02(N) | (LIBXSMM_UP2POT_02(N) >> 4))
#define LIBXSMM_UP2POT_08(N) (LIBXSMM_UP2POT_04(N) | (LIBXSMM_UP2POT_04(N) >> 8))
#define LIBXSMM_UP2POT_16(N) (LIBXSMM_UP2POT_08(N) | (LIBXSMM_UP2POT_08(N) >> 16))
#define LIBXSMM_UP2POT_32(N) (LIBXSMM_UP2POT_16(N) | (LIBXSMM_UP2POT_16(N) >> 32))
#define LIBXSMM_UP2POT(N) (LIBXSMM_UP2POT_32((unsigned long long)(N) - LIBXSMM_MIN(1, N)) + LIBXSMM_MIN(1, N))
#define LIBXSMM_LO2POT(N) (LIBXSMM_UP2POT_32((unsigned long long)(N) >> 1) + LIBXSMM_MIN(1, N))

#define LIBXSMM_UPDIV(N, MULT) (((N) + ((MULT) - 1)) / (MULT))
#define LIBXSMM_UP(N, MULT) (LIBXSMM_UPDIV(N, MULT) * (MULT))
#define LIBXSMM_LO2(N, NPOT) ((N) & ~((NPOT) - 1))
#define LIBXSMM_UP2(N, NPOT) LIBXSMM_LO2((N) + ((NPOT) - 1), NPOT)
/** Examples: N+10%->UPF(N,1,10), N-10%->UPF(N,-1,10), N*90%->UPF(N,-1,10) */
#define LIBXSMM_UPF(N, NOM, DEN) (((N) * ((DEN) + (NOM))) / (DEN))
#define LIBXSMM_SIGN(A) (0 < (A) ? (1) : ( 0 == (A) ? (0) : (-1)))
#define LIBXSMM_ABS(A) (0 <= (A) ? (A) : -(A))
#define LIBXSMM_MIN(A, B) ((A) < (B) ? (A) : (B))
#define LIBXSMM_MAX(A, B) ((A) < (B) ? (B) : (A))
#define LIBXSMM_MOD(A, N) ((A) % (N))
#define LIBXSMM_MOD2(A, NPOT) ((A) & ((NPOT) - 1))
#define LIBXSMM_DELTA(T0, T1) ((T0) < (T1) ? ((T1) - (T0)) : ((T0) - (T1)))
#define LIBXSMM_CLMP(VALUE, LO, HI) ((LO) < (VALUE) ? ((VALUE) <= (HI) ? (VALUE) : LIBXSMM_MIN(VALUE, HI)) : LIBXSMM_MAX(LO, VALUE))
#define LIBXSMM_SIZEOF(START, LAST) (((const char*)(LAST)) - ((const char*)(START)) + sizeof(*LAST))
#define LIBXSMM_FEQ(A, B) ((A) == (B))
#define LIBXSMM_NEQ(A, B) ((A) != (B))
#define LIBXSMM_ISPOT(A) (0 != (A) && !((A) & ((A) - 1)))
#define LIBXSMM_ISWAP(A, B) (((A) ^= (B)), ((B) ^= (A)), ((A) ^= (B)))
#define LIBXSMM_ISNAN(A)  LIBXSMM_NEQ(A, A)
#define LIBXSMM_NOTNAN(A) LIBXSMM_FEQ(A, A)
#define LIBXSMM_ROUNDX(TYPE, A) ((TYPE)((long long)(0 <= (A) ? ((double)(A) + 0.5) : ((double)(A) - 0.5))))
#define LIBXSMM_NEARBYINTX(TYPE, A) ((TYPE)((long long)(LIBXSMM_ROUNDX(TYPE, ((double)(A) / 2.0)) * 2)))
#define LIBXSMM_CONST_VOID_PTR(A) *((const void**)&(A))
#define LIBXSMM_EOR(ENUM_TYPE, ENUM, FLAG) ((ENUM_TYPE)(((int)(ENUM)) | ((int)(FLAG))))

LIBXSMM_API_INLINE   signed long long libxsmm_widen_u32i64(unsigned int value) { return value; }
LIBXSMM_API_INLINE unsigned long long libxsmm_widen_u32u64(unsigned int value) { return value; }
#define LIBXSMM_WIDEN_U32I64(VALUE) libxsmm_widen_u32i64(LIBXSMM_CAST_UINT(VALUE))
#define LIBXSMM_WIDEN_U32U64(VALUE) libxsmm_widen_u32u64(LIBXSMM_CAST_UINT(VALUE))

/** Makes some functions available independent of C99 support. */
#if defined(__STDC_VERSION__) && (199901L/*C99*/ <= __STDC_VERSION__)
# if defined(__PGI)
#   define LIBXSMM_POWF(A, B) ((float)pow((float)(A), (float)(B)))
# else
#   define LIBXSMM_POWF(A, B) powf(A, B)
# endif
# define LIBXSMM_FREXPF(A, B) frexpf(A, B)
# define LIBXSMM_ROUNDF(A) roundf(A)
# define LIBXSMM_ROUND(A) round(A)
# define LIBXSMM_NEARBYINTF(A) nearbyintf(A)
# define LIBXSMM_NEARBYINT(A) nearbyint(A)
# define LIBXSMM_TANHF(A) tanhf(A)
# define LIBXSMM_SQRTF(A) sqrtf(A)
# define LIBXSMM_EXP2F(A) exp2f(A)
# define LIBXSMM_LOG2F(A) log2f(A)
# define LIBXSMM_FABSF(A) fabsf(A)
# define LIBXSMM_FABS(A) fabs(A)
# define LIBXSMM_ERFF(A) erff(A)
# define LIBXSMM_EXP2(A) exp2(A)
# define LIBXSMM_LOG2(A) log2(A)
# define LIBXSMM_EXPF(A) expf(A)
# define LIBXSMM_LOGF(A) logf(A)
#else
# define LIBXSMM_POWF(A, B) ((float)pow((float)(A), (float)(B)))
# define LIBXSMM_FREXPF(A, B) ((float)frexp((float)(A), B))
# define LIBXSMM_ROUNDF(A) LIBXSMM_ROUNDX(float, A)
# define LIBXSMM_ROUND(A) LIBXSMM_ROUNDX(double, A)
# define LIBXSMM_NEARBYINTF(A) LIBXSMM_NEARBYINTX(float, A)
# define LIBXSMM_NEARBYINT(A) LIBXSMM_NEARBYINTX(double, A)
# define LIBXSMM_TANHF(A) ((float)tanh((float)(A)))
# define LIBXSMM_SQRTF(A) ((float)sqrt((float)(A)))
# define LIBXSMM_EXP2F(A) LIBXSMM_POWF(2, A)
# define LIBXSMM_LOG2F(A) ((float)LIBXSMM_LOG2((float)(A)))
# define LIBXSMM_FABSF(A) ((float)fabs((float)(A)))
# define LIBXSMM_FABS(A) fabs(A)
# define LIBXSMM_ERFF(A) ((float)erf((float)(A)))
# define LIBXSMM_EXP2(A) pow(2.0, A)
# define LIBXSMM_LOG2(A) (log(A) * (1.0 / (M_LN2)))
# define LIBXSMM_EXPF(A) ((float)exp((float)(A)))
# define LIBXSMM_LOGF(A) ((float)log((float)(A)))
#endif

#if defined(LIBXSMM_INTEL_COMPILER)
# if (1700 <= LIBXSMM_INTEL_COMPILER)
#   define LIBXSMM_ASSUME(EXPRESSION) __assume(EXPRESSION)
# else
#   define LIBXSMM_ASSUME(EXPRESSION) assert(EXPRESSION)
# endif
#elif defined(_MSC_VER)
# define LIBXSMM_ASSUME(EXPRESSION) __assume(EXPRESSION)
#elif defined(__GNUC__) && !defined(_CRAYC) && (LIBXSMM_VERSION2(4, 5) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__))
# define LIBXSMM_ASSUME(EXPRESSION) do { if (!(EXPRESSION)) __builtin_unreachable(); } while(0)
#else
# define LIBXSMM_ASSUME(EXPRESSION) assert(EXPRESSION)
#endif

#if defined(__INTEL_COMPILER)
# define LIBXSMM_ASSUME_ALIGNED(A, N) __assume_aligned(A, N)
#else
# define LIBXSMM_ASSUME_ALIGNED(A, N) assert(0 == ((uintptr_t)(A)) % (N))
#endif
#define LIBXSMM_ALIGN(POINTER, ALIGNMENT/*POT*/) ((POINTER) + (LIBXSMM_UP2((uintptr_t)(POINTER), ALIGNMENT) - ((uintptr_t)(POINTER))) / sizeof(*(POINTER)))
#define LIBXSMM_FOLD2(POINTER, ALIGNMENT, NPOT) LIBXSMM_MOD2(((uintptr_t)(POINTER) / (ALIGNMENT)), NPOT)

#if defined(_MSC_VER) && !defined(__clang__) && !defined(LIBXSMM_INTEL_COMPILER) /* account for incorrect handling of __VA_ARGS__ */
# define LIBXSMM_SELECT_ELEMENT(INDEX1/*one-based*/, .../*elements*/) LIBXSMM_CONCATENATE(LIBXSMM_SELECT_ELEMENT_, INDEX1)LIBXSMM_EXPAND((__VA_ARGS__))
#else
# define LIBXSMM_SELECT_ELEMENT(INDEX1/*one-based*/, .../*elements*/) LIBXSMM_CONCATENATE(LIBXSMM_SELECT_ELEMENT_, INDEX1)(__VA_ARGS__)
#endif
#define  LIBXSMM_SELECT_ELEMENT_1(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E0
#define  LIBXSMM_SELECT_ELEMENT_2(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E1
#define  LIBXSMM_SELECT_ELEMENT_3(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E2
#define  LIBXSMM_SELECT_ELEMENT_4(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E3
#define  LIBXSMM_SELECT_ELEMENT_5(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E4
#define  LIBXSMM_SELECT_ELEMENT_6(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E5
#define  LIBXSMM_SELECT_ELEMENT_7(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E6
#define  LIBXSMM_SELECT_ELEMENT_8(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E7
#define  LIBXSMM_SELECT_ELEMENT_9(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E8
#define LIBXSMM_SELECT_ELEMENT_10(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E9
#define LIBXSMM_SELECT_HEAD_AUX(A, ...) (A)
#define LIBXSMM_SELECT_HEAD(...) LIBXSMM_EXPAND(LIBXSMM_SELECT_HEAD_AUX(__VA_ARGS__, 0/*dummy*/))
#define LIBXSMM_SELECT_TAIL(A, ...) __VA_ARGS__

/**
 * For VLAs, check EXACTLY for C99 since a C11-conforming compiler may not provide VLAs.
 * However, some compilers (Intel) may signal support for VLA even with strict ANSI (C89).
 * To ultimately disable VLA-support, define LIBXSMM_NO_VLA (make VLA=0).
 * VLA-support is signaled by LIBXSMM_VLA.
 */
#if !defined(LIBXSMM_VLA) && !defined(LIBXSMM_NO_VLA) && !defined(__PGI) && ( \
    (defined(__STDC_VERSION__) && (199901L/*C99*/ == __STDC_VERSION__ || (!defined(__STDC_NO_VLA__) && 199901L/*C99*/ < __STDC_VERSION__))) || \
    (defined(__GNUC__) && LIBXSMM_VERSION2(5, 0) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__) && !defined(__STRICT_ANSI__) && !defined(__cplusplus)) || \
    (defined(LIBXSMM_INTEL_COMPILER) && !defined(_WIN32) && !defined(__cplusplus)) || \
    (defined(__INTEL_COMPILER) && !defined(_WIN32)))
# define LIBXSMM_VLA
#endif

/**
 * LIBXSMM_INDEX1 calculates the linear address for a given set of (multiple) indexes/bounds.
 * Syntax: LIBXSMM_INDEX1(<ndims>, <i0>, ..., <i(ndims-1)>, <s1>, ..., <s(ndims-1)>).
 * Please note that the leading dimension (s0) is omitted in the above syntax!
 * TODO: support leading dimension (pitch/stride).
 */
#if defined(_MSC_VER) && !defined(__clang__) /* account for incorrect handling of __VA_ARGS__ */
# define LIBXSMM_INDEX1(NDIMS, ...) LIBXSMM_CONCATENATE(LIBXSMM_INDEX1_, NDIMS)LIBXSMM_EXPAND((__VA_ARGS__))
#else
# define LIBXSMM_INDEX1(NDIMS, ...) LIBXSMM_CONCATENATE(LIBXSMM_INDEX1_, NDIMS)(__VA_ARGS__)
#endif
#define  LIBXSMM_INDEX1_1(...) ((size_t)LIBXSMM_SELECT_HEAD(__VA_ARGS__))
#define  LIBXSMM_INDEX1_2(I0, I1, S1) (LIBXSMM_INDEX1_1(I0) * ((size_t)S1) + (size_t)I1)
#define  LIBXSMM_INDEX1_3(I0, I1, I2, S1, S2) (LIBXSMM_INDEX1_2(I0, I1, S1) * ((size_t)S2) + (size_t)I2)
#define  LIBXSMM_INDEX1_4(I0, I1, I2, I3, S1, S2, S3) (LIBXSMM_INDEX1_3(I0, I1, I2, S1, S2) * ((size_t)S3) + (size_t)I3)
#define  LIBXSMM_INDEX1_5(I0, I1, I2, I3, I4, S1, S2, S3, S4) (LIBXSMM_INDEX1_4(I0, I1, I2, I3, S1, S2, S3) * ((size_t)S4) + (size_t)I4)
#define  LIBXSMM_INDEX1_6(I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, S5) (LIBXSMM_INDEX1_5(I0, I1, I2, I3, I4, S1, S2, S3, S4) * ((size_t)S5) + (size_t)I5)
#define  LIBXSMM_INDEX1_7(I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, S6) (LIBXSMM_INDEX1_6(I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, S5) * ((size_t)S6) + (size_t)I6)
#define  LIBXSMM_INDEX1_8(I0, I1, I2, I3, I4, I5, I6, I7, S1, S2, S3, S4, S5, S6, S7) (LIBXSMM_INDEX1_7(I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, S6) * ((size_t)S7) + (size_t)I7)
#define  LIBXSMM_INDEX1_9(I0, I1, I2, I3, I4, I5, I6, I7, I8, S1, S2, S3, S4, S5, S6, S7, S8) (LIBXSMM_INDEX1_8(I0, I1, I2, I3, I4, I5, I6, I7, S1, S2, S3, S4, S5, S6, S7) * ((size_t)S8) + (size_t)I8)
#define LIBXSMM_INDEX1_10(I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, S1, S2, S3, S4, S5, S6, S7, S8, S9) (LIBXSMM_INDEX1_9(I0, I1, I2, I3, I4, I5, I6, I7, I8, S1, S2, S3, S4, S5, S6, S7, S8) * ((size_t)S9) + (size_t)I9)

/**
 * LIBXSMM_VLA_DECL declares an array according to the given set of (multiple) bounds.
 * Syntax: LIBXSMM_VLA_DECL(<ndims>, <elem-type>, <var-name>, <init>, <s1>, ..., <s(ndims-1)>).
 * The element type can be "const" or otherwise qualified; initial value must be (const)element-type*.
 * Please note that the syntax is similar to LIBXSMM_INDEX1, and the leading dimension (s0) is omitted!
 *
 * LIBXSMM_VLA_ACCESS gives the array element according to the given set of (multiple) indexes/bounds.
 * Syntax: LIBXSMM_VLA_ACCESS(<ndims>, <array>, <i0>, ..., <i(ndims-1)>, <s1>, ..., <s(ndims-1)>).
 * Please note that the syntax is similar to LIBXSMM_INDEX1, and the leading dimension (s0) is omitted!
 */
#if !defined(LIBXSMM_VLA_POSTFIX)
# define LIBXSMM_VLA_POSTFIX _
#endif
#if defined(LIBXSMM_VLA)
LIBXSMM_API_INLINE int libxsmm_nonconst_int(int i) { return i; }
# define  LIBXSMM_VLA_ACCESS(NDIMS, ARRAY, ...) LIBXSMM_VLA_ACCESS_ND(NDIMS, LIBXSMM_CONCATENATE(ARRAY, LIBXSMM_VLA_POSTFIX), LIBXSMM_VLA_ACCESS_SINK, __VA_ARGS__)
# define  LIBXSMM_VLA_ACCESS_SINK(S) + 0 * (S)
# define  LIBXSMM_VLA_ACCESS_NONCONST(I) libxsmm_nonconst_int(I)
# define  LIBXSMM_VLA_ACCESS_ND(NDIMS, ARRAY, XY, ...) LIBXSMM_CONCATENATE3(LIBXSMM_VLA_ACCESS_, NDIMS, D)(ARRAY, XY, __VA_ARGS__)
# define  LIBXSMM_VLA_ACCESS_0D(ARRAY, XY, ...) (ARRAY)/*scalar*/
# define  LIBXSMM_VLA_ACCESS_1D(ARRAY, XY, ...) (ARRAY)[LIBXSMM_VLA_ACCESS_NONCONST(LIBXSMM_SELECT_HEAD(__VA_ARGS__))]
# define  LIBXSMM_VLA_ACCESS_2D(ARRAY, XY, I0, I1, ...) ((ARRAY) XY(__VA_ARGS__))[I0][LIBXSMM_VLA_ACCESS_NONCONST(I1)]
# define  LIBXSMM_VLA_ACCESS_3D(ARRAY, XY, I0, I1, I2, S1, ...) ((ARRAY) XY(S1) XY(__VA_ARGS__))[I0][I1][LIBXSMM_VLA_ACCESS_NONCONST(I2)]
# define  LIBXSMM_VLA_ACCESS_4D(ARRAY, XY, I0, I1, I2, I3, S1, S2, ...) ((ARRAY) XY(S1) XY(S2) XY(__VA_ARGS__))[I0][I1][I2][LIBXSMM_VLA_ACCESS_NONCONST(I3)]
# define  LIBXSMM_VLA_ACCESS_5D(ARRAY, XY, I0, I1, I2, I3, I4, S1, S2, S3, ...) ((ARRAY) XY(S1) XY(S2) XY(S3) XY(__VA_ARGS__))[I0][I1][I2][I3][LIBXSMM_VLA_ACCESS_NONCONST(I4)]
# define  LIBXSMM_VLA_ACCESS_6D(ARRAY, XY, I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, ...) ((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][LIBXSMM_VLA_ACCESS_NONCONST(I5)]
# define  LIBXSMM_VLA_ACCESS_7D(ARRAY, XY, I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, ...) ((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(S5) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][I5][LIBXSMM_VLA_ACCESS_NONCONST(I6)]
# define  LIBXSMM_VLA_ACCESS_8D(ARRAY, XY, I0, I1, I2, I3, I4, I5, I6, I7, S1, S2, S3, S4, S5, S6, ...) ((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(S5) XY(S6) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][I5][I6][LIBXSMM_VLA_ACCESS_NONCONST(I7)]
# define  LIBXSMM_VLA_ACCESS_9D(ARRAY, XY, I0, I1, I2, I3, I4, I5, I6, I7, I8, S1, S2, S3, S4, S5, S6, S7, ...) ((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(S5) XY(S6) XY(S7) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][I5][I6][I7][LIBXSMM_VLA_ACCESS_NONCONST(I8)]
# define LIBXSMM_VLA_ACCESS_10D(ARRAY, XY, I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, S1, S2, S3, S4, S5, S6, S7, S8, ...) ((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(S5) XY(S6) XY(S7) XY(S8) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][I5][I6][I7][I8][LIBXSMM_VLA_ACCESS_NONCONST(I9)]
# define LIBXSMM_VLA_DECL(NDIMS, ELEMENT_TYPE, ARRAY_VAR, .../*initial value, and bounds*/) \
    ELEMENT_TYPE LIBXSMM_VLA_ACCESS_ND(LIBXSMM_SELECT_ELEMENT(NDIMS, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9), *LIBXSMM_RESTRICT LIBXSMM_CONCATENATE(ARRAY_VAR, LIBXSMM_VLA_POSTFIX), \
      LIBXSMM_ELIDE, LIBXSMM_SELECT_TAIL(__VA_ARGS__, 0)/*bounds*/, LIBXSMM_SELECT_TAIL(__VA_ARGS__, 0)/*dummy*/) = \
   (ELEMENT_TYPE LIBXSMM_VLA_ACCESS_ND(LIBXSMM_SELECT_ELEMENT(NDIMS, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9), *, \
      LIBXSMM_ELIDE, LIBXSMM_SELECT_TAIL(__VA_ARGS__, 0)/*bounds*/, LIBXSMM_SELECT_TAIL(__VA_ARGS__, 0)/*dummy*/))LIBXSMM_SELECT_HEAD(__VA_ARGS__)
#else /* calculate linear index */
# define LIBXSMM_VLA_ACCESS(NDIMS, ARRAY, ...) LIBXSMM_CONCATENATE(ARRAY, LIBXSMM_VLA_POSTFIX)[LIBXSMM_INDEX1(NDIMS, __VA_ARGS__)]
# define LIBXSMM_VLA_DECL(NDIMS, ELEMENT_TYPE, ARRAY_VAR, .../*initial value, and bounds*/) \
    ELEMENT_TYPE *LIBXSMM_RESTRICT LIBXSMM_CONCATENATE(ARRAY_VAR, LIBXSMM_VLA_POSTFIX) = /*(ELEMENT_TYPE*)*/LIBXSMM_SELECT_HEAD(__VA_ARGS__) \
    + 0 * LIBXSMM_INDEX1(NDIMS, LIBXSMM_SELECT_TAIL(__VA_ARGS__, LIBXSMM_SELECT_TAIL(__VA_ARGS__, 0))) /* dummy-shift to "sink" unused arguments */
#endif

/** Address of an ARRAY of elements (of TYPESIZE) using linear index according to LIBXSMM_INDEX1. */
#define LIBXSMM_ACCESS_RAW(NDIMS, TYPESIZE, ARRAY, ...) ((void*)(((char*)(ARRAY)) + (TYPESIZE) * LIBXSMM_INDEX1(NDIMS, __VA_ARGS__)))
/** Address of an ARRAY of TYPE (can be const-qualified) using linear index according to LIBXSMM_INDEX1. */
#define LIBXSMM_ACCESS(NDIMS, TYPE, ARRAY, ...) (((TYPE*)(ARRAY)) + LIBXSMM_INDEX1(NDIMS, __VA_ARGS__))

/** Compare types, e.g., real types. */
#define LIBXSMM_EQUAL(T1, T2) LIBXSMM_CONCATENATE3(LIBXSMM_EQUAL_, T1, T2)
#define LIBXSMM_EQUAL_floatfloat 1
#define LIBXSMM_EQUAL_doubledouble 1
#define LIBXSMM_EQUAL_floatdouble 0
#define LIBXSMM_EQUAL_doublefloat 0

#if !defined(LIBXSMM_UNUSED)
# if 0
#   define LIBXSMM_UNUSED(VARIABLE) LIBXSMM_PRAGMA(unused(VARIABLE))
# else
#   define LIBXSMM_UNUSED(VARIABLE) (void)(VARIABLE)
# endif
#endif
#if defined(NDEBUG)
# define LIBXSMM_UNUSED_NDEBUG(VARIABLE) LIBXSMM_UNUSED(VARIABLE)
# define LIBXSMM_UNUSED_DEBUG(VARIABLE)
#else
# define LIBXSMM_UNUSED_NDEBUG(VARIABLE)
# define LIBXSMM_UNUSED_DEBUG(VARIABLE) LIBXSMM_UNUSED(VARIABLE)
#endif

#if defined(_OPENMP)
# define LIBXSMM_PRAGMA_OMP(...) LIBXSMM_PRAGMA(omp __VA_ARGS__)
# if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#   define LIBXSMM_OMP_VAR(A) LIBXSMM_UNUSED(A) /* suppress warning about "unused" variable */
# elif defined(__clang__)
#   define LIBXSMM_OMP_VAR(A) (A) = 0
# else
# define LIBXSMM_OMP_VAR(A)
# endif
#else
# define LIBXSMM_PRAGMA_OMP(...)
# define LIBXSMM_OMP_VAR(A)
#endif
/** Append "_omp" postfix to the given symbol. */
#define LIBXSMM_USEOMP(FUNCTION) LIBXSMM_CONCATENATE(FUNCTION, _omp)

#if defined(LIBXSMM_BUILD) && (defined(__GNUC__) || defined(__clang__)) && !defined(__CYGWIN__) && !defined(__MINGW32__)
# define LIBXSMM_ATTRIBUTE_WEAK_IMPORT LIBXSMM_ATTRIBUTE(weak_import)
# define LIBXSMM_ATTRIBUTE_WEAK LIBXSMM_ATTRIBUTE(weak)
#else
# define LIBXSMM_ATTRIBUTE_WEAK
# define LIBXSMM_ATTRIBUTE_WEAK_IMPORT
#endif

#if !defined(LIBXSMM_NO_CTOR) && !defined(LIBXSMM_CTOR) && \
    (defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__)) && \
    (defined(__GNUC__) || defined(__clang__))
# define LIBXSMM_ATTRIBUTE_CTOR LIBXSMM_ATTRIBUTE(constructor)
# define LIBXSMM_ATTRIBUTE_DTOR LIBXSMM_ATTRIBUTE(destructor)
/*# define LIBXSMM_CTOR*/
#else
# define LIBXSMM_ATTRIBUTE_CTOR
# define LIBXSMM_ATTRIBUTE_DTOR
#endif

#if defined(LIBXSMM_BUILD)
# define LIBXSMM_API_CTOR LIBXSMM_API LIBXSMM_ATTRIBUTE_CTOR
# define LIBXSMM_API_DTOR LIBXSMM_API LIBXSMM_ATTRIBUTE_DTOR
#else
# define LIBXSMM_API_CTOR LIBXSMM_API
# define LIBXSMM_API_DTOR LIBXSMM_API
#endif

#if defined(__GNUC__) && !defined(__PGI) && !defined(__ibmxl__)
# define LIBXSMM_ATTRIBUTE_NO_TRACE LIBXSMM_ATTRIBUTE(no_instrument_function)
#else
# define LIBXSMM_ATTRIBUTE_NO_TRACE
#endif

#if defined(__GNUC__)
# define LIBXSMM_MAY_ALIAS LIBXSMM_ATTRIBUTE(__may_alias__)
#else
# define LIBXSMM_MAY_ALIAS
#endif

#if !defined(LIBXSMM_MKTEMP_PATTERN)
# define LIBXSMM_MKTEMP_PATTERN "XXXXXX"
#endif

/** Below group is to fix-up some platform/compiler specifics. */
#if defined(_WIN32)
# if !defined(_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES)
#   define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
# endif
# if !defined(_CRT_INTERNAL_NONSTDC_NAMES)
#   define _CRT_INTERNAL_NONSTDC_NAMES 1
# endif
# if !defined(_CRT_SECURE_NO_DEPRECATE)
#   define _CRT_SECURE_NO_DEPRECATE 1
# endif
# if !defined(_USE_MATH_DEFINES)
#   define _USE_MATH_DEFINES 1
# endif
# if !defined(WIN32_LEAN_AND_MEAN)
#   define WIN32_LEAN_AND_MEAN 1
# endif
# if !defined(NOMINMAX)
#   define NOMINMAX 1
# endif
# if defined(__INTEL_COMPILER) && (190023506 <= _MSC_FULL_VER)
#   define __builtin_huge_val() HUGE_VAL
#   define __builtin_huge_valf() HUGE_VALF
#   define __builtin_nan nan
#   define __builtin_nanf nanf
#   define __builtin_nans nan
#   define __builtin_nansf nanf
#   if defined(__cplusplus)
#     define _CMATH_
#   endif
# endif
# if !defined(LIBXSMM_PATH_SEPARATOR)
#   define LIBXSMM_PATH_SEPARATOR '\\'
# endif
#elif !defined(LIBXSMM_PATH_SEPARATOR)
# define LIBXSMM_PATH_SEPARATOR '/'
#endif
#if defined(LIBXSMM_BUILD) && !defined(_WIN32)
# if !defined(_XOPEN_SOURCE) && 0
#   define _XOPEN_SOURCE 500
# endif
# if !defined(_DEFAULT_SOURCE)
#   define _DEFAULT_SOURCE
# endif
# if !defined(_GNU_SOURCE)
#   define _GNU_SOURCE
# endif
#endif
#if !defined(__STDC_FORMAT_MACROS)
# define __STDC_FORMAT_MACROS
#endif
#if defined(__clang__) && !defined(__extern_always_inline)
# define __extern_always_inline LIBXSMM_INLINE
#endif
#if defined(LIBXSMM_INLINE_FIXUP) && !defined(inline)
# define inline LIBXSMM_INLINE_KEYWORD
#endif

#if (0 != LIBXSMM_SYNC)
# if !defined(_REENTRANT)
#   define _REENTRANT
# endif
# if defined(__PGI)
#   if defined(__GCC_ATOMIC_TEST_AND_SET_TRUEVAL)
#     undef __GCC_ATOMIC_TEST_AND_SET_TRUEVAL
#   endif
#   define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
# endif
#endif

#if !defined(__clang__) || 1
# if !defined(__has_feature)
# define __has_feature(A) 0
# endif
# if !defined(__has_builtin)
#   define __has_builtin(A) 0
# endif
#endif

#if (0 != LIBXSMM_SYNC) && !defined(_WIN32) && !defined(__CYGWIN__)
# include <pthread.h>
#endif
#if defined(_WIN32) || defined(__CYGWIN__)
# include <windows.h>
#else
# include <unistd.h>
#endif
#if !defined(LIBXSMM_ASSERT)
# include <assert.h>
# if defined(NDEBUG)
#   define LIBXSMM_ASSERT(EXPR) LIBXSMM_ASSUME(EXPR)
# else
#   define LIBXSMM_ASSERT(EXPR) assert(EXPR)
# endif
#endif
#if !defined(LIBXSMM_ASSERT_MSG)
# if defined(_MSC_VER)
#   define LIBXSMM_ASSERT_MSG(EXPR, MSG) (assert((EXPR) && *MSG), EXPR)
# else
#   define LIBXSMM_ASSERT_MSG(EXPR, MSG) assert((EXPR) && *MSG)
# endif
#endif

#if !defined(LIBXSMM_ELIDE_RESULT)
# define LIBXSMM_ELIDE_RESULT(TYPE, EXPR) do { \
    /*const*/ TYPE libxsmm_elide_result_ = (EXPR); \
    LIBXSMM_UNUSED(libxsmm_elide_result_); \
  } while(0)
#endif
#if !defined(LIBXSMM_EXPECT_ELIDE)
# define LIBXSMM_EXPECT_ELIDE(EXPR) LIBXSMM_ELIDE_RESULT(int, EXPR)
#endif
#if defined(NDEBUG)
# define LIBXSMM_EXPECT LIBXSMM_EXPECT_ELIDE
#else
# define LIBXSMM_EXPECT LIBXSMM_ASSERT
#endif
#if defined(_DEBUG)
# define LIBXSMM_EXPECT_DEBUG LIBXSMM_EXPECT
#else
# define LIBXSMM_EXPECT_DEBUG LIBXSMM_EXPECT_ELIDE
#endif
#if defined(_OPENMP) && defined(LIBXSMM_SYNC_OMP)
# include <omp.h>
#endif
#include <inttypes.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <stdio.h>

#if !defined(__leaf__)
# define __leaf__
#endif
#if defined(__THROW)
# define LIBXSMM_NOTHROW __THROW
#endif
#if defined(__cplusplus)
# if (199711L < __cplusplus)
#   define LIBXSMM_NOEXCEPT noexcept
# else
#   define LIBXSMM_NOEXCEPT throw()
# endif
#else
# define LIBXSMM_NOEXCEPT
#endif
#if !defined(LIBXSMM_NOTHROW)
# define LIBXSMM_NOTHROW
#endif

#if !defined(FLT_MAX)
# if !defined(__FLT_MAX__)
#   define FLT_MAX 3.40282346638528859811704183484516925e+38F
# else
#   define FLT_MAX __FLT_MAX__
# endif
#endif
#if !defined(FLT_MIN)
# if !defined(__FLT_MIN__)
#   define FLT_MIN 1.17549435082228750796873653722224568e-38F
# else
#   define FLT_MIN __FLT_MIN__
# endif
#endif
#if defined(_WIN32) && 0
# define LIBXSMM_SNPRINTF(S, N, ...) _snprintf_s(S, N, _TRUNCATE, __VA_ARGS__)
#elif (defined(__STDC_VERSION__) && 199901L <= __STDC_VERSION__) || defined(__cplusplus)
# define LIBXSMM_SNPRINTF(S, N, ...) snprintf(S, N, __VA_ARGS__)
#else
# define LIBXSMM_SNPRINTF(S, N, ...) sprintf((S) + /*unused*/(N) * 0, __VA_ARGS__)
#endif

#if defined(_WIN32)
# define LIBXSMM_PUTENV(A) _putenv(A)
#else
# define LIBXSMM_PUTENV(A) putenv(A)
# define LIBXSMM_MKTEMP(A) mkstemp(A)
# if defined(__clang__) || !defined(__GNUC__) || (defined(__GNUC__) && LIBXSMM_VERSION2(4, 3) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__))
LIBXSMM_EXTERN int mkstemp(char*);
# else
LIBXSMM_EXTERN int mkstemp(char*) LIBXSMM_NOTHROW;
# endif
LIBXSMM_EXTERN int putenv(char*) LIBXSMM_NOTHROW;
#endif

/* block must be after including above header files */
#if (defined(__GLIBC__) && defined(__GLIBC_MINOR__) && LIBXSMM_VERSION2(__GLIBC__, __GLIBC_MINOR__) < LIBXSMM_VERSION2(2, 26)) \
  || (defined(LIBXSMM_INTEL_COMPILER) && (1802 >= LIBXSMM_INTEL_COMPILER) && !defined(__cplusplus) && defined(__linux__))
/* _Float128 was introduced with GNU GCC 7.0. */
# if !defined(_Float128) && !defined(__SIZEOF_FLOAT128__) && defined(__GNUC__) && !defined(__cplusplus) && defined(__linux__)
#   define _Float128 __float128
# endif
# if !defined(LIBXSMM_GLIBC_FPTYPES) && defined(__GNUC__) && !defined(__cplusplus) && defined(__linux__) \
  && (LIBXSMM_VERSION2(7, 0) > LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__) || \
     (defined(LIBXSMM_INTEL_COMPILER) && (1802 >= LIBXSMM_INTEL_COMPILER)))
#   define LIBXSMM_GLIBC_FPTYPES
# endif
# if !defined(_Float128X) && defined(LIBXSMM_GLIBC_FPTYPES)
#   define _Float128X _Float128
# endif
# if !defined(_Float32) && defined(LIBXSMM_GLIBC_FPTYPES)
#   define _Float32 float
# endif
# if !defined(_Float32x) && defined(LIBXSMM_GLIBC_FPTYPES)
#   define _Float32x _Float32
# endif
# if !defined(_Float64) && defined(LIBXSMM_GLIBC_FPTYPES)
#   define _Float64 double
# endif
# if !defined(_Float64x) && defined(LIBXSMM_GLIBC_FPTYPES)
#   define _Float64x _Float64
# endif
#endif

#if defined(LIBXSMM_GLIBC_FPTYPES)
# if defined(__cplusplus)
#   undef __USE_MISC
#   if !defined(_DEFAULT_SOURCE)
#     define _DEFAULT_SOURCE
#   endif
#   if !defined(_BSD_SOURCE)
#     define _BSD_SOURCE
#   endif
# else
#   if !defined(__PURE_INTEL_C99_HEADERS__)
#     define __PURE_INTEL_C99_HEADERS__
#   endif
# endif
#endif

#if !defined(LIBXSMM_NO_LIBM)
# if (defined(LIBXSMM_INTEL_COMPILER) && (1800 <= LIBXSMM_INTEL_COMPILER)) \
  && !defined(_WIN32) /* error including dfp754.h */
#   include <mathimf.h>
# endif
# if defined(__STRICT_ANSI__)
#   define LIBXSMM_STRICT_ANSI __STRICT_ANSI__
#   undef __STRICT_ANSI__
# endif
# include <math.h>
# if defined(LIBXSMM_STRICT_ANSI)
#   define __STRICT_ANSI__ LIBXSMM_STRICT_ANSI
#   undef LIBXSMM_STRICT_ANSI
# endif
# if (!defined(__STDC_VERSION__) || (199901L/*C99*/ > __STDC_VERSION__)) && !defined(_WIN32)
LIBXSMM_EXTERN double erf(double) LIBXSMM_NOTHROW;
# endif
#endif
#if !defined(M_PI)
# define M_PI 3.14159265358979323846
#endif

#endif /*LIBXSMM_MACROS_H*/
