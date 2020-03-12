/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
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
 * The following interfaces shall be explicitly included,
 * i.e., separate from libxsmm.h:
 * - libxsmm_intrinsics_x86.h
 * - libxsmm_cpuid.h
 * - libxsmm_sync.h
 * - libxsmm_mhd.h
*/
#include "libxsmm_dnn_convolution.h"
#include "libxsmm_dnn_fullyconnected.h"
#include "libxsmm_dnn_fusedbatchnorm.h"
#include "libxsmm_dnn_fusedgroupnorm.h"
#include "libxsmm_dnn_pooling.h"
#include "libxsmm_dnn_rnncell.h"
#include "libxsmm_blocked_gemm.h"
#include "libxsmm_generator.h"
#include "libxsmm_frontend.h"
#include "libxsmm_fsspmdm.h"
#include "libxsmm_malloc.h"
#include "libxsmm_spmdm.h"
#include "libxsmm_cpuid.h"
#include "libxsmm_timer.h"
#include "libxsmm_math.h"
#include "libxsmm_rng.h"


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
/** Set target architecture (arch="0|sse|snb|hsw|knl|knm|skx|clx|cpx", NULL/"0": CPUID). */
LIBXSMM_API void libxsmm_set_target_arch(const char* arch);

/** Get the level of verbosity. */
LIBXSMM_API int libxsmm_get_verbosity(void);
/**
 * Set the level of verbosity (0: off, positive value: verbosity level,
 * negative value: maximum verbosity, which also dumps JIT-code)
 */
LIBXSMM_API void libxsmm_set_verbosity(int level);

/** Get the default prefetch strategy. */
LIBXSMM_API libxsmm_gemm_prefetch_type libxsmm_get_gemm_auto_prefetch(void);
/** Set the default prefetch strategy. */
LIBXSMM_API void libxsmm_set_gemm_auto_prefetch(libxsmm_gemm_prefetch_type strategy);

/** Receive information about JIT-generated code. */
LIBXSMM_API int libxsmm_get_kernel_info(const void* kernel, libxsmm_kernel_info* info);

/** Get information about the matrix multiplication kernel. */
LIBXSMM_API int libxsmm_get_mmkernel_info(libxsmm_xmmfunction kernel, libxsmm_mmkernel_info* info);

/** Get information about the matrix transpose kernel. */
LIBXSMM_API int libxsmm_get_transkernel_info(libxsmm_xtransfunction kernel, libxsmm_transkernel_info* info);

/** Get information about the matrix copy kernel. */
LIBXSMM_API int libxsmm_get_mcopykernel_info(libxsmm_xmcopyfunction kernel, libxsmm_mcopykernel_info* info);

/** Get information about the code registry. */
LIBXSMM_API int libxsmm_get_registry_info(libxsmm_registry_info* info);

/**
 * Register user-defined key-value pair; the value can be then queried per libxsmm_xdispatch.
 * Since the key-type is unknown to LIBXSMM, the key must be binary reproducible. Structured
 * data may be padded (compiler/platform-specific), and key-structure initialization shall be:
 * memset(&mykey, 0, sizeof(mykey)) followed by an element-wise initialization (some compilers
 * leave padded data uninitialized which breaks binary reproducible keys). The size of the key
 * is limited to LIBXSMM_DESCRIPTOR_MAXSIZE. The given value is copied by LIBXSMM and may be
 * initialized at registration-time or when received per libxsmm_xdispatch. Registered data
 * is released at program termination but can be also released if needed (libxsmm_xrelease).
 */
LIBXSMM_API void* libxsmm_xregister(const void* key, size_t key_size, size_t value_size, const void* value_init);
/**
 * Query user-defined value from LIBXSMM's code registry. The value's buffer is owned and
 * managed by LIBXSMM (can be libxsmm_xrelease'd, .e.g., if larger value for the same key
 * must be stored).
 */
LIBXSMM_API void* libxsmm_xdispatch(const void* key, size_t key_size);
/** Remove key-value pair from code registry and release memory. */
LIBXSMM_API void libxsmm_xrelease(const void* key, size_t key_size);

/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (descriptor form). */
LIBXSMM_API libxsmm_xmmfunction libxsmm_xmmdispatch(const libxsmm_gemm_descriptor* descriptor);

/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (double-precision). */
LIBXSMM_API libxsmm_dmmfunction libxsmm_dmmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (single-precision). */
LIBXSMM_API libxsmm_smmfunction libxsmm_smmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (bf16 inputs, fp32-accumulate) */
LIBXSMM_API libxsmm_bsmmfunction libxsmm_bsmmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs) */
LIBXSMM_API libxsmm_bmmfunction libxsmm_bmmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (low/short-precision, int-accumulate) */
LIBXSMM_API libxsmm_wimmfunction libxsmm_wimmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (low/char-precision, int-accumulate) */
LIBXSMM_API libxsmm_ssbimmfunction libxsmm_ssbimmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch);
LIBXSMM_API libxsmm_usbimmfunction libxsmm_usbimmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch);
LIBXSMM_API libxsmm_subimmfunction libxsmm_subimmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch);
LIBXSMM_API libxsmm_uubimmfunction libxsmm_uubimmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (low/char-precision, int-accumulate, int8 outputs) */
LIBXSMM_API libxsmm_sububmmfunction libxsmm_sububmmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch);

/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (double-precision). */
LIBXSMM_API libxsmm_dmmfunction_reducebatch_addr libxsmm_dmmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (single-precision). */
LIBXSMM_API libxsmm_smmfunction_reducebatch_addr libxsmm_smmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate). */
LIBXSMM_API libxsmm_bsmmfunction_reducebatch_addr libxsmm_bsmmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs). */
LIBXSMM_API libxsmm_bmmfunction_reducebatch_addr libxsmm_bmmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int16 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_wimmfunction_reducebatch_addr libxsmm_wimmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_ssbimmfunction_reducebatch_addr libxsmm_ssbimmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_usbimmfunction_reducebatch_addr libxsmm_usbimmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_subimmfunction_reducebatch_addr libxsmm_subimmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_uubimmfunction_reducebatch_addr libxsmm_uubimmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate, int8 outputs). */
LIBXSMM_API libxsmm_sububmmfunction_reducebatch_addr libxsmm_sububmmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);

/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (double-precision). */
LIBXSMM_API libxsmm_dmmfunction_reducebatch_addr libxsmm_dmmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (single-precision). */
LIBXSMM_API libxsmm_smmfunction_reducebatch_addr libxsmm_smmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/* Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate). */
LIBXSMM_API libxsmm_bsmmfunction_reducebatch_addr libxsmm_bsmmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs). */
LIBXSMM_API libxsmm_bmmfunction_reducebatch_addr libxsmm_bmmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int16 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_wimmfunction_reducebatch_addr libxsmm_wimmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_ssbimmfunction_reducebatch_addr libxsmm_ssbimmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_usbimmfunction_reducebatch_addr libxsmm_usbimmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_subimmfunction_reducebatch_addr libxsmm_subimmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_uubimmfunction_reducebatch_addr libxsmm_uubimmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate, int8 outputs). */
LIBXSMM_API libxsmm_sububmmfunction_reducebatch_addr libxsmm_sububmmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);

/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (double-precision). */
LIBXSMM_API libxsmm_dmmfunction_reducebatch_offs libxsmm_dmmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (single-precision). */
LIBXSMM_API libxsmm_smmfunction_reducebatch_offs libxsmm_smmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate). */
LIBXSMM_API libxsmm_bsmmfunction_reducebatch_offs libxsmm_bsmmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs). */
LIBXSMM_API libxsmm_bmmfunction_reducebatch_offs libxsmm_bmmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int16 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_wimmfunction_reducebatch_offs libxsmm_wimmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_ssbimmfunction_reducebatch_offs libxsmm_ssbimmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_usbimmfunction_reducebatch_offs libxsmm_usbimmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_subimmfunction_reducebatch_offs libxsmm_subimmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_uubimmfunction_reducebatch_offs libxsmm_uubimmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate, int8 outputs). */
LIBXSMM_API libxsmm_sububmmfunction_reducebatch_offs libxsmm_sububmmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);

/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (double-precision). */
LIBXSMM_API libxsmm_dmmfunction_reducebatch_offs libxsmm_dmmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (single-precision). */
LIBXSMM_API libxsmm_smmfunction_reducebatch_offs libxsmm_smmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate). */
LIBXSMM_API libxsmm_bsmmfunction_reducebatch_offs libxsmm_bsmmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs). */
LIBXSMM_API libxsmm_bmmfunction_reducebatch_offs libxsmm_bmmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int16 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_wimmfunction_reducebatch_offs libxsmm_wimmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_ssbimmfunction_reducebatch_offs libxsmm_ssbimmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_usbimmfunction_reducebatch_offs libxsmm_usbimmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_subimmfunction_reducebatch_offs libxsmm_subimmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_uubimmfunction_reducebatch_offs libxsmm_uubimmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate, int8 outputs). */
LIBXSMM_API libxsmm_sububmmfunction_reducebatch_offs libxsmm_sububmmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);

/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (double-precision). */
LIBXSMM_API libxsmm_dmmfunction_reducebatch_strd libxsmm_dmmdispatch_reducebatch_strd(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (single-precision). */
LIBXSMM_API libxsmm_smmfunction_reducebatch_strd libxsmm_smmdispatch_reducebatch_strd(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate). */
LIBXSMM_API libxsmm_bsmmfunction_reducebatch_strd libxsmm_bsmmdispatch_reducebatch_strd(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs). */
LIBXSMM_API libxsmm_bmmfunction_reducebatch_strd libxsmm_bmmdispatch_reducebatch_strd(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int16 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_wimmfunction_reducebatch_strd libxsmm_wimmdispatch_reducebatch_strd(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_ssbimmfunction_reducebatch_strd libxsmm_ssbimmdispatch_reducebatch_strd(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_usbimmfunction_reducebatch_strd libxsmm_usbimmdispatch_reducebatch_strd(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_subimmfunction_reducebatch_strd libxsmm_subimmdispatch_reducebatch_strd(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_uubimmfunction_reducebatch_strd libxsmm_uubimmdispatch_reducebatch_strd(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate, int8 outputs). */
LIBXSMM_API libxsmm_sububmmfunction_reducebatch_strd libxsmm_sububmmdispatch_reducebatch_strd(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);

/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (double-precision). */
LIBXSMM_API libxsmm_dmmfunction_reducebatch_strd libxsmm_dmmdispatch_reducebatch_strd_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (single-precision). */
LIBXSMM_API libxsmm_smmfunction_reducebatch_strd libxsmm_smmdispatch_reducebatch_strd_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate). */
LIBXSMM_API libxsmm_bsmmfunction_reducebatch_strd libxsmm_bsmmdispatch_reducebatch_strd_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs). */
LIBXSMM_API libxsmm_bmmfunction_reducebatch_strd libxsmm_bmmdispatch_reducebatch_strd_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int16 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_wimmfunction_reducebatch_strd libxsmm_wimmdispatch_reducebatch_strd_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_ssbimmfunction_reducebatch_strd libxsmm_ssbimmdispatch_reducebatch_strd_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_usbimmfunction_reducebatch_strd libxsmm_usbimmdispatch_reducebatch_strd_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_subimmfunction_reducebatch_strd libxsmm_subimmdispatch_reducebatch_strd_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXSMM_API libxsmm_uubimmfunction_reducebatch_strd libxsmm_uubimmdispatch_reducebatch_strd_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate, int8 outputs). */
LIBXSMM_API libxsmm_sububmmfunction_reducebatch_strd libxsmm_sububmmdispatch_reducebatch_strd_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);

/**
 * Process a series of matrix multiplications (batch). See also libxsmm_gemm_batch/omp.
 * The kind of matrix operands (a, b, c) depend on index_stride:
 * index_stride==0: pointers to pointers of elements e.g., double** for the C matrices.
 * index_stride!=0: pointer to elements e.g., const double* for the A and B matrices.
 */
LIBXSMM_API void libxsmm_mmbatch(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec,
  const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc,
  /** Determines index-base (usually 0, 1 for one-based indexes); uses the same unit as the strides. */
  libxsmm_blasint index_base,
  /**
   * Stride used to walk stride_a, stride_b, and stride_c; zero turns stride_* into scalar values.
   * The index_stride is measured in Bytes (sizeof(libxsmm_blasint) determines packed indexes).
   */
  libxsmm_blasint index_stride,
  /**
   * Depending on index_stride, the meaning of stride_a, stride_b, and stride_c is different.
   * index_stride==0: stride_a, stride_b, and stride_c are pointers to scalar values.
   * index_stride!=0: stride_* are indexes determining the position of a, b, and c operands.
   */
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  /**
   * Number of matrix multiplications. If the size is given as a negative value,
   * then internal synchronization is omitted.
   */
  libxsmm_blasint batchsize,
  /** Thread-ID (TID), and number of threads. */
  /*unsigned*/int tid, /*unsigned*/int nthreads);

/** Process a series of matrix multiplications (batch). See also libxsmm_mmbatch. */
LIBXSMM_API void libxsmm_gemm_batch(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec,
  const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
                     const void* b, const libxsmm_blasint* ldb,
   const void* beta,       void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize);

/** Process a series of matrix multiplications (batch) with OpenMP (libxsmmext). See also libxsmm_mmbatch. */
LIBXSMM_APIEXT void libxsmm_gemm_batch_omp(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec,
  const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
                     const void* b, const libxsmm_blasint* ldb,
   const void* beta,       void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize);

/** Unlike libxsmm_gemm_batch, groups of homogeneous batches are possible (double-precision). */
LIBXSMM_API void libxsmm_dgemm_batch(const char transa_array[], const char transb_array[],
  const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxsmm_blasint lda_array[],
                              const double* b_array[], const libxsmm_blasint ldb_array[],
   const double beta_array[],       double* c_array[], const libxsmm_blasint ldc_array[],
  const libxsmm_blasint* group_count, const libxsmm_blasint group_size[]);

/** Unlike libxsmm_gemm_batch, groups of homogeneous batches are possible (single-precision). */
LIBXSMM_API void libxsmm_sgemm_batch(const char transa_array[], const char transb_array[],
  const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxsmm_blasint lda_array[],
                             const float* b_array[], const libxsmm_blasint ldb_array[],
   const float beta_array[],       float* c_array[], const libxsmm_blasint ldc_array[],
  const libxsmm_blasint* group_count, const libxsmm_blasint group_size[]);

/** Unlike libxsmm_gemm_batch, groups of homogeneous batches are possible (double-precision). */
LIBXSMM_APIEXT void libxsmm_dgemm_batch_omp(const char transa_array[], const char transb_array[],
  const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxsmm_blasint lda_array[],
                              const double* b_array[], const libxsmm_blasint ldb_array[],
   const double beta_array[],       double* c_array[], const libxsmm_blasint ldc_array[],
  const libxsmm_blasint* group_count, const libxsmm_blasint group_size[]);

/** Unlike libxsmm_gemm_batch, groups of homogeneous batches are possible (single-precision). */
LIBXSMM_APIEXT void libxsmm_sgemm_batch_omp(const char transa_array[], const char transb_array[],
  const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxsmm_blasint lda_array[],
                             const float* b_array[], const libxsmm_blasint ldb_array[],
   const float beta_array[],       float* c_array[], const libxsmm_blasint ldc_array[],
  const libxsmm_blasint* group_count, const libxsmm_blasint group_size[]);

/**
 * This function is a no-op unless LIBXSMM is built to intercept GEMM calls.
 * Pointer arguments are used to filter intercepted GEMM calls such that
 * non-NULL values match. Otherwise (NULL) the respective argument is
 * considered a "free value" i.e., every value can match; libxsmmext required.
 */
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

/** Code generation routine for GEMM/packed using a descriptor */
LIBXSMM_API libxsmm_pgemm_xfunction libxsmm_dispatch_pgemm(const libxsmm_pgemm_descriptor* descriptor);

/** Code generation routine for GETRF/packed using a descriptor */
LIBXSMM_API libxsmm_getrf_xfunction libxsmm_dispatch_getrf(const libxsmm_getrf_descriptor* descriptor);

/** Code generation routine for TRMM/packed using a descriptor */
LIBXSMM_API libxsmm_trmm_xfunction libxsmm_dispatch_trmm(const libxsmm_trmm_descriptor* descriptor);

/** Code generation routine for TRSM/packed using a descriptor */
LIBXSMM_API libxsmm_trsm_xfunction libxsmm_dispatch_trsm(const libxsmm_trsm_descriptor* descriptor);

/**
 * Code generation routine for the CSR format which multiplies a dense SOA matrix (each element holds a SIMD-width
 * wide vector) and a sparse matrix or a sparse matrix with a dense SOA matrix.
 * The result is always a SOA matrix. There is no code cache, and user code has to manage the code pointers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_xmmfunction libxsmm_create_xcsr_soa(const libxsmm_gemm_descriptor* descriptor,
   const unsigned int* row_ptr, const unsigned int* column_idx, const void* values, const unsigned int packed_width);

/**
 * Code generation routine for the CSC format which multiplies a dense SOA matrix (each element holds a SIMD-width
 * wide vector) and a sparse matrix or a sparse matrix with a dense SOA matrix.
 * The result is always a SOA matrix. There is no code cache, and user code has to manage the code pointers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_xmmfunction libxsmm_create_xcsc_soa(const libxsmm_gemm_descriptor* descriptor,
   const unsigned int* column_ptr, const unsigned int* row_idx, const void* values, const unsigned int packed_width);

/**
 * Code generation routine for row-major format B matrix which is multiplied by a dense packed matrix (each element holds a SIMD-width
 * wide vector) and the result is another packed matrix. The memory layout of the SOA matrix is [row][col][packed].
 * here is no code cache, and user code has to manage the code pointers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_xmmfunction libxsmm_create_pgemm_ac_rm(const libxsmm_gemm_descriptor* descriptor, const unsigned int packed_width);

/**
 * Code generation routine for row-major format A matrix which is multiplied by a dense packed matrix (each element holds a SIMD-width
 * wide vector) and the result is another packed matrix. The memory layout of the packed matrix is [row][col][packed].
 * here is no code cache, and user code has to manage the code pointers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_xmmfunction libxsmm_create_pgemm_bc_rm(const libxsmm_gemm_descriptor* descriptor, const unsigned int packed_width);

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

/**
 * Deallocates the JIT'ted code as returned by libxsmm_create_* functions,
 * unregisters and releases code from the code registry.
 */
LIBXSMM_API void libxsmm_release_kernel(const void* kernel);

/** Matrix copy function ("in" can be NULL to zero the destination). */
LIBXSMM_API void libxsmm_matcopy(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch);

/** Matrix copy function ("in" can be NULL to zero the destination, per-thread form). */
LIBXSMM_API void libxsmm_matcopy_thread(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch, /*unsigned*/int tid, /*unsigned*/int nthreads);

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
  /*unsigned*/int tid, /*unsigned*/int nthreads);

/** Matrix transposition; MT via libxsmmext (out-of-place form). */
LIBXSMM_APIEXT void libxsmm_otrans_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo);

/** Matrix transposition (in-place form). */
LIBXSMM_API void libxsmm_itrans(void* inout, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld);

/** Initialize GEMM-handle; allows to better amortize setup overhead. */
LIBXSMM_API libxsmm_gemm_handle* libxsmm_gemm_handle_init(libxsmm_gemm_blob* blob,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta, int flags, /*unsigned*/int ntasks);

/** Calculate required scratch buffer size needed to perform libxsmm_gemm_thread. */
LIBXSMM_API size_t libxsmm_gemm_handle_get_scratch_size(const libxsmm_gemm_handle* handle);

/** Low-level type-agnostic GEMM suitable for external threads or tasks. */
LIBXSMM_API void libxsmm_gemm_thread(const libxsmm_gemm_handle* handle, void* scratch,
  const void* a, const void* b, void* c, /*unsigned*/int tid, /*unsigned*/int nthreads);

/** General dense matrix multiplication (sequential). */
LIBXSMM_API void libxsmm_xgemm(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc);

/** General dense matrix multiplication (libxsmmext); available as xgemm (generic), dgemm (DP), and sgemm (SP). */
LIBXSMM_APIEXT void libxsmm_xgemm_omp(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc);

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
/** Dispatched general dense matrix multiplication (BF16 input, F32 result). */
LIBXSMM_API void libxsmm_bsgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const libxsmm_bfloat16* a, const libxsmm_blasint* lda,
  const libxsmm_bfloat16* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc);

#if defined(LIBXSMM_BUILD) && !defined(LIBXSMM_DEFAULT_CONFIG)
$MNK_INTERFACE_LIST
#endif /*defined(LIBXSMM_BUILD)*/

#if defined(__cplusplus)

/** Map a built-in type to libxsmm_gemm_precision (libxsmm_gemm_precision_enum). */
template<typename T> struct LIBXSMM_RETARGETABLE libxsmm_gemm_precision_enum             { static const libxsmm_gemm_precision value = static_cast<libxsmm_gemm_precision>(LIBXSMM_DATATYPE_UNSUPPORTED); };
template<> struct LIBXSMM_RETARGETABLE libxsmm_gemm_precision_enum<double>               { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_F64; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_gemm_precision_enum<float>                { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_F32; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_gemm_precision_enum<int>                  { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_I32; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_gemm_precision_enum</*signed*/short>      { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_I16; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_gemm_precision_enum<libxsmm_bfloat16>     { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_BF16; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_gemm_precision_enum<tensorflow::bfloat16> { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_BF16; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_gemm_precision_enum<signed char>          { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_I8; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_gemm_precision_enum<unsigned char>        { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_I8; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_gemm_precision_enum<char>                 { static const libxsmm_gemm_precision value = LIBXSMM_GEMM_PRECISION_I8; };

template<typename INP_TYPE> struct LIBXSMM_RETARGETABLE libxsmm_gemm_default_output      { typedef INP_TYPE type; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_gemm_default_output</*signed*/short>      { typedef int type; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_gemm_default_output<unsigned short>       { typedef int type; };

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
  const float* alpha, const libxsmm_bfloat16* a, const libxsmm_blasint* lda,
                      const libxsmm_bfloat16* b, const libxsmm_blasint* ldb,
   const float* beta,                  float* c, const libxsmm_blasint* ldc)
{
  libxsmm_bsgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline LIBXSMM_RETARGETABLE void libxsmm_gemm(const char* transa, const char* transb,
  /* by-value */ libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const libxsmm_bfloat16* a, const libxsmm_blasint* lda,
                      const libxsmm_bfloat16* b, const libxsmm_blasint* ldb,
   const float* beta,                  float* c, const libxsmm_blasint* ldc)
{
  libxsmm_bsgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
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

