/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
   Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include "libxsmm_blocked_gemm_types.h"
#include <libxsmm.h>


LIBXSMM_API libxsmm_blocked_gemm_handle* libxsmm_blocked_gemm_handle_create(/*unsigned*/ int nthreads,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* bm, const libxsmm_blasint* bn, const libxsmm_blasint* bk,
  const libxsmm_blasint* b_m1, const libxsmm_blasint* b_n1, const libxsmm_blasint* b_k1, const libxsmm_blasint* b_k2,
  const void* alpha, const void* beta, const int* gemm_flags,
  const libxsmm_gemm_prefetch_type* prefetch,
  const libxsmm_blocked_gemm_order* order)
{
  const char *const env_m = getenv("LIBXSMM_BLOCKED_GEMM_M"), *const env_n = getenv("LIBXSMM_BLOCKED_GEMM_N"), *const env_k = getenv("LIBXSMM_BLOCKED_GEMM_K");
  const libxsmm_blasint mm = LIBXSMM_MIN(0 == bm ? ((NULL == env_m || 0 == *env_m) ? 32 : atoi(env_m)) : *bm, m);
  const libxsmm_blasint kk = LIBXSMM_MIN(0 == bk ? ((NULL == env_k || 0 == *env_k) ? mm : atoi(env_k)) : *bk, k);
  const libxsmm_blasint nn = LIBXSMM_MIN(0 == bn ? ((NULL == env_n || 0 == *env_n) ? kk : atoi(env_n)) : *bn, n);
  libxsmm_blocked_gemm_handle* result = 0;
  static int error_once = 0;

  if (0 < m && 0 < n && 0 < k && 0 < mm && 0 < nn && 0 < kk && 0 < nthreads) {
    libxsmm_blocked_gemm_handle handle;
    memset(&handle, 0, sizeof(handle));
    if (0 == (m % mm) && 0 == (n % nn) && 0 == (k % kk) &&
        0 == (m % *b_m1) && 0 == (n % *b_n1) && 0 == (k % *b_k1) &&
        0 == ((k / *b_k1 / *b_k2) % kk) && 0 == ((n / *b_n1) % nn) && 0 == ((m / *b_m1) % mm))
    { /* check for valid block-size */
      libxsmm_gemm_descriptor* desc;
      libxsmm_descriptor_blob blob;
      if (0 == prefetch) { /* auto-prefetch */
        /* TODO: more sophisticated strategy perhaps according to CPUID */
        const libxsmm_gemm_prefetch_type prefetch_default = LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C;
        const char *const env_p = getenv("LIBXSMM_BLOCKED_GEMM_PREFETCH");
        desc = libxsmm_gemm_descriptor_init2(&blob, iprec, oprec, mm, nn, kk, mm/*lda*/, kk/*ldb*/, mm/*ldc*/,
          alpha, beta, 0 == gemm_flags ? LIBXSMM_GEMM_FLAG_NONE : *gemm_flags,
          (NULL == env_p || 0 == *env_p) ? prefetch_default : libxsmm_gemm_uid2prefetch(atoi(env_p)));
      }
      else { /* user-defined */
        desc = libxsmm_gemm_descriptor_init2(&blob, iprec, oprec, mm, nn, kk, mm/*lda*/, kk/*ldb*/, mm/*ldc*/,
          alpha, beta, 0 == gemm_flags ? LIBXSMM_GEMM_FLAG_NONE : *gemm_flags, *prefetch);
      }
      if (0 != desc) {
        handle.mb = m / mm; handle.nb = n / nn; handle.kb = k / kk;
        if (LIBXSMM_GEMM_PREFETCH_NONE != desc->prefetch) {
          handle.kernel_pf = libxsmm_xmmdispatch(desc);
          desc->prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
          handle.kernel = libxsmm_xmmdispatch(desc);
        }
        else { /* no prefetch */
          handle.kernel = libxsmm_xmmdispatch(desc);
          handle.kernel_pf.xmm = 0;
        }
      }
      if (0 != handle.kernel.xmm) {
        const size_t tls_size = LIBXSMM_UP2((size_t)mm * nn * LIBXSMM_TYPESIZE(oprec), LIBXSMM_CACHELINE) * nthreads;
        const size_t size_locks = (size_t)handle.mb * (size_t)handle.nb * sizeof(libxsmm_blocked_gemm_lock);
        handle.locks = (libxsmm_blocked_gemm_lock*)libxsmm_aligned_malloc(size_locks, LIBXSMM_CACHELINE);
        handle.buffer = libxsmm_aligned_malloc(tls_size, LIBXSMM_CACHELINE);
        result = (libxsmm_blocked_gemm_handle*)malloc(sizeof(libxsmm_blocked_gemm_handle));

        if (224 <= nthreads
#if !defined(__MIC__)
          && LIBXSMM_X86_AVX512_MIC <= libxsmm_target_archid
          && LIBXSMM_X86_AVX512_CORE > libxsmm_target_archid
#endif
          )
        {
          handle.barrier = libxsmm_barrier_create(nthreads / 4, 4);
        }
        else {
          handle.barrier = libxsmm_barrier_create(nthreads / 2, 2);
        }
        if (0 != result && 0 != handle.barrier && 0 != handle.buffer && 0 != handle.locks) {
          handle.m = m; handle.n = n; handle.k = k; handle.bm = mm; handle.bn = nn; handle.bk = kk;
          handle.b_m1 = *b_m1; handle.b_n1 = *b_n1; handle.b_k1 = *b_k1; handle.b_k2 = *b_k2;
          handle.iprec = iprec; handle.oprec = oprec;
          memset(handle.locks, 0, size_locks);
          handle.order = (0 == order ? LIBXSMM_BLOCKED_GEMM_ORDER_JIK : *order);
          handle.nthreads = nthreads;
          *result = handle;
        }
        else {
          if (0 != libxsmm_verbosity /* library code is expected to be mute */
            && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
          {
            fprintf(stderr, "LIBXSMM ERROR: BGEMM handle allocation failed!\n");
          }
          libxsmm_barrier_release(handle.barrier);
          libxsmm_free(handle.buffer);
          libxsmm_free(handle.locks);
          free(result);
          result = 0;
        }
      }
      else if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: unsupported BGEMM kernel requested!\n");
      }
    }
    else if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: BGEMM block-size is invalid!\n");
    }
  }
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_blocked_gemm_handle_create!\n");
  }

  return result;
}


LIBXSMM_API void libxsmm_blocked_gemm_handle_destroy(const libxsmm_blocked_gemm_handle* handle)
{
  if (0 != handle) {
    libxsmm_barrier_release(handle->barrier);
    libxsmm_free(handle->buffer);
    libxsmm_free(handle->locks);
    free((libxsmm_blocked_gemm_handle*)handle);
  }
}


LIBXSMM_API int libxsmm_blocked_gemm_copyin_a(const libxsmm_blocked_gemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
#if 0 /* TODO: support leading dimension for the source buffer */
    const libxsmm_blasint ild = (0 == ld ? handle->m : *ld);
    assert(ild >= handle->m);
#else
    LIBXSMM_UNUSED(ld);
#endif
    switch (handle->iprec) {
      case LIBXSMM_GEMM_PRECISION_F64: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE double
#       include "template/libxsmm_blocked_gemm_copyin_a.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE float
#       include "template/libxsmm_blocked_gemm_copyin_a.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_I16: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE short
#       include "template/libxsmm_blocked_gemm_copyin_a.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      default: {
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: BGEMM precision of matrix A is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: BGEMM-handle cannot be NULL!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API int libxsmm_blocked_gemm_copyin_b(const libxsmm_blocked_gemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
#if 0 /* TODO: support leading dimension for the source buffer */
    const libxsmm_blasint ild = (0 == ld ? handle->k : *ld);
    assert(ild >= handle->k);
#else
    LIBXSMM_UNUSED(ld);
#endif
    switch (handle->iprec) {
      case LIBXSMM_GEMM_PRECISION_F64: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE double
#       include "template/libxsmm_blocked_gemm_copyin_b.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE float
#       include "template/libxsmm_blocked_gemm_copyin_b.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_I16: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE short
#       include "template/libxsmm_blocked_gemm_copyin_b.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      default: {
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: BGEMM precision of matrix B is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: BGEMM-handle cannot be NULL!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API int libxsmm_blocked_gemm_copyin_c(const libxsmm_blocked_gemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
#if 0 /* TODO: support leading dimension for the source buffer */
    const libxsmm_blasint ild = (0 == ld ? handle->m : *ld);
    assert(ild >= handle->m);
#else
    LIBXSMM_UNUSED(ld);
#endif
    switch (handle->oprec) {
      case LIBXSMM_GEMM_PRECISION_F64: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE double
#       include "template/libxsmm_blocked_gemm_copyin_c.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE float
#       include "template/libxsmm_blocked_gemm_copyin_c.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_I16: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE int
#       include "template/libxsmm_blocked_gemm_copyin_c.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      default: {
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: BGEMM precision of matrix A is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: BGEMM-handle cannot be NULL!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API int libxsmm_blocked_gemm_copyout_c(const libxsmm_blocked_gemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
#if 0 /* TODO: support leading dimension for the source buffer */
    const libxsmm_blasint ild = (0 == ld ? handle->m : *ld);
    assert(ild >= handle->m);
#else
    LIBXSMM_UNUSED(ld);
#endif
    switch (handle->oprec) {
      case LIBXSMM_GEMM_PRECISION_F64: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE double
#       include "template/libxsmm_blocked_gemm_copyout_c.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE float
#       include "template/libxsmm_blocked_gemm_copyout_c.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_I16: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE int
#       include "template/libxsmm_blocked_gemm_copyout_c.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      default: {
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: BGEMM precision of matrix A is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: BGEMM-handle cannot be NULL!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API int libxsmm_blocked_gemm_convert_b_to_a(const libxsmm_blocked_gemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
#if 0 /* TODO: support leading dimension for the source buffer */
    const libxsmm_blasint ild = (0 == ld ? handle->k : *ld);
    assert(ild >= handle->k);
#else
    LIBXSMM_UNUSED(ld);
#endif
    switch (handle->iprec) {
      case LIBXSMM_GEMM_PRECISION_F64: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE double
#       include "template/libxsmm_blocked_gemm_convert_b_to_a.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE float
#       include "template/libxsmm_blocked_gemm_convert_b_to_a.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_I16: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE short
#       include "template/libxsmm_blocked_gemm_convert_b_to_a.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      default: {
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: BGEMM precision of matrix B is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: BGEMM-handle cannot be NULL!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API int libxsmm_blocked_gemm_transpose_b(const libxsmm_blocked_gemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
#if 0 /* TODO: support leading dimension for the source buffer */
    const libxsmm_blasint ild = (0 == ld ? handle->k : *ld);
    assert(ild >= handle->k);
#else
    LIBXSMM_UNUSED(ld);
#endif
    switch (handle->iprec) {
      case LIBXSMM_GEMM_PRECISION_F64: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE double
#       include "template/libxsmm_blocked_gemm_transpose_b.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE float
#       include "template/libxsmm_blocked_gemm_transpose_b.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_I16: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE short
#       include "template/libxsmm_blocked_gemm_transpose_b.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      default: {
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: BGEMM precision of matrix B is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: BGEMM-handle cannot be NULL!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_INLINE void internal_bgemm_order(libxsmm_blocked_gemm_order order,
  libxsmm_blasint w_i, libxsmm_blasint nw_i, libxsmm_blasint nw_j, libxsmm_blasint nw_k,
  libxsmm_blasint* i2, libxsmm_blasint* j2, libxsmm_blasint* k2)
{
  switch (order) {
    case LIBXSMM_BLOCKED_GEMM_ORDER_JIK: {
      *j2 = (w_i / (nw_i * nw_k));
      *i2 = (w_i - (*j2) * (nw_i * nw_k)) / nw_k;
      *k2 = (w_i % nw_k);
    } break;
    case LIBXSMM_BLOCKED_GEMM_ORDER_IJK: {
      *i2 = (w_i / (nw_j * nw_k));
      *j2 = (w_i - (*i2) * (nw_j * nw_k)) / nw_k;
      *k2 = (w_i % nw_k);
    } break;
    case LIBXSMM_BLOCKED_GEMM_ORDER_JKI: {
      *j2 = (w_i / (nw_k * nw_i));
      *k2 = (w_i - (*j2) * (nw_k * nw_i)) / nw_i;
      *i2 = (w_i % nw_i);
    } break;
    case LIBXSMM_BLOCKED_GEMM_ORDER_IKJ: {
      *i2 = (w_i / (nw_k * nw_j));
      *k2 = (w_i - (*i2) * (nw_k * nw_j)) / nw_j;
      *j2 = (w_i % nw_j);
    } break;
    case LIBXSMM_BLOCKED_GEMM_ORDER_KJI: {
      *k2 = (w_i / (nw_j * nw_i));
      *j2 = (w_i - (*k2) * (nw_j * nw_i)) / nw_i;
      *i2 = (w_i % nw_i);
    } break;
    case LIBXSMM_BLOCKED_GEMM_ORDER_KIJ: {
      *k2 = (w_i / (nw_i * nw_j));
      *i2 = (w_i - (*k2) * (nw_i * nw_j)) / nw_j;
      *j2 = (w_i % nw_j);
    } break;
    default: assert(0/*should never happen*/);
  }
}

LIBXSMM_API void libxsmm_blocked_gemm_st(const libxsmm_blocked_gemm_handle* handle, const void* a, const void* b, void* c,
  /*unsigned*/int start_thread, /*unsigned*/int tid)
{
  static int error_once = 0;
#if defined(LIBXSMM_BLOCKED_GEMM_CHECKS)
  if (0 != handle && 0 != a && 0 != b && 0 != c && start_thread <= tid && 0 <= tid)
#endif
  {
    const int ltid = tid - start_thread;
    if (handle->nthreads > 1) {
      libxsmm_barrier_init(handle->barrier, ltid);
    }
    switch (handle->iprec) {
      case LIBXSMM_GEMM_PRECISION_F64: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE_AB double
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE_C  double
#       include "template/libxsmm_blocked_gemm.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE_AB
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE_C
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE_AB float
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE_C  float
#       include "template/libxsmm_blocked_gemm.tpl.c"
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE_AB
#       undef  LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE_C
      } break;
      case LIBXSMM_GEMM_PRECISION_I16: {
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE_AB short
#       define LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE_C  int
#       include "template/libxsmm_blocked_gemm.tpl.c"
#       undef LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE_C
#       undef LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE_AB
      } break;
      default: if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: BGEMM precision is not supported!\n");
      }
    }
    if (handle->nthreads > 1) {
      libxsmm_barrier_wait(handle->barrier, ltid);
    }
  }
#if defined(LIBXSMM_BLOCKED_GEMM_CHECKS)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_blocked_gemm!\n");
  }
#endif
}

