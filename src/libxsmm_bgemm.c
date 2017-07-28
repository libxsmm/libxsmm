/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
   Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm_bgemm.h>
#include "libxsmm_gemm.h"
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_BGEMM_MAX_NTHREADS)
# define LIBXSMM_BGEMM_MAX_NTHREADS 512
#endif


typedef union LIBXSMM_RETARGETABLE libxsmm_bgemm_lock {
  volatile int instance, pad[16];
} libxsmm_bgemm_lock;

struct LIBXSMM_RETARGETABLE libxsmm_bgemm_handle {
  union { double d; float s; int w; } alpha, beta;
  libxsmm_xmmfunction kernel_pf;
  libxsmm_xmmfunction kernel;
  void* buffer;
  libxsmm_bgemm_lock* locks;
  libxsmm_blasint m, n, k, bm, bn, bk;
  libxsmm_blasint b_m1, b_n1, b_k1, b_k2;
  libxsmm_blasint mb, nb, kb;
  libxsmm_gemm_precision precision;
  libxsmm_bgemm_order order;
  int typesize;
  int flags;
};


LIBXSMM_API_DEFINITION libxsmm_bgemm_handle* libxsmm_bgemm_handle_create(
  libxsmm_gemm_precision precision, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* bm, const libxsmm_blasint* bn, const libxsmm_blasint* bk,
  const libxsmm_blasint* b_m1, const libxsmm_blasint* b_n1, const libxsmm_blasint* b_k1, const libxsmm_blasint* b_k2,
  const void* alpha, const void* beta, const int* gemm_flags,
  const libxsmm_gemm_prefetch_type* strategy,
  const libxsmm_bgemm_order* order)
{
  const char *const env_m = getenv("LIBXSMM_BGEMM_M"), *const env_n = getenv("LIBXSMM_BGEMM_N"), *const env_k = getenv("LIBXSMM_BGEMM_K");
  const libxsmm_blasint mm = LIBXSMM_MIN(0 == bm ? ((0 == env_m || 0 == *env_m) ? 32 : atoi(env_m)) : *bm, m);
  const libxsmm_blasint kk = LIBXSMM_MIN(0 == bk ? ((0 == env_k || 0 == *env_k) ? mm : atoi(env_k)) : *bk, k);
  const libxsmm_blasint nn = LIBXSMM_MIN(0 == bn ? ((0 == env_n || 0 == *env_n) ? kk : atoi(env_n)) : *bn, n);
  libxsmm_bgemm_handle handle, *result = 0;
  libxsmm_gemm_descriptor descriptor = { 0 };
  static int error_once = 0;

  if (0 < m && 0 < n && 0 < k && 0 < mm && 0 < nn && 0 < kk) {
    memset(&handle, 0, sizeof(handle));

    if (EXIT_SUCCESS == libxsmm_gemm_descriptor_init(&descriptor,
      precision, mm, nn, kk, &mm/*lda*/, &kk/*ldb*/, &mm/*ldc*/,
      alpha, beta, gemm_flags, 0/*prefetch*/))
    {
      handle.typesize = LIBXSMM_TYPESIZE(precision);
      handle.mb = m / mm; handle.nb = n / nn; handle.kb = k / kk;
      assert(0 < handle.typesize);

      if (0 == (m % mm) && 0 == (n % nn) && 0 == (k % kk) &&
          0 == (m % *b_m1) && 0 == (n % *b_n1) && 0 == (k % *b_k1) &&
          0 == ((k / *b_k1 / *b_k2) % kk) && 0 == ((n / *b_n1) % nn) && 0 == ((m / *b_m1) % mm)) { /* check for valid block-size */
        const libxsmm_gemm_prefetch_type prefetch = (0 == strategy ? ((libxsmm_gemm_prefetch_type)LIBXSMM_PREFETCH) : *strategy);
        handle.b_m1 = *b_m1; handle.b_n1 = *b_n1;
        handle.b_k1 = *b_k1; handle.b_k2 = *b_k2;
        handle.kernel = libxsmm_xmmdispatch(&descriptor);
        if (0 != handle.kernel.smm && LIBXSMM_PREFETCH_NONE != prefetch && LIBXSMM_PREFETCH_SIGONLY != prefetch) {
          if (LIBXSMM_PREFETCH_AUTO == prefetch) { /* automatically chosen */
            /* TODO: more sophisticated strategy perhaps according to CPUID */
            const char *const env_p = getenv("LIBXSMM_BGEMM_PREFETCH");
            const int uid = ((0 == env_p || 0 == *env_p) ? 7/*LIBXSMM_PREFETCH_AL2BL2_VIA_C*/ : atoi(env_p));
            descriptor.prefetch = (unsigned short)libxsmm_gemm_uid2prefetch(uid);
          }
          else { /* user-defined */
            descriptor.prefetch = (unsigned short)prefetch;
          }
          handle.kernel_pf = libxsmm_xmmdispatch(&descriptor);
        }
        if (0 != handle.kernel.smm && (LIBXSMM_PREFETCH_NONE == descriptor.prefetch || 0 != handle.kernel_pf.smm)) {
          const size_t tls_size = ((mm * nn * handle.typesize + LIBXSMM_CACHELINE_SIZE - 1) & ~(LIBXSMM_CACHELINE_SIZE - 1)) * LIBXSMM_BGEMM_MAX_NTHREADS;
          const libxsmm_blasint size_locks = handle.mb * handle.nb * sizeof(libxsmm_bgemm_lock);
          handle.locks = (libxsmm_bgemm_lock*)libxsmm_aligned_malloc(size_locks, LIBXSMM_ALIGNMENT);
          handle.buffer = libxsmm_aligned_malloc(tls_size, LIBXSMM_ALIGNMENT);
          result = (libxsmm_bgemm_handle*)malloc(sizeof(libxsmm_bgemm_handle));

          if (0 != result && 0 != handle.buffer && 0 != handle.locks) {
            handle.precision = precision;
            handle.m = m; handle.n = n; handle.k = k; handle.bm = mm; handle.bn = nn; handle.bk = kk;
            memset(handle.locks, 0, size_locks);
            handle.order = (0 == order ? LIBXSMM_BGEMM_ORDER_JIK : *order);
            *result = handle;
          }
          else {
            if (0 != libxsmm_verbosity /* library code is expected to be mute */
             && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
            {
              fprintf(stderr, "LIBXSMM ERROR: BGEMM handle allocation failed!\n");
            }
            libxsmm_free(handle.buffer);
            libxsmm_free(handle.locks);
            free(result);
            result = 0;
          }
        }
        else if (0 != libxsmm_verbosity /* library code is expected to be mute */
              && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: BGEMM kernel generation failed!\n");
        }
      }
      else {
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
         && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: BGEMM block-size is invalid!\n");
        }
      }
    }
    else if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: BGEMM precision is not supported!\n");
    }
  }
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_bgemm_handle_create!\n");
  }

  return result;
}


LIBXSMM_API_DEFINITION void libxsmm_bgemm_handle_destroy(const libxsmm_bgemm_handle* handle)
{
  if (0 != handle) {
    libxsmm_free(handle->buffer);
    libxsmm_free(handle->locks);
    free((libxsmm_bgemm_handle*)handle);
  }
}


LIBXSMM_API_DEFINITION int libxsmm_bgemm_copyin_a(const libxsmm_bgemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
    const libxsmm_blasint ild = (0 == ld ? handle->m : *ld);
    /* TODO: support leading dimension for the source buffer */
    assert(ild >= handle->m); LIBXSMM_UNUSED(ild);

    switch (handle->precision) {
      case LIBXSMM_GEMM_PRECISION_F64: {
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE double
#       include "template/libxsmm_bgemm_copyin_a.tpl.c"
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE float
#       include "template/libxsmm_bgemm_copyin_a.tpl.c"
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_I16: {
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE short
#       include "template/libxsmm_bgemm_copyin_a.tpl.c"
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE
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


LIBXSMM_API_DEFINITION int libxsmm_bgemm_copyin_b(const libxsmm_bgemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
    const libxsmm_blasint ild = (0 == ld ? handle->k : *ld);
    /* TODO: support leading dimension for the source buffer */
    assert(ild >= handle->k); LIBXSMM_UNUSED(ild);

    switch (handle->precision) {
      case LIBXSMM_GEMM_PRECISION_F64: {
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE double
#       include "template/libxsmm_bgemm_copyin_b.tpl.c"
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE float
#       include "template/libxsmm_bgemm_copyin_b.tpl.c"
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_I16: {
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE short
#       include "template/libxsmm_bgemm_copyin_b.tpl.c"
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE
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


LIBXSMM_API_DEFINITION int libxsmm_bgemm_copyin_c(const libxsmm_bgemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
    const libxsmm_blasint ild = (0 == ld ? handle->m : *ld);
    /* TODO: support leading dimension for the source buffer */
    assert(ild >= handle->m); LIBXSMM_UNUSED(ild);

    switch (handle->precision) {
      case LIBXSMM_GEMM_PRECISION_F64: {
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE double
#       include "template/libxsmm_bgemm_copyin_c.tpl.c"
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE float
#       include "template/libxsmm_bgemm_copyin_c.tpl.c"
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_I16: {
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE int
#       include "template/libxsmm_bgemm_copyin_c.tpl.c"
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE
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


LIBXSMM_API_DEFINITION int libxsmm_bgemm_copyout_c(const libxsmm_bgemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
    const libxsmm_blasint ild = (0 == ld ? handle->m : *ld);
    /* TODO: support leading dimension for the source buffer */
    assert(ild >= handle->m); LIBXSMM_UNUSED(ild);

    switch (handle->precision) {
      case LIBXSMM_GEMM_PRECISION_F64: {
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE double
#       include "template/libxsmm_bgemm_copyout_c.tpl.c"
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE float
#       include "template/libxsmm_bgemm_copyout_c.tpl.c"
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXSMM_GEMM_PRECISION_I16: {
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE int
#       include "template/libxsmm_bgemm_copyout_c.tpl.c"
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE
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


LIBXSMM_API_INLINE void internal_bgemm_order(libxsmm_bgemm_order order,
  libxsmm_blasint w_i, libxsmm_blasint nw_i, libxsmm_blasint nw_j, libxsmm_blasint nw_k,
  libxsmm_blasint* i2, libxsmm_blasint* j2, libxsmm_blasint* k2)
{
  switch (order) {
    case LIBXSMM_BGEMM_ORDER_JIK: {
      *j2 = (w_i / (nw_i * nw_k));
      *i2 = (w_i - (*j2) * (nw_i * nw_k)) / nw_k;
      *k2 = (w_i % nw_k);
    } break;
    case LIBXSMM_BGEMM_ORDER_IJK: {
      *i2 = (w_i / (nw_j * nw_k));
      *j2 = (w_i - (*i2) * (nw_j * nw_k)) / nw_k;
      *k2 = (w_i % nw_k);
    } break;
    case LIBXSMM_BGEMM_ORDER_JKI: {
      *j2 = (w_i / (nw_k * nw_i));
      *k2 = (w_i - (*j2) * (nw_k * nw_i)) / nw_i;
      *i2 = (w_i % nw_i);
    } break;
    case LIBXSMM_BGEMM_ORDER_IKJ: {
      *i2 = (w_i / (nw_k * nw_j));
      *k2 = (w_i - (*i2) * (nw_k * nw_j)) / nw_j;
      *j2 = (w_i % nw_j);
    } break;
    case LIBXSMM_BGEMM_ORDER_KJI: {
      *k2 = (w_i / (nw_j * nw_i));
      *j2 = (w_i - (*k2) * (nw_j * nw_i)) / nw_i;
      *i2 = (w_i % nw_i);
    } break;
    case LIBXSMM_BGEMM_ORDER_KIJ: {
      *k2 = (w_i / (nw_i * nw_j));
      *i2 = (w_i - (*k2) * (nw_i * nw_j)) / nw_j;
      *j2 = (w_i % nw_j);
    } break;
    default: assert(0/*should never happen*/);
  }
}

LIBXSMM_API_DEFINITION void libxsmm_bgemm(const libxsmm_bgemm_handle* handle,
  const void* a, const void* b, void* c, int tid, int nthreads)
{
  static int error_once = 0;
#if !defined(NDEBUG) /* intentionally no errror check in release build */
  if (0 != handle && 0 != a && 0 != b && 0 != c && 0 <= tid && tid < nthreads)
#endif
  {
    switch (handle->precision) {
      case LIBXSMM_GEMM_PRECISION_F64: {
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_AB double
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_C  double
#       include "template/libxsmm_bgemm.tpl.c"
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_AB
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_C
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_AB float
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_C  float
#       include "template/libxsmm_bgemm.tpl.c"
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_AB
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_C
      } break;
      case LIBXSMM_GEMM_PRECISION_I16: {
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_AB short
#       define LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_C  int
#       include "template/libxsmm_bgemm.tpl.c"
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_AB
#       undef  LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_C
      } break;
      default: if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: BGEMM precision is not supported!\n");
      }
    }
  }
#if !defined(NDEBUG) /* intentionally no errror check in release build */
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_bgemm!\n");
  }
#endif
}
