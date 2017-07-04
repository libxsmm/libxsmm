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
#include <libxsmm.h>
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
  volatile int instance, var[16], pad[16];
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


LIBXSMM_API_DEFINITION libxsmm_bgemm_handle* libxsmm_bgemm_handle_create(libxsmm_gemm_precision precision,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint bm, libxsmm_blasint bn, libxsmm_blasint bk,
  libxsmm_blasint b_m1, libxsmm_blasint b_n1, libxsmm_blasint b_k1, libxsmm_blasint b_k2,
  const void* alpha, const void* beta, const int* gemm_flags,
  const libxsmm_gemm_prefetch_type* strategy,
  const libxsmm_bgemm_order* order)
{
  libxsmm_bgemm_handle handle, *result = 0;
  libxsmm_gemm_descriptor descriptor = { 0 };
  static int error_once = 0;

  if (0 < m && 0 < n && 0 < k && 0 < bm && 0 < bn && 0 < bk) {
    memset(&handle, 0, sizeof(handle));
    handle.flags = (0 == gemm_flags ? LIBXSMM_FLAGS : *gemm_flags);

    switch (precision) {
      case LIBXSMM_GEMM_PRECISION_F64: {
        handle.alpha.d = (0 != alpha ? *((const double*)alpha) : LIBXSMM_ALPHA);
        handle.beta.d = (0 != beta ? *((const double*)beta) : LIBXSMM_BETA);
        assert(LIBXSMM_FEQ(1, handle.alpha.d) && LIBXSMM_FEQ(1, handle.beta.d)/*TODO*/);
        LIBXSMM_GEMM_DESCRIPTOR(descriptor, precision, handle.flags, bm, bn, bk, bm/*lda*/, bk/*ldb*/, bm/*ldc*/,
          handle.alpha.d, handle.beta.d, LIBXSMM_PREFETCH_NONE);
        handle.typesize = 8;
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
        handle.alpha.s = (0 != alpha ? *((const float*)alpha) : LIBXSMM_ALPHA);
        handle.beta.s = (0 != beta ? *((const float*)beta) : LIBXSMM_BETA);
        assert(LIBXSMM_FEQ(1, handle.alpha.s) && LIBXSMM_FEQ(1, handle.beta.s)/*TODO*/);
        LIBXSMM_GEMM_DESCRIPTOR(descriptor, precision, handle.flags, bm, bn, bk, bm/*lda*/, bk/*ldb*/, bm/*ldc*/,
          handle.alpha.s, handle.beta.s, LIBXSMM_PREFETCH_NONE);
        handle.typesize = 4;
      } break;
      case LIBXSMM_GEMM_PRECISION_I16: {
        /*
         * Take alpha and beta as short data although wgemm takes full integers.
         * However, alpha and beta are only JIT-supported for certain value,
         * and the call-side may not distinct different input and output types
         * (integer/short), hence it is safer to only read short data.
         */
        handle.alpha.w = (0 != alpha ? *((const short*)alpha) : LIBXSMM_ALPHA);
        handle.beta.w = (0 != beta ? *((const short*)beta) : LIBXSMM_BETA);
        assert(LIBXSMM_FEQ(1, handle.alpha.w) && LIBXSMM_FEQ(1, handle.beta.w)/*TODO*/);
        LIBXSMM_GEMM_DESCRIPTOR(descriptor, precision, handle.flags, bm, bn, bk, bm/*lda*/, bk/*ldb*/, bm/*ldc*/,
          handle.alpha.w, handle.beta.w, LIBXSMM_PREFETCH_NONE);
        handle.typesize = 2;
      } break;
      default: ;
    }

    if (0 < handle.typesize) {
      handle.mb = m / bm; handle.nb = n / bn; handle.kb = k / bk;

      if (0 == (m % bm) && 0 == (n % bn) && 0 == (k % bk)) { /* check for valid block-size */
        const libxsmm_gemm_prefetch_type prefetch = (0 == strategy ? ((libxsmm_gemm_prefetch_type)LIBXSMM_PREFETCH) : *strategy);
        const libxsmm_blasint sm = m / handle.mb, sn = n / handle.nb, size = sm * sn;
        handle.b_m1 = b_m1; handle.b_n1 = b_n1; handle.b_k1 = b_k1; handle.b_k2 = b_k2;
        assert(0 == (m % handle.b_m1) && 0 == (n % handle.b_n1) && 0 == (k % handle.b_k1));
        assert(0 == ((k / handle.b_k1 / handle.b_k2) % bk));
        assert(0 == ((n / handle.b_n1) % bn));
        assert(0 == ((m / handle.b_m1) % bm));
        handle.kernel = libxsmm_xmmdispatch(&descriptor);
        if (0 != handle.kernel.smm && LIBXSMM_PREFETCH_NONE != prefetch && LIBXSMM_PREFETCH_SIGONLY != prefetch) {
          if (LIBXSMM_PREFETCH_AUTO == prefetch) { /* automatically chosen */
            /* TODO: more sophisticated strategy perhaps according to CPUID */
            descriptor.prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C;
          }
          else { /* user-defined */
            descriptor.prefetch = (unsigned short)prefetch;
          }
          handle.kernel_pf = libxsmm_xmmdispatch(&descriptor);
        }
        if (0 != handle.kernel.smm && (LIBXSMM_PREFETCH_NONE == descriptor.prefetch || 0 != handle.kernel_pf.smm)) {
          result = (libxsmm_bgemm_handle*)malloc(sizeof(libxsmm_bgemm_handle));
          handle.buffer = libxsmm_aligned_malloc(LIBXSMM_BGEMM_MAX_NTHREADS * bm * bn * handle.typesize, LIBXSMM_ALIGNMENT);
          handle.locks = (libxsmm_bgemm_lock*)libxsmm_aligned_malloc(size * sizeof(libxsmm_bgemm_lock), LIBXSMM_ALIGNMENT);

          if (0 != result && 0 != handle.buffer && 0 != handle.locks) {
            handle.precision = precision;
            handle.m = m; handle.n = n; handle.k = k; handle.bm = bm; handle.bn = bn; handle.bk = bk;
            memset(handle.locks, 0, size * sizeof(libxsmm_bgemm_lock));
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
