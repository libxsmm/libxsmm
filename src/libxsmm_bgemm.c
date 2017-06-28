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
#if !defined(LIBXSMM_BGEMM_PREFETCH)
# define LIBXSMM_BGEMM_PREFETCH
#endif


typedef union LIBXSMM_RETARGETABLE libxsmm_bgemm_lock {
  volatile int instance, pad[16];
} libxsmm_bgemm_lock;

struct LIBXSMM_RETARGETABLE libxsmm_bgemm_handle {
  union { double d; float s; } alpha, beta;
#if defined(LIBXSMM_BGEMM_PREFETCH)
  libxsmm_xmmfunction kernel_pf;
#endif
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
  const void* alpha, const void* beta, const int* gemm_flags, const libxsmm_bgemm_order* order)
{
  libxsmm_bgemm_handle handle = { 0 }, *result = 0;
  libxsmm_gemm_descriptor descriptor = { 0 };
  static int error_once = 0;

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
      handle.alpha.s = (0 != alpha ? *((const double*)alpha) : LIBXSMM_ALPHA);
      handle.beta.s = (0 != beta ? *((const double*)beta) : LIBXSMM_BETA);
      assert(LIBXSMM_FEQ(1, handle.alpha.s) && LIBXSMM_FEQ(1, handle.beta.s)/*TODO*/);
      LIBXSMM_GEMM_DESCRIPTOR(descriptor, precision, handle.flags, bm, bn, bk, bm/*lda*/, bk/*ldb*/, bm/*ldc*/,
        handle.alpha.s, handle.beta.s, LIBXSMM_PREFETCH_NONE);
      handle.typesize = 4;
    } break;
    default: handle.typesize = 0;
  }

  if (0 < handle.typesize) {
    if (0 == (m % bm) && 0 == (n % bn) && 0 == (k % bk)) { /* check for valid block-size */
      handle.b_m1 = 1; handle.b_n1 = 1; handle.b_k1 = 1; handle.b_k2 = 1;
      assert(0 == (m % handle.b_m1) && 0 == (n % handle.b_n1) && 0 == (k % handle.b_k1));
      assert(0 == ((k / handle.b_k1 / handle.b_k2) % bk));
      assert(0 == ((n / handle.b_n1) % bn));
      assert(0 == ((m / handle.b_m1) % bm));
      result = (libxsmm_bgemm_handle*)malloc(sizeof(libxsmm_bgemm_handle));

      if (0 != result) {
        const int sm = m / handle.mb, sn = n / handle.nb, size = sm * sn;
        handle.precision = precision;
        handle.m = m; handle.n = n; handle.k = k; handle.bm = bm; handle.bn = bn; handle.bk = bk;
        handle.mb = m / bm; handle.nb = n / bn; handle.kb = k / bk;
        handle.buffer = libxsmm_aligned_malloc(LIBXSMM_BGEMM_MAX_NTHREADS * bm * bn * handle.typesize, LIBXSMM_ALIGNMENT);
        handle.locks = (libxsmm_bgemm_lock*)libxsmm_aligned_malloc(size * sizeof(libxsmm_bgemm_lock), LIBXSMM_ALIGNMENT);
        memset(handle.locks, 0, size * sizeof(libxsmm_bgemm_lock));
        handle.order = (0 == order ? LIBXSMM_BGEMM_ORDER_JIK : *order);
        handle.kernel = libxsmm_xmmdispatch(&descriptor);
#if defined(LIBXSMM_BGEMM_PREFETCH)
        descriptor.prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C;
        handle.kernel_pf = libxsmm_xmmdispatch(&descriptor);
#endif
        *result = handle;
      }
    }
    else {
      if (0 != libxsmm_verbosity /* library code is expected to be mute */
       && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM: BGEMM block-size is invalid!\n");
      }
    }
  }
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM: BGEMM precision is not supported!\n");
  }

  return result;
}


LIBXSMM_API_DEFINITION void libxsmm_bgemm_handle_destroy(const libxsmm_bgemm_handle* handle)
{
  if (0 != handle) {
    /* TODO: release internal structures */
    free((libxsmm_bgemm_handle*)handle);
  }
}


LIBXSMM_API_DEFINITION int libxsmm_bgemm_copyin_a(const libxsmm_bgemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
    libxsmm_blasint mb, kb, bm, bk;
    const libxsmm_blasint ild = (0 == ld ? handle->m : *ld);
    /* TODO: support leading dimension for the source buffer */
    assert(ild >= handle->m); LIBXSMM_UNUSED(ild);

    switch (handle->precision) {
      case LIBXSMM_GEMM_PRECISION_F64: {
        LIBXSMM_VLA_DECL(4, double, real_dst, dst, handle->mb, handle->bk, handle->bm);
        LIBXSMM_VLA_DECL(2, const double, real_src, src, handle->m);
        for (kb = 0; kb < handle->kb; ++kb) {
          for (mb = 0; mb < handle->mb; ++mb) {
            for (bk = 0; bk < handle->bk; ++bk) {
              for (bm = 0; bm < handle->bm; ++bm) {
                LIBXSMM_VLA_ACCESS(4, real_dst, kb, mb, bk, bm, handle->mb, handle->bk, handle->bm) =
                LIBXSMM_VLA_ACCESS(2, real_src, kb * handle->bk + bk, mb * handle->bm + bm, handle->m);
              }
            }
          }
        }
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
        LIBXSMM_VLA_DECL(4, float, real_dst, dst, handle->mb, handle->bk, handle->bm);
        LIBXSMM_VLA_DECL(2, const float, real_src, src, handle->m);
        for (kb = 0; kb < handle->kb; ++kb) {
          for (mb = 0; mb < handle->mb; ++mb) {
            for (bk = 0; bk < handle->bk; ++bk) {
              for (bm = 0; bm < handle->bm; ++bm) {
                LIBXSMM_VLA_ACCESS(4, real_dst, kb, mb, bk, bm, handle->mb, handle->bk, handle->bm) =
                LIBXSMM_VLA_ACCESS(2, real_src, kb * handle->bk + bk, mb * handle->bm + bm, handle->m);
              }
            }
          }
        }
      } break;
      default: {
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
         && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM: BGEMM precision of matrix A is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM: BGEMM-handle cannot be NULL!\n");
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
    libxsmm_blasint kb, nb, bk, bn;
    const libxsmm_blasint ild = (0 == ld ? handle->k : *ld);
    /* TODO: support leading dimension for the source buffer */
    assert(ild >= handle->k); LIBXSMM_UNUSED(ild);

    switch (handle->precision) {
      case LIBXSMM_GEMM_PRECISION_F64: {
        LIBXSMM_VLA_DECL(4, double, real_dst, dst, handle->kb, handle->bn, handle->bk);
        LIBXSMM_VLA_DECL(2, const double, real_src, src, handle->k);
        for (nb = 0; nb < handle->nb; ++nb) {
          for (kb = 0; kb < handle->kb; ++kb) {
            for (bn = 0; bn < handle->bn; ++bn) {
              for (bk = 0; bk < handle->bk; ++bk) {
                LIBXSMM_VLA_ACCESS(4, real_dst, nb, kb, bn, bk, handle->kb, handle->bn, handle->bk) =
                LIBXSMM_VLA_ACCESS(2, real_src, nb * handle->bn + bn, kb * handle->bk + bk, handle->k);
              }
            }
          }
        }
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
        LIBXSMM_VLA_DECL(4, float, real_dst, dst, handle->kb, handle->bn, handle->bk);
        LIBXSMM_VLA_DECL(2, const float, real_src, src, handle->k);
        for (nb = 0; nb < handle->nb; ++nb) {
          for (kb = 0; kb < handle->kb; ++kb) {
            for (bn = 0; bn < handle->bn; ++bn) {
              for (bk = 0; bk < handle->bk; ++bk) {
                LIBXSMM_VLA_ACCESS(4, real_dst, nb, kb, bn, bk, handle->kb, handle->bn, handle->bk) =
                LIBXSMM_VLA_ACCESS(2, real_src, nb * handle->bn + bn, kb * handle->bk + bk, handle->k);
              }
            }
          }
        }
      } break;
      default: {
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
         && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM: BGEMM precision of matrix B is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM: BGEMM-handle cannot be NULL!\n");
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
    libxsmm_blasint mb, nb, bm, bn;
    const libxsmm_blasint ild = (0 == ld ? handle->m : *ld);
    /* TODO: support leading dimension for the source buffer */
    assert(ild >= handle->m); LIBXSMM_UNUSED(ild);

    switch (handle->precision) {
      case LIBXSMM_GEMM_PRECISION_F64: {
        LIBXSMM_VLA_DECL(4, double, real_dst, dst, handle->mb, handle->bn, handle->bm);
        LIBXSMM_VLA_DECL(2, const double, real_src, src, handle->m);
        for (nb = 0; nb < handle->nb; ++nb) {
          for (mb = 0; mb < handle->mb; ++mb) {
            for (bn = 0; bn < handle->bn; ++bn) {
              for (bm = 0; bm < handle->bm; ++bm) {
                LIBXSMM_VLA_ACCESS(4, real_dst, nb, mb, bn, bm, handle->mb, handle->bn, handle->bm) =
                LIBXSMM_VLA_ACCESS(2, real_src, nb * handle->bn + bn, mb * handle->bm + bm, handle->m);
              }
            }
          }
        }
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
        LIBXSMM_VLA_DECL(4, double, real_dst, dst, handle->mb, handle->bn, handle->bm);
        LIBXSMM_VLA_DECL(2, const double, real_src, src, handle->m);
        for (nb = 0; nb < handle->nb; ++nb) {
          for (mb = 0; mb < handle->mb; ++mb) {
            for (bn = 0; bn < handle->bn; ++bn) {
              for (bm = 0; bm < handle->bm; ++bm) {
                LIBXSMM_VLA_ACCESS(4, real_dst, nb, mb, bn, bm, handle->mb, handle->bn, handle->bm) =
                LIBXSMM_VLA_ACCESS(2, real_src, nb * handle->bn + bn, mb * handle->bm + bm, handle->m);
              }
            }
          }
        }
      } break;
      default: {
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
         && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM: BGEMM precision of matrix A is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM: BGEMM-handle cannot be NULL!\n");
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

#if 0
# define _KERNEL libxsmm_dmmfunction
typedef double real_type;
#else
# define _KERNEL libxsmm_smmfunction
typedef float real_type;
#endif

LIBXSMM_API_DEFINITION void libxsmm_bgemm(const libxsmm_bgemm_handle* handle,
  const void* a, const void* b, void* c, int tid, int nthreads)
{
  LIBXSMM_VLA_DECL(2, libxsmm_bgemm_lock, locks, handle->locks, handle->nb);
  /* TODO: align thread-local buffer portion with the size of a cache-line in order to avoid "Ping-Pong" */
  LIBXSMM_VLA_DECL(2, real_type, l_out, (real_type*)(((char*)handle->buffer) + tid * handle->bm * handle->bn * handle->typesize), handle->bm);
  LIBXSMM_VLA_DECL(4, const real_type, real_a, a, handle->mb, handle->bk, handle->bm);
  LIBXSMM_VLA_DECL(4, const real_type, real_b, b, handle->kb, handle->bn, handle->bk);
  LIBXSMM_VLA_DECL(4, real_type, real_c, c, handle->mb, handle->bn, handle->bm);

  _KERNEL l_kernel = handle->kernel.smm;
#if defined(LIBXSMM_BGEMM_PREFETCH)
  _KERNEL l_kernel_pf = handle->kernel_pf.smm;
#endif

  const libxsmm_blasint b_m1 = handle->b_m1;
  const libxsmm_blasint b_n1 = handle->b_n1;
  const libxsmm_blasint b_k1 = handle->b_k1;
  const libxsmm_blasint b_k2 = handle->b_k2;

  const libxsmm_blasint mm = handle->m / b_m1;
  const libxsmm_blasint nn = handle->n / b_n1;
  const libxsmm_blasint kk = handle->k / b_k1;

  const libxsmm_blasint nw_i = mm / handle->bm;
  const libxsmm_blasint nw_j = nn / handle->bn;
  libxsmm_blasint nw_k = kk / handle->bk; /* TODO: check */
  const libxsmm_blasint nw = nw_i * nw_j * nw_k;

  libxsmm_blasint m, n, k, mb, nb, kb;
  libxsmm_blasint ki, kj, w_i, _ki;

  /* TODO: take transa and transb into account (flags) */

  for (ki = 0; ki < handle->bn; ++ki) {
    LIBXSMM_PRAGMA_SIMD
    for (kj = 0; kj < handle->bm; ++kj) {
      LIBXSMM_VLA_ACCESS(2, l_out, ki, kj, handle->bm) = 0;
    }
  }

  for (mb = 0, m = 0; mb < b_m1; ++mb, m += nw_i) {
    for (nb = 0, n = 0; nb < b_n1; ++nb, n += nw_j) {
      for (kb = 0, k = 0; kb < b_k1; ++kb, k += nw_k) {
        int s = (tid * nw) / nthreads;
        int e = ((tid + 1) * nw) / nthreads;
        int o_i2 = 0, o_j2 = 0;
        nw_k /= b_k2; /* TODO: check */

        for (w_i = s; w_i < e; ++w_i) {
          int i2 = 0, j2 = 0, k2 = 0;
          internal_bgemm_order(handle->order, w_i, nw_i, nw_j, nw_k, &i2, &j2, &k2);

          i2 = m + i2;
          j2 = n + j2;
          k2 = k + k2;

          if (w_i == s) {
            o_i2 = i2;
            o_j2 = j2;
          }
          else {
            if ((o_i2 != i2) || (o_j2 != j2)) {
              libxsmm_bgemm_lock *const lock = &LIBXSMM_VLA_ACCESS(2, locks, o_i2, o_j2, handle->nb);
              LIBXSMM_SYNC_SET(lock->instance);
              for (ki = 0; ki < handle->bn; ++ki) {
                LIBXSMM_PRAGMA_SIMD
                for (kj = 0; kj < handle->bm; ++kj) {
                  LIBXSMM_VLA_ACCESS(4, real_c, o_j2, o_i2, ki, kj, handle->mb, handle->bn, handle->bm) +=
                  LIBXSMM_VLA_ACCESS(2, l_out, ki, kj, handle->bm);
                }
              }
              LIBXSMM_SYNC_UNSET(lock->instance);
              for (ki = 0; ki < handle->bn; ++ki) {
                LIBXSMM_PRAGMA_SIMD
                for (kj = 0; kj < handle->bm; ++kj) {
                  LIBXSMM_VLA_ACCESS(2, l_out, ki, kj, handle->bm) = 0;
                }
              }
              o_i2 = i2;
              o_j2 = j2;
            }
          }
          for (_ki = 0, ki = (b_k2 * k2); _ki < b_k2 ; ++_ki, ++ki) {
#if !defined(LIBXSMM_BGEMM_PREFETCH)
            l_kernel(&LIBXSMM_VLA_ACCESS(4, real_a, ki, i2, 0, 0, handle->mb, handle->bk, handle->bm),
                     &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki, 0, 0, handle->kb, handle->bn, handle->bk), l_out);
#else
            if (k2 < (nw_k - 2)) { /* avoid prefetching of untouched data */
#if defined(__AVX2__)
              l_kernel_pf(&LIBXSMM_VLA_ACCESS(4, real_a, ki, i2, 0, 0, handle->mb, handle->bk, handle->bm),
                          &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki, 0, 0, handle->kb, handle->bn, handle->bk), l_out,
                          &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki+1, 0, 0, handle->kb, handle->bn, handle->bk),
                          &LIBXSMM_VLA_ACCESS(4, real_a, ki+1, i2, 0, 0, handle->mb, handle->bk, handle->bm), NULL);
#else
              l_kernel_pf(&LIBXSMM_VLA_ACCESS(4, real_a, ki, i2, 0, 0, handle->mb, handle->bk, handle->bm),
                          &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki, 0, 0, handle->kb, handle->bn, handle->bk), l_out,
                          &LIBXSMM_VLA_ACCESS(4, real_a, ki+1, i2, 0, 0, handle->mb, handle->bk, handle->bm),
                          &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki+1, 0, 0, handle->kb, handle->bn, handle->bk), NULL);
#endif
            }
            else {
              l_kernel(&LIBXSMM_VLA_ACCESS(4, real_a, ki, i2, 0, 0, handle->mb, handle->bk, handle->bm),
                       &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki, 0, 0, handle->kb, handle->bn, handle->bk), l_out);
            }
#endif
          }

          if (w_i == (e - 1)) {
            libxsmm_bgemm_lock* lock;
            o_i2 = i2;
            o_j2 = j2;

            lock = &LIBXSMM_VLA_ACCESS(2, locks, o_i2, o_j2, handle->nb);
            LIBXSMM_SYNC_SET(lock->instance);
            for (ki = 0; ki < handle->bn; ++ki) {
              LIBXSMM_PRAGMA_SIMD
              for (kj = 0; kj < handle->bm; ++kj) {
                LIBXSMM_VLA_ACCESS(4, real_c, o_j2, o_i2, ki, kj, handle->mb, handle->bn, handle->bm) +=
                LIBXSMM_VLA_ACCESS(2, l_out, ki, kj, handle->bm);
              }
            }
            LIBXSMM_SYNC_UNSET(lock->instance);
            for (ki = 0; ki < handle->bn; ++ki) {
              LIBXSMM_PRAGMA_SIMD
              for (kj = 0; kj < handle->bm; ++kj) {
                LIBXSMM_VLA_ACCESS(2, l_out, ki, kj, handle->bm) = 0;
              }
            }
          }
        }
      }
    }
  }
}

