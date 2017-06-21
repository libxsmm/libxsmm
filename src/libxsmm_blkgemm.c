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

#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif
#include "libxsmm_blkgemm.h"

#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl_service.h>
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
/* note: this does not reproduce 48-bit RNG quality */
# define drand48() ((double)rand() / RAND_MAX)
# define srand48 srand
#endif


/** Function prototype for SGEMM; this way any kind of LAPACK/BLAS library is sufficient at link-time. */

LIBXSMM_API_DEFINITION  void libxsmm_blksgemm_init_a( libxsmm_blkgemm_handle* handle,
                                             real* libxsmm_mat_dst,
                                             real* colmaj_mat_src ) {
  LIBXSMM_VLA_DECL(4, real, dst, libxsmm_mat_dst, handle->mb, handle->bk, handle->bm);
  LIBXSMM_VLA_DECL(2, const real, src, colmaj_mat_src, handle->m);
  int mb, kb, bm, bk;

  for ( kb = 0; kb < handle->kb; kb++ ) {
    for ( mb = 0; mb < handle->mb; mb++ ) {
      for ( bk = 0; bk < handle->bk; bk++ ) {
        for ( bm = 0; bm < handle->bm; bm++ ) {
          LIBXSMM_VLA_ACCESS(4, dst, kb, mb, bk, bm, handle->mb, handle->bk, handle->bm) =
          LIBXSMM_VLA_ACCESS(2, src, kb * handle->bk + bk, mb * handle->bm + bm, handle->m);
        }
      }
    }
  }
}

LIBXSMM_API_DEFINITION  void libxsmm_blksgemm_init_b( libxsmm_blkgemm_handle* handle,
                                             real* libxsmm_mat_dst,
                                             real* colmaj_mat_src ) {
  LIBXSMM_VLA_DECL(4, real, dst, libxsmm_mat_dst, handle->kb, handle->bn, handle->bk);
  LIBXSMM_VLA_DECL(2, const real, src, colmaj_mat_src, handle->k);
  int kb, nb, bk, bn;

  for ( nb = 0; nb < handle->nb; nb++ ) {
    for ( kb = 0; kb < handle->kb; kb++ ) {
      for ( bn = 0; bn < handle->bn; bn++ ) {
        for ( bk = 0; bk < handle->bk; bk++ ) {
          LIBXSMM_VLA_ACCESS(4, dst, nb, kb, bn, bk, handle->kb, handle->bn, handle->bk) =
          LIBXSMM_VLA_ACCESS(2, src, nb * handle->bn + bn, kb * handle->bk + bk, handle->k);
        }
      }
    }
  }
}

LIBXSMM_API_DEFINITION  void libxsmm_blksgemm_init_c( libxsmm_blkgemm_handle* handle,
                                             real* libxsmm_mat_dst,
                                             real* colmaj_mat_src ) {
  LIBXSMM_VLA_DECL(4, real, dst, libxsmm_mat_dst, handle->mb, handle->bn, handle->bm);
  LIBXSMM_VLA_DECL(2, const real, src, colmaj_mat_src, handle->m);
  int mb, nb, bm, bn;

  for ( nb = 0; nb < handle->nb; nb++ ) {
    for ( mb = 0; mb < handle->mb; mb++ ) {
      for ( bn = 0; bn < handle->bn; bn++ ) {
        for ( bm = 0; bm < handle->bm; bm++ ) {
          LIBXSMM_VLA_ACCESS(4, dst, nb, mb, bn, bm, handle->mb, handle->bn, handle->bm) =
          LIBXSMM_VLA_ACCESS(2, src, nb * handle->bn + bn, mb * handle->bm + bm, handle->m);
        }
      }
    }
  }
}

LIBXSMM_API_DEFINITION  void libxsmm_blksgemm_check_c( libxsmm_blkgemm_handle* handle,
                                              real* libxsmm_mat_dst,
                                              real* colmaj_mat_src ) {
  LIBXSMM_VLA_DECL(4, real, dst, libxsmm_mat_dst, handle->mb, handle->bn, handle->bm);
  LIBXSMM_VLA_DECL(2, const real, src, colmaj_mat_src, handle->m);
  int mb, nb, bm, bn;
  double max_error = 0.0;
  double src_norm = 0.0;
  double dst_norm = 0.0;

  for ( nb = 0; nb < handle->nb; nb++ ) {
    for ( mb = 0; mb < handle->mb; mb++ ) {
      for ( bn = 0; bn < handle->bn; bn++ ) {
        for ( bm = 0; bm < handle->bm; bm++ ) {
          const double dstval = (double)LIBXSMM_VLA_ACCESS(4, dst, nb, mb, bn, bm, handle->mb, handle->bn, handle->bm);
          const double srcval = (double)LIBXSMM_VLA_ACCESS(2, src, nb * handle->bn + bn, mb * handle->bm + bm, handle->m);
          const double local_error = fabs(dstval - srcval);
          if (local_error > max_error) {
            max_error = local_error;
            /*printf("src:%f dst:%f %d %d %d %d\n", srcval, dstval, nb, mb, bn, bm);*/
          }
          src_norm += srcval;
          dst_norm += dstval;
        }
      }
    }
  }
  printf(" max error: %f, sum BLAS: %f, sum LIBXSMM: %f \n", max_error, src_norm, dst_norm );
}

LIBXSMM_INLINE void libxsmm_order (int w_i, int nw_i, int nw_j, int nw_k, int _order, int *i2, int *j2, int *k2)
{
  switch (_order) {
    case jik: /*jik*/
      *j2 = (int)(w_i/(nw_i*nw_k));
      *i2 = (int)((w_i-(*j2)*(nw_i*nw_k))/nw_k);
      *k2 = w_i%nw_k;
      break;
    case ijk: /*ijk*/
      *i2 = (int)(w_i/(nw_j*nw_k));
      *j2 = (int)((w_i-(*i2)*(nw_j*nw_k))/nw_k);
      *k2 = w_i%nw_k;
      break;
    case jki: /*jki*/
      *j2 = (int)(w_i/(nw_k*nw_i));
      *k2 = (int)((w_i-(*j2)*(nw_k*nw_i))/nw_i);
      *i2 = w_i%nw_i;
      break;
    case ikj: /*ikj*/
      *i2 = (int)(w_i/(nw_k*nw_j));
      *k2 = (int)((w_i-(*i2)*(nw_k*nw_j))/nw_j);
      *j2 = w_i%nw_j;
      break;
    case kji: /*kji*/
      *k2 = (int)(w_i/(nw_j*nw_i));
      *j2 = (int)((w_i-(*k2)*(nw_j*nw_i))/nw_i);
      *i2 = w_i%nw_i;
      break;
    case kij: /*kij*/
      *k2 = (int)(w_i/(nw_i*nw_j));
      *i2 = (int)((w_i-(*k2)*(nw_i*nw_j))/nw_j);
      *j2 = w_i%nw_j;
      break;
    default:
      *j2 = (int)(w_i/(nw_i*nw_k));
      *i2 = (int)((w_i-(*j2)*(nw_i*nw_k))/nw_k);
      *k2 = w_i%nw_k;
      break;
  }
}

/******************************************************************************
  Fine grain parallelized version(s) of BGEMM - LIBXSMM_BGEMM
  - Requires block structure layout for A,B matrices
  - Parallelized across all three dims - M, N, K
  - Uses fine-grain on-demand locks for write to C and fast barrier
  - Allows for calling multiple GEMMs, specified by 'count'
 ******************************************************************************/
void libxsmm_bgemm( const int _M,
               const int _N,  
               const int _K,
               const int B_M,
               const int B_N,
               const int B_K,
               const real *Ap,
               const real *Bp,
               real *Cp,
               int tid,
               int nthrds,
               const libxsmm_blkgemm_handle* handle ) {
  _KERNEL l_kernel = handle->_l_kernel;
#ifdef _USE_LIBXSMM_PREFETCH
  _KERNEL l_kernel_pf = handle->_l_kernel_pf;
#endif

  LOCK_T (* locks)[_N/B_N] = (LOCK_T (*)[*])handle->_wlock;
  real l_out[B_N][B_M];
  int ki;
  int kj;
  for (ki = 0; ki < B_N; ki++) {
    LIBXSMM_PRAGMA_SIMD
    for (kj = 0; kj < B_M; kj++) {
      l_out[ki][kj] = 0.f;
    }
  }

  LIBXSMM_VLA_DECL(4, real, A, Ap, _M/B_M, B_K, B_M);
  LIBXSMM_VLA_DECL(4, real, B, Bp, _K/B_K, B_N, B_K);
  LIBXSMM_VLA_DECL(4, real, C, Cp, _M/B_M, B_N, B_M);

  int B_M1 = handle->b_m1;
  int B_N1 = handle->b_n1;
  int B_K1 = handle->b_k1;
  int B_K2 = handle->b_k2;
  int ORDER = handle->_ORDER;

  int M = _M/B_M1;
  int N = _N/B_N1;
  int K = _K/B_K1;

  int nw_i = (M/B_M);
  int nw_j = (N/B_N);
  int nw_k = (K/B_K);
  int nw = nw_i*nw_j*nw_k;

  int _mb, _nb, _kb;
  int _m, _n, _k;
  int w_i, _ki;
  for (_mb=0, _m=0; _mb < B_M1; _mb++, _m+=nw_i) {
    for (_nb=0, _n=0; _nb < B_N1; _nb++, _n+=nw_j) {
      for (_kb=0, _k=0; _kb < B_K1; _kb++, _k+=nw_k) {
        nw_k = (K/B_K)/B_K2;
        int nw = nw_i*nw_j*nw_k;
        int s = (tid*nw)/nthrds;
        int e = ((tid+1)*nw)/nthrds;

        int o_i2, o_j2, o_k2;
        for (w_i = s; w_i < e; w_i++) {
          int i2, j2, k2;
          libxsmm_order(w_i, nw_i, nw_j, nw_k, ORDER, &i2, &j2, &k2);

          i2 = _m + i2;
          j2 = _n + j2;
          k2 = _k + k2;

          if (w_i == s) { 
            o_i2 = i2;
            o_j2 = j2;
            o_k2 = k2;
          } else {
            if ((o_i2 != i2) || (o_j2 != j2)) {
              LOCK_SET(&locks[o_i2][o_j2]);
              for (ki = 0; ki < B_N; ki++) {
                LIBXSMM_PRAGMA_SIMD
                for (kj = 0; kj < B_M; kj++) {
                  LIBXSMM_VLA_ACCESS(4, C, o_j2, o_i2, ki, kj, _M/B_M, B_N, B_M) += l_out[ki][kj];
                }
              }
              LOCK_UNSET(&locks[o_i2][o_j2]);
              for (ki = 0; ki < B_N; ki++) {
                LIBXSMM_PRAGMA_SIMD
                for (kj = 0; kj < B_M; kj++) {
                  l_out[ki][kj] = 0.0f;
                }
              }
              o_i2 = i2;
              o_j2 = j2;
            }
          }
          for (_ki = 0, ki=B_K2*k2; _ki < B_K2 ; _ki++, ki++) {
#ifndef _USE_LIBXSMM_PREFETCH
            l_kernel((const real*)&LIBXSMM_VLA_ACCESS(4, A, ki, i2, 0, 0, _M/B_M, B_K, B_M), 
                     (const real*)&LIBXSMM_VLA_ACCESS(4, B, j2, ki, 0, 0, _K/B_K, B_N, B_K), (real*)l_out);
#else
            /* avoiding prefetch for untouched data */
            if ( k2 < (K/B_K)-2 ) {
#if defined(__AVX2__)
              l_kernel_pf((const real*)&LIBXSMM_VLA_ACCESS(4, A, ki, i2, 0, 0, _M/B_M, B_K, B_M), 
                          (const real*)&LIBXSMM_VLA_ACCESS(4, B, j2, ki, 0, 0, _K/B_K, B_N, B_K), (real*)l_out, 
                          (const real*)&LIBXSMM_VLA_ACCESS(4, B, j2, ki+1, 0, 0, _K/B_K, B_N, B_K), 
                          (const real*)&LIBXSMM_VLA_ACCESS(4, A, ki+1, i2, 0, 0, _M/B_M, B_K, B_M), NULL);
#else
              l_kernel_pf((const real*)&LIBXSMM_VLA_ACCESS(4, A, ki, i2, 0, 0, _M/B_M, B_K, B_M), 
                          (const real*)&LIBXSMM_VLA_ACCESS(4, B, j2, ki, 0, 0, _K/B_K, B_N, B_K), (real*)l_out, 
                          (const real*)&LIBXSMM_VLA_ACCESS(4, A, ki+1, i2, 0, 0, _M/B_M, B_K, B_M), 
                          (const real*)&LIBXSMM_VLA_ACCESS(4, B, j2, ki+1, 0, 0, _K/B_K, B_N, B_K), NULL);
#endif
            } else {
              l_kernel((const real*)&LIBXSMM_VLA_ACCESS(4, A, ki, i2, 0, 0, _M/B_M, B_K, B_M), 
                       (const real*)&LIBXSMM_VLA_ACCESS(4, B, j2, ki, 0, 0, _K/B_K, B_N, B_K), (real*)l_out);
            }
#endif
          }

          if (w_i == e-1) {
            o_i2 = i2;
            o_j2 = j2;
            LOCK_SET(&locks[o_i2][o_j2]);
            for (ki = 0; ki < B_N; ki++) {
              LIBXSMM_PRAGMA_SIMD
              for (kj = 0; kj < B_M; kj++) {
                LIBXSMM_VLA_ACCESS(4, C, o_j2, o_i2, ki, kj, _M/B_M, B_N, B_M) += l_out[ki][kj];
              }
            }
            LOCK_UNSET(&locks[o_i2][o_j2]);
            for (ki = 0; ki < B_N; ki++) {
              LIBXSMM_PRAGMA_SIMD
              for (kj = 0; kj < B_M; kj++) {
                l_out[ki][kj] = 0.0f;
              }
            }
          }
        }
      }
    }
  }
}

/* TODO: To be used with per-thread jitting */
void libxsmm_bgemm_dry_run( const int _M,
                       const int _N,  
                       const int _K,
                       const int B_M,
                       const int B_N,
                       const int B_K,
                       const real *Ap,
                       const real *Bp,
                       real *Cp,
                       int tid,
                       int nthrds,
                       const libxsmm_blkgemm_handle* handle ) {
  _KERNEL l_kernel = handle->_l_kernel;
#ifdef _USE_LIBXSMM_PREFETCH
  _KERNEL l_kernel_pf = handle->_l_kernel_pf;
#endif

  LOCK_T (* locks)[_N/B_N] = (LOCK_T (*)[*])handle->_wlock;
  real l_out[B_N][B_M];
  int ki;
  int kj;
  for (ki = 0; ki < B_N; ki++) {
    LIBXSMM_PRAGMA_SIMD
    for (kj = 0; kj < B_M; kj++) {
      l_out[ki][kj] = 0.f;
    }
  }

  LIBXSMM_VLA_DECL(4, real, A, Ap, _M/B_M, B_K, B_M);
  LIBXSMM_VLA_DECL(4, real, B, Bp, _K/B_K, B_N, B_K);
  LIBXSMM_VLA_DECL(4, real, C, Cp, _M/B_M, B_N, B_M);

  int B_M1 = handle->b_m1;
  int B_N1 = handle->b_n1;
  int B_K1 = handle->b_k1;
  int B_K2 = handle->b_k2;
  int ORDER = handle->_ORDER;

  int M = _M/B_M1;
  int N = _N/B_N1;
  int K = _K/B_K1;

  int nw_i = (M/B_M);
  int nw_j = (N/B_N);
  int nw_k = (K/B_K);
  int nw = nw_i*nw_j*nw_k;

  int _mb, _nb, _kb;
  int _m, _n, _k;
  int w_i, _ki;
  for (_mb=0, _m=0; _mb < B_M1; _mb++, _m+=nw_i) {
    for (_nb=0, _n=0; _nb < B_N1; _nb++, _n+=nw_j) {
      for (_kb=0, _k=0; _kb < B_K1; _kb++, _k+=nw_k) {
        nw_k = (K/B_K)/B_K2;
        int nw = nw_i*nw_j*nw_k;
        int s = (tid*nw)/nthrds;
        int e = ((tid+1)*nw)/nthrds;

        int o_i2, o_j2, o_k2;
        for (w_i = s; w_i < e; w_i++) {
          int i2, j2, k2;
          libxsmm_order(w_i, nw_i, nw_j, nw_k, ORDER, &i2, &j2, &k2);

          i2 = _m + i2;
          j2 = _n + j2;
          k2 = _k + k2;

          if (w_i == s) { 
            o_i2 = i2;
            o_j2 = j2;
            o_k2 = k2;
          } else {
            if ((o_i2 != i2) || (o_j2 != j2)) {
              LOCK_SET(&locks[o_i2][o_j2]);
              for (ki = 0; ki < B_N; ki++) {
                LIBXSMM_PRAGMA_SIMD
                for (kj = 0; kj < B_M; kj++) {
                  LIBXSMM_VLA_ACCESS(4, C, o_j2, o_i2, ki, kj, _M/B_M, B_N, B_M) += l_out[ki][kj];
                }
              }
              LOCK_UNSET(&locks[o_i2][o_j2]);
              for (ki = 0; ki < B_N; ki++) {
                LIBXSMM_PRAGMA_SIMD
                for (kj = 0; kj < B_M; kj++) {
                  l_out[ki][kj] = 0.0f;
                }
              }
              o_i2 = i2;
              o_j2 = j2;
            }
          }
          for (_ki = 0, ki=B_K2*k2; _ki < B_K2 ; _ki++, ki++) {
#ifndef _USE_LIBXSMM_PREFETCH
            /*l_kernel((const real*)&LIBXSMM_VLA_ACCESS(4, A, ki, i2, 0, 0, _M/B_M, B_K, B_M), 
                     (const real*)&LIBXSMM_VLA_ACCESS(4, B, j2, ki, 0, 0, _K/B_K, B_N, B_K), (real*)l_out);*/
#else
            /* avoiding prefetch for untouched data */
            if ( k2 < (K/B_K)-2 ) {
#if defined(__AVX2__)
              /*l_kernel_pf((const real*)&LIBXSMM_VLA_ACCESS(4, A, ki, i2, 0, 0, _M/B_M, B_K, B_M), 
                          (const real*)&LIBXSMM_VLA_ACCESS(4, B, j2, ki, 0, 0, _K/B_K, B_N, B_K), (real*)l_out, 
                          (const real*)&LIBXSMM_VLA_ACCESS(4, B, j2, ki+1, 0, 0, _K/B_K, B_N, B_K), 
                          (const real*)&LIBXSMM_VLA_ACCESS(4, A, ki+1, i2, 0, 0, _M/B_M, B_K, B_M), NULL);*/
#else
              /*Put memory address computation code here*/
              /*l_kernel_pf((const real*)&LIBXSMM_VLA_ACCESS(4, A, ki, i2, 0, 0, _M/B_M, B_K, B_M), 
                          (const real*)&LIBXSMM_VLA_ACCESS(4, B, j2, ki, 0, 0, _K/B_K, B_N, B_K), (real*)l_out, 
                          (const real*)&LIBXSMM_VLA_ACCESS(4, A, ki+1, i2, 0, 0, _M/B_M, B_K, B_M), 
                          (const real*)&LIBXSMM_VLA_ACCESS(4, B, j2, ki+1, 0, 0, _K/B_K, B_N, B_K), NULL);*/
#endif
            } else {
              /*Put memory address computation code here*/
              /*l_kernel((const real*)&LIBXSMM_VLA_ACCESS(4, A, ki, i2, 0, 0, _M/B_M, B_K, B_M), 
                       (const real*)&LIBXSMM_VLA_ACCESS(4, B, j2, ki, 0, 0, _K/B_K, B_N, B_K), (real*)l_out);*/
            }
#endif
          }

          if (w_i == e-1) {
            o_i2 = i2;
            o_j2 = j2;
            LOCK_SET(&locks[o_i2][o_j2]);
            for (ki = 0; ki < B_N; ki++) {
              LIBXSMM_PRAGMA_SIMD
              for (kj = 0; kj < B_M; kj++) {
                LIBXSMM_VLA_ACCESS(4, C, o_j2, o_i2, ki, kj, _M/B_M, B_N, B_M) += l_out[ki][kj];
              }
            }
            LOCK_UNSET(&locks[o_i2][o_j2]);
            for (ki = 0; ki < B_N; ki++) {
              LIBXSMM_PRAGMA_SIMD
              for (kj = 0; kj < B_M; kj++) {
                l_out[ki][kj] = 0.0f;
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_API_DEFINITION  void libxsmm_blksgemm_exec( const libxsmm_blkgemm_handle* handle,
                                           const char transA,
                                           const char transB,
                                           const real* alpha,
                                           const real* a,
                                           const real* b,
                                           const real* beta,
                                           real* c ) {
  /* TODO: take transpose into account */
  LIBXSMM_UNUSED(transA);
  LIBXSMM_UNUSED(transB);

  if ( !(LIBXSMM_FEQ(*beta, (real)1.0) && LIBXSMM_FEQ(*alpha, (real)1.0)) ) {
    printf(" alpha and beta need to be 1.0\n" );
    exit(-1);
  }

  /*The following value can be >1 for RNNs*/
  int count = 1;

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
#if defined(_OPENMP)
    int tid = omp_get_thread_num();
    int nthrds = omp_get_num_threads();
#else
    int tid = 0;
    int nthrds = 1;
#endif
    int i;
    for (i = 0; i < count; i++) {
      libxsmm_bgemm (handle->m, handle->n, handle->k, handle->bm, handle->bn, handle->bk, a, b, c, tid, nthrds, handle);
      BG_BARRIER (handle->bar, tid);
    }
  }
}

LIBXSMM_API_DEFINITION void libxsmm_blkgemm_handle_alloc(libxsmm_blkgemm_handle* handle, const int M, const int MB, const int N, const int NB)
{
  int i;
  /* allocating lock array */
  if (handle->_wlock == NULL)
    handle->_wlock = (LOCK_T*)_mm_malloc((M/MB)*(N/NB)*sizeof(LOCK_T), 64);

  for (i = 0; i < (M/MB)*(N/NB); i++)
    LOCK_INIT(&handle->_wlock[i]);

  BG_BARRIER_INIT(handle->bar);
}
