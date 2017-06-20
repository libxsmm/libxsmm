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
#include "block_xgemm.h"

#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl_service.h>
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
/* note: this does not reproduce 48-bit RNG quality */
# define drand48() ((double)rand() / RAND_MAX)
# define srand48 srand
#endif

typedef float real;

/** Function prototype for SGEMM; this way any kind of LAPACK/BLAS library is sufficient at link-time. */
void LIBXSMM_FSYMBOL(sgemm)(const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const real*, const real*, const libxsmm_blasint*, const real*, const libxsmm_blasint*,
  const real*, real*, const libxsmm_blasint*);

typedef struct libxsmm_blkgemm_handle {
  int m;
  int n;
  int k;
  int bm;
  int bn;
  int bk;
  int mb;
  int nb;
  int kb;
  /* The following have been added by KB */
  int b_m1, b_n1, b_k1, b_k2;
  int _ORDER;
  int C_pre_init;
  LOCK_T* _wlock;
  _KERNEL _l_kernel;
#ifdef _USE_LIBXSMM_PREFETCH
  _KERNEL _l_kernel_pf;
#endif
  libxsmm_barrier* bar;
} libxsmm_blkgemm_handle;

LIBXSMM_INLINE void libxsmm_blksgemm_init_a( libxsmm_blkgemm_handle* handle,
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

LIBXSMM_INLINE void libxsmm_blksgemm_init_b( libxsmm_blkgemm_handle* handle,
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

LIBXSMM_INLINE void libxsmm_blksgemm_init_c( libxsmm_blkgemm_handle* handle,
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

LIBXSMM_INLINE void libxsmm_blksgemm_check_c( libxsmm_blkgemm_handle* handle,
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

void order (int w_i, int nw_i, int nw_j, int nw_k, int _order, int *i2, int *j2, int *k2)
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
  Fine grain parallelized version(s) of BGEMM - BGEMM2_7
  - Requires block structure layout for A,B matrices
  - Parallelized across all three dims - M, N, K
  - Uses fine-grain on-demand locks for write to C and fast barrier
  - Allows for calling multiple GEMMs, specified by 'count'
 ******************************************************************************/
void bgemm2_7( const int _M,
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
          order(w_i, nw_i, nw_j, nw_k, ORDER, &i2, &j2, &k2);

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
                  C[o_j2][o_i2][ki][kj] += l_out[ki][kj];
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
void bgemm2_7_dry_run( const int _M,
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
          order(w_i, nw_i, nw_j, nw_k, ORDER, &i2, &j2, &k2);

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
                  C[o_j2][o_i2][ki][kj] += l_out[ki][kj];
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

LIBXSMM_INLINE void libxsmm_blksgemm_exec( const libxsmm_blkgemm_handle* handle,
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
      bgemm2_7 (handle->m, handle->n, handle->k, handle->bm, handle->bn, handle->bk, a, b, c, tid, nthrds, handle);
      BG_BARRIER (handle->bar, tid);
    }
  }
}

void BGEMM_HANDLE_alloc(libxsmm_blkgemm_handle* handle, const int M, const int MB, const int N, const int NB)
{
  int i;
  /* allocating lock array */
  if (handle->_wlock == NULL)
    handle->_wlock = (LOCK_T*)_mm_malloc((M/MB)*(N/NB)*sizeof(LOCK_T), 64);

  for (i = 0; i < (M/MB)*(N/NB); i++)
    LOCK_INIT(&handle->_wlock[i]);

  BG_BARRIER_INIT(handle->bar);
}

int main(int argc, char* argv []) {
  real *a, *b, *c, *a_gold, *b_gold, *c_gold;
  int M, N, K, LDA, LDB, LDC;
  real alpha, beta;
  unsigned long long start, end;
  double total, flops;
  int i, reps;
  size_t l;
  char trans;
  libxsmm_blkgemm_handle handle;

  /* init */
/*
  a = 0;
  b = 0;
  c = 0;
  a_gold = 0;
  b_gold = 0;
  c_gold = 0;
*/
  M = 0;
  N = 0;
  K = 0;
  LDA = 0;
  LDB = 0;
  LDC = 0;
  alpha = (real)1.0;
  beta = (real)1.0;
  start = 0;
  end = 0;
  total = 0.0;
  flops = 0.0;
  i = 0;
  l = 0;
  reps = 0;
  trans = 'N';

  /* check command line */
  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("\nUsage: ./block_gemm [M] [N] [K] [bm] [bn] [bk] [b_m1] [b_n1] [b_k1] [b_k2] [reps]\n\n");
    return 0;
  }

  /* setup defaults */
  handle.m = 2048;
  handle.n = 2048;
  handle.k = 2048;
  handle.bm = 32;
  handle.bn = 32;
  handle.bk = 32;
  handle.b_m1 = 1;
  handle.b_n1 = 1;
  handle.b_k1 = 1;
  handle.b_k2 = 1;
  reps = 100;

  /* reading new values from cli */
  i = 1;
  if (argc > i) handle.m      = atoi(argv[i++]);
  if (argc > i) handle.n      = atoi(argv[i++]);
  if (argc > i) handle.k      = atoi(argv[i++]);
  if (argc > i) handle.bm     = atoi(argv[i++]);
  if (argc > i) handle.bn     = atoi(argv[i++]);
  if (argc > i) handle.bk     = atoi(argv[i++]);
  if (argc > i) handle.b_m1   = atoi(argv[i++]);
  if (argc > i) handle.b_n1   = atoi(argv[i++]);
  if (argc > i) handle.b_k1   = atoi(argv[i++]);
  if (argc > i) handle.b_k2   = atoi(argv[i++]);
  if (argc > i) reps          = atoi(argv[i++]);
  M = handle.m;
  LDA = handle.m;
  N = handle.n;
  LDB = handle.k;
  K = handle.k;
  LDC = handle.m;
  alpha = (real)1.0;
  beta = (real)1.0;
  flops = (double)M * (double)N * (double)K * (double)2.0 * (double)reps;

  /* check for valid blocking and JIT-kernel */
  if ( handle.m % handle.bm != 0 ) {
    printf( " M needs to be a multiple of bm... exiting!\n" );
    return -1;
  }
  if ( handle.n % handle.bn != 0 ) {
    printf( " N needs to be a multiple of bn... exiting!\n" );
    return -2;
  }
  if ( handle.k % handle.bk != 0 ) {
    printf( " K needs to be a multiple of bk... exiting!\n" );
    return -3;
  }
  if ( handle.m % handle.b_m1 != 0 ) {
    printf( " M needs to be a multiple of b_m1... exiting!\n" );
    return -4;
  }
  if ( handle.n % handle.b_n1 != 0 ) {
    printf( " N needs to be a multiple of b_n1... exiting!\n" );
    return -5;
  }
  if ( handle.k % handle.b_k1 != 0 ) {
    printf( " K needs to be a multiple of b_k1... exiting!\n" );
    return -6;
  }
  if ( handle.m/handle.b_m1 % handle.bm != 0 ) {
    printf( " m/b_m1 needs to be a multiple of bm... exiting!\n" );
    return -7;
  }
  if ( handle.n/handle.b_n1 % handle.bn != 0 ) {
    printf( " n/b_n1 needs to be a multiple of bn... exiting!\n" );
    return -8;
  }
  if ( handle.k/handle.b_k1/handle.b_k2 % handle.bk != 0 ) {
    printf( " k/b_k1/b_k2 needs to be a multiple of bk... exiting!\n" );
    return -9;
  }
  handle.mb = handle.m / handle.bm;
  handle.nb = handle.n / handle.bn;
  handle.kb = handle.k / handle.bk;

  /* init random seed and print some info */
  printf(" running with: M=%i, N=%i, K=%i, bm=%i, bn=%i, bk=%i, b_m1=%i, b_n1=%i, b_k1=%i, b_k2=%i, reps=%i\n", M, N, K, handle.bm, handle.bn, handle.bk, handle.b_m1, handle.b_n1, handle.b_k1, handle.b_k2, reps );
  printf(" working set size: A: %f, B: %f, C: %f, Total: %f in MiB\n", ((double)(M*K*sizeof(real)))/(1024.0*1024.0),
                                                                       ((double)(K*N*sizeof(real)))/(1024.0*1024.0),
                                                                       ((double)(M*N*sizeof(real)))/(1024.0*1024.0),
                                                                       ((double)(M*N*sizeof(real)+M*K*sizeof(real)+N*K*sizeof(real)))/(1024.0*1024.0) );
  srand48(1);

#if defined(MKL_ENABLE_AVX512) /* AVX-512 instruction support */
  mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif

  /* allocate data */
  a      = (real*)libxsmm_aligned_malloc( M*K*sizeof(real), 2097152 );
  b      = (real*)libxsmm_aligned_malloc( K*N*sizeof(real), 2097152 );
  c      = (real*)libxsmm_aligned_malloc( M*N*sizeof(real), 2097152 );
  a_gold = (real*)libxsmm_aligned_malloc( M*K*sizeof(real), 2097152 );
  b_gold = (real*)libxsmm_aligned_malloc( K*N*sizeof(real), 2097152 );
  c_gold = (real*)libxsmm_aligned_malloc( M*N*sizeof(real), 2097152 );

  /* init data */
  for ( l = 0; l < (size_t)M * (size_t)K; l++ ) {
    a_gold[l] = (real)drand48();
  }
  for ( l = 0; l < (size_t)K * (size_t)N; l++ ) {
    b_gold[l] = (real)drand48();
  }
  for ( l = 0; l < (size_t)M * (size_t)N; l++ ) {
    c_gold[l] = (real)0.0;
  }
  for ( l = 0; l < (size_t)M * (size_t)N; l++ ) {
    c[l]      = (real)0.0;
  }
  libxsmm_blksgemm_init_a( &handle, a, a_gold );
  libxsmm_blksgemm_init_b( &handle, b, b_gold );

  handle._ORDER = 0;
  handle.C_pre_init = 0;
  handle._wlock = NULL; 
  handle.bar = NULL;
  
  BGEMM_HANDLE_alloc(&handle, handle.m, handle.bm, handle.n, handle.bn);

  handle._l_kernel =_KERNEL_JIT(handle.bm, handle.bn, handle.bk, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
#ifdef _USE_LIBXSMM_PREFETCH
  libxsmm_prefetch_type l_prefetch_op = LIBXSMM_PREFETCH_AL2BL2_VIA_C;
  handle._l_kernel_pf =_KERNEL_JIT(handle.bm, handle.bn, handle.bn, NULL, NULL, NULL, NULL, NULL, NULL, &l_prefetch_op );
#endif

  /* check result */
  /* run LIBXSEMM, trans, alpha and beta are ignored */
  libxsmm_blksgemm_exec( &handle, trans, trans, &alpha, a, b, &beta, c );
  /* run BLAS */
  LIBXSMM_FSYMBOL(sgemm)(&trans, &trans, &M, &N, &K, &alpha, a_gold, &LDA, b_gold, &LDB, &beta, c_gold, &LDC);
  /* compare result */
  libxsmm_blksgemm_check_c( &handle, c, c_gold );

  /* time BLAS */
  start = libxsmm_timer_tick();
  for ( i = 0; i < reps; i++ ) {
    LIBXSMM_FSYMBOL(sgemm)(&trans, &trans, &M, &N, &K, &alpha, a_gold, &LDA, b_gold, &LDB, &beta, c_gold, &LDC);
  }
  end = libxsmm_timer_tick();
  total = libxsmm_timer_duration(start, end);
  printf("GFLOPS  (BLAS)    = %.5g\n", (flops*1e-9)/total);

  /* time libxsmm */
  start = libxsmm_timer_tick();
  for ( i = 0; i < reps; i++ ) {
    libxsmm_blksgemm_exec( &handle, trans, trans, &alpha, a, b, &beta, c );
  }
  end = libxsmm_timer_tick();
  total = libxsmm_timer_duration(start, end);
  printf("GFLOPS  (LIBXSMM) = %.5g\n", (flops*1e-9)/total);

  /* free data */
  libxsmm_free( a );
  libxsmm_free( b );
  libxsmm_free( c );
  libxsmm_free( a_gold );
  libxsmm_free( b_gold );
  libxsmm_free( c_gold );

  return 0;
}

