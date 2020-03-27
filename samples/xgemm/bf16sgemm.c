/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <mkl.h>

void sgemm_trup( const char*  transa,
                     const char*  transb,
                     const int*   m,
                     const int*   n,
                     const int*   k,
                     const float* alpha,
                     const float* a,
                     const int*   lda,
                     const float* b,
                     const int*   ldb,
                     const float* beta,
                     float* c,
                     const int*   ldc,
                     float*       scratch ) {
  /* tmpa and tmpb pointers for block storage */
  float* tmpa = scratch;
  float* tmpb = scratch+((*k)*(*lda));

  /* (fixed) blocking factors */
  int bm = 64;
  int bn = 64;
  int bk = 64;
  int Bm = (*m)/bm;
  int Bn = (*n)/bn;
  int Bk = (*k)/bk;
  unsigned long long Bkbr = (unsigned long long)Bk;
  int BnB = 8;
  int BmB = Bm;
  /* mult-dim array definitons for readable code */
  LIBXSMM_VLA_DECL( 2, const float, origa,    a,         (*lda) );
  LIBXSMM_VLA_DECL( 2, const float, origb,    b,         (*ldb) );
  LIBXSMM_VLA_DECL( 2,       float, origc,    c,         (*ldc) );
  LIBXSMM_VLA_DECL( 4,       float,  blka, tmpa, Bk, bk,    bm );
  LIBXSMM_VLA_DECL( 4,       float,  blkb, tmpb, Bk, bn,    bk );

  /* jitted libxsmm batch reduce kernel for compute */
  libxsmm_smmfunction_reducebatch_strd fluxcapacitor = libxsmm_smmdispatch_reducebatch_strd( bm, bn, bk, bk*bm*sizeof(float), bk*bn*sizeof(float), &bm, &bk, ldc, NULL, NULL, NULL, NULL);

  /* tmp counters */
  int lm1, ln1, lk1, lm2, ln2, lk2, lno, Bne, lmo, Bme;

  /* some checks */
  assert( ((*m)   % 64) == 0 );
  assert( ((*n)   % 64) == 0 );
  assert( ((*k)   % 64) == 0 );
  assert( ((*lda) % 64) == 0 );
  assert( ((*ldb) % 64) == 0 );
  assert( ((*ldc) % 64) == 0 );
  assert( *alpha == -1.0f );
  assert( *beta  ==  1.0f );
  assert( *transa == 'N' );
  assert( *transb == 'N' );

  #pragma omp parallel private(lm1, lm2, ln1, ln2, lk1, lk2, lno, Bne, lmo, Bme)
  {
    for ( lmo = 0; lmo < Bm; lmo += BmB ) {
      Bme = (lmo+BmB > Bm) ? Bm : lmo+BmB;
      for ( lno = 0; lno < Bn; lno += BnB ) {
        Bne = (lno+BnB > Bn) ? Bn : lno+BnB;

        #pragma omp for private(ln1, ln2, lk1, lk2) collapse(2)
        for ( ln1 = lno; ln1 < Bne; ++ln1 ) {
          for ( lk1 = 0; lk1 < Bk; ++lk1 ) {
            for ( ln2 = 0; ln2 < bn; ++ln2 ) {
#if 1
              float* tmpaddr1 = &LIBXSMM_VLA_ACCESS( 4, blkb, ln1, lk1, ln2, 0, Bk, bn, bk );
              const float* tmpaddr2 = &LIBXSMM_VLA_ACCESS( 2, origb, (ln1*bn)+ln2, (lk1*bk), (*ldb) );
              _mm512_storeu_ps( tmpaddr1,    _mm512_loadu_ps( tmpaddr2 ) );
              _mm512_storeu_ps( tmpaddr1+16, _mm512_loadu_ps( tmpaddr2+16 ) );
              _mm512_storeu_ps( tmpaddr1+32, _mm512_loadu_ps( tmpaddr2+32 ) );
              _mm512_storeu_ps( tmpaddr1+48, _mm512_loadu_ps( tmpaddr2+48 ) );
#else
              for ( lk2 = 0; lk2 < bk; ++lk2 ) {
                LIBXSMM_VLA_ACCESS( 4, blkb, ln1, lk1, ln2, lk2, Bk, bn, bk ) =
                  LIBXSMM_VLA_ACCESS( 2, origb, (ln1*bn)+ln2, (lk1*bk)+lk2, (*ldb) );
              }
#endif
            }
          }
        }

        #pragma omp for private(lm1, ln1, lk1, lk2) collapse(2)
        for ( lm1 = lmo; lm1 < Bme; ++lm1 ) {
          /* we prepare a bm*K tile of A in L1/L2 cache */
          for ( lk1 = 0; lk1 < Bk; ++lk1 ) {
            __m512 vmone = _mm512_set1_ps( -1.0f );
            for ( lk2 = 0; lk2 < bk; ++lk2 ) {
#if 1
              float* tmpaddr1 = &LIBXSMM_VLA_ACCESS( 4, blka, lm1, lk1, lk2, 0, Bk, bk, bm );
              const float* tmpaddr2 = &LIBXSMM_VLA_ACCESS( 2, origa, (lk1*bk)+lk2, (lm1*bm), (*lda) );
              _mm512_storeu_ps( tmpaddr1,    _mm512_mul_ps( vmone, _mm512_loadu_ps( tmpaddr2 ) ) );
              _mm512_storeu_ps( tmpaddr1+16, _mm512_mul_ps( vmone, _mm512_loadu_ps( tmpaddr2+16 ) ) );
              _mm512_storeu_ps( tmpaddr1+32, _mm512_mul_ps( vmone, _mm512_loadu_ps( tmpaddr2+32 ) ) );
              _mm512_storeu_ps( tmpaddr1+48, _mm512_mul_ps( vmone, _mm512_loadu_ps( tmpaddr2+48 ) ) );
#else
              for ( lm2 = 0; lm2 < bm;  ) {
                LIBXSMM_VLA_ACCESS( 4, blka, lm1, lk1, lk2, lm2, Bk, bk, bm ) =
                  (*alpha)*LIBXSMM_VLA_ACCESS( 2, origa, (lk1*bk)+lk2, (lm1*bm)+lm2, (*lda) );
              }
#endif
            }
          }
        }

        #pragma omp for private(lm1, ln1) collapse(2)
        for ( lm1 = lmo; lm1 < Bme; ++lm1 ) {
           for ( ln1 = lno; ln1 < Bne; ++ln1 ) {
            fluxcapacitor( &LIBXSMM_VLA_ACCESS( 4,  blka, lm1, 0, 0, 0, Bk, bk, bm ),
                           &LIBXSMM_VLA_ACCESS( 4,  blkb, ln1, 0, 0, 0, Bk, bn, bk ),
                           &LIBXSMM_VLA_ACCESS( 2, origc, (ln1*bn), (lm1*bm), (*ldc) ),
                           &Bkbr );
          }
        }
      }
    }
  }
}

int main(int argc, char* argv []) {
  int M, N, K, LDA, LDB, LDC, iters;
  float alpha = -1.0f, beta = 1.0f;
  char transa = 'N', transb = 'N';
  float *A, *B, *C, *Cgold, *scratch;
  size_t i;
  double max_error;
  libxsmm_timer_tickint l_start;
  double l_runtime;
  double l_gflops;

  if ( argc != 5 ) {
    printf("wrong arguments, required: ./%s sM N K iters\n", argv[0]);
    return EXIT_FAILURE;
  }

  M = atoi(argv[1]);
  N = atoi(argv[2]);
  K = atoi(argv[3]);
  iters = atoi(argv[4]);
  LDA = M;
  LDB = K;
  LDC = M;

  A        = (float*)libxsmm_aligned_malloc( (size_t)M * (size_t)K * sizeof(float),             2097152 );
  B        = (float*)libxsmm_aligned_malloc( (size_t)N * (size_t)K * sizeof(float),             2097152 );
  C        = (float*)libxsmm_aligned_malloc( (size_t)M * (size_t)N * sizeof(float),             2097152 );
  Cgold    = (float*)libxsmm_aligned_malloc( (size_t)M * (size_t)N * sizeof(float),             2097152 );
  scratch  = (float*)libxsmm_aligned_malloc( (size_t)K * ((size_t)M+(size_t)N) * sizeof(float), 2097152 );
  l_gflops = ((double)M*(double)N*(double)K*2.0)/(double)1e9;

  /* init data */
  for (i = 0; i < (size_t)M*(size_t)K; i++) {
    A[i] = (float)libxsmm_rng_f64();
  }
  for (i = 0; i < (size_t)N*(size_t)K; i++) {
    B[i] = (float)libxsmm_rng_f64();
  }
  for (i = 0; i < (size_t)N*(size_t)M; i++) {
    Cgold[i] = (float)libxsmm_rng_f64();
  }
  for (i = 0; i < (size_t)N*(size_t)M; i++) {
    C[i] = Cgold[i];
  }

  /* call MKL and custom trup for correctness */
  sgemm( &transa, &transb, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, Cgold, &LDC );
  sgemm_trup( &transa, &transb, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, C, &LDC, scratch );

  /* check max error */
  max_error = 0.0;
  for (i = 0; i < (size_t)N*(size_t)M; i++) {
    if ( fabs( Cgold[i] - C[i] ) > max_error ) {
      max_error = fabs( Cgold[i] - C[i] );
    }
  }

  /* Print total max error */
  printf("\n\n Total Max Error %f\n\n", max_error );

  /* benchmark */
  l_start = libxsmm_timer_tick();
  for( i = 0; i < iters; ++i ) {
    sgemm( &transa, &transb, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, Cgold, &LDC );
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  l_runtime = l_runtime / (double)iters;
  printf(" Performance SGEMM:  %f GFLOPS  %f s \n", l_gflops/l_runtime, l_runtime );

  l_start = libxsmm_timer_tick();
  for( i = 0; i < iters; ++i ) {
    sgemm_trup( &transa, &transb, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, Cgold, &LDC, scratch );
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  l_runtime = l_runtime / (double)iters;
  printf(" Performance custom: %f GFLOPS  %f s \n\n", l_gflops/l_runtime, l_runtime );

  libxsmm_free( A );
  libxsmm_free( B );
  libxsmm_free( C );
  libxsmm_free( Cgold );
  libxsmm_free( scratch );

  if ( max_error >= 0.00005 ) {
    return EXIT_FAILURE;
  } else {
    return EXIT_SUCCESS;
  }
}

