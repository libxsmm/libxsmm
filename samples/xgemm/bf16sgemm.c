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

size_t sgemm_trup_get_scratch( const int*   m,
                               const int*   n,
                               const int*   k ) {
  size_t memam = 0;

  assert( ((*m)   % 64) == 0 );
  assert( (((*n)-1) % 64) == 0 );
  assert( ((*k)   % 64) == 0 );

  memam += (*m)*(*k)*sizeof(float);
  memam += (*n)*(*k)*sizeof(float);
  memam += ((*m)*(*k)*sizeof(size_t))/(64*64);
  memam += ((*n-1)*(*k)*sizeof(size_t))/(64*64);
  memam += ((*n-1)*(*m)*sizeof(size_t))/(64*64);

  return memam;
}

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
                       void*  scratch ) {
  /* tmpa and tmpb pointers for block storage */
  float* tmpa = (float*)scratch;
  float* tmpb = tmpa+((*k)*(*lda));
  /* (fixed) blocking factors */
  int bm = 64;
  int bn = 64;
  int bk = 64;
  int bn2 = 65;
  int Bm = (*m)/bm;
  int Bn = ((*n)-1)/bn;
  int Bk = (*k)/bk;
  unsigned long long Bkbr = (unsigned long long)Bk;
  int BnB = 8;
  int BmB = Bm;

  /* helper arrays for mixed shaped tile tensors */
  size_t* poa = (size_t*)(tmpb + (*n)*(*ldb));
  size_t* pob = poa + (Bm*Bk);
  size_t* poc = pob + (Bn*Bk);

  /* mult-dim array definitons for readable code */
  LIBXSMM_VLA_DECL( 2, const float, origa,    a,         (*lda) );
  LIBXSMM_VLA_DECL( 2, const float, origb,    b,         (*ldb) );

  /* organization of tile offsets */
  LIBXSMM_VLA_DECL( 2,       size_t, offa, poa, Bk);
  LIBXSMM_VLA_DECL( 2,       size_t, offb, pob, Bk);
  LIBXSMM_VLA_DECL( 2,       size_t, offc, poc, Bm);

  /* jitted libxsmm batch reduce kernel for compute */
  libxsmm_smmfunction_reducebatch_strd fluxcapacitor =  libxsmm_smmdispatch_reducebatch_strd( bm, bn,  bk, bk*bm*sizeof(float), bk*bn*sizeof(float),  &bm, &bk, ldc, NULL, NULL, NULL, NULL);
  libxsmm_smmfunction_reducebatch_strd fluxcapacitor2 = libxsmm_smmdispatch_reducebatch_strd( bm, bn2, bk, bk*bm*sizeof(float), bk*bn2*sizeof(float), &bm, &bk, ldc, NULL, NULL, NULL, NULL);

  /* tmp counters */
  int lm1, ln1, lk1, lm2, ln2, lk2, lno, Bne, lmo, Bme;

  /* some checks */
  assert( ((*m)   % 64) == 0 );
  assert( (((*n)-1) % 64) == 0 );
  assert( ((*k)   % 64) == 0 );
  assert( ((*lda) % 64) == 0 );
  assert( ((*ldb) % 64) == 0 );
  assert( ((*ldc) % 64) == 0 );
  assert( *alpha == -1.0f );
  assert( *beta  ==  1.0f );
  assert( *transa == 'N' );
  assert( *transb == 'N' );

  for ( lm1 = 0; lm1 < Bm; ++lm1 ) {
    for ( lk1 = 0; lk1 < Bk; ++lk1 ) {
      LIBXSMM_VLA_ACCESS( 2, offa, lm1, lk1, Bk ) = ((size_t)bm*bk*lk1) + ((size_t)lm1*bm*(*k));
    }
  }

  for ( ln1 = 0; ln1 < Bn; ++ln1 ) {
    for ( lk1 = 0; lk1 < Bk; ++lk1 ) {
      if ( ln1 == Bn-1 ) {
        LIBXSMM_VLA_ACCESS( 2, offb, ln1, lk1, Bk ) = ((size_t)bn2*bk*lk1) + ((size_t)ln1*bn*(*k));
      } else {
        LIBXSMM_VLA_ACCESS( 2, offb, ln1, lk1, Bk ) = ((size_t)bn*bk*lk1) + ((size_t)ln1*bn*(*k));
      }
    }
  }

  for ( ln1 = 0; ln1 < Bn; ++ln1 ) {
    for ( lm1 = 0; lm1 < Bm; ++lm1 ) {
      LIBXSMM_VLA_ACCESS( 2, offc, ln1, lm1, Bm ) = ((size_t)bm*lm1) + ((size_t)ln1*bn*(*m));
    }
  }

  #pragma omp parallel private(lm1, lm2, ln1, ln2, lk1, lk2, lno, Bne, lmo, Bme)
  {
    for ( lmo = 0; lmo < Bm; lmo += BmB ) {
      Bme = (lmo+BmB > Bm) ? Bm : lmo+BmB;
      for ( lno = 0; lno < Bn; lno += BnB ) {
        Bne = (lno+BnB > Bn) ? Bn : lno+BnB;

        #pragma omp for private(ln1, ln2, lk1, lk2) collapse(2)
        for ( ln1 = lno; ln1 < Bne; ++ln1 ) {
          for ( lk1 = 0; lk1 < Bk; ++lk1 ) {
            int mybn = ( ln1 == Bn-1 ) ? bn2 : bn;
            for ( ln2 = 0; ln2 < mybn; ++ln2 ) {
              float* tmpaddr1 = tmpb + LIBXSMM_VLA_ACCESS( 2, offb, ln1, lk1, Bk ) + ln2*bk;
              const float* tmpaddr2 = &LIBXSMM_VLA_ACCESS( 2, origb, (ln1*bn)+ln2, (lk1*bk), (*ldb) );
              _mm512_storeu_ps( tmpaddr1,    _mm512_loadu_ps( tmpaddr2 ) );
              _mm512_storeu_ps( tmpaddr1+16, _mm512_loadu_ps( tmpaddr2+16 ) );
              _mm512_storeu_ps( tmpaddr1+32, _mm512_loadu_ps( tmpaddr2+32 ) );
              _mm512_storeu_ps( tmpaddr1+48, _mm512_loadu_ps( tmpaddr2+48 ) );
            }
          }
        }

        #pragma omp for private(lm1, ln1, lk1, lk2) collapse(2)
        for ( lm1 = lmo; lm1 < Bme; ++lm1 ) {
          /* we prepare a bm*K tile of A in L1/L2 cache */
          for ( lk1 = 0; lk1 < Bk; ++lk1 ) {
            __m512 vmone = _mm512_set1_ps( -1.0f );
            for ( lk2 = 0; lk2 < bk; ++lk2 ) {
              float* tmpaddr1 = tmpa + LIBXSMM_VLA_ACCESS( 2, offa, lm1, lk1, Bk ) + lk2*bm;
              const float* tmpaddr2 = &LIBXSMM_VLA_ACCESS( 2, origa, (lk1*bk)+lk2, (lm1*bm), (*lda) );
              _mm512_storeu_ps( tmpaddr1,    _mm512_mul_ps( vmone, _mm512_loadu_ps( tmpaddr2 ) ) );
              _mm512_storeu_ps( tmpaddr1+16, _mm512_mul_ps( vmone, _mm512_loadu_ps( tmpaddr2+16 ) ) );
              _mm512_storeu_ps( tmpaddr1+32, _mm512_mul_ps( vmone, _mm512_loadu_ps( tmpaddr2+32 ) ) );
              _mm512_storeu_ps( tmpaddr1+48, _mm512_mul_ps( vmone, _mm512_loadu_ps( tmpaddr2+48 ) ) );
            }
          }
        }

        #pragma omp for private(lm1, ln1) collapse(2)
        for ( lm1 = lmo; lm1 < Bme; ++lm1 ) {
           for ( ln1 = lno; ln1 < Bne; ++ln1 ) {
             if ( ln1 == (Bn - 1) ) {
               fluxcapacitor2( tmpa + LIBXSMM_VLA_ACCESS( 2, offa, lm1,   0,     Bk ),
                               tmpb + LIBXSMM_VLA_ACCESS( 2, offb, ln1,   0,     Bk ),
                               c    + LIBXSMM_VLA_ACCESS( 2, offc, ln1, lm1, (*ldc) ),
                               &Bkbr );

             } else {
               fluxcapacitor( tmpa + LIBXSMM_VLA_ACCESS( 2, offa, lm1,   0,     Bk ),
                              tmpb + LIBXSMM_VLA_ACCESS( 2, offb, ln1,   0,     Bk ),
                              c    + LIBXSMM_VLA_ACCESS( 2, offc, ln1, lm1, (*ldc) ),
                              &Bkbr );
            }
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

  if ( argc != 4 ) {
    printf("wrong arguments, required: ./%s N K iters\n", argv[0]);
    return EXIT_FAILURE;
  }

  M = atoi(argv[1]);
  N = M+1;
  K = atoi(argv[2]);
  iters = atoi(argv[3]);
  LDA = M;
  LDB = K;
  LDC = M;

  A        = (float*)libxsmm_aligned_malloc( (size_t)M * (size_t)K * sizeof(float),             2097152 );
  B        = (float*)libxsmm_aligned_malloc( (size_t)N * (size_t)K * sizeof(float),             2097152 );
  C        = (float*)libxsmm_aligned_malloc( (size_t)M * (size_t)N * sizeof(float),             2097152 );
  Cgold    = (float*)libxsmm_aligned_malloc( (size_t)M * (size_t)N * sizeof(float),             2097152 );
  scratch  = (void*)libxsmm_aligned_malloc( sgemm_trup_get_scratch( &M, &N, &K ) * sizeof(char), 2097152 );
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

