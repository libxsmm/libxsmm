/******************************************************************************
** Copyright (c) 2014-2017, Intel Corporation                                **
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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include <immintrin.h>

#ifdef __USE_MKL
#define MKL_DIRECT_CALL_SEQ
#include <mkl.h>
#endif

#ifndef __INTEL_COMPILER
/* based on: http://www.mcs.anl.gov/~kazutomo/rdtsc.html */
static __inline__ unsigned long long _rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}
#endif

/*#define STREAM_A_B*/
#ifdef STREAM_A_B
#define STREAM_A_B_SIZE 1000
#define STREAM_A_B_PREFETCH
#endif

#ifdef USE_ASM_DIRECT
void dense_test_mul(const REALTYPE* a, const REALTYPE* b, REALTYPE* c);
#else
#include GEMM_HEADER
#endif

#ifndef MY_M
#define MY_M 20
#endif

#ifndef MY_N
#define MY_N 9
#endif

#ifndef MY_K
#define MY_K MY_N
#endif

#ifndef MY_LDA
#define MY_LDA MY_M
#endif

#ifndef MY_LDB
#define MY_LDB MY_K
#endif

#ifndef MY_LDC
#define MY_LDC MY_M
#endif

#define REPS 100000
/*#define REPS 1*/

static double sec(struct timeval start, struct timeval end) {
  return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)))) / 1.0e6;
}

void run_test(void) {
  /* allocate */
#ifdef STREAM_A_B
  REALTYPE* l_a = (REALTYPE*)_mm_malloc(MY_LDA * MY_K * sizeof(REALTYPE) * STREAM_A_B_SIZE, 64);
  REALTYPE* l_b = (REALTYPE*)_mm_malloc(MY_LDB * MY_N * sizeof(REALTYPE) * STREAM_A_B_SIZE, 64);
  unsigned int l_s;
#else
  REALTYPE* l_a = (REALTYPE*)_mm_malloc(MY_LDA * MY_K * sizeof(REALTYPE), 64);
  REALTYPE* l_b = (REALTYPE*)_mm_malloc(MY_LDB * MY_N * sizeof(REALTYPE), 64);
#endif
  REALTYPE* l_c = (REALTYPE*)_mm_malloc(MY_LDC * MY_N * sizeof(REALTYPE), 64);
  REALTYPE* l_c_gold = (REALTYPE*)_mm_malloc(MY_LDC * MY_N * sizeof(REALTYPE), 64);
  REALTYPE l_max_error = 0.0;

  unsigned int l_i;
  unsigned int l_j;
  unsigned int l_t;
  unsigned int l_m;
  unsigned int l_n;
  unsigned int l_k;

  struct timeval l_start, l_end;
  double l_total;

#ifdef STREAM_A_B
  for ( l_s = 0; l_s < STREAM_A_B_SIZE; l_s++ ) {
    REALTYPE* l_p_a = l_a + (l_s * MY_K * MY_LDA);
#else
    REALTYPE* l_p_a = l_a;
#endif
    /* touch A */
    for ( l_i = 0; l_i < MY_LDA; l_i++) {
      for ( l_j = 0; l_j < MY_K; l_j++) {
#if REPS==1
        l_p_a[(l_j * MY_LDA) + l_i] = (REALTYPE)drand48();
#else
        l_p_a[(l_j * MY_LDA) + l_i] = (REALTYPE)(l_i + (l_j * MY_M));
#endif
      }
    }
#ifdef STREAM_A_B
  }
#endif

#ifdef STREAM_A_B
  for ( l_s = 0; l_s < STREAM_A_B_SIZE; l_s++ ) {
    REALTYPE* l_p_b = l_b + (l_s * MY_N * MY_LDB);
#else
    {
      REALTYPE* l_p_b = l_b;
#endif
      /* touch B */
      for ( l_i = 0; l_i < MY_LDB; l_i++ ) {
        for ( l_j = 0; l_j < MY_N; l_j++ ) {
#if REPS==1
          l_p_b[(l_j * MY_LDB) + l_i] = (REALTYPE)drand48();
#else
          l_p_b[(l_j * MY_LDB) + l_i] = (REALTYPE)(l_i + (l_j * MY_K));
#endif
        }
      }
    }
#ifdef STREAM_A_B
  }
#endif

  /* touch C */
  for ( l_i = 0; l_i < MY_LDC; l_i++) {
    for ( l_j = 0; l_j < MY_N; l_j++) {
      l_c[(l_j * MY_LDC) + l_i] = (REALTYPE)0.0;
      l_c_gold[(l_j * MY_LDC) + l_i] = (REALTYPE)0.0;
    }
  }

#ifdef __USE_MKL
  {
    char l_trans = 'N';
    int l_M = MY_M;
    int l_N = MY_N;
    int l_K = MY_K;
    int l_lda = MY_LDA;
    int l_ldb = MY_LDB;
    int l_ldc = MY_LDC;
    if (sizeof(REALTYPE) == sizeof(double)) {
      double l_one = 1.0;
      dgemm(&l_trans, &l_trans, &l_M, &l_N, &l_K, &l_one, (double*)l_a, &l_lda, (double*)l_b, &l_ldb, &l_one, (double*)l_c_gold, &l_ldc);
    } else {
      float l_one = 1.0f;
      sgemm(&l_trans, &l_trans, &l_M, &l_N, &l_K, &l_one, (float*)l_a, &l_lda, (float*)l_b, &l_ldb, &l_one, (float*)l_c_gold, &l_ldc);
    }
  }

  /* touch C */
  for ( l_i = 0; l_i < MY_LDC; l_i++) {
    for ( l_j = 0; l_j < MY_N; l_j++) {
      l_c[(l_j * MY_LDC) + l_i] = (REALTYPE)0.0;
      l_c_gold[(l_j * MY_LDC) + l_i] = (REALTYPE)0.0;
    }
  }
#endif

  /* C routine */
  gettimeofday(&l_start, NULL);
#ifndef __USE_MKL
  #pragma nounroll_and_jam
  for ( l_t = 0; l_t < REPS; l_t++ ) {
#ifdef STREAM_A_B
    REALTYPE* l_p_a = l_a - (MY_K * MY_LDA);
    REALTYPE* l_p_b = l_b - (MY_N * MY_LDB);
    for ( l_s = 0; l_s < STREAM_A_B_SIZE; l_s++ ) {
      l_p_a += (MY_K * MY_LDA);
      l_p_b += (MY_N * MY_LDB);
#else
      REALTYPE* l_p_a = l_a;
      REALTYPE* l_p_b = l_b;
#endif
      for ( l_n = 0; l_n < MY_N; l_n++ ) {
        for ( l_k = 0; l_k < MY_K; l_k++ ) {
          #pragma vector always
          for ( l_m = 0; l_m < MY_M; l_m++ ) {
            l_c_gold[(l_n * MY_LDC) + l_m] += l_p_a[(l_k * MY_LDA) + l_m] * l_p_b[(l_n * MY_LDB) + l_k];
          }
        }
      }
#ifdef STREAM_A_B
    }
#endif
  }
#else
  char l_trans = 'N';
  int l_M = MY_M;
  int l_N = MY_N;
  int l_K = MY_K;
  int l_lda = MY_LDA;
  int l_ldb = MY_LDB;
  int l_ldc = MY_LDC;
  if (sizeof(REALTYPE) == sizeof(double)) {
    double l_one = 1.0;
    for ( l_t = 0; l_t < REPS; l_t++ ) {
#ifdef STREAM_A_B
      REALTYPE* l_p_a = l_a - (MY_K * MY_LDA);
      REALTYPE* l_p_b = l_b - (MY_N * MY_LDB);
      for ( l_s = 0; l_s < STREAM_A_B_SIZE; l_s++ ) {
        l_p_a += (MY_K * MY_LDA);
        l_p_b += (MY_N * MY_LDB);
#else
        REALTYPE* l_p_a = l_a;
        REALTYPE* l_p_b = l_b;
#endif
        dgemm(&l_trans, &l_trans, &l_M, &l_N, &l_K, &l_one, (double*)l_p_a, &l_lda, (double*)l_p_b, &l_ldb, &l_one, (double*)l_c_gold, &l_ldc);
#ifdef STREAM_A_B
      }
#endif
    }
  } else {
    float l_one = 1.0f;
    for ( l_t = 0; l_t < REPS; l_t++ ) {
#ifdef STREAM_A_B
      REALTYPE* l_p_a = l_a - (MY_K * MY_LDA);
      REALTYPE* l_p_b = l_b - (MY_N * MY_LDB);
      for ( l_s = 0; l_s < STREAM_A_B_SIZE; l_s++ ) {
        l_p_a += (MY_K * MY_LDA);
        l_p_b += (MY_N * MY_LDB);
#else
        REALTYPE* l_p_a = l_a;
        REALTYPE* l_p_b = l_b;
#endif
        sgemm(&l_trans, &l_trans, &l_M, &l_N, &l_K, &l_one, (float*)l_p_a, &l_lda, (float*)l_p_b, &l_ldb, &l_one, (float*)l_c_gold, &l_ldc);
#ifdef STREAM_A_B
      }
#endif
    }
  }
#endif
  gettimeofday(&l_end, NULL);

  l_total = sec(l_start, l_end);
#ifndef __USE_MKL
  printf("%fs for C\n", l_total);
#ifdef STREAM_A_B
  printf("%f GFLOPS for C\n", ((double)((double)REPS * (double)MY_M * (double)MY_N * (double)MY_K) * 2.0 * ((double)STREAM_A_B_SIZE)) / (l_total * 1.0e9));
#else
  printf("%f GFLOPS for C\n", ((double)((double)REPS * (double)MY_M * (double)MY_N * (double)MY_K) * 2.0) / (l_total * 1.0e9));
#endif
#else
  printf("%fs for MKL\n", l_total);
#ifdef STREAM_A_B
  printf("%f GFLOPS for MKL\n", ((double)((double)REPS * (double)MY_M * (double)MY_N * (double)MY_K) * 2.0 * ((double)STREAM_A_B_SIZE)) / (l_total * 1.0e9));
#else
  printf("%f GFLOPS for MKL\n", ((double)((double)REPS * (double)MY_M * (double)MY_N * (double)MY_K) * 2.0) / (l_total * 1.0e9));
#endif
#endif

  gettimeofday(&l_start, NULL);
  size_t l_cyc_start = _rdtsc();

  for ( l_t = 0; l_t < REPS; l_t++ ) {
#ifdef STREAM_A_B
    REALTYPE* l_p_a = l_a - (MY_K * MY_LDA);
    REALTYPE* l_p_b = l_b - (MY_N * MY_LDB);
    for ( l_s = 0; l_s < STREAM_A_B_SIZE; l_s++ ) {
      l_p_a += (MY_K * MY_LDA);
      l_p_b += (MY_N * MY_LDB);
#else
      REALTYPE* l_p_a = l_a;
      REALTYPE* l_p_b = l_b;
#endif
#ifdef STREAM_A_B_PREFETCH
      dense_test_mul(l_p_a, l_p_b, l_c, l_p_a + (MY_K * MY_LDA), l_p_b + (MY_N * MY_LDB), NULL);
#else
      dense_test_mul(l_p_a, l_p_b, l_c);
#endif
#ifdef STREAM_A_B
    }
#endif
  }
  size_t l_cyc_end = _rdtsc();
  gettimeofday(&l_end, NULL);
  l_total = sec(l_start, l_end);

  printf("%fs for assembly\n", l_total);
#ifdef STREAM_A_B
  printf("%f GFLOPS for assembly\n", ((double)((double)REPS * (double)MY_M * (double)MY_N * (double)MY_K) * 2.0 * ((double)STREAM_A_B_SIZE)) / (l_total * 1.0e9));
#else
  printf("%f GFLOPS for assembly\n", ((double)((double)REPS * (double)MY_M * (double)MY_N * (double)MY_K) * 2.0) / (l_total * 1.0e9));
  printf("%f FLOPS/cycle for assembly (using _rdtsc())\n", ((double)((double)REPS * (double)MY_M * (double)MY_N * (double)MY_K) * 2.0) / ((double)(l_cyc_end - l_cyc_start)));
#endif

  /* check result */
  for ( l_i = 0; l_i < MY_M; l_i++) {
    for ( l_j = 0; l_j < MY_N; l_j++) {
#if 0
      printf("Entries in row %i, column %i, gold: %f, assembly: %f\n", l_i+1, l_j+1, l_c_gold[(l_j*MY_M)+l_i], l_c[(l_j*MY_M)+l_i]);
#endif
      if (l_max_error < fabs( l_c_gold[(l_j * MY_LDC) + l_i] - l_c[(l_j * MY_LDC) + l_i]))
        l_max_error = fabs( l_c_gold[(l_j * MY_LDC) + l_i] - l_c[(l_j * MY_LDC) + l_i]);
    }
  }

  printf("max. error: %f\n", l_max_error);

  /* free */
  _mm_free(l_a);
  _mm_free(l_b);
  _mm_free(l_c);
  _mm_free(l_c_gold);
}

int main(int argc, char* argv[]) {
  printf("------------------------------------------------\n");
  printf("RUNNING (%ix%i) X (%ix%i) = (%ix%i)", MY_M, MY_K, MY_K, MY_N, MY_M, MY_N);
#ifdef STREAM_A_B
  printf(", STREAM_A_B");
#endif
  if (sizeof(REALTYPE) == sizeof(double)) {
    printf(", DP\n");
  } else {
    printf(", SP\n");
  }
  printf("------------------------------------------------\n");
  run_test();
  printf("------------------------------------------------\n");
  return 0;
}
