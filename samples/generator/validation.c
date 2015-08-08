/*
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include <immintrin.h>

#ifdef __USE_MKL
#define MKL_DIRECT_CALL
#define MKL_DIRECT_CALL_SEQ
#include <mkl.h>
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

#define REPS 10000
//#define REPS 1

inline double sec(struct timeval start, struct timeval end) {
  return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)))) / 1.0e6;
}

void run_test() {
  // allocate
  REALTYPE* l_a = (REALTYPE*)_mm_malloc(MY_M * MY_K * sizeof(REALTYPE), 64);
  REALTYPE* l_b = (REALTYPE*)_mm_malloc(MY_K * MY_N * sizeof(REALTYPE), 64);
  REALTYPE* l_c = (REALTYPE*)_mm_malloc(MY_M * MY_N * sizeof(REALTYPE), 64);
  REALTYPE* l_c_gold = (REALTYPE*)_mm_malloc(MY_M * MY_N * sizeof(REALTYPE), 64);

  unsigned int l_i;
  unsigned int l_j;
  unsigned int l_t;
  unsigned int l_m;
  unsigned int l_n;
  unsigned int l_k;

  // touch A
  for ( l_i = 0; l_i < MY_M; l_i++) {
    for ( l_j = 0; l_j < MY_K; l_j++) {
#if REPS==1
      l_a[(l_j * MY_M) + l_i] = (REALTYPE)drand48();
#else
      l_a[(l_j * MY_M) + l_i] = (REALTYPE)(l_i + (l_j * MY_M));
#endif
    }
  }
  // touch B
  for ( l_i = 0; l_i < MY_K; l_i++ ) {
    for ( l_j = 0; l_j < MY_N; l_j++ ) {
#if REPS==1
      l_b[(l_j * MY_K) + l_i] = (REALTYPE)drand48();
#else
      l_b[(l_j * MY_K) + l_i] = (REALTYPE)(l_i + (l_j * MY_K));
#endif
    }
  }
  // touch C
  for ( l_i = 0; l_i < MY_M; l_i++) {
    for ( l_j = 0; l_j < MY_N; l_j++) {
      l_c[(l_j * MY_M) + l_i] = (REALTYPE)0.0;
      l_c_gold[(l_j * MY_M) + l_i] = (REALTYPE)0.0;
    }
  }

  // C routine
  struct timeval l_start, l_end;

  gettimeofday(&l_start, NULL);
#ifndef __USE_MKL
  for ( l_t = 0; l_t < REPS; l_t++  ) {
    for ( l_n = 0; l_n < MY_N; l_n++  ) {
      for ( l_k = 0; l_k < MY_K; l_k++  ) {
        for ( l_m = 0; l_m < MY_M; l_m++ ) {
          l_c_gold[(l_n * MY_M) + l_m] += l_a[(l_k * MY_M) + l_m] * l_b[(l_n * MY_K) + l_k];
        }
      }
    }
  }
#else
  char l_trans = 'N';
  int l_M = MY_M;
  int l_N = MY_N;
  int l_K = MY_K;
  if (sizeof(REALTYPE) == sizeof(double)) {
    for ( l_t = 0; l_t < REPS; l_t++  ) {
      double l_one = 1.0;    
      dgemm(&l_trans, &l_trans, &l_M, &l_N, &l_K, &l_one, (double*)l_a, &l_M, (double*)l_b, &l_K, &l_one, (double*)l_c_gold, &l_M);
    } 
  } else {
    for ( l_t = 0; l_t < REPS; l_t++  ) {
      float l_one = 1.0f;    
      sgemm(&l_trans, &l_trans, &l_M, &l_N, &l_K, &l_one, (float*)l_a, &l_M, (float*)l_b, &l_K, &l_one, (float*)l_c_gold, &l_M);
    }
  }
#endif
  gettimeofday(&l_end, NULL);
  
  double l_total = sec(l_start, l_end);
#ifndef __USE_MKL
  printf("%fs for C\n", l_total);
  printf("%f GFLOPS for C\n", ((double)((double)REPS * (double)MY_M * (double)MY_N * (double)MY_K) * 2.0) / (l_total * 1.0e9));
#else
  printf("%fs for MKL\n", l_total);
  printf("%f GFLOPS for MKL\n", ((double)((double)REPS * (double)MY_M * (double)MY_N * (double)MY_K) * 2.0) / (l_total * 1.0e9));
#endif

  gettimeofday(&l_start, NULL);

  for ( l_t = 0; l_t < REPS; l_t++ ) {
    dense_test_mul(l_a, l_b, l_c);
  }
    
  gettimeofday(&l_end, NULL);
  l_total = sec(l_start, l_end);


  printf("%fs for assembly\n", l_total);
  printf("%f GFLOPS for assembly\n", ((double)((double)REPS * (double)MY_M * (double)MY_N * (double)MY_K) * 2.0) / (l_total * 1.0e9));

  // check result
  REALTYPE l_max_error = 0.0;

  for ( l_i = 0; l_i < MY_M; l_i++) {
    for ( l_j = 0; l_j < MY_N; l_j++) {
#if 0
      printf("Entries in row %i, column %i, gold: %f, assembly: %f\n", l_i+1, l_j+1, l_c_gold[(l_j*MY_M)+l_i], l_c[(l_j*MY_M)+l_i]);
#endif
      if (l_max_error < fabs( l_c_gold[(l_j * MY_M) + l_i] - l_c[(l_j * MY_M) + l_i]))
        l_max_error = fabs( l_c_gold[(l_j * MY_M) + l_i] - l_c[(l_j * MY_M) + l_i]);
    }
  }

  printf("max. error: %f\n", l_max_error);

  // free
  _mm_free(l_a);
  _mm_free(l_b);
  _mm_free(l_c);
  _mm_free(l_c_gold);
}

int main(int argc, char* argv[]) {
  printf("------------------------------------------------\n");
  printf("RUNNING (%ix%i) X (%ix%i) = (%ix%i)", MY_M, MY_K, MY_K, MY_N, MY_M, MY_N);
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
