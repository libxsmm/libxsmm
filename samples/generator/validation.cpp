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

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <immintrin.h>
#include <sys/time.h>
#ifdef __USE_MKL
#define MKL_DIRECT_CALL
#define MKL_DIRECT_CALL_SEQ
#include <mkl.h>
#endif

#ifdef USE_ASM_DIRECT
extern "C" { void dense_test_mul(const REALTYPE* a, const REALTYPE* b, REALTYPE* c); }
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
  REALTYPE* a = (REALTYPE*)_mm_malloc(MY_M * MY_K * sizeof(REALTYPE), 64);
  REALTYPE* b = (REALTYPE*)_mm_malloc(MY_K * MY_N * sizeof(REALTYPE), 64);
  REALTYPE* c = (REALTYPE*)_mm_malloc(MY_M * MY_N * sizeof(REALTYPE), 64);
  REALTYPE* c_gold = (REALTYPE*)_mm_malloc(MY_M * MY_N * sizeof(REALTYPE), 64);

  // touch A
  for (int i = 0; i < MY_M; i++) {
    for (int j = 0; j < MY_K; j++) {
#if REPS==1
      a[(j * MY_M) + i] = (REALTYPE)drand48();
#else
      a[(j * MY_M) + i] = (REALTYPE)(i + (j * MY_M));
#endif
    }
  }
  // touch B
  for (int i = 0; i < MY_K; i++) {
    for (int j = 0; j < MY_N; j++) {
#if REPS==1
      b[(j * MY_K) + i] = (REALTYPE)drand48();
#else
      b[(j * MY_K) + i] = (REALTYPE)(i + (j * MY_K));
#endif
    }
  }
  // touch C
  for (int i = 0; i < MY_M; i++) {
    for (int j = 0; j < MY_N; j++) {
      c[(j * MY_M) + i] = (REALTYPE)0.0;
      c_gold[(j * MY_M) + i] = (REALTYPE)0.0;
    }
  }

  // C routine
  struct timeval start, end;
  gettimeofday(&start, NULL);

  for (int t = 0; t < REPS; t++) {
#ifndef __USE_MKL
    for (int n = 0; n < MY_N; n++) {
      for (int k = 0; k < MY_K; k++) {
        for (int m = 0; m < MY_M; m++) {
          c_gold[(n * MY_M) + m] += a[(k * MY_M) + m] * b[(n * MY_K) + k];
        }
      }
    }
#else
    char trans = 'N';
    int M = MY_M;
    int N = MY_N;
    int K = MY_K;
    double one = 1.0;
    
    dgemm(&trans, &trans, &M, &N, &K, &one, a, &M, b, &K, &one, c_gold, &M);
#endif
  }

  gettimeofday(&end, NULL);
  double total = sec(start, end);
#ifndef __USE_MKL
  std::cout << total << "s for C" << std::endl;
  std::cout << ((double)((double)REPS * (double)MY_M * (double)MY_N * (double)MY_K) * 2.0) / (total * 1.0e9) << " GFLOPS for C" << std::endl;
#else
  std::cout << total << "s for MKL" << std::endl;
  std::cout << ((double)((double)REPS * (double)MY_M * (double)MY_N * (double)MY_K) * 2.0) / (total * 1.0e9) << " GFLOPS for MKL" << std::endl;
#endif

  gettimeofday(&start, NULL);

  for (int t = 0; t < REPS; t++) {
    dense_test_mul(a, b, c);
  }
    
  gettimeofday(&end, NULL);
  total = sec(start, end);

  std::cout << total << "s for assembly" << std::endl;
  std::cout << ((double)((double)REPS * (double)MY_M * (double)MY_N * (double)MY_K) * 2.0) / (total * 1.0e9) << " GFLOPS for assembly" << std::endl;

  // check result
  REALTYPE max_error = 0.0;

  for (int i = 0; i < MY_M; i++) {
    for (int j = 0; j < MY_N; j++) {
      //std::cout << c_gold[(j*MY_M)+i] << " " << c[(j*MY_M)+i] << std::endl;
      if (max_error < fabs( c_gold[(j * MY_M) + i] - c[(j * MY_M) + i]))
        max_error = fabs( c_gold[(j * MY_M) + i] - c[(j * MY_M) + i]);
    }
  }

  std::cout << "max. error: " << max_error << std::endl;

  // free
  _mm_free(a);
  _mm_free(b);
  _mm_free(c);
  _mm_free(c_gold);
}

int main(int argc, char* argv[]) {
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "RUNNING (" << MY_M << "x" << MY_K << ") X (" << MY_K << "x" << MY_N << ") = (" << MY_M << "x" << MY_N << ")";
  if (sizeof(REALTYPE) == sizeof(double)) {
    std::cout << ", DP" << std::endl;
  } else {
    std::cout << ", SP" << std::endl;
  }
  std::cout << "------------------------------------------------" << std::endl;
  run_test();
  std::cout << "------------------------------------------------" << std::endl;
  return 0;
}
