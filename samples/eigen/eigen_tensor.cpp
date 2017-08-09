/******************************************************************************
** Copyright (c) 2017, Intel Corporation                                     **
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
/* Hans Pabst (Intel Corp.)
******************************************************************************/

#if !defined(LIBXSMM_FALLBACK_MMFUNCTION) && 0
# define LIBXSMM_FALLBACK_MMFUNCTION
#endif

/** This sample uses LIBXSMM's header-only implementation. */
#include <libxsmm_source.h>

#if !defined(USE_LIBXSMM)
# define USE_LIBXSMM
#endif

#if defined(USE_LIBXSMM)
# if !defined(EIGEN_VECTORIZE_AVX)
#   define EIGEN_VECTORIZE_AVX
# endif
# if !defined(EIGEN_USE_LIBXSMM)
#   define EIGEN_USE_LIBXSMM
# endif
#endif

#if !defined(__EIGEN) && !defined(__EIGEN_UNSUPPORTED) && 0
# define __EIGEN_UNSUPPORTED
# define __EIGEN
#endif

#if !defined(EIGEN_USE_THREADS) && defined(__EIGEN) && (defined(_OPENMP) || (defined(__BLAS) && 1 < (__BLAS)))
# define EIGEN_USE_THREADS
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if defined(__EIGEN_UNSUPPORTED)
# include <unsupported/Eigen/CXX11/Tensor>
# include <unsupported/Eigen/CXX11/ThreadPool>
#endif
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(REAL_TYPE)
# define REAL_TYPE float
#endif


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  try {
#if !defined(__EIGEN_UNSUPPORTED)
    LIBXSMM_UNUSED(argc); LIBXSMM_UNUSED(argv);
    throw std::runtime_error("Eigen or Eigen/unsupported not found!");
#else
    const libxsmm_blasint m = (1 < argc ? std::atoi(argv[1]) : 512);
    const libxsmm_blasint k = (3 < argc ? atoi(argv[3]) : m);
    const libxsmm_blasint n = (2 < argc ? atoi(argv[2]) : k);
    const int nrepeat = LIBXSMM_MAX(4 < argc ? atoi(argv[4]) : 13 / LIBXSMM_MAX(1, libxsmm_icbrt(1ULL * m * n * k) >> 10), 3);
    const char *const env_check = getenv("CHECK"), *const env_nthreads = getenv("NTHREADS");
    const double check = (0 == env_check ? 1.0 : LIBXSMM_ABS(atof(env_check)));
    const double gflops = 2.0 * m * n * k * 1E-9;
    const int nthreads = LIBXSMM_MAX(0 == env_nthreads ? 0 : atoi(env_nthreads), 1);
# if defined(LIBXSMM_OFFLOAD_TARGET)
#   pragma offload target(LIBXSMM_OFFLOAD_TARGET)
# endif
    {
# if defined(MKL_ENABLE_AVX512)
      mkl_enable_instructions(MKL_ENABLE_AVX512);
# endif
# if defined(_OPENMP)
      Eigen::NonBlockingThreadPool threadpool(1 == nthreads ? omp_get_max_threads() : nthreads);
# else
      Eigen::NonBlockingThreadPool threadpool(nthreads);
# endif
      Eigen::ThreadPoolDevice device(&threadpool, threadpool.NumThreads());
      Eigen::Tensor<REAL_TYPE,2/*nindices*/,0/*options*/,libxsmm_blasint> ta(m, k), tb(k, n), tc(m, n), td(m, n);
      const char transa = 'N', transb = 'N';
      const REAL_TYPE alpha = 1, beta = 0;
      libxsmm_matdiff_info diff;
      unsigned long long start;
      double d1, d2;
      {
        std::array<Eigen::IndexPair<libxsmm_blasint>,1> product_dims = {
          Eigen::IndexPair<libxsmm_blasint>(1, 0),
        };
        ta.setRandom(); tb.setRandom();
        start = libxsmm_timer_tick();
        for (int i = 0; i < nrepeat; ++i) {
          tc.device(device) = ta.contract(tb, product_dims);
        }
        d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      }
      libxsmm_gemm_print(stdout, LIBXSMM_GEMM_PRECISION(REAL_TYPE), &transa, &transb,
        &m, &n, &k, &alpha, ta.data(), &m, tb.data(), &k, &beta, tc.data(), &m);
      fprintf(stdout, "\n\n");
      {
        start = libxsmm_timer_tick();
        for (int i = 0; i < nrepeat; ++i) {
          LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k,
            &alpha, ta.data(), &m, tb.data(), &k,
             &beta, td.data(), &m);
        }
        d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      }
      if (0 < d1) {
        fprintf(stdout, "\tEigen"
#if !defined(USE_LIBXSMM)
          "+XSMM"
#endif
          ": %.1f GFLOPS/s\n", gflops * nrepeat / d1);
      }
      if (0 < d2) {
        fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / d2);
      }
      if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(REAL_TYPE), m, n, td.data(), tc.data(), &m, &m, &diff)) {
        fprintf(stdout, "\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
        if (check < 100.0 * diff.normf_rel) {
          fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
          result = EXIT_FAILURE;
        }
      }
    }
    fprintf(stdout, "Finished\n");
#endif /*defined(__EIGEN_UNSUPPORTED)*/
  }
  catch(const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    result = EXIT_FAILURE;
  }
  catch(const char* message) {
    fprintf(stderr, "Error: %s\n", message);
    result = EXIT_FAILURE;
  }
  catch(...) {
    fprintf(stderr, "Error: unknown exception caught!\n");
    result = EXIT_FAILURE;
  }

  return result;
}

