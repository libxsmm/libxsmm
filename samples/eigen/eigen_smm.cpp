/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/

/** This sample uses LIBXSMM's header-only implementation. */
#include <libxsmm_source.h>

#if !defined(__EIGEN) && 0
# define __EIGEN
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if defined(__EIGEN)
# include <Eigen/Dense>
#endif
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if 0 /* enable padding on a per-matrix basis */
# define PAD(TYPE, VALUE) (LIBXSMM_UP2((VALUE) * sizeof(TYPE), LIBXSMM_ALIGNMENT) / sizeof(TYPE))
#else
# define PAD(TYPE, VALUE) (VALUE)
#endif

#if !defined(RANDOMIZED) && 0
# define RANDOMIZED
#endif

#if !defined(ITYPE)
# define ITYPE double
#endif
#if !defined(OTYPE)
# define OTYPE ITYPE
#endif


#if defined(__EIGEN)

template<typename itype, typename otype>
LIBXSMM_INLINE LIBXSMM_RETARGETABLE
void smm_eigen_dynamic(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const itype *LIBXSMM_RESTRICT a, const itype *LIBXSMM_RESTRICT b, otype *LIBXSMM_RESTRICT c)
{
  typedef Eigen::Matrix<itype,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor|Eigen::AutoAlign> matrix_itype;
  typedef Eigen::Matrix<otype,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor|Eigen::AutoAlign> matrix_otype;
  matrix_otype::Map(c, m, n).noalias() = matrix_itype::Map(a, m, k) * matrix_itype::Map(b, k, n);
}

#endif /*defined(__EIGEN)*/


template<typename itype, typename otype>
LIBXSMM_INLINE LIBXSMM_RETARGETABLE
void smm_xsmm_specialized(const libxsmm_mmfunction<itype,otype>& xmm,
  const itype* a, const itype* b, otype* c, const itype* next_a, const itype* next_b, const otype* next_c)
{
#if (0 != LIBXSMM_PREFETCH)
  xmm(a, b, c, next_a, next_b, next_c);
#else
  xmm(a, b, c);
  LIBXSMM_UNUSED(next_a);
  LIBXSMM_UNUSED(next_b);
  LIBXSMM_UNUSED(next_c);
#endif
}


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  try {
    const libxsmm_blasint benchmark = (1 < argc ? std::atoi(argv[1]) : 0);
    const libxsmm_blasint m = (2 < argc ? std::atoi(argv[2]) : 23);
    const libxsmm_blasint k = (4 < argc ? std::atoi(argv[4]) : m);
    const libxsmm_blasint n = (3 < argc ? std::atoi(argv[3]) : k);
    const libxsmm_blasint q = (5 < argc ? std::atoi(argv[5]) : 0/*auto*/);
    const libxsmm_blasint nrepeat = (6 < argc ? std::atoi(argv[6]) : (0 >= q ? 13 : 1));

    const libxsmm_blasint lda = m, ldb = k, ldc = m;
    const char transa = 'N', transb = 'N';
    const OTYPE alpha = 1, beta = 1;

    const libxsmm_blasint asize = PAD(ITYPE, lda * k), bsize = PAD(ITYPE, ldb * n), csize = PAD(OTYPE, ldc * n);
    const libxsmm_blasint max_size = ((2ULL << 30/*2 GB*/) / ((static_cast<size_t>(asize) + bsize) * sizeof(ITYPE) + csize * sizeof(OTYPE)));
    const libxsmm_blasint s = LIBXSMM_MIN(0 < q ? q : max_size, max_size);
    const libxsmm_blasint aspace = LIBXSMM_ALIGNMENT / sizeof(ITYPE);
    const size_t bwsize = (static_cast<size_t>(asize)/*load*/ + static_cast<size_t>(bsize)/*load*/) * sizeof(ITYPE)
                        + (sizeof(OTYPE) * static_cast<size_t>(csize) * 2/*RFO*/);
    const double gflops = 2E-9 * s * m * n * k;
#if LIBXSMM_TYPEINFO(ITYPE, FP)
    const char ops[] = "FLOPS";
    const double scale = 1.0 / s;
#else
    const char ops[] = "OPS";
    const double scale = 1;
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
#   pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
    {
#if defined(_OPENMP)
      const libxsmm_blasint chunksize = s / omp_get_max_threads();
#endif
      struct raii { // avoid std::vector (first-touch init. causes NUMA issue)
        ITYPE *a, *b;
        OTYPE *c;
        size_t m_size, m_shuffle;
        raii(libxsmm_blasint asize_, libxsmm_blasint bsize_, libxsmm_blasint csize_, libxsmm_blasint size_)
          : a(new ITYPE[static_cast<size_t>(asize_)]), b(new ITYPE[static_cast<size_t>(bsize_)])
          , c(new OTYPE[static_cast<size_t>(csize_)])
          , m_size(static_cast<size_t>(size_)), m_shuffle(libxsmm_shuffle(static_cast<unsigned int>(size_)))
        {}
        ~raii() { delete[] a; delete[] b; delete[] c; }
#if defined(RANDOMIZED)
        libxsmm_blasint shuffle(libxsmm_blasint i) const { return (i * m_shuffle) % m_size; }
#else
        libxsmm_blasint shuffle(libxsmm_blasint i) const { return i; }
#endif
      } helper(s * asize + aspace - 1, s * bsize + aspace - 1, s * csize + aspace - 1, s);

      ITYPE *const a = LIBXSMM_ALIGN(helper.a, LIBXSMM_ALIGNMENT);
      ITYPE *const b = LIBXSMM_ALIGN(helper.b, LIBXSMM_ALIGNMENT);
      OTYPE *const c = LIBXSMM_ALIGN(helper.c, LIBXSMM_ALIGNMENT);
#if defined(_OPENMP)
#     pragma omp parallel for schedule(static)
#endif
      for (libxsmm_blasint i = 0; i < s; ++i) {
        LIBXSMM_MATINIT(ITYPE, 42 + helper.shuffle(i), a + static_cast<size_t>(asize) * helper.shuffle(i), m, k, lda, scale);
        LIBXSMM_MATINIT(ITYPE, 24 + helper.shuffle(i), b + static_cast<size_t>(bsize) * helper.shuffle(i), k, n, ldb, scale);
        LIBXSMM_MATINIT(OTYPE, 22 + i, c + static_cast<size_t>(csize) * i, m, n, ldc, scale);
      }

      // initialize LIBXSMM
      libxsmm_init();

      fprintf(stdout, "m=%lli n=%lli k=%lli size=%lli memory=%.1f MB (%s)\n\n",
        static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k), static_cast<long long>(s),
        1.0 * (s * ((static_cast<size_t>(asize) + bsize) * sizeof(ITYPE) + csize * sizeof(OTYPE))) / (1ULL << 20), 8 == sizeof(ITYPE) ? "DP" : "SP");

      const libxsmm_mmfunction<ITYPE,OTYPE> xmm(LIBXSMM_GEMM_FLAGS(transa, transb), m, n, k, lda, ldb, ldc, alpha, beta, LIBXSMM_PREFETCH_AUTO);

      switch (benchmark) {
      case 0: if (xmm) {
        fprintf(stdout, "LIBXSMM batched (A,B,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
            const ITYPE *const ai = a + static_cast<size_t>(asize) * helper.shuffle(i), *const bi = b + static_cast<size_t>(bsize) * helper.shuffle(i);
            OTYPE *const ci = c + static_cast<size_t>(csize) * i;
            smm_xsmm_specialized<ITYPE,OTYPE>(xmm, ai, bi, ci,
              LIBXSMM_GEMM_PREFETCH_A(ai + asize), LIBXSMM_GEMM_PREFETCH_B(bi + bsize),
              LIBXSMM_GEMM_PREFETCH_C(ci + csize));
          }
        }
        const unsigned long long ncycles = libxsmm_timer_ncycles(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /* fallthrough */

#if defined(__EIGEN)
      case 1: {
        fprintf(stdout, "Eigen/dynamic batched (A,B,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
            const ITYPE *const ai = a + static_cast<size_t>(asize) * helper.shuffle(i), *const bi = b + static_cast<size_t>(bsize) * helper.shuffle(i);
            OTYPE *const ci = c + static_cast<size_t>(csize) * i;
            smm_eigen_dynamic(m, n, k, ai, bi, ci);
          }
        }
        const unsigned long long ncycles = libxsmm_timer_ncycles(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }
#endif /*defined(__EIGEN)*/
      break;
      case 2: if (xmm) {
        fprintf(stdout, "LIBXSMM streamed (A,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
            const ITYPE *const ai = a + static_cast<size_t>(asize) * helper.shuffle(i);
            OTYPE *const ci = c + static_cast<size_t>(csize) * i;
            smm_xsmm_specialized<ITYPE,OTYPE>(xmm, ai, b, ci,
              LIBXSMM_GEMM_PREFETCH_A(ai + asize), LIBXSMM_GEMM_PREFETCH_B(b),
              LIBXSMM_GEMM_PREFETCH_C(ci + csize));
          }
        }
        const unsigned long long ncycles = libxsmm_timer_ncycles(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - bsize * sizeof(ITYPE)) / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /* fallthrough */

#if defined(__EIGEN)
      case 3: {
        fprintf(stdout, "Eigen/dynamic streamed (A,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
            const ITYPE *const ai = a + static_cast<size_t>(asize) * helper.shuffle(i);
            OTYPE *const ci = c + static_cast<size_t>(csize) * i;
            smm_eigen_dynamic(m, n, k, ai, b, ci);
          }
        }
        const unsigned long long ncycles = libxsmm_timer_ncycles(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - bsize * sizeof(ITYPE)) / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }
#endif /*defined(__EIGEN)*/
      break;
      case 4: if (xmm) {
        fprintf(stdout, "LIBXSMM streamed (B,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
            const ITYPE *const bi = b + static_cast<size_t>(bsize) * helper.shuffle(i);
            OTYPE *const ci = c + static_cast<size_t>(csize) * i;
            smm_xsmm_specialized<ITYPE,OTYPE>(xmm, a, bi, ci,
              LIBXSMM_GEMM_PREFETCH_A(a), LIBXSMM_GEMM_PREFETCH_B(bi + bsize),
              LIBXSMM_GEMM_PREFETCH_C(ci + csize));
          }
        }
        const unsigned long long ncycles = libxsmm_timer_ncycles(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - asize * sizeof(ITYPE)) / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /* fallthrough */

#if defined(__EIGEN)
      case 5: {
        fprintf(stdout, "Eigen/dynamic streamed (B,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
            const ITYPE *const bi = b + static_cast<size_t>(bsize) * helper.shuffle(i);
            OTYPE *const ci = c + static_cast<size_t>(csize) * i;
            smm_eigen_dynamic(m, n, k, a, bi, ci);
          }
        }
        const unsigned long long ncycles = libxsmm_timer_ncycles(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - asize * sizeof(ITYPE)) / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }
#endif /*defined(__EIGEN)*/
      break;
      case 6: if (xmm) {
        fprintf(stdout, "LIBXSMM streamed (A,B)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
            const libxsmm_blasint j = omp_get_thread_num() * chunksize * csize;
#else
            const libxsmm_blasint j = 0;
#endif
            const ITYPE *const ai = a + static_cast<size_t>(asize) * helper.shuffle(i), *const bi = b + static_cast<size_t>(bsize) * helper.shuffle(i);
            smm_xsmm_specialized<ITYPE,OTYPE>(xmm, ai, bi, c + j,
              LIBXSMM_GEMM_PREFETCH_A(ai + asize), LIBXSMM_GEMM_PREFETCH_B(bi + bsize),
              LIBXSMM_GEMM_PREFETCH_C(c + j));
          }
        }
        const unsigned long long ncycles = libxsmm_timer_ncycles(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - sizeof(OTYPE) * csize * 2) / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /* fallthrough */

#if defined(__EIGEN)
      case 7: {
        fprintf(stdout, "Eigen/dynamic streamed (A,B)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
            const libxsmm_blasint j = omp_get_thread_num() * chunksize * csize;
#else
            const libxsmm_blasint j = 0;
#endif
            const ITYPE *const ai = a + static_cast<size_t>(asize) * helper.shuffle(i), *const bi = b + static_cast<size_t>(bsize) * helper.shuffle(i);
            smm_eigen_dynamic(m, n, k, ai, bi, c + j);
          }
        }
        const unsigned long long ncycles = libxsmm_timer_ncycles(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - sizeof(OTYPE) * csize * 2) / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }
#endif /*defined(__EIGEN)*/
      break;
      case 8: if (xmm) {
        fprintf(stdout, "LIBXSMM cached...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
            const libxsmm_blasint j = omp_get_thread_num() * chunksize * csize;
#else
            const libxsmm_blasint j = 0;
#endif
            smm_xsmm_specialized<ITYPE,OTYPE>(xmm, a, b, c + j,
              LIBXSMM_GEMM_PREFETCH_A(a), LIBXSMM_GEMM_PREFETCH_B(b),
              LIBXSMM_GEMM_PREFETCH_C(c + j));
          }
        }
        const unsigned long long ncycles = libxsmm_timer_ncycles(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /* fallthrough */

#if defined(__EIGEN)
      case 9: {
        fprintf(stdout, "Eigen/dynamic cached...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
            const libxsmm_blasint j = omp_get_thread_num() * chunksize * csize;
#else
            const libxsmm_blasint j = 0;
#endif
            smm_eigen_dynamic(m, n, k, a, b, c + j);
          }
        }
        const unsigned long long ncycles = libxsmm_timer_ncycles(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }
#endif /*defined(__EIGEN)*/
      break;
      default: throw "invalid case selected!";
      } /*switch*/

      // finalize LIBXSMM
      libxsmm_finalize();
      fprintf(stdout, "Finished\n");
    }
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

