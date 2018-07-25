/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
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
#include <libxsmm.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>
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


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  try {
    const libxsmm_blasint benchmark = 1 < argc ? std::atoi(argv[1]) : 0;
    const libxsmm_blasint m = (2 < argc ? std::atoi(argv[2]) : 23);
    const libxsmm_blasint k = (4 < argc ? std::atoi(argv[4]) : m);
    const libxsmm_blasint n = (3 < argc ? std::atoi(argv[3]) : k);
    const libxsmm_blasint q = (5 < argc ? std::atoi(argv[5]) : 0/*auto*/);
    const libxsmm_blasint nrepeat = (6 < argc ? std::atoi(argv[6]) : (0 >= q ? 13 : 1));

    const libxsmm_blasint lda = m, ldb = k, ldc = m;
    const char transa = 'N', transb = 'N';
    const OTYPE alpha = 1, beta = 1;

    const libxsmm_blasint asize = PAD(ITYPE, lda * k), bsize = PAD(ITYPE, ldb * n), csize = PAD(OTYPE, ldc * n);
    const libxsmm_blasint max_size = ((2ULL << 30/*2 GB*/) / ((asize + bsize) * sizeof(ITYPE) + csize * sizeof(OTYPE)));
    const libxsmm_blasint s = LIBXSMM_MIN(0 < q ? q : max_size, max_size);
    const libxsmm_blasint aspace = LIBXSMM_ALIGNMENT / sizeof(ITYPE);
    const size_t bwsize = static_cast<size_t>((asize/*load*/ + bsize/*load*/) * sizeof(ITYPE) + 2/*RFO*/ * csize * sizeof(OTYPE));
    const double gflops = 2E-9 * s * m * n * k;
#if LIBXSMM_TYPEINFO(ITYPE, FP)
    const char *const ops = "FLOPS";
    const double scale = 1.0 / s;
#else
    const char *const ops = "OPS";
    const double scale = 1;
#endif
#if !defined(_DEBUG)
    const char *const env_check = getenv("CHECK");
    const int check = (0 == env_check ? 0 : atoi(env_check));
#else
    /*const*/ int check = 1;
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
        OTYPE *c, *d;
        libxsmm_blasint *m_shuffle;
        raii(libxsmm_blasint asize_, libxsmm_blasint bsize_, libxsmm_blasint csize_, libxsmm_blasint size_)
          : a(new ITYPE[static_cast<size_t>(asize_)]), b(new ITYPE[static_cast<size_t>(bsize_)])
          , c(new OTYPE[static_cast<size_t>(csize_)]), d(new OTYPE[static_cast<size_t>(csize_)])
          , m_shuffle(new libxsmm_blasint[size_])
        {
# if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
# endif
          for (libxsmm_blasint i = 0; i < size_; ++i) m_shuffle[i] = libxsmm_rand_u32(size_);
        }
        ~raii() { delete[] a; delete[] b; delete[] c; delete[] d; delete[] m_shuffle; }
#if defined(RANDOMIZED)
        libxsmm_blasint shuffle(libxsmm_blasint i) const { return m_shuffle[i]; }
#else
        libxsmm_blasint shuffle(libxsmm_blasint i) const { return i; }
#endif
      } helper(s * asize + aspace - 1, s * bsize + aspace - 1, s * csize + aspace - 1, s);

      ITYPE *const a = LIBXSMM_ALIGN(helper.a, LIBXSMM_ALIGNMENT);
      ITYPE *const b = LIBXSMM_ALIGN(helper.b, LIBXSMM_ALIGNMENT);
      OTYPE *const c = LIBXSMM_ALIGN(helper.c, LIBXSMM_ALIGNMENT);
      OTYPE *const d = LIBXSMM_ALIGN(helper.d, LIBXSMM_ALIGNMENT);
#if defined(_OPENMP)
#     pragma omp parallel for schedule(static)
#endif
      for (libxsmm_blasint i = 0; i < s; ++i) {
        LIBXSMM_MATINIT(ITYPE, 42 + helper.shuffle(i), a + helper.shuffle(i) * asize, m, k, lda, scale);
        LIBXSMM_MATINIT(ITYPE, 24 + helper.shuffle(i), b + helper.shuffle(i) * bsize, k, n, ldb, scale);
        LIBXSMM_MATINIT(OTYPE, 22 + i, c + i * csize, m, n, ldc, scale);
        LIBXSMM_MATINIT(OTYPE, 22 + i, d + i * csize, m, n, ldc, scale);
      }

#if defined(MKL_ENABLE_AVX512)
      mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
      // initialize LIBXSMM
      libxsmm_init();

      fprintf(stdout, "m=%lli n=%lli k=%lli size=%lli memory=%.1f MB (input=%s output=%s)\n\n",
        static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k), static_cast<long long>(s),
        1.0 * (s * ((asize + bsize) * sizeof(ITYPE) + csize * sizeof(OTYPE))) / (1 << 20),
        LIBXSMM_TYPENAME(ITYPE), LIBXSMM_TYPENAME(OTYPE));

      const libxsmm_mmfunction<ITYPE,OTYPE> xmm(LIBXSMM_GEMM_FLAGS(transa, transb),
        m, n, k, lda, ldb, ldc, alpha, beta, LIBXSMM_PREFETCH);
      if (!xmm) throw "no specialized routine found!";

      // arrays needed for the batch interface (indirect)
      std::vector<const ITYPE*> va_array(static_cast<size_t>(s)), vb_array(static_cast<size_t>(s));
      std::vector<OTYPE*> vc_array(static_cast<size_t>(s));
      const ITYPE* *const a_array = &va_array[0];
      const ITYPE* *const b_array = &vb_array[0];
      OTYPE* *const c_array = &vc_array[0];

      switch (benchmark) {
      case 0: { // batched
        fprintf(stdout, "Batched (A,B,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
            const ITYPE *const ai = a + helper.shuffle(i) * asize, *const bi = b + helper.shuffle(i) * bsize;
            OTYPE *const ci = c + i * csize;
#if (0 != LIBXSMM_PREFETCH)
            xmm(ai, bi, ci,
              LIBXSMM_GEMM_PREFETCH_A(ai + asize),
              LIBXSMM_GEMM_PREFETCH_B(bi + bsize),
              LIBXSMM_GEMM_PREFETCH_C(ci + csize));
#else
            xmm(ai, bi, ci);
#endif
          }
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2 * k - 1) * (double)(s * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /* fallthrough */

      case 1: { // batched/indirect
        fprintf(stdout, "Indirect (A,B,C)...\n");
        for (libxsmm_blasint i = 0; i < s; ++i) {
          a_array[i] = a + helper.shuffle(i) * asize; b_array[i] = b + helper.shuffle(i) * bsize; c_array[i] = d + i * csize;
        }
        const libxsmm_blasint ptrsize = sizeof(void*);
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
          libxsmm_gemm_batch2_omp(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(OTYPE), &transa, &transb,
            m, n, k, &alpha, &a_array[0], &lda, &b_array[0], &ldb, &beta, &c_array[0], &ldc,
            0/*index_base*/, 0/*index_stride*/, &ptrsize, &ptrsize, &ptrsize, s);
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2 * k - 1) * (double)(s * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        if (0 == benchmark) { /* Gold result is available */
          libxsmm_matdiff_info diff;
          memset(&diff, 0, sizeof(diff));
          for (libxsmm_blasint h = 0; h < s; ++h) {
            const OTYPE *const u = c + h * csize, *const v = c_array[h];
            libxsmm_matdiff_info dv;
            result = libxsmm_matdiff(LIBXSMM_DATATYPE(OTYPE), m, n, u, v, &ldc, &ldc, &dv);
            if (EXIT_SUCCESS == result) {
              libxsmm_matdiff_reduce(&diff, &dv);
            }
          }
          if (0 < diff.normf_rel) fprintf(stdout, "\tdiff: %.0f%%\n", 100.0 * diff.normf_rel);
        }
      } break;

      case 2: { // streaming A and C
        fprintf(stdout, "Streamed (A,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
            const ITYPE *const ai = a + helper.shuffle(i) * asize;
            OTYPE *const ci = c + i * csize;
#if (0 != LIBXSMM_PREFETCH)
            xmm(ai, b, ci,
              LIBXSMM_GEMM_PREFETCH_A(ai + asize), LIBXSMM_GEMM_PREFETCH_B(b),
              LIBXSMM_GEMM_PREFETCH_C(ci + csize));
#else
            xmm(ai, b, ci);
#endif
          }
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2 * k - 1) * (double)(s * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - bsize * sizeof(ITYPE)) / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /* fallthrough */

      case 3: { // indirect A and C
        fprintf(stdout, "Indirect (A,C)...\n");
        for (libxsmm_blasint i = 0; i < s; ++i) { a_array[i] = a + helper.shuffle(i) * asize; b_array[i] = b; c_array[i] = d + i * csize; }
        const libxsmm_blasint ptrsize = sizeof(void*);
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
          libxsmm_gemm_batch2_omp(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(OTYPE), &transa, &transb,
            m, n, k, &alpha, &a_array[0], &lda, &b_array[0], &ldb, &beta, &c_array[0], &ldc,
            0/*index_base*/, 0/*index_stride*/, &ptrsize, &ptrsize, &ptrsize, s);
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2 * k - 1) * (double)(s * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - bsize * sizeof(ITYPE)) / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        libxsmm_matdiff_info diff;
        memset(&diff, 0, sizeof(diff));
        for (libxsmm_blasint h = 0; h < s; ++h) {
          const OTYPE *const u = c + h * csize, *const v = c_array[h];
          libxsmm_matdiff_info dv;
          result = libxsmm_matdiff(LIBXSMM_DATATYPE(OTYPE), m, n, u, v, &ldc, &ldc, &dv);
          if (EXIT_SUCCESS == result) {
            libxsmm_matdiff_reduce(&diff, &dv);
          }
        }
        if (0 < diff.normf_rel) fprintf(stdout, "\tdiff: %.0f%%\n", 100.0 * diff.normf_rel);
      } break;

      case 4: { // streaming B and C
        fprintf(stdout, "Streamed (B,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxsmm_blasint i = 0; i < s; ++i) {
            const ITYPE *const bi = b + helper.shuffle(i) * bsize;
            OTYPE *const ci = c + i * csize;
#if (0 != LIBXSMM_PREFETCH)
            xmm(a, bi, ci,
              LIBXSMM_GEMM_PREFETCH_A(a), LIBXSMM_GEMM_PREFETCH_B(bi + bsize),
              LIBXSMM_GEMM_PREFETCH_C(ci + csize));
#else
            xmm(a, bi, ci);
#endif
          }
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2 * k - 1) * (double)(s * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - asize * sizeof(ITYPE)) / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /* fallthrough */

      case 5: { // indirect B and C
        fprintf(stdout, "Indirect (B,C)...\n");
        for (libxsmm_blasint i = 0; i < s; ++i) { a_array[i] = a; b_array[i] = b + helper.shuffle(i) * bsize; c_array[i] = d + i * csize; }
        const libxsmm_blasint ptrsize = sizeof(void*);
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
          libxsmm_gemm_batch2_omp(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(OTYPE), &transa, &transb,
            m, n, k, &alpha, &a_array[0], &lda, &b_array[0], &ldb, &beta, &c_array[0], &ldc,
            0/*index_base*/, 0/*index_stride*/, &ptrsize, &ptrsize, &ptrsize, s);
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2 * k - 1) * (double)(s * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - asize * sizeof(ITYPE)) / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        libxsmm_matdiff_info diff;
        memset(&diff, 0, sizeof(diff));
        for (libxsmm_blasint h = 0; h < s; ++h) {
          const OTYPE *const u = c + h * csize, *const v = c_array[h];
          libxsmm_matdiff_info dv;
          result = libxsmm_matdiff(LIBXSMM_DATATYPE(OTYPE), m, n, u, v, &ldc, &ldc, &dv);
          if (EXIT_SUCCESS == result) {
            libxsmm_matdiff_reduce(&diff, &dv);
          }
        }
        if (0 < diff.normf_rel) fprintf(stdout, "\tdiff: %.0f%%\n", 100.0 * diff.normf_rel);
      } break;

      case 6: { // streaming A and B
        fprintf(stdout, "Streamed (A,B)...\n");
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
            const ITYPE *const ai = a + helper.shuffle(i) * asize, *const bi = b + helper.shuffle(i) * bsize;
#if (0 != LIBXSMM_PREFETCH)
            xmm(ai, bi, c + j,
              LIBXSMM_GEMM_PREFETCH_A(ai + asize),
              LIBXSMM_GEMM_PREFETCH_B(bi + bsize),
              LIBXSMM_GEMM_PREFETCH_C(c + j));
#else
            xmm(ai, bi, c + j);
#endif
          }
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2 * k - 1) * (double)(s * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - 2 * csize * sizeof(OTYPE)) / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /* fallthrough */

      case 7: { // indirect A and B
        fprintf(stdout, "Indirect (A,B)...\n");
#if defined(_OPENMP)
#       pragma omp parallel for schedule(static)
#endif
        for (libxsmm_blasint i = 0; i < s; ++i) {
          a_array[i] = a + helper.shuffle(i) * asize; b_array[i] = b + helper.shuffle(i) * bsize;
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
          c_array[i] = d + omp_get_thread_num() * chunksize * csize;
#else
          c_array[i] = d;
#endif
        }
        const libxsmm_blasint ptrsize = sizeof(void*);
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
          libxsmm_gemm_batch2_omp(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(OTYPE), &transa, &transb,
            m, n, k, &alpha, &a_array[0], &lda, &b_array[0], &ldb, &beta, &c_array[0], &ldc,
            0/*index_base*/, 0/*index_stride*/, &ptrsize, &ptrsize, &ptrsize, s);
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2 * k - 1) * (double)(s * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - 2 * csize * sizeof(OTYPE)) / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } break;

      case 8: { // cached
        fprintf(stdout, "Cached...\n");
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
#if (0 != LIBXSMM_PREFETCH)
            xmm(a, b, c + j,
              LIBXSMM_GEMM_PREFETCH_A(a),
              LIBXSMM_GEMM_PREFETCH_B(b),
              LIBXSMM_GEMM_PREFETCH_C(c + j));
#else
            xmm(a, b, c + j);
#endif
          }
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2 * k - 1) * (double)(s * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /* fallthrough */

      case 9: { // indirect cached
        fprintf(stdout, "Indirect cached...\n");
#if defined(_OPENMP)
#       pragma omp parallel for schedule(static)
#endif
        for (libxsmm_blasint i = 0; i < s; ++i) {
          a_array[i] = a; b_array[i] = b;
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
          c_array[i] = d + omp_get_thread_num() * chunksize * csize;
#else
          c_array[i] = d;
#endif
        }
        const libxsmm_blasint ptrsize = sizeof(void*);
        const unsigned long long start = libxsmm_timer_tick();
        for (libxsmm_blasint r = 0; r < nrepeat; ++r) {
          libxsmm_gemm_batch2_omp(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(OTYPE), &transa, &transb,
            m, n, k, &alpha, &a_array[0], &lda, &b_array[0], &ldb, &beta, &c_array[0], &ldc,
            0/*index_base*/, 0/*index_stride*/, &ptrsize, &ptrsize, &ptrsize, s);
        }
        const unsigned long long ncycles = libxsmm_timer_diff(start, libxsmm_timer_tick());
        const double duration = libxsmm_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2 * k - 1) * (double)(s * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } break;
      default: throw "invalid case selected!";
      } /*switch*/

      if (0 != check) {
        libxsmm_matdiff_info diff;
        result = libxsmm_matdiff(LIBXSMM_DATATYPE(OTYPE), m, n, 0 == (benchmark & 1) ? c : d, NULL, &ldc, &ldc, &diff);
        if (EXIT_SUCCESS == result) {
          fprintf(stdout, "\tcheck: %f\n", diff.l1_ref);
        }
      }
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

