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
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm_source.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl_trans.h>
# include <mkl_service.h>
#elif defined(__OPENBLAS77)
# include <openblas/f77blas.h>
#endif
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(ELEM_TYPE)
# define ELEM_TYPE double
#endif

#if (defined(_OPENMP) || (defined(__BLAS) && 1 < (__BLAS)))
# if !defined(OTRANS_THREAD) && defined(_OPENMP) && 0
#   define OTRANS_THREAD libxsmm_otrans_thread
# endif
# define OTRANS libxsmm_otrans_omp
#else
# define OTRANS libxsmm_otrans
#endif
#define ITRANS libxsmm_itrans

#if !defined(USE_SELF_VALIDATION) && \
  ((!LIBXSMM_EQUAL(ELEM_TYPE, float) && !LIBXSMM_EQUAL(ELEM_TYPE, double)) || (0 == __BLAS))
# define USE_SELF_VALIDATION
#endif

#if !defined(USE_SELF_VALIDATION)
# if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
#   define OTRANS_GOLD(M, N, A, LDI, B, LDO) \
      LIBXSMM_CONCATENATE(mkl_, LIBXSMM_TPREFIX(ELEM_TYPE, omatcopy))('C', 'T', \
        *(M), *(N), (ELEM_TYPE)1, A, *(LDI), B, *(LDO))
#   define ITRANS_GOLD(M, N, A, LDI, LDO) \
      LIBXSMM_CONCATENATE(mkl_, LIBXSMM_TPREFIX(ELEM_TYPE, imatcopy))('C', 'T', \
        *(M), *(N), (ELEM_TYPE)1, A, *(LDI), *(LDO))
# elif defined(__OPENBLAS)
#   define OTRANS_GOLD(M, N, A, LDI, B, LDO) { \
      /*const*/char otrans_gold_tc_ = 'C', otrans_gold_tt_ = 'T'; \
      /*const*/ELEM_TYPE otrans_gold_alpha_ = 1; \
      LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(ELEM_TYPE, omatcopy))(&otrans_gold_tc_, &otrans_gold_tt_, \
        (libxsmm_blasint*)(M), (libxsmm_blasint*)(N), &otrans_gold_alpha_, A, \
        (libxsmm_blasint*)(LDI), B, (libxsmm_blasint*)(LDO)); \
    }
#   define ITRANS_GOLD(M, N, A, LDI, LDO) { \
      /*const*/char itrans_gold_tc_ = 'C', itrans_gold_tt_ = 'T'; \
      /*const*/ELEM_TYPE itrans_gold_alpha_ = 1; \
      LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(ELEM_TYPE, imatcopy))(&itrans_gold_tc_, &itrans_gold_tt_, \
        (libxsmm_blasint*)(M), (libxsmm_blasint*)(N), &itrans_gold_alpha_, A, \
        (libxsmm_blasint*)(LDI), (libxsmm_blasint*)(LDO)); \
    }
# else
#   define USE_SELF_VALIDATION
# endif
#endif


LIBXSMM_INLINE LIBXSMM_RETARGETABLE ELEM_TYPE initial_value(libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ld)
{
  return (ELEM_TYPE)(i * ld + j);
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE libxsmm_blasint randstart(libxsmm_blasint start, libxsmm_blasint value)
{
  return (rand() % LIBXSMM_MAX(value - start, 1)) + LIBXSMM_MAX(start, 1);
}


int main(int argc, char* argv[])
{
  const char t = (char)(1 < argc ? *argv[1] : 'o');
  const libxsmm_blasint m = (2 < argc ? atoi(argv[2]) : 4096);
#if 0 /* TODO: enable when in-place transpose is fully supported */
  const libxsmm_blasint n = (3 < argc ? atoi(argv[3]) : m);
#else
  const libxsmm_blasint n = (3 < argc ? (('o' == t || 'O' == t) ? atoi(argv[3]) : m) : m);
#endif
  const libxsmm_blasint ldi = LIBXSMM_MAX/*sanitize ld*/(4 < argc ? atoi(argv[4]) : 0, m);
  const libxsmm_blasint ldo = LIBXSMM_MAX/*sanitize ld*/(5 < argc ? atoi(argv[5]) : 0, n);
  const int r = (6 < argc ? atoi(argv[6]) : 0), s = LIBXSMM_ABS(r);
  const libxsmm_blasint lower = LIBXSMM_MAX(7 < argc ? atoi(argv[7]) : 0, 0);
  libxsmm_blasint km = m, kn = n, kldi = ldi, kldo = (('o' == t || 'O' == t) ? ldo : ldi);
  int result = EXIT_SUCCESS, k;

  if (0 == strchr("oOiI", t)) {
    fprintf(stderr, "%s [<transpose-kind:o|i>] [<m>] [<n>] [<ld-in>] [<ld-out>] [random:0|nruns] [lbound]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
  {
    const char *const env_tasks = getenv("TASKS"), *const env_check = getenv("CHECK");
    const int tasks = (0 == env_tasks || 0 == *env_tasks) ? 0/*default*/ : atoi(env_tasks);
    ELEM_TYPE *const a = (ELEM_TYPE*)libxsmm_malloc(ldi * (('o' == t || 'O' == t) ? n : ldo) * sizeof(ELEM_TYPE));
    ELEM_TYPE *const b = (ELEM_TYPE*)libxsmm_malloc(ldo * (('o' == t || 'O' == t) ? m : ldi) * sizeof(ELEM_TYPE));
#if !defined(USE_SELF_VALIDATION) /* check against an alternative/external implementation */
    ELEM_TYPE *const c = (ELEM_TYPE*)((0 == env_check || 0 != atoi(env_check))
      ? libxsmm_malloc(ldo * (('o' == t || 'O' == t) ? m : ldi) * sizeof(ELEM_TYPE))
      : 0);
    double duration2 = 0;
#endif
    double duration = 0;
    unsigned long long start;
    libxsmm_blasint i = 0, j;
    size_t size = 0;
#if defined(MKL_ENABLE_AVX512)
    mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
    fprintf(stdout, "m=%lli n=%lli ldi=%lli ldo=%lli size=%.fMB (%s, %s)\n",
      (long long)m, (long long)n, (long long)ldi, (long long)ldo,
      1.0 * (m * n * sizeof(ELEM_TYPE)) / (1 << 20), LIBXSMM_STRINGIFY(ELEM_TYPE),
      ('o' == t || 'O' == t) ? "out-of-place" : "in-place");

    for (i = 0; i < n; ++i) {
      for (j = 0; j < m; ++j) {
        a[i*ldi+j] = initial_value(i, j, m);
      }
    }

    for (k = (0 == r ? -1 : 0); k < s && EXIT_SUCCESS == result; ++k) {
      if (0 < r) {
        const libxsmm_blasint rldi = randstart(lower, ldi);
        km = randstart(lower, m);
        kldi = LIBXSMM_MAX(rldi, km);
        if (('o' == t || 'O' == t)) {
          const libxsmm_blasint rldo = randstart(lower, ldo);
          kn = randstart(lower, n);
          kldo = LIBXSMM_MAX(rldo, kn);
          /* trigger JIT-generated code */
          OTRANS(b, a, sizeof(ELEM_TYPE), km, kn, kldi, kldo);
        }
        else {
#if 0 /* TODO: enable when in-place transpose is fully supported */
          kn = randstart(lower, n);
#else
          kn = km;
#endif
          kldo = kldi;
          /* trigger JIT-generated code */
          ITRANS(b, sizeof(ELEM_TYPE), km, kn, kldi);
        }
      }
      size += km * kn * sizeof(ELEM_TYPE);

      if (('o' == t || 'O' == t)) {
        if (0 == tasks) { /* library-internal parallelization */
          start = libxsmm_timer_tick();
#if defined(OTRANS_THREAD)
#         pragma omp parallel
          OTRANS_THREAD(b, a, sizeof(ELEM_TYPE), km, kn, kldi, kldo, omp_get_thread_num(), omp_get_num_threads());
#else
          result = OTRANS(b, a, sizeof(ELEM_TYPE), km, kn, kldi, kldo);
#endif
          duration += libxsmm_timer_duration(start, libxsmm_timer_tick());
        }
        else { /* external parallelization */
          start = libxsmm_timer_tick();
#if defined(_OPENMP)
#         pragma omp parallel
#         pragma omp single nowait
#endif
          result = OTRANS(b, a, sizeof(ELEM_TYPE), km, kn, kldi, kldo);
          duration += libxsmm_timer_duration(start, libxsmm_timer_tick());
        }
#if !defined(USE_SELF_VALIDATION)
        if (0 != c) { /* check */
          start = libxsmm_timer_tick();
          OTRANS_GOLD(&km, &kn, a, &kldi, c, &kldo);
          duration2 += libxsmm_timer_duration(start, libxsmm_timer_tick());
        }
#endif
      }
      else {
        assert(('i' == t || 'I' == t) && kldo == kldi);
        memcpy(b, a, kldi * kn * sizeof(ELEM_TYPE));

        if (2 > tasks) { /* library-internal parallelization */
          start = libxsmm_timer_tick();
          result = ITRANS(b, sizeof(ELEM_TYPE), km, kn, kldi);
          duration += libxsmm_timer_duration(start, libxsmm_timer_tick());
        }
        else { /* external parallelization */
          start = libxsmm_timer_tick();
#if defined(_OPENMP)
#         pragma omp parallel
#         pragma omp single
#endif
          result = ITRANS(b, sizeof(ELEM_TYPE), km, kn, kldi);
          duration += libxsmm_timer_duration(start, libxsmm_timer_tick());
        }
#if !defined(USE_SELF_VALIDATION)
        if (0 != c) { /* check */
          memcpy(c, a, kldi * kn * sizeof(ELEM_TYPE));
          start = libxsmm_timer_tick();
          ITRANS_GOLD(&km, &kn, c, &kldi, &kldo);
          duration2 += libxsmm_timer_duration(start, libxsmm_timer_tick());
        }
#endif
      }
#if defined(USE_SELF_VALIDATION)
      if (0 == env_check || 0 != atoi(env_check)) { /* check */
        for (i = 0; i < km; ++i) {
          for (j = 0; j < kn; ++j) {
            const ELEM_TYPE u = b[i*kldo+j];
            const ELEM_TYPE v = a[j*kldi+i];
            if (0 == LIBXSMM_FEQ(u, v)) {
              i += km; /* leave outer loop as well */
              result = EXIT_FAILURE;
              break;
            }
          }
        }
      }
#endif
    }

    if (EXIT_SUCCESS == result) {
      if (0 < duration) {
        /* out-of-place transpose bandwidth assumes RFO */
        fprintf(stdout, "\tbandwidth: %.1f GB/s\n", size
          * ((('o' == t || 'O' == t)) ? 3 : 2) / (duration * (1 << 30)));
      }
      fprintf(stdout, "\tduration: %.0f ms\n", 1000.0
          * (0 == lower ? (duration / (0 == r ? (s + 1) : s)) : duration));
#if !defined(USE_SELF_VALIDATION)
      if (0 < duration2 && 0 != c) {
        fprintf(stdout, "\treference: %.1fx\n", duration / duration2);
      }
#endif
    }
    else if ((0 == env_check || 0 != atoi(env_check))) { /* check */
      fprintf(stderr, "Error: "
#if defined(USE_SELF_VALIDATION)
        "self-"
#endif
        "validation failed for m=%lli, n=%lli, ldi=%lli, and ldo=%lli!\n",
          (long long)km, (long long)kn, (long long)kldi, (long long)kldo);
    }

    libxsmm_free(a);
    libxsmm_free(b);
#if !defined(USE_SELF_VALIDATION)
    libxsmm_free(c);
#endif
  }
  return result;
}

