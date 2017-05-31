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
#elif defined(__OPENBLAS)
# include <openblas/f77blas.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(ELEM_TYPE)
# define ELEM_TYPE float
#endif

#if !defined(OTRANS1)
# if (defined(__BLAS) && 1 < (__BLAS))
#   define OTRANS1 libxsmm_otrans_omp
# else
#   define OTRANS1 libxsmm_otrans
# endif
#endif
#if !defined(ITRANS1)
# define ITRANS1 libxsmm_itrans
#endif

#if !defined(USE_SELF_VALIDATION) && !defined(__BLAS) || (0 != __BLAS)
# if !LIBXSMM_EQUAL(ELEM_TYPE, float) && !LIBXSMM_EQUAL(ELEM_TYPE, double)
#   define USE_SELF_VALIDATION
# endif
#endif

#if !defined(USE_SELF_VALIDATION)
# if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
#   if !defined(OTRANS2)
#     define OTRANS2(TC, TT, M, N, ALPHA, A, LDI, B, LDO) \
        LIBXSMM_CONCATENATE(mkl_, LIBXSMM_TPREFIX(ELEM_TYPE, omatcopy)) \
        (*(TC), *(TT), *(M), *(N), (ELEM_TYPE)(*(ALPHA)), A, *(LDI), B, *(LDO))
#   endif
#   if !defined(ITRANS2)
#     define ITRANS2(TC, TT, M, N, ALPHA, A, LDI, LDO) \
        LIBXSMM_CONCATENATE(mkl_, LIBXSMM_TPREFIX(ELEM_TYPE, imatcopy)) \
        (*(TC), *(TT), *(M), *(N), (ELEM_TYPE)(*(ALPHA)), A, *(LDI), *(LDO))
#   endif
# elif defined(__OPENBLAS)
#   if !defined(OTRANS2)
#     define OTRANS2(TC, TT, M, N, ALPHA, A, LDI, B, LDO) \
        LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(ELEM_TYPE, omatcopy)) \
        ((char*)(TC), (char*)(TT), (libxsmm_blasint*)(M), (libxsmm_blasint*)(N), \
          (ELEM_TYPE*)(ALPHA), A, (libxsmm_blasint*)(LDI), B, (libxsmm_blasint*)(LDO))
#   endif
#   if !defined(ITRANS2)
#     define ITRANS2(TC, TT, M, N, ALPHA, A, LDI, LDO) \
        LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(ELEM_TYPE, imatcopy)) \
        ((char*)(TC), (char*)(TT), (libxsmm_blasint*)(M), (libxsmm_blasint*)(N), \
          (ELEM_TYPE*)(ALPHA), A, (libxsmm_blasint*)(LDI), (libxsmm_blasint*)(LDO))
#   endif
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
    const char tc = 'C', tt = 'T';
    const double alpha = 1;
    double duration2 = 0;
#endif
    double duration = 0;
    unsigned int size = 0;
    unsigned long long start;
    libxsmm_blasint i = 0, j;
#if defined(MKL_ENABLE_AVX512)
    mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
    fprintf(stdout, "m=%i n=%i ldi=%i ldo=%i size=%.fMB (%s, %s)\n", m, n, ldi, ldo,
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
          OTRANS1(b, a, sizeof(ELEM_TYPE), km, kn, kldi, kldo);
        }
        else {
#if 0 /* TODO: enable when in-place transpose is fully supported */
          kn = randstart(lower, n);
#else
          kn = km;
#endif
          kldo = kldi;
          /* trigger JIT-generated code */
          ITRANS1(b, sizeof(ELEM_TYPE), km, kn, kldi);
        }
      }
      size += km * kn * sizeof(ELEM_TYPE);

      if (('o' == t || 'O' == t)) {
        if (0 == tasks) { /* library-internal parallelization */
          start = libxsmm_timer_tick();
          result = OTRANS1(b, a, sizeof(ELEM_TYPE), km, kn, kldi, kldo);
          duration += libxsmm_timer_duration(start, libxsmm_timer_tick());
        }
        else { /* external parallelization */
          start = libxsmm_timer_tick();
#if defined(_OPENMP)
#         pragma omp parallel
#         pragma omp single nowait
#endif
          result = OTRANS1(b, a, sizeof(ELEM_TYPE), km, kn, kldi, kldo);
          duration += libxsmm_timer_duration(start, libxsmm_timer_tick());
        }
#if !defined(USE_SELF_VALIDATION)
        if (0 != c) { /* check */
          start = libxsmm_timer_tick();
          OTRANS2(&tc, &tt, &km, &kn, &alpha, a, &kldi, c, &kldo);
          duration2 += libxsmm_timer_duration(start, libxsmm_timer_tick());
        }
#endif
      }
      else {
        assert(('i' == t || 'I' == t) && kldo == kldi);
        memcpy(b, a, kldi * kn * sizeof(ELEM_TYPE));

        if (2 > tasks) { /* library-internal parallelization */
          start = libxsmm_timer_tick();
          result = ITRANS1(b, sizeof(ELEM_TYPE), km, kn, kldi);
          duration += libxsmm_timer_duration(start, libxsmm_timer_tick());
        }
        else { /* external parallelization */
          start = libxsmm_timer_tick();
#if defined(_OPENMP)
#         pragma omp parallel
#         pragma omp single
#endif
          result = ITRANS1(b, sizeof(ELEM_TYPE), km, kn, kldi);
          duration += libxsmm_timer_duration(start, libxsmm_timer_tick());
        }
#if !defined(USE_SELF_VALIDATION)
        if (0 != c) { /* check */
          memcpy(c, a, kldi * kn * sizeof(ELEM_TYPE));
          start = libxsmm_timer_tick();
          ITRANS2(&tc, &tt, &km, &kn, &alpha, c, &kldi, &kldo);
          duration2 += libxsmm_timer_duration(start, libxsmm_timer_tick());
        }
#endif
      }

      if ((0 == env_check || 0 != atoi(env_check))) { /* check */
        for (i = 0; i < km; ++i) {
          for (j = 0; j < kn; ++j) {
            const ELEM_TYPE u = b[i*kldo+j];
#if defined(USE_SELF_VALIDATION)
            const ELEM_TYPE v = a[j*kldi+i];
#else /* check against an alternative/external implementation */
            const ELEM_TYPE v = c[i*kldo+j];
#endif
            if (0 == LIBXSMM_FEQ(u, v)) {
              i += km; /* leave outer loop as well */
              result = EXIT_FAILURE;
              break;
            }
          }
        }
      }
    }

    if (EXIT_SUCCESS == result) {
      if (0 < duration) {
        fprintf(stdout, "\tbandwidth: %.1f GB/s\n", size / (duration * (1 << 30)));
      }
      fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
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
        "validation failed for m=%i, n=%i, ldi=%i, and ldo=%i!\n", km, kn, kldi, kldo);
    }

    libxsmm_free(a);
    libxsmm_free(b);
#if !defined(USE_SELF_VALIDATION)
    libxsmm_free(c);
#endif
  }
  return result;
}

