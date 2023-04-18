/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm_source.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

#if !defined(ELEMTYPE)
# define ELEMTYPE double
#endif

#if !defined(RAND_SEED)
# define RAND_SEED 25071975
#endif

#if !defined(BATCH_SIZE)
# define BATCH_SIZE 100
#endif

#if (defined(__BLAS) && 1 < (__BLAS))
# if !defined(OTRANS_THREAD) && defined(_OPENMP) && 0
#   define OTRANS_THREAD libxsmm_otrans_task
# endif
# define OTRANS libxsmm_otrans_omp
#else
# define OTRANS libxsmm_otrans
#endif
#define ITRANS libxsmm_itrans

#if defined(__BLAS) && (0 != __BLAS) && \
  (LIBXSMM_EQUAL(ELEMTYPE, float) || LIBXSMM_EQUAL(ELEMTYPE, double))
# if defined(__MKL)
#   include <mkl_trans.h>
#   define OTRANS_GOLD(M, N, A, LDI, B, LDO) \
      LIBXSMM_CONCATENATE(mkl_, LIBXSMM_TPREFIX(ELEMTYPE, omatcopy))('C', 'T', \
        (size_t)(*(M)), (size_t)(*(N)), (ELEMTYPE)1, A, (size_t)(*(LDI)), B, (size_t)(*(LDO)))
#   define ITRANS_GOLD(M, N, A, LDI, LDO) \
      LIBXSMM_CONCATENATE(mkl_, LIBXSMM_TPREFIX(ELEMTYPE, imatcopy))('C', 'T', \
        (size_t)(*(M)), (size_t)(*(N)), (ELEMTYPE)1, A, (size_t)(*(LDI)), (size_t)(*(LDO)))
#   if !defined(USE_REFERENCE)
#     define USE_REFERENCE
#   endif
# elif defined(__OPENBLAS77) && 0/* issue #390 */
#   include <f77blas.h>
#   define OTRANS_GOLD(M, N, A, LDI, B, LDO) { \
      /*const*/char otrans_gold_tc_ = 'C', otrans_gold_tt_ = 'T'; \
      /*const*/ELEMTYPE otrans_gold_alpha_ = 1; \
      LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(ELEMTYPE, omatcopy))(&otrans_gold_tc_, &otrans_gold_tt_, \
        (libxsmm_blasint*)(M), (libxsmm_blasint*)(N), &otrans_gold_alpha_, A, \
        (libxsmm_blasint*)(LDI), B, (libxsmm_blasint*)(LDO)); \
    }
#   define ITRANS_GOLD(M, N, A, LDI, LDO) { \
      /*const*/char itrans_gold_tc_ = 'C', itrans_gold_tt_ = 'T'; \
      /*const*/ELEMTYPE itrans_gold_alpha_ = 1; \
      LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(ELEMTYPE, imatcopy))(&itrans_gold_tc_, &itrans_gold_tt_, \
        (libxsmm_blasint*)(M), (libxsmm_blasint*)(N), &itrans_gold_alpha_, A, \
        (libxsmm_blasint*)(LDI), (libxsmm_blasint*)(LDO)); \
    }
#   if !defined(USE_REFERENCE)
#     define USE_REFERENCE
#   endif
# endif
#endif


LIBXSMM_INLINE ELEMTYPE initial_value(libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ld)
{
  return (ELEMTYPE)i * ld + j;
}


LIBXSMM_INLINE libxsmm_blasint randstart(libxsmm_blasint start, libxsmm_blasint value)
{
  const libxsmm_blasint s = (start < value ? start : 0), r = LIBXSMM_MIN(s + (rand() % (value - s)) + 1, value);
  assert(0 < r && s <= r && r <= value);
  return r;
}


#if !defined(USE_REFERENCE)
LIBXSMM_INLINE void matrix_transpose(ELEMTYPE *LIBXSMM_RESTRICT dst, const ELEMTYPE *LIBXSMM_RESTRICT src, libxsmm_blasint rows, libxsmm_blasint cols)
{
  libxsmm_blasint i, j;
  LIBXSMM_VLA_DECL(2, const ELEMTYPE, src_2d, src, cols);
  LIBXSMM_VLA_DECL(2, ELEMTYPE, dst_2d, dst, rows);
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(i); LIBXSMM_OMP_VAR(j);
# pragma omp parallel for private(i, j)
#endif
  for (i = 0; i < rows; ++i) {
    for (j = 0; j < cols; ++j) {
      LIBXSMM_VLA_ACCESS(2, dst_2d, j, i, rows) = LIBXSMM_VLA_ACCESS(2, src_2d, i, j, cols);
    }
  }
}
#endif


int main(int argc, char* argv[])
{
  const char t = (char)(1 < argc ? *argv[1] : 'o');
  const libxsmm_blasint m = (2 < argc ? atoi(argv[2]) : 4096);
  const libxsmm_blasint n = (3 < argc ? atoi(argv[3]) : m);
  const libxsmm_blasint ldi = LIBXSMM_MAX/*sanitize ld*/(4 < argc ? atoi(argv[4]) : 0, m);
  const libxsmm_blasint ldo = LIBXSMM_MAX/*sanitize ld*/(5 < argc ? atoi(argv[5]) : 0,
    ('o' == t || 'O' == t) ? n : LIBXSMM_MAX(n, ldi));
  const int r = (6 < argc ? atoi(argv[6]) : 0), s = LIBXSMM_ABS(r);
  const libxsmm_blasint lower = (7 < argc ? atoi(argv[7]) : 0);
  libxsmm_blasint km = m, kn = n, kldi = ldi, kldo = ldo;
  int result = EXIT_SUCCESS, k;

  if (0 != strchr("oOiI", t)) {
    const char *const env_tasks = getenv("TASKS"), *const env_check = getenv("CHECK");
    const int tasks = (NULL == env_tasks || 0 == *env_tasks) ? 0/*default*/ : atoi(env_tasks);
    const int check = (NULL == env_check || 0 == *env_check) ? 1/*default*/ : atoi(env_check);
    ELEMTYPE *const a = (ELEMTYPE*)libxsmm_malloc((size_t)ldi * (size_t)(('o' == t || 'O' == t) ? n : ldo) * sizeof(ELEMTYPE));
    ELEMTYPE *const b = (ELEMTYPE*)libxsmm_malloc((size_t)ldo * (size_t)(('o' == t || 'O' == t) ? m : ldi) * sizeof(ELEMTYPE));
    libxsmm_timer_tickint start, duration = 0, duration2 = 0;
    libxsmm_blasint i;
    size_t size = 0;

    fprintf(stdout, "m=%lli n=%lli ldi=%lli ldo=%lli size=%.fMB (%s, %s)\n",
      (long long)m, (long long)n, (long long)ldi, (long long)ldo,
      1.0 * (sizeof(ELEMTYPE) * m * n) / (1ULL << 20), LIBXSMM_STRINGIFY(ELEMTYPE),
      ('o' == t || 'O' == t) ? "out-of-place" : "in-place");

#if defined(_OPENMP)
    LIBXSMM_OMP_VAR(i);
#   pragma omp parallel for private(i)
#endif
    for (i = 0; i < n; ++i) {
      libxsmm_blasint j;
      for (j = 0; j < m; ++j) {
        a[i*ldi+j] = initial_value(i, j, m);
      }
    }

    if (0 != check) { /* repeatable (reference) */
      srand(RAND_SEED);
    }
    else { /* randomized selection */
      srand(libxsmm_timer_tick() % ((unsigned int)-1));
    }
    for (k = (0 == r ? -1 : 0); k < s && EXIT_SUCCESS == result; ++k) {
      if (0 < r) {
        const libxsmm_blasint rldi = 0 <= lower ? randstart(lower, ldi) : 0;
        const libxsmm_blasint rldo = 0 <= lower ? randstart(lower, ldo) : 0;
        km = randstart(LIBXSMM_ABS(lower), m);
        kldi = LIBXSMM_MAX(rldi, km);
        kn = randstart(LIBXSMM_ABS(lower), n);
        kldo = LIBXSMM_MAX(rldo, kn);
        /* warmup: trigger JIT-generated code */
        if ('o' == t || 'O' == t) {
          OTRANS(b, a, sizeof(ELEMTYPE), km, kn, kldi, kldo);
        }
        else {
          ITRANS(b, sizeof(ELEMTYPE), km, kn, kldi, kldo);
        }
      }
      size += (size_t)(sizeof(ELEMTYPE) * km * kn);

      if ('o' == t || 'O' == t) {
#if !defined(USE_REFERENCE)
        kldi = km; kldo = kn;
#endif
        if (0 == tasks) { /* library-internal parallelization */
          start = libxsmm_timer_tick();
#if defined(OTRANS_THREAD)
#         pragma omp parallel
          OTRANS_THREAD(b, a, sizeof(ELEMTYPE), km, kn, kldi, kldo, omp_get_thread_num(), omp_get_num_threads());
#else
          OTRANS(b, a, sizeof(ELEMTYPE), km, kn, kldi, kldo);
#endif
          duration += libxsmm_timer_ncycles(start, libxsmm_timer_tick());
        }
        else { /* external parallelization */
          start = libxsmm_timer_tick();
#if defined(_OPENMP)
#         pragma omp parallel
#         pragma omp single nowait
#endif
          OTRANS(b, a, sizeof(ELEMTYPE), km, kn, kldi, kldo);
          duration += libxsmm_timer_ncycles(start, libxsmm_timer_tick());
        }
      }
      else {
        assert(('i' == t || 'I' == t));
        memcpy(b, a, (size_t)(sizeof(ELEMTYPE) * kldi * kn));

        if (2 > tasks) { /* library-internal parallelization */
          start = libxsmm_timer_tick();
          ITRANS(b, sizeof(ELEMTYPE), km, kn, kldi, kldo);
          duration += libxsmm_timer_ncycles(start, libxsmm_timer_tick());
        }
        else { /* external parallelization */
          start = libxsmm_timer_tick();
#if defined(_OPENMP)
#         pragma omp parallel
#         pragma omp single
#endif
          ITRANS(b, sizeof(ELEMTYPE), km, kn, kldi, kldo);
          duration += libxsmm_timer_ncycles(start, libxsmm_timer_tick());
        }
      }
      if (0 != check) { /* check */
        for (i = 0; i < km; ++i) {
          libxsmm_blasint j;
          for (j = 0; j < kn; ++j) {
            const ELEMTYPE u = b[i*kldo+j];
            const ELEMTYPE v = a[j*kldi+i];
            if (LIBXSMM_NEQ(u, v)) {
              i += km; /* leave outer loop as well */
              result = EXIT_FAILURE;
              break;
            }
          }
        }
      }
    }

    if (0 < check) { /* check shall imply reference (performance-)test */
      srand(RAND_SEED); /* reproduce the same sequence as above */
      for (k = (0 == r ? -1 : 0); k < s && EXIT_SUCCESS == result; ++k) {
        if (0 < r) {
          const libxsmm_blasint rldi = 0 <= lower ? randstart(lower, ldi) : 0;
          km = randstart(LIBXSMM_ABS(lower), m);
          kldi = LIBXSMM_MAX(rldi, km);
          if ('o' == t || 'O' == t) {
            const libxsmm_blasint rldo = 0 <= lower ? randstart(lower, ldo) : 0;
            kn = randstart(LIBXSMM_ABS(lower), n);
            kldo = LIBXSMM_MAX(rldo, kn);
          }
          else {
            kn = randstart(LIBXSMM_ABS(lower), n);
            kldo = kldi;
          }
        }

        if ('o' == t || 'O' == t) {
#if defined(USE_REFERENCE)
          start = libxsmm_timer_tick();
          OTRANS_GOLD(&km, &kn, a, &kldi, b, &kldo);
#else
          kldi = km; kldo = kn;
          start = libxsmm_timer_tick();
          matrix_transpose(b, a, km, kn);
#endif
          duration2 += libxsmm_timer_ncycles(start, libxsmm_timer_tick());
        }
        else {
          assert(('i' == t || 'I' == t));
#if defined(USE_REFERENCE)
          memcpy(b, a, (size_t)(kldi * kn * sizeof(ELEMTYPE)));
          start = libxsmm_timer_tick();
          ITRANS_GOLD(&km, &kn, b, &kldi, &kldo);
          duration2 += libxsmm_timer_ncycles(start, libxsmm_timer_tick());
#else
          fprintf(stderr, "Warning: no validation routine available!\n");
          continue;
#endif
        }
        if (1 < check || 0 > check) { /* check */
          for (i = 0; i < km; ++i) {
            libxsmm_blasint j;
            for (j = 0; j < kn; ++j) {
              const ELEMTYPE u = b[i*kldo+j];
              const ELEMTYPE v = a[j*kldi+i];
              if (LIBXSMM_NEQ(u, v)) {
                i += km; /* leave outer loop as well */
                result = EXIT_FAILURE;
                break;
              }
            }
          }
        }
      }
    }
    if (EXIT_SUCCESS == result) {
      const double d = libxsmm_timer_duration(0, duration), mbyte = (1U << 30);
      const size_t bwsize = size * ((('o' == t || 'O' == t)) ? 3 : 2);
      if (0 < duration) {
        /* out-of-place transpose bandwidth assumes RFO */
        fprintf(stdout, "\tbandwidth: %.2f GB/s\n", bwsize / (d * mbyte));
#if defined(BATCH_SIZE) && (0 < (BATCH_SIZE))
        if (0 >= r && ('i' == t || 'I' == t) &&
          /* limit evaluation to batches of small matrices */
          km <= LIBXSMM_CONFIG_MAX_DIM && kn <= LIBXSMM_CONFIG_MAX_DIM)
        {
          double dbatch;
          start = libxsmm_timer_tick();
          libxsmm_itrans_batch(b, sizeof(ELEMTYPE), km, kn, kldi, kldo,
            0/*index_base*/, 0/*index_stride*/, NULL/*stride*/,
            BATCH_SIZE, 0/*tid*/, 1/*ntasks*/);
          dbatch = libxsmm_timer_duration(start, libxsmm_timer_tick());
          if (0 < dbatch) {
            fprintf(stdout, "\tbatch: %.1f GB/s\n", bwsize * BATCH_SIZE / (dbatch * mbyte));
          }
        }
#endif
      }
      if (0 == lower) {
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * (d / (0 == r ? (s + 1) : s)));
      }
      else {
        fprintf(stdout, "\tduration: %f ms\n", 1000.0 * d);
      }
      if (0 < duration2) {
        fprintf(stdout, "\treference: %.1fx\n", (1.0 * duration) / duration2);
      }
    }
    else if (0 != check) { /* check */
      fprintf(stderr, "Error: validation failed for m=%lli, n=%lli, ldi=%lli, and ldo=%lli!\n",
        (long long)km, (long long)kn, (long long)kldi, (long long)kldo);
    }

    libxsmm_free(a);
    libxsmm_free(b);
  }
  else {
    fprintf(stderr, "%s [<transpose-kind:o|i>] [<m>] [<n>] [<ld-in>] [<ld-out>] [random:0|nruns] [lbound]\n", argv[0]);
    result = EXIT_FAILURE;
  }

  return result;
}

