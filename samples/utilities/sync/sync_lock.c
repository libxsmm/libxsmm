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
#include <utils/libxsmm_timer.h>
#include <libxsmm.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

/* measure non-contended latency of RO-lock */
#define MEASURE_LATENCY_RO(LOCK_KIND, LOCKPTR, NREPEAT, NR) do { \
  libxsmm_timer_tickint latency = 0; \
  double duration; \
  int i; \
  for (i = 0; i < (NREPEAT) / 4; ++i) { \
    const libxsmm_timer_tickint tick = libxsmm_timer_tick(); \
    LIBXSMM_LOCK_ACQREAD(LOCK_KIND, LOCKPTR); \
    LIBXSMM_LOCK_RELREAD(LOCK_KIND, LOCKPTR); \
    LIBXSMM_LOCK_ACQREAD(LOCK_KIND, LOCKPTR); \
    LIBXSMM_LOCK_RELREAD(LOCK_KIND, LOCKPTR); \
    LIBXSMM_LOCK_ACQREAD(LOCK_KIND, LOCKPTR); \
    LIBXSMM_LOCK_RELREAD(LOCK_KIND, LOCKPTR); \
    LIBXSMM_LOCK_ACQREAD(LOCK_KIND, LOCKPTR); \
    LIBXSMM_LOCK_RELREAD(LOCK_KIND, LOCKPTR); \
    latency += libxsmm_timer_ncycles(tick, libxsmm_timer_tick()); \
  } \
  duration = libxsmm_timer_duration(0, latency); \
  if (0 < duration) { \
    printf("\tro-latency: %.0f ns (call/s %.0f MHz, %.0f cycles)\n", \
      duration * (NR) * 1e9, (NREPEAT) / (1e6 * duration), latency * (NR)); \
  } \
} while(0)

/* measure non-contended latency of RW-lock */
#define MEASURE_LATENCY_RW(LOCK_KIND, LOCKPTR, NREPEAT, NR) do { \
  libxsmm_timer_tickint latency = 0; \
  double duration; \
  int i; \
  for (i = 0; i < (NREPEAT) / 4; ++i) { \
    const libxsmm_timer_tickint tick = libxsmm_timer_tick(); \
    LIBXSMM_LOCK_ACQUIRE(LOCK_KIND, LOCKPTR); \
    LIBXSMM_LOCK_RELEASE(LOCK_KIND, LOCKPTR); \
    LIBXSMM_LOCK_ACQUIRE(LOCK_KIND, LOCKPTR); \
    LIBXSMM_LOCK_RELEASE(LOCK_KIND, LOCKPTR); \
    LIBXSMM_LOCK_ACQUIRE(LOCK_KIND, LOCKPTR); \
    LIBXSMM_LOCK_RELEASE(LOCK_KIND, LOCKPTR); \
    LIBXSMM_LOCK_ACQUIRE(LOCK_KIND, LOCKPTR); \
    LIBXSMM_LOCK_RELEASE(LOCK_KIND, LOCKPTR); \
    latency += libxsmm_timer_ncycles(tick, libxsmm_timer_tick()); \
  } \
  duration = libxsmm_timer_duration(0, latency); \
  if (0 < duration) { \
    printf("\trw-latency: %.0f ns (call/s %.0f MHz, %.0f cycles)\n", \
      duration * (NR) * 1e9, (NREPEAT) / (1e6 * duration), latency * (NR)); \
  } \
} while(0)

#if defined(_OPENMP)
# define MEASURE_THROUGHPUT_PARALLEL(NTHREADS) LIBXSMM_PRAGMA(omp parallel num_threads(NTHREADS))
# define MEASURE_THROUGHPUT_ATOMIC LIBXSMM_PRAGMA(omp atomic)
#else
# define MEASURE_THROUGHPUT_PARALLEL(NTHREADS)
# define MEASURE_THROUGHPUT_ATOMIC
#endif

#define MEASURE_THROUGHPUT(LOCK_KIND, LOCKPTR, NREPEAT, NTHREADS, WORK_R, WORK_W, NW, NT) do { \
  libxsmm_timer_tickint throughput = 0; \
  double duration; \
  MEASURE_THROUGHPUT_PARALLEL(NTHREADS) \
  { \
    int n, nn; \
    libxsmm_timer_tickint t1, t2, d = 0; \
    const libxsmm_timer_tickint t0 = libxsmm_timer_tick(); \
    for (n = 0; n < (NREPEAT); n = nn) { \
      nn = n + 1; \
      if (0 != (nn % (NW))) { /* read */ \
        LIBXSMM_LOCK_ACQREAD(LOCK_KIND, LOCKPTR); \
        t1 = libxsmm_timer_tick(); \
        t2 = work(t1, WORK_R); \
        LIBXSMM_LOCK_RELREAD(LOCK_KIND, LOCKPTR); \
        d += libxsmm_timer_ncycles(t1, t2); \
      } \
      else { /* write */ \
        LIBXSMM_LOCK_ACQUIRE(LOCK_KIND, LOCKPTR); \
        t1 = libxsmm_timer_tick(); \
        t2 = work(t1, WORK_W); \
        LIBXSMM_LOCK_RELEASE(LOCK_KIND, LOCKPTR); \
        d += libxsmm_timer_ncycles(t1, t2); \
      } \
    } \
    t1 = libxsmm_timer_ncycles(t0, libxsmm_timer_tick()); \
    MEASURE_THROUGHPUT_ATOMIC \
    throughput += t1 - d; \
  } \
  duration = libxsmm_timer_duration(0, throughput); \
  if (0 < duration) { \
    const double r = 1.0 / (NT); \
    printf("\tthroughput: %.0f us (call/s %.0f kHz, %.0f cycles)\n", \
      duration * r * 1e6, (NT) / (1e3 * duration), throughput * r); \
  } \
} while(0)

#define BENCHMARK(LOCK_KIND, IMPL, NTHREADS, WORK_R, WORK_W, WRATIOPERC, NREPEAT_LAT, NREPEAT_TPT) do { \
  const int nw = 0 < (WRATIOPERC) ? (100 / (WRATIOPERC)) : ((NREPEAT_TPT) + 1); \
  const int nt = (NREPEAT_TPT) * (NTHREADS); \
  const double nr = 1.0 / (NREPEAT_LAT); \
  LIBXSMM_LOCK_ATTR_TYPE(LOCK_KIND) attr; \
  LIBXSMM_LOCK_TYPE(LOCK_KIND) lock; \
  LIBXSMM_ASSERT(0 < nt); \
  printf("Latency and throughput of \"%s\" (%s) for nthreads=%i wratio=%i%% work_r=%i work_w=%i nlat=%i ntpt=%i\n", \
    LIBXSMM_STRINGIFY(LOCK_KIND), IMPL, NTHREADS, WRATIOPERC, WORK_R, WORK_W, NREPEAT_LAT, NREPEAT_TPT); \
  LIBXSMM_LOCK_ATTR_INIT(LOCK_KIND, &attr); \
  LIBXSMM_LOCK_INIT(LOCK_KIND, &lock, &attr); \
  LIBXSMM_LOCK_ATTR_DESTROY(LOCK_KIND, &attr); \
  MEASURE_LATENCY_RO(LOCK_KIND, &lock, NREPEAT_LAT, nr); \
  MEASURE_LATENCY_RW(LOCK_KIND, &lock, NREPEAT_LAT, nr); \
  MEASURE_THROUGHPUT(LOCK_KIND, &lock, NREPEAT_TPT, NTHREADS, WORK_R, WORK_W, nw, nt); \
  LIBXSMM_LOCK_DESTROY(LOCK_KIND, &lock); \
} while(0)


libxsmm_timer_tickint work(libxsmm_timer_tickint start, libxsmm_timer_tickint duration);
libxsmm_timer_tickint work(libxsmm_timer_tickint start, libxsmm_timer_tickint duration)
{
  const libxsmm_timer_tickint end = start + duration;
  libxsmm_timer_tickint tick = start;
  do {
    libxsmm_timer_tickint i, s = 0;
    for (i = 0; i < ((end - tick) / 4); ++i) s += i;
    tick = libxsmm_timer_tick();
  }
  while(tick < end);
  return tick;
}


int main(int argc, char* argv[])
{
#if defined(_OPENMP)
  const int max_nthreads = omp_get_max_threads();
#else
  const int max_nthreads = 1;
#endif
  const int nthreads = LIBXSMM_MAX(1 < argc ? atoi(argv[1]) : max_nthreads, 1);
  const int wratioperc = LIBXSMM_CLMP(2 < argc ? atoi(argv[2]) : 5, 0, 100);
  const int work_r = LIBXSMM_MAX(3 < argc ? atoi(argv[3]) : 100, 1);
  const int work_w = LIBXSMM_MAX(4 < argc ? atoi(argv[4]) : (10 * work_r), 1);
  const int nlat = LIBXSMM_MAX(5 < argc ? atoi(argv[5]) : 2000000, 1);
  const int ntpt = LIBXSMM_MAX(6 < argc ? atoi(argv[6]) : 10000, 1);

  libxsmm_init();
  printf("LIBXSMM: default lock-kind \"%s\" (%s)\n\n", LIBXSMM_STRINGIFY(LIBXSMM_LOCK_DEFAULT),
#if defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK)
    "OS-native");
#else
    "Other");
#endif

#if defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK)
  BENCHMARK(LIBXSMM_LOCK_SPINLOCK, "OS-native", nthreads, work_r, work_w, wratioperc, nlat, ntpt);
#else
  BENCHMARK(LIBXSMM_LOCK_SPINLOCK, "Other", nthreads, work_r, work_w, wratioperc, nlat, ntpt);
#endif
#if defined(LIBXSMM_LOCK_SYSTEM_MUTEX)
  BENCHMARK(LIBXSMM_LOCK_MUTEX, "OS-native", nthreads, work_r, work_w, wratioperc, nlat, ntpt);
#else
  BENCHMARK(LIBXSMM_LOCK_MUTEX, "Other", nthreads, work_r, work_w, wratioperc, nlat, ntpt);
#endif
#if defined(LIBXSMM_LOCK_SYSTEM_RWLOCK)
  BENCHMARK(LIBXSMM_LOCK_RWLOCK, "OS-native", nthreads, work_r, work_w, wratioperc, nlat, ntpt);
#else
  BENCHMARK(LIBXSMM_LOCK_RWLOCK, "Other", nthreads, work_r, work_w, wratioperc, nlat, ntpt);
#endif

  return EXIT_SUCCESS;
}

