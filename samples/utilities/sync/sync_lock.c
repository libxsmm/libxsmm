/******************************************************************************
** Copyright (c) 2018-2019, Intel Corporation                                **
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
#include <stdio.h>

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

#define BENCHMARK(LOCK_KIND, IMPL, NREPEAT, NTHREADS, WORK_R, WORK_W, NW, NT, NR, WRATIOPERC) do { \
  LIBXSMM_LOCK_ATTR_TYPE(LOCK_KIND) attr; \
  LIBXSMM_LOCK_TYPE(LOCK_KIND) lock; \
  LIBXSMM_ASSERT(0 < (NT)); \
  fprintf(stdout, "Latency and throughput of %s (%s) for nthreads=%i wratio=%i%% work_r=%i work_w=%i nrepeat=%i\n", \
    LIBXSMM_STRINGIFY(LOCK_KIND), IMPL, NTHREADS, WRATIOPERC, WORK_R, WORK_W, NREPEAT); \
  LIBXSMM_LOCK_ATTR_INIT(LOCK_KIND, &attr); \
  LIBXSMM_LOCK_INIT(LOCK_KIND, &lock, &attr); \
  LIBXSMM_LOCK_ATTR_DESTROY(LOCK_KIND, &attr); \
  MEASURE_LATENCY_RO(LOCK_KIND, &lock, NREPEAT, NR); \
  MEASURE_LATENCY_RW(LOCK_KIND, &lock, NREPEAT, NR); \
  MEASURE_THROUGHPUT(LOCK_KIND, &lock, NREPEAT, NTHREADS, WORK_R, WORK_W, NW, NT); \
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
  const int nrepeat = LIBXSMM_MAX(5 < argc ? atoi(argv[5]) : 1000000, 1);
  const int nw = 0 < wratioperc ? (100 / wratioperc) : (nrepeat + 1);
  const int nt = nrepeat * nthreads;
  const double nr = 1.0 / nrepeat;

#if defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK)
  BENCHMARK(LIBXSMM_LOCK_SPINLOCK, "OS-native", nrepeat, nthreads, work_r, work_w, nw, nt, nr, wratioperc);
#else
  BENCHMARK(LIBXSMM_LOCK_SPINLOCK, "LIBXSMM", nrepeat, nthreads, work_r, work_w, nw, nt, nr, wratioperc);
#endif
#if defined(LIBXSMM_LOCK_SYSTEM_MUTEX)
  BENCHMARK(LIBXSMM_LOCK_MUTEX, "OS-native", nrepeat, nthreads, work_r, work_w, nw, nt, nr, wratioperc);
#else
  BENCHMARK(LIBXSMM_LOCK_MUTEX, "LIBXSMM", nrepeat, nthreads, work_r, work_w, nw, nt, nr, wratioperc);
#endif
#if defined(LIBXSMM_LOCK_SYSTEM_RWLOCK)
  BENCHMARK(LIBXSMM_LOCK_RWLOCK, "OS-native", nrepeat, nthreads, work_r, work_w, nw, nt, nr, wratioperc);
#else
  BENCHMARK(LIBXSMM_LOCK_RWLOCK, "LIBXSMM", nrepeat, nthreads, work_r, work_w, nw, nt, nr, wratioperc);
#endif

  return EXIT_SUCCESS;
}

