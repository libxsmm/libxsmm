/******************************************************************************
* Copyright (c) Friedrich-Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Antonio Noack (FSU Jena)
******************************************************************************/

/**
 * This file should be compatible with all correctness tests, and used as an additional benchmark.
 * You need to define the following values as arguments to BENCHMARK_RUN()
 *  - BENCHMARKED_CALL your kernel call, e.g. unary_kernel( &unary_param );
 *  - FLOPS_PER_ITERATION as the number of operations inside your kernel call
 *  - BANDWIDTH_PER_ITERATION as the number of bytes transferred inside your kernel call, shall be > 0
 *
 * This benchmarking only works if no architectures shall be compared, or calling the kernel and creating the kernel need to be in the same function.
 * You can disable benchmarking by setting BENCHMARK_DURATION to 0.
 */

/*

Notes:

compiling libxsmm without BLAS:
make BLAS=0 STATIC=0 -j 48

set target architecture:
LIBXSMM_TARGET=A64FX

create object dumps:
LIBXSMM_VERBOSE=-1

inspect a dumped file:
objdump -D -b binary -maarch64 <fileName>

*/

/* how many architectures will be tested at max; more archs would need to be implemented in getBenchmarkedArch() */
#define MAX_BENCHMARK_ARCHITECTURES 2

const char* getBenchmarkedArch(int index) {
  if (index < 0 || index >= MAX_BENCHMARK_ARCHITECTURES) return 0;
  static const char* archs[MAX_BENCHMARK_ARCHITECTURES] = { NULL };
  int i;
  if (archs[0] == 0) {
    /* init all other architectures to zero, just in case */
    for (i = 1; i < MAX_BENCHMARK_ARCHITECTURES; i++) {
      archs[i] = NULL;
    }
    /* find main architecture */
    archs[0] = libxsmm_get_target_arch();
    /* find secondary/comparison architectures (currently only one) */
    const char* arch1 = getenv("ARCH1");
    if (arch1) {
      libxsmm_set_target_arch(arch1);
      archs[1] = libxsmm_get_target_arch();
    }
  }
  return archs[index];
}

/* returns the target duration of every single benchmark run; if the duration is <= 0 or NaN, no benchmarks will be run */
double getBenchmarkDuration(){
  static double duration = -1;
  if (duration < 0) {
    duration = 0.1;
    const char* dur = getenv("BENCHMARK_DURATION");
    if(dur){
      duration = atof(dur);
    }
  }
  return duration;
}

#define BENCHMARK_INIT() \
  double l_targetRuntimeSeconds = getBenchmarkDuration(); \
  size_t l_warmupRuns = 1000, l_warmupIndex; \
  size_t l_benchmarkRuns, l_benchmarkIndex; \
  double l_warmupDuration, l_duration; \
  double l_gflops[MAX_BENCHMARK_ARCHITECTURES]; \
  double l_gbandwidth[MAX_BENCHMARK_ARCHITECTURES]; \
  const char* l_archNames[MAX_BENCHMARK_ARCHITECTURES]; \
  const char* l_arch; \
  libxsmm_timer_tickint l_startTime0, l_endTime0, l_startTime, l_endTime; /* loop over architectures */ \
  for (int l_archIndex = 0; l_archIndex < MAX_BENCHMARK_ARCHITECTURES; l_archIndex++) { \
    if (l_targetRuntimeSeconds > 0){\
      l_arch = l_archNames[l_archIndex] = getBenchmarkedArch(l_archIndex); \
      if (!l_arch) break; \
      libxsmm_finalize(); \
      libxsmm_init(); \
      libxsmm_set_target_arch(l_arch);\
    }

#define BENCHMARK_RUN(BENCHMARKED_CALL, BANDWIDTH_PER_ITERATION, FLOPS_PER_ITERATION) \
  if (l_targetRuntimeSeconds > 0) { \
    /* warmup and computation how many steps are required */ \
    l_startTime0 = libxsmm_timer_tick(); \
    for (l_warmupIndex = 0; l_warmupIndex < l_warmupRuns; l_warmupIndex++) { \
      BENCHMARKED_CALL; \
    } \
    l_endTime0 = libxsmm_timer_tick(); \
    l_warmupDuration = libxsmm_timer_duration(l_startTime0, l_endTime0); \
    if (l_warmupDuration <= 1e-9) l_warmupDuration = 1e-9; \
    l_benchmarkRuns = (size_t)(l_targetRuntimeSeconds * l_warmupRuns / l_warmupDuration); \
\
    /* running the actual benchmark */ \
    l_startTime = libxsmm_timer_tick(); \
    for (l_benchmarkIndex = 0; l_benchmarkIndex < l_benchmarkRuns; l_benchmarkIndex++) { \
      BENCHMARKED_CALL; \
    } \
    l_endTime = libxsmm_timer_tick(); \
    l_duration = libxsmm_timer_duration(l_startTime, l_endTime); \
    l_gflops[l_archIndex] = FLOPS_PER_ITERATION * (double)l_benchmarkRuns / l_duration * 1e-9; \
    l_gbandwidth[l_archIndex] = BANDWIDTH_PER_ITERATION * (double)l_benchmarkRuns / l_duration * 1e-9; \
\
    /* printing results */ \
    if (getBenchmarkedArch(1)) printf("Architecture  : %s\n", l_arch); \
    printf("GB/s Bandwidth: %.24f\n", l_gbandwidth[l_archIndex]); \
    printf("GFlops        : %.24f\n", l_gflops[l_archIndex]); \
    printf("Runs          : %ld\n", l_benchmarkRuns); /* how often the kernel was run; could be interesting */ \
    if (l_archIndex > 0) /* comparison with the first/main architecture */ \
      printf("Speedup       : %.24fx\n", l_gbandwidth[l_archIndex] / l_gbandwidth[0]); \
  }

#define BENCHMARK_FINALIZE() \
  } // end of loop over architectures
