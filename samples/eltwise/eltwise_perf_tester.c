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

#include <libxsmm_source.h>

#define T float

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


struct SVE_UnaryBenchmark {
  libxsmm_meltw_unary_type unaryType;
  char* name;
  unsigned int flopsPerElement;
  unsigned int bandwidthPerElement;
  double gflops; /* compute performance in GFlop/s */
  double gbandwidth; /* bandwidth in GiB */
};

void benchmark( int m, int n, struct SVE_UnaryBenchmark *io_result ) {

  double l_targetRuntimeSeconds = 0.1;/* could be an environment variable */
  size_t l_warmupRuns = 1000;/* warmup also measures how long it needs approximately */
  size_t l_benchmarkRuns;
  size_t l_dataSize = m * n;

  /* allocate buffers */
  T* l_input = malloc(sizeof(T) * l_dataSize);
  T* l_output = malloc(sizeof(T) * l_dataSize);
  for(int i=0;i<l_dataSize;i++) l_input[i] = (T) i;

  libxsmm_meltwfunction_unary l_kernel = libxsmm_dispatch_meltw_unary(m, n, &m, &m,
     LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
     LIBXSMM_MELTW_FLAG_UNARY_NONE, io_result->unaryType );

  libxsmm_meltw_unary_param param;

  param.in.primary  = l_input;
  param.out.primary = l_output;

  libxsmm_timer_tickint l_startTime0 = libxsmm_timer_tick();

  /* warmup runs */
  for(size_t i=0;i<l_warmupRuns;i++){
    l_kernel( &param );
  }

  libxsmm_timer_tickint l_endTime0 = libxsmm_timer_tick();
  double l_warmupDuration = libxsmm_timer_duration(l_startTime0, l_endTime0);
  if(l_warmupDuration <= 1e-6) l_warmupDuration = 1e-6;
  l_benchmarkRuns = (size_t) (l_targetRuntimeSeconds * l_warmupRuns / l_warmupDuration);

  /* actual measurement runs */
  libxsmm_timer_tickint l_startTime = libxsmm_timer_tick();

  for(size_t i=0;i<l_benchmarkRuns;i++){
    l_kernel( &param );
  }

  libxsmm_timer_tickint l_endTime = libxsmm_timer_tick();

  /* compute delta time */
  double l_duration = libxsmm_timer_duration(l_startTime, l_endTime);

  double l_flopsPerRun = io_result->flopsPerElement * (m * n);
  double l_bandwidthPerRun = io_result->bandwidthPerElement * (m * n);

  /* compute performance metrics */
  io_result->gflops = (double) l_flopsPerRun * l_benchmarkRuns / l_duration * 1e-9;
  io_result->gbandwidth = (double) l_bandwidthPerRun * l_benchmarkRuns / l_duration * 1e-9;

  free(l_input);
  free(l_output);

}

void printPerformance(struct SVE_UnaryBenchmark bench, const char* architecture){
  if(bench.gflops > 0){
    printf("  %s: %f GFlop/s, %f GB/s\n", architecture, bench.gflops, bench.gbandwidth);
  } else {
    printf("  %s: %f GB/s\n", architecture, bench.gbandwidth);
  }
}

int main(int argc, const char* argv[]) {

  if(argc < 3 && argc > 5){
    printf("Usage: %s <m> <n> [<arch0> <arch1>]\n", argv[0]);
    return -1;
  }

  int m = atol(argv[1]);
  int n = atol(argv[2]);
  const char* arch0 = argc > 3 ? argv[3] : libxsmm_get_target_arch();
  const char* arch1 = argc > 4 ? argv[4] : NULL;
  int has_second_arch = arch1 != NULL;

  if(m <= 0 || n <= 0){
    printf("m and n must be positive!\n");
    return -1;
  }

  printf("Starting benchmark\n");

  /* could be customized in the future */
  #define num_tests 7
  struct SVE_UnaryBenchmark types[num_tests] = {
    { LIBXSMM_MELTW_TYPE_UNARY_XOR, "xor", 0, sizeof(T) },
    { LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, "copy", 0, sizeof(T) * 2 },
    { LIBXSMM_MELTW_TYPE_UNARY_X2, "x²", 1, sizeof(T) * 2 },
    { LIBXSMM_MELTW_TYPE_UNARY_INC, "+=1", 1, sizeof(T) * 2 },
    { LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL, "1/x", 1, sizeof(T) * 2 },
    { LIBXSMM_MELTW_TYPE_UNARY_SQRT, "sqrt", 1, sizeof(T) * 2 },
    { LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT, "1/sqrt", 1, sizeof(T) * 2 },
  };

  if(!has_second_arch){
    libxsmm_init();
    libxsmm_set_target_arch(arch0);
    arch0 = libxsmm_get_target_arch();
  }

  for(int i=0;i<num_tests;i++){

    struct SVE_UnaryBenchmark type0 = types[i], type1 = types[i];

    if(has_second_arch){

      /* run with arch0 */
      libxsmm_init();
      libxsmm_set_target_arch(arch0); /* e.g. LIBXSMM_AARCH64_V82 */
      benchmark(m, n, &type0);
      if(i == 0) arch0 = libxsmm_get_target_arch();
      libxsmm_finalize();

      /* run with arch1 */
      libxsmm_init();
      libxsmm_set_target_arch(arch1); /* e.g. LIBXSMM_AARCH64_A64FX */
      benchmark(m, n, &type1);
      if(i == 0) arch1 = libxsmm_get_target_arch();
      libxsmm_finalize();

      /* print benchmark results */
      printf("Testing %s:\n", type0.name);
      printPerformance(type0, arch0);
      printPerformance(type1, arch1);
      /* bandwidth is always not-zero, gflops can be */
      printf("  Comparison: %s is %fx faster\n", arch0, type0.gbandwidth/type1.gbandwidth);

    } else {

      /* run with arch0 */
      benchmark(m, n, &type0);

      /* print benchmark results */
      printf("Testing %s:\n", type0.name);
      printPerformance(type0, arch0);

    }

  }

  if(!has_second_arch){
    libxsmm_finalize();
  }

  printf("Finished benchmark\n");

}
