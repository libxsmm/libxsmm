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

// main directory:
// make BLAS=0 STATIC=0 -j 48

// inspect a dumped file:
// objdump -D -b binary -maarch64 libxsmm_x86_64_tsize1_20x5_20x5_opcode12_flags0_params29.meltw

// testing kernels:
// cd libxsmm
// make BLAS=0 LIBXSMM_NO_BLAS=1 STATIC=0 -j 48
// cd samples/eltwise
// make clean
// make BLAS=0 LIBXSMM_NO_BLAS=1 STATIC=0 -j 48
// LIBXSMM_TARGET=A64FX ./kernel_test/unary_rcp_sqrt_32b_eqld.sh


struct SVE_UnaryBenchmark {
  libxsmm_meltw_unary_type unaryType;
  char* name;
  unsigned int flopsPerElement;
  unsigned int bandwidthPerElement;
  double gflops; /* compute performance in GFlop/s */
  double gbandwidth; /* bandwidth in GiB */
};

void benchmark( int m, int n, struct SVE_UnaryBenchmark *io_result ) {

  size_t l_warmupRuns = 50;
  size_t l_benchmarkRuns = 1000 * 1000;
  size_t l_dataSize = m * n;

  // allocate buffers
  T* l_input = malloc(sizeof(T) * l_dataSize);
  T* l_output = malloc(sizeof(T) * l_dataSize);
  for(int i=0;i<l_dataSize;i++) l_input[i] = (T) i;

  libxsmm_meltwfunction_unary l_kernel = libxsmm_dispatch_meltw_unary(m, n, &m, &m,
     LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
     LIBXSMM_MELTW_FLAG_UNARY_NONE, io_result->unaryType );

  libxsmm_meltw_unary_param param;

  param.in.primary  = l_input;
  param.out.primary = l_output;

  // warm up runs
  for(size_t i=0;i<l_warmupRuns;i++){
    l_kernel( &param );
  }

  // start timer
  libxsmm_timer_tickint l_startTime = libxsmm_timer_tick();

  for(size_t i=0;i<l_benchmarkRuns;i++){
    l_kernel( &param );
  }

  // stop timer
  libxsmm_timer_tickint l_endTime = libxsmm_timer_tick();

  // compute delta time
  double l_duration = libxsmm_timer_duration(l_startTime, l_endTime);

  double l_flopsPerRun = io_result->flopsPerElement * (m * n);
  double l_bandwidthPerRun = io_result->bandwidthPerElement * (m * n);

  // compute performance metrics
  io_result->gflops = (double) l_flopsPerRun * l_benchmarkRuns / l_duration * 1e-9;
  io_result->gbandwidth = (double) l_bandwidthPerRun * l_benchmarkRuns / l_duration * 1e-9;

  free(l_input);
  free(l_output);

}

int main(/*int argc, char* argv[]*/) {

  printf("Starting benchmark\n");

  int m = 64, n = 64;

  // start the benchmarking
  #define numTests 7
  struct SVE_UnaryBenchmark types[numTests] = {
    { LIBXSMM_MELTW_TYPE_UNARY_XOR, "xor", 0, sizeof(T) },
    { LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, "copy", 0, sizeof(T) * 2 },
    { LIBXSMM_MELTW_TYPE_UNARY_X2, "x²", 1, sizeof(T) * 2 },
    { LIBXSMM_MELTW_TYPE_UNARY_INC, "+=1", 1, sizeof(T) * 2 },
    { LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL, "1/x", 1, sizeof(T) * 2 },
    { LIBXSMM_MELTW_TYPE_UNARY_SQRT, "sqrt", 1, sizeof(T) * 2 },
    { LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT, "1/sqrt", 1, sizeof(T) * 2 },
  };

  for(int i=0;i<numTests;i++){

    struct SVE_UnaryBenchmark asimd = types[i], sve = types[i];

    /* run with SVE */
    libxsmm_init();
    libxsmm_set_target_archid( LIBXSMM_AARCH64_A64FX );
    benchmark(m, n, &sve);
    libxsmm_finalize();

    /* run with ASIMD */
    libxsmm_init();
    libxsmm_set_target_archid( LIBXSMM_AARCH64_V82 );
    benchmark(m, n, &asimd);
    libxsmm_finalize();

    /* print benchmark results */
    printf("Testing %s: %f gflops, %f GB/s\n", sve.name, sve.gflops, sve.gbandwidth);
    printf("  Comparison: %fx faster\n", sve.gbandwidth/asimd.gbandwidth);/* bandwidth is always not-zero, gflops can be */

  }

  printf("Finished benchmark\n");

}
