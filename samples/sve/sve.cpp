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

#include <libxsmm_source.h>
#include <vector>
#include <iostream>
#include <chrono>

// main directory:
// make BLAS=0 STATIC=0 -j 48


// g++ -DBLAS=0 -DLIBXSMM_NO_BLAS=1 -I../../include sve.cpp -L../../lib -lxsmm -pthread -o compiled
// LD_LIBRARY_PATH=../../lib LIBXSMM_VERBOSE=-1 ./compiled

// objdump -D -b binary -maarch64 libxsmm_x86_64_tsize1_20x5_20x5_opcode12_flags0_params29.meltw

struct SVE_UnaryBenchmark {
  libxsmm_meltw_unary_type unaryType;
  std::string name;
  unsigned int flopsPerElement;
  unsigned int bandwidthPerElement;
  double gflops = 0; /* compute performance in GFlop/s */
  double gbandwidth = 0; /* bandwidth in GiB */
};

template <typename T>
void benchmark( int m, int n, struct SVE_UnaryBenchmark &io_result ) {

  size_t l_warmupRuns = 50;
  size_t l_benchmarkRuns = 10 * 1000 * 1000;
  size_t l_dataSize = m * n;

  // allocate buffers
  std::vector<T> l_input(l_dataSize), l_output(l_dataSize);
  for(int i=0;i<l_dataSize;i++) l_input[i] = (T) i;

  libxsmm_meltwfunction_unary l_kernel = libxsmm_dispatch_meltw_unary(m, n, &m, &m,
     LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
     LIBXSMM_MELTW_FLAG_UNARY_NONE, io_result.unaryType );

  libxsmm_meltw_unary_param param;

  param.in.primary  = (void*) l_input.data();
  param.out.primary = (void*) l_output.data();

  // warm up runs
  for(size_t i=0;i<l_warmupRuns;i++){
    l_kernel( &param );
  }

  // start timer
  auto l_startTime = std::chrono::high_resolution_clock::now();

  for(size_t i=0;i<l_benchmarkRuns;i++){
    l_kernel( &param );
  }

  // stop timer
  auto l_endTime = std::chrono::high_resolution_clock::now();

  // compute delta time
  double l_duration = std::chrono::duration<double>(l_endTime - l_startTime).count();

  double l_flopsPerRun = io_result.flopsPerElement * (m * n);
  double l_bandwidthPerRun = io_result.bandwidthPerElement * (m * n);

  // compute performance metrics
  io_result.gflops = (double) l_flopsPerRun * l_benchmarkRuns / l_duration * 1e-9;
  io_result.gbandwidth = (double) l_bandwidthPerRun * l_benchmarkRuns / l_duration * 1e-9;

}

int testSVEKernels(){

  typedef float T;

  // goldig, endlich funktioniert es :)
  // also:
  // generate vector
  int size = 100;
  int size2 = size * 20;
  std::vector<T> a(size2), b(size2), c(size2);
  // fill vector with data
  for(int i=0;i<size;i++){
    a[i] = i;
    b[i] = 15 - (i & 15);
    c[i] = 1;
  }

  // generate functions, which we can call
  int m = (size / 5) & ~3, n = (size / m) & ~3;
  libxsmm_datatype f32 = LIBXSMM_DATATYPE_F32;
  libxsmm_meltw_unary_flags none = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltwfunction_unary set_zero = libxsmm_dispatch_meltw_unary(m, n, &m, &m, f32, f32, f32, none, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  libxsmm_meltwfunction_unary copy     = libxsmm_dispatch_meltw_unary(m, n, &m, &m, f32, f32, f32, none, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  libxsmm_meltwfunction_unary square   = libxsmm_dispatch_meltw_unary(m, n, &m, &m, f32, f32, f32, none, LIBXSMM_MELTW_TYPE_UNARY_X2);
  libxsmm_meltwfunction_unary trans    = libxsmm_dispatch_meltw_unary(m, n, &m, &n, f32, f32, f32, none, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
  libxsmm_meltwfunction_unary inc      = libxsmm_dispatch_meltw_unary(m, n, &m, &m, f32, f32, f32, none, LIBXSMM_MELTW_TYPE_UNARY_INC);
  libxsmm_meltwfunction_unary recipro  = libxsmm_dispatch_meltw_unary(m, n, &m, &m, f32, f32, f32, none, LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL);
  libxsmm_meltwfunction_unary sqrt     = libxsmm_dispatch_meltw_unary(m, n, &m, &m, f32, f32, f32, none, LIBXSMM_MELTW_TYPE_UNARY_SQRT);

  if(set_zero == nullptr || copy == nullptr || square == nullptr || trans == nullptr || inc == nullptr ){
    std::cerr << "generated kernel is null!!" << std::endl;
    return -1;
  }

  std::cout << "m: " << m << ", n: " << n << std::endl;

  // apply several functions on the vectors
  libxsmm_meltw_unary_param param;

  param.in.primary  = (void*) a.data();
  param.out.primary = (void*) b.data();

  set_zero( &param );
  std::cout << "set zero" << std::endl;

  param.in.primary  = (void*) a.data();
  param.out.primary = (void*) c.data();

  copy( &param );
  std::cout << "copy" << std::endl;

  param.in.primary  = (void*) a.data();
  param.out.primary = (void*) a.data();

  square( &param );
  std::cout << "square" << std::endl;

  param.in.primary  = (void*) b.data();
  param.out.primary = (void*) b.data();

  inc( &param );
  std::cout << "inc" << std::endl;

  param.in.primary  = (void*) c.data();
  param.out.primary = (void*) c.data();

  recipro( &param );
  std::cout << "1/x" << std::endl;

  /* undoing the x² */
  param.in.primary  = (void*) a.data();
  param.out.primary = (void*) a.data();

  sqrt( &param );
  std::cout << "sqrt" << std::endl;

  // todo compare vector with expected result
  for(int i=0;i<size;i++){
    std::cout << "[" << i << "] a: " << a[i] << ", b: " << b[i] << ", c: " << c[i] << std::endl;
  }
  std::cout << "done" << std::endl;
  // todo done (or free resources ^^)

  return EXIT_SUCCESS;
}

int main(/*int argc, char* argv[]*/) {

  typedef float T;
  std::cout << "Hello world!" << std::endl;

  libxsmm_init();

  int target_arch0 = libxsmm_get_target_archid();
  std::cout << "Detected architecture " << target_arch0 << std::endl;

  // somehow was only LIBXSMM_AARCH64_V82 even if it should have been the A64FX
  libxsmm_set_target_archid( LIBXSMM_AARCH64_A64FX );

  int r0 = testSVEKernels();
  if( r0 != EXIT_SUCCESS ) return r0;

  int m = 19, n = 27;

  // start the benchmarking
  std::vector<struct SVE_UnaryBenchmark> types;
  types.push_back( { LIBXSMM_MELTW_TYPE_UNARY_XOR, "xor", 0, sizeof(T) } );
  types.push_back( { LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, "copy", 0, sizeof(T) * 2 } );
  types.push_back( { LIBXSMM_MELTW_TYPE_UNARY_X2, "x²", 1, sizeof(T) * 2 } );
  types.push_back( { LIBXSMM_MELTW_TYPE_UNARY_INC, "+=1", 1, sizeof(T) * 2 });
  types.push_back( { LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL, "1/x", 1, sizeof(T) * 2 } );
  types.push_back( { LIBXSMM_MELTW_TYPE_UNARY_SQRT, "sqrt", 1, sizeof(T) * 2 } );

  libxsmm_finalize();

  for(auto &type : types){
    struct SVE_UnaryBenchmark asimd = type, sve = type;
    libxsmm_init();
    libxsmm_set_target_archid( LIBXSMM_AARCH64_A64FX );
    benchmark<T>(m, n, sve);
    libxsmm_finalize();
    libxsmm_init();
    libxsmm_set_target_archid( LIBXSMM_AARCH64_V82 );
    benchmark<T>(m, n, asimd);
    libxsmm_finalize();
    // print benchmark results
    std::cout << "bench " << type.name << ": " << sve.gflops << " gflops, " << sve.gbandwidth << " GB/s" << std::endl;
    std::cout << "  comparison: " << sve.gbandwidth / asimd.gbandwidth << "x faster" << std::endl;
  }

  std::cout << "finished benchmark" << std::endl;

}
