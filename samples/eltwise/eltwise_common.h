/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
*               Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.), Antonio Noack (FSU Jena)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

libxsmm_datatype char_to_libxsmm_datatype( const char* dt ) {
  libxsmm_datatype dtype = LIBXSMM_DATATYPE_UNSUPPORTED;

  if ( (strcmp(dt, "F64") == 0) ) {
    dtype = LIBXSMM_DATATYPE_F64;
  } else if ( (strcmp(dt, "I64") == 0) ) {
    dtype = LIBXSMM_DATATYPE_I64;
  } else if ( (strcmp(dt, "F32") == 0) ) {
    dtype = LIBXSMM_DATATYPE_F32;
  } else if ( (strcmp(dt, "I32") == 0) ) {
    dtype = LIBXSMM_DATATYPE_I32;
  } else if ( (strcmp(dt, "F16") == 0) ) {
    dtype = LIBXSMM_DATATYPE_F16;
  } else if ( (strcmp(dt, "BF16") == 0) ) {
    dtype = LIBXSMM_DATATYPE_BF16;
  } else if ( (strcmp(dt, "I16") == 0) ) {
    dtype = LIBXSMM_DATATYPE_I16;
  } else if ( (strcmp(dt, "BF8") == 0) ) {
    dtype = LIBXSMM_DATATYPE_BF8;
  } else if ( (strcmp(dt, "I8") == 0) ) {
    dtype = LIBXSMM_DATATYPE_I8;
  } else {
    dtype = LIBXSMM_DATATYPE_UNSUPPORTED;
  }

  return dtype;
}

void init_random_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n, const libxsmm_blasint neg_values ) {
  double* d_data = (double*) data;
  float* f_data = (float*) data;
  libxsmm_bfloat16* bf_data = (libxsmm_bfloat16*) data;
  libxsmm_bfloat8* bf8_data = (libxsmm_bfloat8*) data;
  int* i_data = (int*) data;
  short* s_data = (short*) data;
  char* c_data = (char*) data;
  libxsmm_blasint l_r, l_i, l_j;

  for (l_r = 0; l_r < br; l_r++) {
    for (l_i = 0; l_i < ld; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        if ( dtype == LIBXSMM_DATATYPE_F64 ) {
          d_data[(l_r * ld * n) + (l_j * ld) + l_i] = (neg_values) ? (0.05 - libxsmm_rng_f64()/10.0) : libxsmm_rng_f64();
        } else if ( dtype == LIBXSMM_DATATYPE_F32 ) {
          f_data[(l_r * ld * n) + (l_j * ld) + l_i] = (neg_values) ? (float)(0.05 - libxsmm_rng_f64()/10.0) : (float)libxsmm_rng_f64();
        } else if ( dtype == LIBXSMM_DATATYPE_BF16 ) {
          libxsmm_bfloat16_f32 tmp/* = { 0 }*/;
          tmp.f = (neg_values) ? (float)(0.05 - libxsmm_rng_f64()/10.0) : (float)libxsmm_rng_f64();
          bf_data[(l_r * ld * n) + (l_j * ld) + l_i] = tmp.i[1];
        } else if ( dtype == LIBXSMM_DATATYPE_BF8 ) {
          union libxsmm_bfloat8_f16 tmp;
          tmp.hf = libxsmm_convert_f32_to_f16 ((neg_values) ? (float)(0.05 - libxsmm_rng_f64()/10.0) : (float)libxsmm_rng_f64());
          bf8_data[(l_r * ld * n) + (l_j * ld) + l_i] = tmp.i[1];
        } else if ( dtype == LIBXSMM_DATATYPE_I32 ) {
          i_data[(l_r * ld * n) + (l_j * ld) + l_i] = (int)  (libxsmm_rng_f64() * 20.0);
        } else if ( dtype == LIBXSMM_DATATYPE_I16 ) {
          s_data[(l_r * ld * n) + (l_j * ld) + l_i] = (short)(libxsmm_rng_f64() * 20.0);
        } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
          c_data[(l_r * ld * n) + (l_j * ld) + l_i] = (char) (libxsmm_rng_f64() * 20.0);
        } else {
        }
      }
    }
  }
}

void init_zero_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n ) {
  char* l_data = (char*) data;
  memset( l_data, 0x0, (size_t)br*ld*n*LIBXSMM_TYPESIZE(dtype) );
}

void init_garbage_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n ) {
  char* l_data = (char*) data;
  memset( l_data, 0xdeadbeef, (size_t)br*ld*n*LIBXSMM_TYPESIZE(dtype) );
}

void apply_row_bcast_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint ld, const libxsmm_blasint m, const libxsmm_blasint n ) {
  double* d_data = (double*) data;
  float* f_data = (float*) data;
  unsigned short* s_data = (unsigned short*) data;
  unsigned char* c_data = (unsigned char*) data;
  libxsmm_blasint i,j;

  for ( i = 0; i < n; ++i ) {
    for ( j = 0; j < LIBXSMM_MAX(m,ld); ++j ) {
      if ( dtype == LIBXSMM_DATATYPE_F64 ) {
        d_data[(i*ld)+j] = d_data[i*ld];
      } else if ( (dtype == LIBXSMM_DATATYPE_F32) || (dtype == LIBXSMM_DATATYPE_I32) ) {
        f_data[(i*ld)+j] = f_data[i*ld];
      } else if ( (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_I16) ) {
        s_data[(i*ld)+j] = s_data[i*ld];
      } else if ( (dtype == LIBXSMM_DATATYPE_I8) || (dtype == LIBXSMM_DATATYPE_BF8) ) {
        c_data[(i*ld)+j] = c_data[i*ld];
      } else {
      }
    }
  }
}

void apply_col_bcast_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint ld, const libxsmm_blasint m, const libxsmm_blasint n ) {
  double* d_data = (double*) data;
  float* f_data = (float*) data;
  unsigned short* s_data = (unsigned short*) data;
  unsigned char* c_data = (unsigned char*) data;
  libxsmm_blasint i,j;

  for ( i = 0; i < n; ++i ) {
    for ( j = 0; j < LIBXSMM_MAX(m,ld); ++j ) {
      if ( dtype == LIBXSMM_DATATYPE_F64 ) {
        d_data[(i*ld)+j] = d_data[j];
      } else if ( (dtype == LIBXSMM_DATATYPE_F32) || (dtype == LIBXSMM_DATATYPE_I32) ) {
        f_data[(i*ld)+j] = f_data[j];
      } else if ( (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_I16) ) {
        s_data[(i*ld)+j] = s_data[j];
      } else if ( (dtype == LIBXSMM_DATATYPE_I8) || (dtype == LIBXSMM_DATATYPE_BF8) ) {
        c_data[(i*ld)+j] = c_data[j];
      } else {
      }
    }
  }
}

void apply_scalar_bcast_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint ld, const libxsmm_blasint m, const libxsmm_blasint n ) {
  double* d_data = (double*) data;
  float* f_data = (float*) data;
  unsigned short* s_data = (unsigned short*) data;
  unsigned char* c_data = (unsigned char*) data;
  libxsmm_blasint i,j;

  for ( i = 0; i < n; ++i ) {
    for ( j = 0; j < LIBXSMM_MAX(m,ld); ++j ) {
      if ( dtype == LIBXSMM_DATATYPE_F64 ) {
        d_data[(i*ld)+j] = d_data[0];
      } else if ( (dtype == LIBXSMM_DATATYPE_F32) || (dtype == LIBXSMM_DATATYPE_I32) ) {
        f_data[(i*ld)+j] = f_data[0];
      } else if ( (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_I16) ) {
        s_data[(i*ld)+j] = s_data[0];
      } else if ( (dtype == LIBXSMM_DATATYPE_I8) || (dtype == LIBXSMM_DATATYPE_BF8) ) {
        c_data[(i*ld)+j] = c_data[0];
      } else {
      }
    }
  }
}

libxsmm_matdiff_info check_matrix( const libxsmm_datatype dtype, const void* data_gold, const void* data, const libxsmm_blasint ld, const libxsmm_blasint m, const libxsmm_blasint n ) {
  libxsmm_matdiff_info l_diff;

  libxsmm_matdiff_clear(&l_diff);

  if ( dtype == LIBXSMM_DATATYPE_F64 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, m, n, data_gold, data, &ld, &ld);
  } else if ( dtype == LIBXSMM_DATATYPE_F32 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, data_gold, data, &ld, &ld);
  } else if ( dtype == LIBXSMM_DATATYPE_BF16 ) {
    float* f_data_gold = (float*) malloc( sizeof(float)*n*ld );
    float* f_data      = (float*) malloc( sizeof(float)*n*ld );
    libxsmm_convert_bf16_f32( data_gold, f_data_gold, n*ld );
    libxsmm_convert_bf16_f32( data,      f_data,      n*ld );
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, f_data_gold, f_data, &ld, &ld);
    free( f_data );
    free( f_data_gold );
  } else if ( dtype == LIBXSMM_DATATYPE_BF8 ) {
    float* f_data_gold = (float*) malloc( sizeof(float)*n*ld );
    float* f_data      = (float*) malloc( sizeof(float)*n*ld );
    libxsmm_convert_bf8_f32( data_gold, f_data_gold, n*ld );
    libxsmm_convert_bf8_f32( data,      f_data,      n*ld );
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, f_data_gold, f_data, &ld, &ld);
    free( f_data );
    free( f_data_gold );
  } else if ( dtype == LIBXSMM_DATATYPE_I32 ) {
    const int* i_data_gold = (const int*)data_gold;
    const int* i_data      = (const int*)data;
    double* f_data_gold = (double*) malloc( sizeof(double)*n*ld );
    double* f_data      = (double*) malloc( sizeof(double)*n*ld );
    libxsmm_blasint i;
    assert(NULL != f_data_gold && NULL != f_data);
    for ( i = 0; i < n*ld; ++i ) f_data_gold[i] = (double)i_data_gold[i];
    for ( i = 0; i < n*ld; ++i ) f_data[i]      = (double)i_data[i];
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, m, n, f_data_gold, f_data, &ld, &ld);
    free( f_data );
    free( f_data_gold );
  } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
    const char* i_data_gold = (const char*)data_gold;
    const char* i_data      = (const char*)data;
    double* f_data_gold = (double*) malloc( sizeof(double)*n*ld );
    double* f_data      = (double*) malloc( sizeof(double)*n*ld );
    libxsmm_blasint i;
    assert(NULL != f_data_gold && NULL != f_data);
    for ( i = 0; i < n*ld; ++i ) f_data_gold[i] = (double)i_data_gold[i];
    for ( i = 0; i < n*ld; ++i ) f_data[i]      = (double)i_data[i];
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, m, n, f_data_gold, f_data, &ld, &ld);
    free( f_data );
    free( f_data_gold );
  } else {
  }

  return l_diff;
}


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
    duration = 0.0;/* benchmarking is deactivated by default */
    const char* dur = getenv("BENCHMARK_DURATION");
    if(dur){
      duration = atof(dur);
    }
  }
  return duration;
}

void benchmark_unary( libxsmm_meltw_unary_type  unary_type,
                      libxsmm_meltw_unary_shape unary_shape,
                      libxsmm_meltw_unary_flags unary_flags,
                      libxsmm_meltw_unary_param unary_param ){

  libxsmm_meltwfunction_unary unary_kernel;
  double l_targetRuntimeSeconds = getBenchmarkDuration();
  size_t l_warmupRuns = 1000, l_warmupIndex;
  size_t l_benchmarkRuns, l_benchmarkIndex;
  double l_warmupDuration, l_duration;
  double l_performance[MAX_BENCHMARK_ARCHITECTURES] = { 0 };
  const char* l_archNames[MAX_BENCHMARK_ARCHITECTURES] = { NULL };
  const char* l_arch = NULL;
  int l_archIndex = 0;
  libxsmm_timer_tickint l_startTime0, l_endTime0, l_startTime, l_endTime; /* loop over architectures */
  for (l_archIndex = 0; l_archIndex < MAX_BENCHMARK_ARCHITECTURES; l_archIndex++) {
    if (l_targetRuntimeSeconds > 0){
      l_arch = l_archNames[l_archIndex] = getBenchmarkedArch(l_archIndex);
      if (!l_arch) break;
      libxsmm_finalize();
      libxsmm_init();
      libxsmm_set_target_arch(l_arch);
      unary_kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, unary_flags );

      /* warmup and computation how many steps are required */
      l_startTime0 = libxsmm_timer_tick();
      for (l_warmupIndex = 0; l_warmupIndex < l_warmupRuns; l_warmupIndex++) {
        unary_kernel(&unary_param);
      }
      l_endTime0 = libxsmm_timer_tick();
      l_warmupDuration = libxsmm_timer_duration(l_startTime0, l_endTime0);
      if (l_warmupDuration <= 1e-9) l_warmupDuration = 1e-9;
      l_benchmarkRuns = (size_t)(l_targetRuntimeSeconds * l_warmupRuns / l_warmupDuration);

      /* running the actual benchmark */
      l_startTime = libxsmm_timer_tick();
      for (l_benchmarkIndex = 0; l_benchmarkIndex < l_benchmarkRuns; l_benchmarkIndex++) {
        unary_kernel(&unary_param);
      }
      l_endTime = libxsmm_timer_tick();
      l_duration = libxsmm_timer_duration(l_startTime, l_endTime);
      l_performance[l_archIndex] = (double)l_benchmarkRuns / l_duration;

      /* printing results */
      if (getBenchmarkedArch(1)) printf("Architecture  : %s\n", l_arch);
      printf("Iterations/s  : %.3f\n", l_performance[l_archIndex]);
      /* how often the kernel was run; could be interesting */
      printf("Runs          : %" PRIuPTR "\n", (uintptr_t)l_benchmarkRuns);
      if (l_archIndex > 0) /* comparison with the first/main architecture */
        printf("Speedup       : %.6fx\n", l_performance[l_archIndex] / l_performance[0]);
      printf("\n");
    }
  } /* end of loop over architectures */
}

void benchmark_binary( libxsmm_meltw_binary_type  binary_type,
                       libxsmm_meltw_binary_shape binary_shape,
                       libxsmm_meltw_binary_flags binary_flags,
                       libxsmm_meltw_binary_param binary_param ){

  libxsmm_meltwfunction_binary binary_kernel;
  double l_targetRuntimeSeconds = getBenchmarkDuration();
  size_t l_warmupRuns = 1000, l_warmupIndex;
  size_t l_benchmarkRuns, l_benchmarkIndex;
  double l_warmupDuration, l_duration;
  double l_performance[MAX_BENCHMARK_ARCHITECTURES] = { 0 };
  const char* l_archNames[MAX_BENCHMARK_ARCHITECTURES] = { NULL };
  const char* l_arch = NULL;
  int l_archIndex = 0;
  libxsmm_timer_tickint l_startTime0, l_endTime0, l_startTime, l_endTime; /* loop over architectures */
  for (l_archIndex = 0; l_archIndex < MAX_BENCHMARK_ARCHITECTURES; l_archIndex++) {
    if (l_targetRuntimeSeconds > 0){
      l_arch = l_archNames[l_archIndex] = getBenchmarkedArch(l_archIndex);
      if (!l_arch) break;
      libxsmm_finalize();
      libxsmm_init();
      libxsmm_set_target_arch(l_arch);
      binary_kernel = libxsmm_dispatch_meltw_binary_v2( binary_type, binary_shape, binary_flags );

      /* warmup and computation how many steps are required */
      l_startTime0 = libxsmm_timer_tick();
      for (l_warmupIndex = 0; l_warmupIndex < l_warmupRuns; l_warmupIndex++) {
        binary_kernel(&binary_param);
      }
      l_endTime0 = libxsmm_timer_tick();
      l_warmupDuration = libxsmm_timer_duration(l_startTime0, l_endTime0);
      if (l_warmupDuration <= 1e-9) l_warmupDuration = 1e-9;
      l_benchmarkRuns = (size_t)(l_targetRuntimeSeconds * l_warmupRuns / l_warmupDuration);

      /* running the actual benchmark */
      l_startTime = libxsmm_timer_tick();
      for (l_benchmarkIndex = 0; l_benchmarkIndex < l_benchmarkRuns; l_benchmarkIndex++) {
        binary_kernel(&binary_param);
      }
      l_endTime = libxsmm_timer_tick();
      l_duration = libxsmm_timer_duration(l_startTime, l_endTime);
      l_performance[l_archIndex] = (double)l_benchmarkRuns / l_duration;

      /* printing results */
      if (getBenchmarkedArch(1)) printf("Architecture  : %s\n", l_arch);
      printf("Iterations/s  : %.3f\n", l_performance[l_archIndex]);
      /* how often the kernel was run; could be interesting */
      printf("Runs          : %" PRIuPTR "\n", (uintptr_t)l_benchmarkRuns);
      if (l_archIndex > 0) /* comparison with the first/main architecture */
        printf("Speedup       : %.6fx\n", l_performance[l_archIndex] / l_performance[0]);
      printf("\n");
    }
  } /* end of loop over architectures */
}
