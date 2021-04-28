#ifndef MC_FUNCS
#define MC_FUNCS

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <string>

#include <iomanip>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

#include <torch/extension.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>


#include <vector>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#pragma message "Using OpenMP"
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif

#include <libxsmm.h>
//#include <libxsmm_intrinsics_x86.h>
#include <immintrin.h>



static thread_local unsigned int *rnd_state = NULL;

void set_rnd_seed(unsigned int seed)
{
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    if(rnd_state) {
      libxsmm_rng_destroy_extstate(rnd_state);
      rnd_state = NULL;
    }
    rnd_state = libxsmm_rng_create_extstate(seed+tid);
  }
}


void init_libxsmm()
{
  libxsmm_init();
  set_rnd_seed(0);
}

struct f32
{
  std::vector<at::Tensor> dropout_forward(torch::Tensor input, float p, bool train);
  at::Tensor dropout_backward(torch::Tensor input, torch::Tensor dropout_mask, float p);
};


// --------------------------------------- copy() -----------------------------------------------------------------

inline void f32_copy(int N, int M, int LDO, int LDI, libxsmm_meltw_unary_param  *params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  libxsmm_datatype compute_dtype = LIBXSMM_DATATYPE_F32;

  libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary(M, N, &LDI, &LDO, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, compute_dtype, unary_flags, unary_type);

  if ( kernel == NULL )
  {
    fprintf( stderr, "JIT for f32 to f32 copy failed. Bailing...!\n");
    exit(-1);
  }
  kernel(params);
}

inline void zero(int M, libxsmm_meltw_unary_param  *params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type = LIBXSMM_MELTW_TYPE_UNARY_XOR;
  libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32;

  libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary(M, 1, &M, &M, dtype, dtype, dtype, unary_flags, unary_type);

  if ( kernel == NULL )
  {
    fprintf( stderr, "JIT for zero kernel failed. Bailing...!\n");
    exit(-1);
  }
  kernel(params);
}

#endif
