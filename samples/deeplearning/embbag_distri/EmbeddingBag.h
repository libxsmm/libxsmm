/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Dhiraj Kalamkar, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#if defined(USE_LIBXSMM_JIT)
#include <libxsmm.h>
#endif
#include "utils.h"
#include "rtm.h"

template <typename T>
class EmbeddingBagImpl
{
public:
  EmbeddingBagImpl(long M, long E) : M(M), E(E)
  {
#ifdef USE_LIBXSMM_JIT
    _ld = E;
    libxsmm_meltw_unary_shape unary_shape_f32 = libxsmm_create_meltw_unary_shape( E, 0, _ld, _ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltw_unary_shape unary_shape_f16 = libxsmm_create_meltw_unary_shape( E, 0, _ld, _ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltw_binary_shape binary_shape_f32 = libxsmm_create_meltw_binary_shape( E, 1, _ld, _ld, _ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    weight_ = (T*)my_malloc((size_t)M * E * sizeof(T), alignment);

    if (sizeof(T) == 4) {
      kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD, unary_shape_f32, (sizeof(long) == 8) ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES : LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES );
    } else {
      kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD, unary_shape_f16, (sizeof(long) == 8) ? LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES : LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES );
    }
    kernel1 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR, unary_shape_f32, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    kernel2 = libxsmm_dispatch_meltw_binary_v2( LIBXSMM_MELTW_TYPE_BINARY_MULADD, binary_shape_f32, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0 );
#else
    weight_ = (T*)my_malloc((size_t)M * E * sizeof(T), alignment);
#endif

  }

  ~EmbeddingBagImpl()
  {
    my_free(weight_);
    weight_ = 0;
  }

  void init(T low = -0.1, T high = 0.1)
  {
    init_random(M * E, weight_, low, high);
  }

#ifdef USE_LIBXSMM_JIT
  void forward(long N, long NS, const long *offsets, const long *indices, T *output_)
  {
    T(*__restrict weight)[E] = (T(*)[E])weight_;
    T(*__restrict output)[E] = (T(*)[E])output_;

    #pragma omp parallel for
    for (int n = 0; n < N; n++)
    {
      libxsmm_meltw_unary_param params;
      auto start = offsets[n];
      auto end = (n < N - 1 ? offsets[n + 1] : NS);
      unsigned long long __n = end-start;

      params.in.primary = weight;
      params.in.secondary = (void*)&indices[start];
      params.in.tertiary = &__n;
      params.out.primary = &output[n][0];
      kernel( &params );
    }
  }
#else
  void forward(long N, long NS, const long *offsets, const long *indices, T *output_)
  {
    T(*__restrict weight)[E] = (T(*)[E])weight_;
    T(*__restrict output)[E] = (T(*)[E])output_;

#pragma omp parallel for
    for (long n = 0; n < N; n++)
    {
      auto start = offsets[n];
      auto end = (n < N - 1 ? offsets[n + 1] : NS);
#pragma omp simd
      for (long v = 0; v < E; v++)
        output[n][v] = 0;
      for (long s = start; s < end; s++)
      {
        auto ind = indices[s];
#pragma omp simd
        for (long v = 0; v < E; v++)
        {
          output[n][v] += weight[ind][v];
        }
      }
    }
  }
#endif

#ifdef USE_LIBXSMM_JIT
  void backward(long N, long NS, const T *gradout_, const long *offsets, const long *indices, T *values_)
  {
    T(*__restrict gradout)[E] = (T(*)[E])gradout_;
    T(*__restrict values)[E] = (T(*)[E])values_;
    int _ld = E;
#pragma omp parallel for
    for (long n = 0; n < N; n++)
    {
      libxsmm_meltw_unary_param unary_param;
      auto start = offsets[n];
      auto end = (n < N - 1 ? offsets[n + 1] : NS);
      unsigned long long _N = end-start;

      unary_param.in.primary    = (void*)&gradout[n][0];
      unary_param.out.primary   = (void*)&values[start][0];
      unary_param.op.primary = (void*)&_N;

      kernel1(&unary_param);
    }
  }
#else
  void backward(long N, long NS, const T *gradout_, const long *offsets, const long *indices, T *values_)
  {
    T(*__restrict gradout)[E] = (T(*)[E])gradout_;
    T(*__restrict values)[E] = (T(*)[E])values_;

#pragma omp parallel for
    for (long n = 0; n < N; n++)
    {
      auto start = offsets[n];
      auto end = (n < N - 1 ? offsets[n + 1] : NS);
      for (long s = start; s < end; s++)
      {
#pragma omp simd
#ifdef STREAMING_WRITES
#pragma vector nontemporal(values)
#endif
        for (long v = 0; v < E; v++)
          values[s][v] = gradout[n][v];
      }
    }
  }
#endif

#ifdef USE_LIBXSMM_JIT
  void update(long NS, const T *grads_, const long *indices, float lr, long M, int use_rtm)
  {
    int use_lock_free = use_rtm == 0 ? 1: 0;
    T(*__restrict weight)[E] = (T(*)[E])weight_;
    T(*__restrict grads)[E] = (T(*)[E])grads_;
    int _ld = E;
    if (use_lock_free) {
      /*printf("Using lock free update\n");*/
      int max_thr = omp_get_max_threads();
      if (M < max_thr) max_thr = M;
#pragma omp parallel num_threads(max_thr)
      {
        int tid = omp_get_thread_num();
        for (long i = 0; i < NS; i++) {
          auto ind = indices[i];
          if (ind % max_thr == tid) {
            libxsmm_meltw_binary_param binary_param;
            binary_param.in0.primary  = (void*)&lr;
            binary_param.in1.primary  = (void*)&grads[i][0];
            binary_param.out.primary  = (void*)&weight[ind][0];
            {
              kernel2(&binary_param);
            }
          }
        }
      }
    } else {
      SimpleSpinLock fallBackLock;
#pragma omp parallel for
      for (long i = 0; i < NS; i++)
      {
        libxsmm_meltw_binary_param binary_param;
        long ind = indices[i];
        binary_param.in0.primary  = (void*)&lr;
        binary_param.in1.primary  = (void*)&grads[i][0];
        binary_param.out.primary  = (void*)&weight[ind][0];
        {
          TransactionScope guard(fallBackLock, 100, 0);
          kernel2(&binary_param);
        }
      }
    }
  }
#else
  void update(long NS, const T *grads_, const long *indices, float lr, long M, int use_rtm)
  {
    T(*__restrict weight)[E] = (T(*)[E])weight_;
    T(*__restrict grads)[E] = (T(*)[E])grads_;

    int use_lock_free = use_rtm == 0 ? 1: 0;

    if (use_lock_free) {
      int max_thr = omp_get_max_threads();
      if (M < max_thr) max_thr = M;
#pragma omp parallel num_threads(max_thr)
      {
        int tid = omp_get_thread_num();
        for (long i = 0; i < NS; i++) {
          auto ind = indices[i];
          if (ind % max_thr == tid) {
#pragma omp simd
            for (long v = 0; v < E; v++)
              weight[ind][v] += lr * grads[i][v];
          }
        }
      }
    } else {
      SimpleSpinLock fallBackLock;
#pragma omp parallel for
      for (long i = 0; i < NS; i++)
      {
        long ind = indices[i];
        {
          TransactionScope guard(fallBackLock, 100, 0);
#pragma omp simd
          for (long v = 0; v < E; v++)
            weight[ind][v] += lr * grads[i][v];
        }
      }
    }
  }
#endif

  T *weight_;
  long M;
  long E;

#ifdef USE_LIBXSMM_JIT
  int _ld;
  libxsmm_meltwfunction_unary kernel;
  libxsmm_meltwfunction_unary kernel1;
  libxsmm_meltwfunction_binary kernel2;
#endif
};

