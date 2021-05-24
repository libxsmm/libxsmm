/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
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
  EmbeddingBagImpl(int M, int E) : M(M), E(E)
  {
    weight_ = (T*)my_malloc((size_t)M * E * sizeof(T), alignment);

#ifdef USE_LIBXSMM_JIT
    _ld = E;
    if (sizeof(T) == 4) {
      kernel = libxsmm_dispatch_meltw_reduce_cols_idx(E, &_ld, &_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, (sizeof(long) == 8) ? LIBXSMM_DATATYPE_I64 : LIBXSMM_DATATYPE_I32);
    } else {
      kernel = libxsmm_dispatch_meltw_reduce_cols_idx(E, &_ld, &_ld, LIBXSMM_DATATYPE_F16, LIBXSMM_DATATYPE_F16, (sizeof(long) == 8) ? LIBXSMM_DATATYPE_I64 : LIBXSMM_DATATYPE_I32);
    }
    kernel1 = libxsmm_dispatch_meltw_unary(E, 0, &_ld, &_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR);
    kernel2 = libxsmm_dispatch_meltw_binary(E, 1, &_ld, &_ld, &_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0, LIBXSMM_MELTW_TYPE_BINARY_MULADD);
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
  void forward(int N, int NS, const long *offsets, const long *indices, T *output_)
  {
    T(*__restrict weight)[E] = (T(*)[*])weight_;
    T(*__restrict output)[E] = (T(*)[*])output_;

    #pragma omp parallel for
    for (int n = 0; n < N; n++)
    {
      libxsmm_meltw_reduce_cols_idx_param params;
      auto start = offsets[n];
      auto end = (n < N - 1 ? offsets[n + 1] : NS);
      params.n = end - start;
      params.ind_ptr = &indices[start];
      params.inp_ptr = weight;
      params.out_ptr = &output[n][0];
      kernel( &params );
    }
  }
#else
  void forward(int N, int NS, const long *offsets, const long *indices, T *output_)
  {
    T(*__restrict weight)[E] = (T(*)[*])weight_;
    T(*__restrict output)[E] = (T(*)[*])output_;

#pragma omp parallel for
    for (int n = 0; n < N; n++)
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
  void backward(int N, int NS, const T *gradout_, const long *offsets, const long *indices, T *values_)
  {
    T(*__restrict gradout)[E] = (T(*)[*])gradout_;
    T(*__restrict values)[E] = (T(*)[*])values_;
    int _ld = E;
#pragma omp parallel for
    for (int n = 0; n < N; n++)
    {
      libxsmm_meltw_unary_param unary_param;
      auto start = offsets[n];
      auto end = (n < N - 1 ? offsets[n + 1] : NS);
      unsigned long long _N = end-start;

      unary_param.in.primary    = (void*)&gradout[n][0];
      unary_param.out.primary   = (void*)&values[start][0];
      unary_param.out.secondary = (void*)&_N;

      kernel1(&unary_param);
    }
  }
#else
  void backward(int N, int NS, const T *gradout_, const long *offsets, const long *indices, T *values_)
  {
    T(*__restrict gradout)[E] = (T(*)[*])gradout_;
    T(*__restrict values)[E] = (T(*)[*])values_;

#pragma omp parallel for
    for (int n = 0; n < N; n++)
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
  void update(int NS, const T *grads_, const long *indices, float lr, int M, int use_rtm)
  {
    int use_lock_free = use_rtm == 0 ? 1: 0;
    T(*__restrict weight)[E] = (T(*)[*])weight_;
    T(*__restrict grads)[E] = (T(*)[*])grads_;
    int _ld = E;
    if(use_lock_free) {
      /*printf("Using lock free update\n");*/
      int max_thr = omp_get_max_threads();
      if(M < max_thr) max_thr = M;
#pragma omp parallel num_threads(max_thr)
      {
        int tid = omp_get_thread_num();
        for(long i = 0; i < NS; i++) {
          auto ind = indices[i];
          if(ind % max_thr == tid) {
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
  void update(int NS, const T *grads_, const long *indices, float lr, int M, int use_rtm)
  {
    T(*__restrict weight)[E] = (T(*)[*])weight_;
    T(*__restrict grads)[E] = (T(*)[*])grads_;

    int use_lock_free = use_rtm == 0 ? 1: 0;

    if(use_lock_free) {
      int max_thr = omp_get_max_threads();
      if(M < max_thr) max_thr = M;
#pragma omp parallel num_threads(max_thr)
      {
        int tid = omp_get_thread_num();
        for(long i = 0; i < NS; i++) {
          auto ind = indices[i];
          if(ind % max_thr == tid) {
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
  int M;
  int E;

#ifdef USE_LIBXSMM_JIT
  int _ld;
  libxsmm_meltwfunction_reduce_cols_idx kernel;
  libxsmm_meltwfunction_unary kernel1;
  libxsmm_meltwfunction_binary kernel2;
#endif
};

