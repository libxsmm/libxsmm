/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/

#include "utils.h"
#include "rtm.h"

template <typename T>
class EmbeddingBagImpl
{
public:
  EmbeddingBagImpl(int M, int E) : M(M), E(E)
  {
    weight_ = (T*)my_malloc((size_t)M * E * sizeof(T), alignment);
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

  void update(int NS, const T *grads_, const long *indices, float lr)
  {
    T(*__restrict weight)[E] = (T(*)[*])weight_;
    T(*__restrict grads)[E] = (T(*)[*])grads_;

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

  T *weight_;
  int M;
  int E;
};

typedef EmbeddingBagImpl<FTyp> EmbeddingBag;
