/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/
#include <vector>
#include <time.h>
#include <sys/syscall.h>
#include <algorithm>
#include <iterator>
#include <set>
#include <parallel/algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>
#include <assert.h>
#include "utils.h"
#include "EmbeddingBag.h"
#include "dist.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_num_threads() (1)
#define omp_get_thread_num() (0)
#define omp_get_max_threads() (1)
#endif

#ifdef RTM_DEBUG
int rtm_stats[1000][16];
#endif

#ifdef USE_FP16
typedef Half DTyp;
#else
typedef FTyp DTyp;
#endif

typedef EmbeddingBagImpl<DTyp> EmbeddingBag;

int my_rank = 0;
int my_size = 1;

struct EmbeddingInOut {
  int N, NS, E, U;
  ITyp *offsets;
  ITyp *indices;
  DTyp *output;
  DTyp *gradout;
  DTyp *grads;
};

// based on https://www.csee.usf.edu/~kchriste/tools/genzipf.c
int zipf_dist(double alpha, int M)
{
  static int init_done = 0;
  static double k = 0;
  static double *sum_probs;
  static int prev_M = 0;
  double z;
  int value;
  int    i;
  int low, high, mid;

  if (prev_M != M) {
    init_done = 0;
    prev_M = M;
  }

  if (!init_done) {
    for (i=1; i<=M; i++)
      k = k + (1.0 / pow((double) i, alpha));
    k = 1.0 / k;

    sum_probs = (double *) my_malloc((M+1)*sizeof(double), alignment);
    sum_probs[0] = 0;
    for (i=1; i<=M; i++) {
      sum_probs[i] = sum_probs[i-1] + k / pow((double) i, alpha);
    }
    init_done = 1;
  }

  do {
    drand48_r(&rand_buf, &z);
  } while ((z == 0) || (z == 1));

  low = 1, high = M, mid;
  do {
    mid = floor((low+high)/2);
    if (sum_probs[mid] >= z && sum_probs[mid-1] < z) {
      value = mid;
      break;
    } else if (sum_probs[mid] >= z) {
      high = mid-1;
    } else {
      low = mid+1;
    }
  } while (low <= high);

  assert((value >=1) && (value <= M));

  return(value);
}

int find_unique(EmbeddingInOut *eio)
{
  int N = eio->N;
  int NS = eio->NS;

  std::vector<std::pair<int, int>> tmpBuf(NS);
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    int start = eio->offsets[i];
    int end = eio->offsets[i+1];
    for (int j = start; j < end; j++) {
      tmpBuf[j].first = eio->indices[j];
      tmpBuf[j].second = i;
    }
  }
  __gnu_parallel::sort(tmpBuf.begin(), tmpBuf.end(), [](const std::pair<int, int> &a, const std::pair<int, int> &b) { return a.first < b.first; });

  int max_thds = omp_get_max_threads();
  int num_uniq[max_thds];

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    num_uniq[tid] = 0;
#pragma omp for schedule(static)
    for (int i = 1; i < NS; i++) {
      if (tmpBuf[i].first != tmpBuf[i-1].first) num_uniq[tid]++;
    }
  }
  num_uniq[0] += 1;
  for (int i = 1; i < max_thds; i++)
    num_uniq[i] += num_uniq[i-1];
  int U = num_uniq[max_thds-1];
  return U;
}

void allocate_buffers_and_generte_rnd_input(int N, int P, double alpha, EmbeddingBag *eb, EmbeddingInOut *eio)
{
  int E = eb->E;
  int M = eb->M;
  int NS = 0;
  eio->N = N;
  eio->E = E;

  eio->offsets = (ITyp*)my_malloc((N+1) * sizeof(ITyp), alignment);
  eio->output = (DTyp*)my_malloc(N * E * sizeof(DTyp), alignment);
  eio->gradout = (DTyp*)my_malloc(N * E * sizeof(DTyp), alignment);

  eio-> offsets[0] = 0;
  for (int i = 1; i <= N; i++) {
    double randval;
    drand48_r(&rand_buf, &randval);
    int cp = (int)(randval * P * 2);
    if (cp == 0) cp = 1;
    NS += cp;
    eio->offsets[i] = NS;
  }
  eio->NS = NS;
  eio->indices = (ITyp*)my_malloc(NS * sizeof(ITyp), alignment);
  eio->grads = (DTyp*)my_malloc(NS * E * sizeof(DTyp), alignment);
#pragma omp parallel for
  for (int n = 0; n < N; n++)
  {
    int start = eio->offsets[n];
    int end = eio->offsets[n+1];
    std::set<ITyp> s_ind;
    ITyp ind;
    double randval;
    while(s_ind.size() < (end - start)) {
      if (alpha == 0.0) {
        drand48_r(&rand_buf, &randval);
        ind = (ITyp)(randval * M);
      } else {
        ind = (ITyp) zipf_dist(alpha, M);
      }
      if (ind == M)
        ind--;
      s_ind.insert(ind);
    }

    int i = start;
    for (std::set<ITyp>::iterator itr = s_ind.begin(); itr != s_ind.end(); itr++, i++) {
      eio->indices[i] = *itr;
    }
    //set iterator gives elements in sorted order
    //std::sort(&eio->indices[start], &eio->indices[end]);
  }

#ifdef COUNT_UNIQUE
  eio->U = find_unique(eio);
#endif
}

void free_buffers(EmbeddingInOut *eio)
{
  my_free(eio->grads);
  my_free(eio->indices);
  my_free(eio->gradout);
  my_free(eio->output);
  my_free(eio->offsets);
}

void pack_for_a2a(int LS, int N, int E, EmbeddingInOut **emb, DTyp *a2aSrc)
{
  for (int i = 0; i < LS; i++)
  {
    for (int n = 0; n < N; n++)
    {
      for (int v = 0; v < E; v++)
      {
        a2aSrc[n * LS * E + i * E + v] = emb[i]->output[n * E + v];
      }
    }
  }
}

void unpack_from_a2a(int LS, int N, int E, EmbeddingInOut **emb, DTyp *a2aGDst)
{
  for (int i = 0; i < LS; i++)
  {
    for (int n = 0; n < N; n++)
    {
      for (int v = 0; v < E; v++)
      {
        emb[i]->gradout[n * E + v] = a2aGDst[n * LS * E + i * E + v];
      }
    }
  }
}

void alltoall(size_t size, DTyp *src, DTyp *dst)
{
  if (my_size <= 1)
    return;
  size_t sz = size / my_size;
#ifdef USE_FP16
    printf("Only FP32 Datatype supported in alltoall\n");
    exit(1);
#else
    dist_alltoall(sz, src, dst);
#endif
}

double get_checksum(DTyp *buf, size_t sz)
{
  double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
  for (size_t i = 0; i < sz; i++) {
    sum += buf[i];
  }
  return sum;
}

int iters = 100;
int N = 2048;
int E = 64;
int P = 100;
int M = 1000000;
int S = 8;
double alpha = 0.0;

#define my_printf(fmt, args...) printf("[%d] " fmt, my_rank, args)

int main(int argc, char * argv[]) {
  int NP = 1;
  dist_init(&argc, &argv);
  my_rank = dist_get_rank();
  my_size = dist_get_size();

  if (argc > 1 && strncmp(argv[1], "-h", 3) == 0) {
    printf("Usage: %s iters N E M S P alpha \n", argv[0]);
    printf("iters: Number of iterations (= %d)\n", iters);
    printf("N: Minibatch (= %d)\n", N);
    printf("E: embedding row width (= %d)\n", E);
    printf("M: Number of rows per table (= %d)\n", M);
    printf("S: Number of Tables (= %d)\n", S);
    printf("P: Average number of indices per look up (= %d)\n", P);
    printf("alpha: Alpha value for Zipf distribution to generate Indices. Use 0 for uniform distribution");
    dist_fini();
    exit(0);
  }

  {
    int i = 1;
    if (argc > i) iters = atoi(argv[i++]);
    if (argc > i) N = atoi(argv[i++]);
    if (argc > i) E = atoi(argv[i++]);
    if (argc > i) M = atoi(argv[i++]);
    if (argc > i) S = atoi(argv[i++]);
    if (argc > i) P = atoi(argv[i++]);
    if (argc > i) alpha = atof(argv[i++]);
  }

  printf("Using: iters: %d N: %d E: %d M: %d S: %d P: %d alpha: %.3f\n", iters, N, E, M, S, P, alpha);

#ifdef USE_RTM
  int use_rtm = 1;
#else
  int use_rtm = 0;
#endif

#if defined(USE_RTM) && defined(RTM_DEBUG)
  clear_rtm_stats();
#endif

  double checksum = 0.0;

  int LS = S / my_size;
  int LN = N / my_size;
  set_random_seed(777+my_rank);

  EmbeddingInOut *eio[iters][LS];
  EmbeddingBag *eb[LS];
  DTyp *A2Asrc, *A2Adst;
  DTyp *A2Agsrc, *A2Agdst;
  size_t tNS = 0;
  size_t tU = 0;

  A2Asrc = (DTyp*)my_malloc(LS*N*E*sizeof(DTyp), alignment);
  A2Agsrc = (DTyp*)my_malloc(S*LN*E*sizeof(DTyp), alignment);

#ifdef USE_FP16
  DTyp low = 0, high = 128;
#else
  DTyp low = -0.01f, high = 0.01f;
#endif

  init_random<DTyp>(S*LN*E, A2Agsrc, low, high);

  if (my_size > 1)
  {
    A2Adst = (DTyp *)my_malloc(S * LN * E * sizeof(DTyp), alignment);
    A2Agdst = (DTyp *)my_malloc(LS * N * E * sizeof(DTyp), alignment);
  }
  else
  {
    A2Adst = A2Asrc;
    A2Agdst = A2Agsrc;
  }

  for (int i = 0; i < LS; i++)
  {
    eb[i] = new EmbeddingBag(M, E);
    eb[i]->init();
    for (int j = 0; j < iters; j++)
    {
      eio[j][i] = new EmbeddingInOut();
      allocate_buffers_and_generte_rnd_input(N, P, alpha, eb[i], eio[j][i]);
      tNS += eio[j][i]->NS;
      tU += eio[j][i]->U;
    }
  }

  double t0 = get_time();
  double fwdTime = 0.0, bwdTime = 0.0, updTime = 0.0;
  double packTime = 0.0, unpackTime = 0.0, fwdA2ATime = 0.0, bwdA2ATime = 0.0;

  for (int i = 0; i < iters; i++) {
    double t0 = get_time();
    for (int s = 0; s < LS; s++) {
      eb[s]->forward(N, eio[i][s]->NS, eio[i][s]->offsets, eio[i][s]->indices, eio[i][s]->output);
    }

    double t1 = get_time();
    pack_for_a2a(LS, N, E, eio[i], A2Asrc);
    double t2 = get_time();
    alltoall(LS*N*E, A2Asrc, A2Adst);
    double t3 = get_time();
    alltoall(LS*N*E, A2Agsrc, A2Agdst);
    double t4 = get_time();
    unpack_from_a2a(LS, N, E, eio[i], A2Agdst);
    double t5 = get_time();

    for (int s = LS-1; s >= 0; s--) {
      eb[s]->backward(N, eio[i][s]->NS, eio[i][s]->gradout, eio[i][s]->offsets, eio[i][s]->indices, eio[i][s]->grads);
    }
    double t6 = get_time();
    for (int s = 0; s < LS; s++) {
      eb[s]->update(eio[i][s]->NS, eio[i][s]->grads, eio[i][s]->indices, -0.1, M, use_rtm);
    }
    double t7 = get_time();
    //printf("Iter %4d: F = %.3f   B = %.3f   U = %.3f\n", i, t1-t0, t2-t1, t3-t2);
    fwdTime += t1-t0;
    bwdTime += t6-t5;
    updTime += t7-t6;
    packTime += t2-t1;
    unpackTime += t5-t4;
    fwdA2ATime += t3-t2;
    bwdA2ATime += t4-t3;
  }
  double t1 = get_time();
#ifdef VERIFY_CORRECTNESS
  for (int s = 0; s < LS; s++) {
    double psum = get_checksum(eb[s]->weight_, (size_t)M*E);
    //my_printf("PSUM %d: %g\n", SS+s, psum);
    checksum += psum;
  }
#endif
#ifdef STREAMING_WRITES
  const size_t rfo = 1;
#else
  const size_t rfo = 2;
#endif

  size_t fwdBytes = ((size_t)tU*E + (size_t)rfo*iters*LS*N*E) * sizeof(DTyp) + ((size_t)tNS + (size_t)iters*LS*N) * sizeof(ITyp);
  size_t bwdBytes = ((size_t)rfo*tNS*E + (size_t)iters*LS*N*E) * sizeof(DTyp) + ((size_t)tNS) * sizeof(ITyp);
  size_t updBytes = ((size_t)2*tU*E + (size_t)tNS*E) * sizeof(DTyp) + ((size_t)tNS) * sizeof(ITyp);

  my_printf("USE RTM = %d  STREAMING STORES = %d\n", use_rtm, rfo == 1 ? 1 : 0);
#ifdef COUNT_UNIQUE
  my_printf("Iters = %d, LS = %d, N = %d, M = %d, E = %d, avgNS = %d, avgU = %d, P = %d, alpha = %.3f\n", iters, LS, N, M, E, tNS/(iters*LS), tU/(iters*LS), P, alpha);
#else
  my_printf("Iters = %d, LS = %d, N = %d, M = %d, E = %d, avgNS = %d, P = %d\n", iters, LS, N, M, E, tNS/(iters*LS), P);
#endif

  my_printf("Per Iter  Time: Fwd: %.3f ms Bwd: %.3f ms Upd: %.3f  A2A: %.3f ms Total: %.3f ms\n", fwdTime/(iters), bwdTime/(iters), updTime/(iters), (fwdA2ATime+bwdA2ATime+packTime+unpackTime)/(iters), (t1-t0)/(iters));
  my_printf("Per Table Time: Fwd: %.3f ms Bwd: %.3f ms Upd: %.3f  Total: %.3f ms\n", fwdTime/(iters*LS), bwdTime/(iters*LS), updTime/(iters*LS), (t1-t0)/(iters*LS));

  my_printf("Per Iter  A2ATime: Fwd: %.3f ms Bwd: %.3f ms Pack: %.3f ms Unpack: %.3f ms \n", fwdA2ATime/(iters), bwdA2ATime/(iters), packTime/(iters), unpackTime/(iters));
  my_printf("BW: FWD: %.3f   BWD: %.3f GB/s   UPD: %.3f GB/s\n", fwdBytes*1e-6/fwdTime, bwdBytes*1e-6/bwdTime, updBytes*1e-6/updTime);


#ifdef VERIFY_CORRECTNESS
  printf("Checksum = %g\n", checksum);
#endif

#if defined(USE_RTM) && defined(RTM_DEBUG)
  print_rtm_stats();
#endif

  for (int i = 0; i < LS; i++)
  {
    for (int j = 0; j < iters; j++)
    {
      free_buffers(eio[j][i]);
      delete eio[j][i];
    }
    delete eb[i];
  }
  if (my_size > 1)
  {
    my_free(A2Agdst);
    my_free(A2Adst);
  }

  my_free(A2Agsrc);
  my_free(A2Asrc);
  dist_fini();
  return 0;
}
