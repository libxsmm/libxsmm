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
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <immintrin.h>
#include <libxsmm.h>

#ifdef USE_PERF_COUNTERS
#include "counters.h"
#endif

#include "radix_sort.h"
#include "utils.h"

const int alignment = 64;
typedef long ITyp;
typedef float FTyp;

thread_local struct drand48_data rand_buf;

void set_random_seed(int seed)
{
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    srand48_r(seed+tid, &rand_buf);
  }
}

template<typename T>
void init_zero(size_t sz, T *buf)
{
#pragma omp parallel for
  for(size_t i = 0; i < sz; i++)
    buf[i] = (T)0;
}

template<typename T>
void init_random(size_t sz, T *buf, T low = -0.1, T high = 0.1)
{
  T range = high - low;
#pragma omp parallel for schedule(static)
  for(size_t i = 0; i < sz; i++) {
    double randval;
    drand48_r(&rand_buf, &randval);
    buf[i] = randval * range - low;
  }
}

double get_checksum(FTyp *buf, size_t sz)
{
  double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
  for(size_t i = 0; i < sz; i++) {
    sum += buf[i];
  }
  return sum;
}

inline void *my_malloc(size_t sz, size_t align)
{
  return _mm_malloc(sz, align);
}

inline void my_free(void *p)
{
    if(!p) return;
    _mm_free(p);
}

#define DECL_VLA_PTR(type, name, dims, ptr) type (*name)dims = (type (*)dims)ptr

template <typename T>
class EmbeddingBagImpl
{
public:
  EmbeddingBagImpl(int M, int E) : M(M), E(E)
  {
    weight_ = (T*)my_malloc((size_t)M * E * sizeof(T), alignment);
    h = (T*)my_malloc((size_t)M * sizeof(T), alignment);

#ifdef USE_LIBXSMM_JIT
    _ld = E;
    kernel = libxsmm_dispatch_meltw_reduce_cols_idx(E, &_ld, &_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, (sizeof(long) == 8) ? LIBXSMM_DATATYPE_I64 : LIBXSMM_DATATYPE_I32);
    kernel1 = libxsmm_dispatch_meltw_unary(E, 1, &_ld, &_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD);
    kernel2 = libxsmm_dispatch_meltw_binary(E, 1, &_ld, &_ld, &_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0, LIBXSMM_MELTW_TYPE_BINARY_MULADD);
#endif
  }

  ~EmbeddingBagImpl()
  {
    my_free(weight_);
    my_free(h);
    weight_ = 0;
    h = 0;
  }

  void init(T low = -0.1, T high = 0.1)
  {
    //init_random(M * E, weight_, low, high);
    init_zero(M * E, weight_);
    init_zero(M, h);
  }

  void fused_backward_update_adagrad(int U, int NS, int N, long *mb_offsets, long *mb_indices, long *wt_indices, T *outGrad_, float lr, float eps)
  {
    DECL_VLA_PTR(T, outGrad, [E], outGrad_);
    DECL_VLA_PTR(T, wt, [E], weight_);

#pragma omp parallel for
    for (int u = 0; u < U; u++) {
      int start = mb_offsets[u];
      int end = mb_offsets[u+1];
      float g_sum[E];
      float sum = 0.0;

#ifdef USE_LIBXSMM_JIT

     // lookup reduction kernel
      libxsmm_meltw_reduce_cols_idx_param params;
      params.n = end - start;
      params.ind_ptr = &mb_indices[start];
      params.inp_ptr = outGrad;
      params.out_ptr = &g_sum[0];
      kernel( &params );

      // squared + reduction kernel
      libxsmm_meltw_unary_param    params1;
      params1.in.primary = g_sum;
      params1.out.primary = &sum;
      kernel1( &params1 );

      sum /= E;
      int idx = wt_indices[u];
      float hi = h[idx];
      hi += sum;
      h[idx] = hi;
      float scale = lr / (sqrt(hi) + eps);

      // scale and accumulate kernel
      libxsmm_meltw_binary_param binary_param;
      binary_param.in0.primary  = (void*)&scale;
      binary_param.in1.primary  = (void*)g_sum;
      binary_param.out.primary  = (void*)&wt[idx][0];
      kernel2(&binary_param);

#else
      for (int l = start; l < end; l++) {
        int idx = mb_indices[l];
        for (int e = 0; e < E; e++) {
          g_sum[e] += outGrad[idx][e];
        }
      }
      for (int e = 0; e < E; e++) {
        sum += g_sum[e] * g_sum[e];
      }
      sum /= E;
      int idx = wt_indices[u];
      float hi = h[idx];
      hi += sum;
      h[idx] = hi;
      float scale = lr / (sqrt(hi) + eps);
      for (int e = 0; e < E; e++) {
        wt[idx][e] += g_sum[e] * scale;
      }
#endif
    }
  }

  T *weight_;
  T *h;
  int M;
  int E;

#ifdef USE_LIBXSMM_JIT
  int _ld;
  libxsmm_meltwfunction_reduce_cols_idx kernel;
  libxsmm_meltwfunction_unary kernel1;
  libxsmm_meltwfunction_binary kernel2;
#endif
};

typedef EmbeddingBagImpl<FTyp> EmbeddingBag;


struct EmbeddingInOut {
  int M, N, NS, E, U;
  ITyp *offsets;
  ITyp *indices;
  FTyp *output;
  FTyp *gradout;
  FTyp *grads;
  ITyp * mb_offsets;
  ITyp * mb_indices;
  ITyp * wt_indices;
};

void sparse_transpose_radix(EmbeddingInOut *eio)
{
  int M = eio->M;
  int N = eio->N;
  int NS = eio->NS;
  Key_Value_Pair<int>* tmpBuf = (Key_Value_Pair<int>*)my_malloc((NS) * sizeof(Key_Value_Pair<int>), alignment);
  Key_Value_Pair<int>* tmpBuf1 = (Key_Value_Pair<int>*)my_malloc((NS) * sizeof(Key_Value_Pair<int>), alignment);

  auto t0 = get_time();
#pragma omp parallel for
  for(int i = 0; i < N; i++) {
    int start = eio->offsets[i];
    int end = eio->offsets[i+1];
    for (int j = start; j < end; j++) {
      tmpBuf[j].first = eio->indices[j];
      tmpBuf[j].second = i;
    }
  }
  auto t1 = get_time();
#ifdef DEBUG_TIME
  printf("Keypair buffer fill Time = %.3f ms\n", t1-t0);
#endif

  t0 = get_time();
  Key_Value_Pair<int>* tmpBuf2 = radix_sort_parallel<int>(&tmpBuf[0], &tmpBuf1[0], NS, M);
  t1 = get_time();
#ifdef DEBUG_TIME
  printf("Radix Sort Time = %.3f ms\n", t1-t0);
#endif

  int max_thds = omp_get_max_threads();
  int num_uniq[max_thds];

  t0 = get_time();
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    num_uniq[tid] = 0;
#pragma omp for schedule(static)
    for (int i = 1; i < NS; i++) {
      if (tmpBuf2[i].first != tmpBuf2[i-1].first) num_uniq[tid]++;
    }
  }

  num_uniq[0] += 1;
  for(int i = 1; i < max_thds; i++)
    num_uniq[i] += num_uniq[i-1];
  int U = num_uniq[max_thds-1];
  t1 = get_time();
#ifdef DEBUG_TIME
  printf("Num Unique Index Time = %.3f ms\n", t1-t0);
#endif

  t0 = get_time();
  eio->mb_offsets[0] = 0;
  eio->mb_indices[0] = tmpBuf2[0].second;
  eio->wt_indices[0] = tmpBuf2[0].first;
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    ITyp *tstart = (tid == 0 ? eio->wt_indices + 1 : eio->wt_indices + num_uniq[tid-1]);
    ITyp *t_offs = (tid == 0 ? eio->mb_offsets + 1 : eio->mb_offsets + num_uniq[tid-1]);
#pragma omp for schedule(static)
    for (int i = 1; i < NS; i++) {
      eio->mb_indices[i] = tmpBuf2[i].second;
      if (tmpBuf2[i].first != tmpBuf2[i-1].first) {
        *tstart = tmpBuf2[i].first;
        *t_offs = i;
        tstart++;
        t_offs++;
      }
    }
  }
  t1 = get_time();
#ifdef DEBUG_TIME
  printf("Offset/Index array construction Time = %.3f ms\n", t1-t0);
#endif
  eio->mb_offsets[U] = NS;
  eio->U = U;
  my_free(tmpBuf);
  my_free(tmpBuf1);
}

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

void allocate_buffers_and_generte_rnd_input(int N, int P, double alpha, EmbeddingBag *eb, EmbeddingInOut *eio)
{
  int E = eb->E;
  int M = eb->M;
  int NS = 0;
  eio->M = M;
  eio->N = N;
  eio->E = E;

  eio->offsets = (ITyp*)my_malloc((N+1) * sizeof(ITyp), alignment);
  eio->output = (FTyp*)my_malloc(N * E * sizeof(FTyp), alignment);
  eio->gradout = (FTyp*)my_malloc(N * E * sizeof(FTyp), alignment);
  init_zero(N * E, eio->output);
  init_random(N * E, eio->gradout, -0.01f, 0.01f);

  eio-> offsets[0] = 0;
  for(int i = 1; i <= N; i++) {
    double randval;
    drand48_r(&rand_buf, &randval);
    int cp = (int)(randval * P * 2);
    if (cp == 0) cp = 1;
    NS += cp;
    eio->offsets[i] = NS;
  }
  eio->NS = NS;
  eio->indices = (ITyp*)my_malloc(NS * sizeof(ITyp), alignment);
  eio->grads = (FTyp*)my_malloc(NS * E * sizeof(FTyp), alignment);
  init_zero(NS * E, eio->grads);
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
  eio->U = -1;
  eio->mb_offsets = (ITyp*)my_malloc(NS * sizeof(ITyp), alignment);
  eio->mb_indices = (ITyp*)my_malloc(NS * sizeof(ITyp), alignment);
  eio->wt_indices = (ITyp*)my_malloc(NS * sizeof(ITyp), alignment);
  init_zero(NS, eio->mb_offsets);
  init_zero(NS, eio->mb_indices);
  init_zero(NS, eio->wt_indices);
  auto t0 = get_time();
  sparse_transpose_radix(eio);
  auto t1 = get_time();
#ifdef DEBUG_TIME
  printf("Trans Time = %.3f ms\n", t1-t0);
#endif
}

void free_buffers(EmbeddingInOut *eio)
{
  my_free(eio->grads);
  my_free(eio->indices);
  my_free(eio->gradout);
  my_free(eio->output);
  my_free(eio->offsets);
  my_free(eio->mb_offsets);
  my_free(eio->mb_indices);
  my_free(eio->wt_indices);
}

int iters = 100;
int N = 2048;
int E = 64;
int P = 100;
int M = 1000000;
int S = 8;
double alpha = 0.0;

#define my_printf(fmt, args...) printf(fmt, args)

int main(int argc, char * argv[]) {
  if(argc > 1 && strncmp(argv[1], "-h", 3) == 0) {
    printf("Usage: %s iters N E M S P alpha \n", argv[0]);
    printf("iters: Number of iterations (= %d)\n", iters);
    printf("N: Minibatch (= %d)\n", N);
    printf("E: embedding row width (= %d)\n", E);
    printf("M: Number of rows per table (= %d)\n", M);
    printf("S: Number of Tables (= %d)\n", S);
    printf("P: Average number of indices per look up (= %d)\n", P);
    printf("alpha: Alpha value for Zipf distribution to generate Indices. Use 0 for uniform distribution");
    exit(0);
  }

  {
    int i = 1;
    if(argc > i) iters = atoi(argv[i++]);
    if(argc > i) N = atoi(argv[i++]);
    if(argc > i) E = atoi(argv[i++]);
    if(argc > i) M = atoi(argv[i++]);
    if(argc > i) S = atoi(argv[i++]);
    if(argc > i) P = atoi(argv[i++]);
    if(argc > i) alpha = atof(argv[i++]);
  }

  printf("Using: iters: %d N: %d E: %d M: %d S: %d P: %d alpha: %f\n", iters, N, E, M, S, P, alpha);

  double checksum = 0.0;

  int LS = S;
  int LN = N;
  set_random_seed(777);

#ifdef USE_PERF_COUNTERS
  ctrs_skx_uc a, b, s;
  bw_gibs bw_avg;

#ifdef USE_LLC_COUNTERS
  setup_skx_uc_ctrs( CTRS_EXP_CHA_LLC_LOOKUP );
#else
  setup_skx_uc_ctrs( CTRS_EXP_DRAM_CAS );
#endif
  zero_skx_uc_ctrs( &a );
  zero_skx_uc_ctrs( &b );
  zero_skx_uc_ctrs( &s );
#endif

  EmbeddingInOut *eio[iters][LS];
  EmbeddingBag *eb[LS];
  size_t tNS = 0;
  size_t tU = 0;

  for(int i = 0; i < LS; i++)
  {
    eb[i] = new EmbeddingBag(M, E);
    eb[i]->init();
    for(int j = 0; j < iters; j++)
    {
      eio[j][i] = new EmbeddingInOut();
      auto t0 = get_time();
      allocate_buffers_and_generte_rnd_input(N, P, alpha, eb[i], eio[j][i]);
      auto t1 = get_time();
#ifdef DEBUG_TIME
      printf("Rand init time = %.3f ms\n", t1 - t0);
#endif
      tNS += eio[j][i]->NS;
      tU += eio[j][i]->U;
    }
  }
  int warmup = (iters > 2 ? 2 : iters);

  for(int i = 0; i < warmup; i++) {
    double t0 = get_time();
    for(int s = 0; s < LS; s++) {
      eb[s]->fused_backward_update_adagrad(eio[i][s]->U, eio[i][s]->NS, N, eio[i][s]->mb_offsets, eio[i][s]->mb_indices, eio[i][s]->wt_indices, eio[i][s]->gradout, -0.1, 1.0e-6);
    }
    double t1 = get_time();
#ifdef DEBUG_TIME
    printf("Warmup Iter %4d: Time = %.3f ms\n", i, t1-t0);
#endif
  }

#ifdef USE_PERF_COUNTERS
  read_skx_uc_ctrs( &a );
#endif
  double t0 = get_time();
  double bwdupdTime = 0.0;

  for(int i = 0; i < iters; i++) {
    double t0 = get_time();
    for(int s = 0; s < LS; s++) {
      //printf("Gradout checksum = %g\n", get_checksum(eio[i][s]->gradout, N*E));
      eb[s]->fused_backward_update_adagrad(eio[i][s]->U, eio[i][s]->NS, N, eio[i][s]->mb_offsets, eio[i][s]->mb_indices, eio[i][s]->wt_indices, eio[i][s]->gradout, -0.1, 1.0e-6);
    }
    double t1 = get_time();
    //printf("Iter %4d: Time = %.3f ms\n", i, t1-t0);
    bwdupdTime += t1-t0;
  }
  double t1 = get_time();
#ifdef USE_PERF_COUNTERS
  read_skx_uc_ctrs( &b );
  difa_skx_uc_ctrs( &a, &b, &s );
  divi_skx_uc_ctrs( &s, iters );
#endif
#ifdef VERIFY_CORRECTNESS
  for(int s = 0; s < LS; s++) {
    double psum = get_checksum(eb[s]->weight_, M*E);
    //my_printf("PSUM %d: %g\n", s, psum);
    checksum += psum;
  }
#endif

  //  tU*E wt RW + N*E gO R + N mb_offsets + NS mb_ind + U wt_ind
  size_t bwdupdBytesMinRd = ((size_t)tU*(E+1)) * sizeof(FTyp) + ((size_t)tNS+tU) * sizeof(ITyp) + ((size_t)iters*LS*N*E) * sizeof(FTyp) + ((size_t)iters*LS*N) * sizeof(ITyp);
  size_t bwdupdBytesMaxRd = ((size_t)tU*(E+16)) * sizeof(FTyp) + ((size_t)tNS+tU) * sizeof(ITyp) + ((size_t)iters*LS*N*E) * sizeof(FTyp) + ((size_t)iters*LS*N) * sizeof(ITyp);

  size_t bwdupdBytesMinWr = ((size_t)tU*(E+1)) * sizeof(FTyp);
  size_t bwdupdBytesMaxWr = ((size_t)tU*(E+16)) * sizeof(FTyp);

  size_t bwdupdBytesMin = bwdupdBytesMinRd + bwdupdBytesMinWr;
  size_t bwdupdBytesMax = bwdupdBytesMaxRd + bwdupdBytesMaxWr;

  my_printf("Iters = %d, LS = %d, N = %d, M = %d, E = %d, avgNS = %ld, avgU = %ld, P = %d\n", iters, LS, N, M, E, tNS/(iters*LS), tU/(iters*LS), P);
  my_printf("Per Iter  Time: %.3f ms  Total: %.3f ms\n", bwdupdTime/(iters), (t1-t0)/(iters));
  my_printf("Per Table Time: %.3f ms  Total: %.3f ms\n", bwdupdTime/(iters*LS), (t1-t0)/(iters*LS));

  double scale = 1000.0/(1024.0*1024.0*1024.0);
  my_printf("BW: RD Min: %.3f GB/s   Max: %.3f GB/s \n", bwdupdBytesMinRd*scale/bwdupdTime, bwdupdBytesMaxRd*scale/bwdupdTime);
  my_printf("BW: WR Min: %.3f GB/s   Max: %.3f GB/s \n", bwdupdBytesMinWr*scale/bwdupdTime, bwdupdBytesMaxWr*scale/bwdupdTime);
  my_printf("BW: TT Min: %.3f GB/s   Max: %.3f GB/s \n", bwdupdBytesMin*scale/bwdupdTime, bwdupdBytesMax*scale/bwdupdTime);

#ifdef USE_PERF_COUNTERS

#ifdef USE_LLC_COUNTERS
  get_llc_bw_skx( &s, bwdupdTime/iters/1000.0, &bw_avg );
  printf("Measured LLC AVG GB/s: RD %f   WR %f   TT %f\n", bw_avg.rd, bw_avg.wr, bw_avg.rd + bw_avg.wr);
#else
  get_cas_ddr_bw_skx( &s, bwdupdTime/iters/1000.0, &bw_avg );
  printf("Measured MEM AVG GB/s: RD %f   WR %f   TT %f\n", bw_avg.rd, bw_avg.wr, bw_avg.rd + bw_avg.wr);
#endif
#endif

#ifdef VERIFY_CORRECTNESS
  printf("Checksum = %g\n", checksum);
#endif

  for(int i = 0; i < LS; i++)
  {
    for(int j = 0; j < iters; j++)
    {
      free_buffers(eio[j][i]);
      delete eio[j][i];
    }
    delete eb[i];
  }
  return 0;
}
