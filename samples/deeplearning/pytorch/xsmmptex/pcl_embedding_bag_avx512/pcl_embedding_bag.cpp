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

#include <torch/extension.h>
#ifdef OLD_PYTORCH
#include <torch/csrc/autograd/record_function.h>
#else
#include <ATen/record_function.h>
#endif
#include <torch/csrc/autograd/VariableTypeUtils.h>

#include <vector>
#include <iostream>
#include <time.h>
#include <immintrin.h>
#include <sys/syscall.h>
#include <omp.h>
#include "rtm.h"
#include <libxsmm.h>
#include <libxsmm_rng.h>
#include <libxsmm_intrinsics_x86.h>

#include "xsmm_functors.h"

#define NO_TPP_FWD
#define NO_TPP_BWD
#define NO_TPP_UPD
#define NO_TPP_DOT

using namespace pcl;

#define MYASSERT(x) do { if(!(x)) {printf("Assert failed %s\n", #x); exit(1);} } while(0)

#define PCL_ASSERT(cond, x...) do { if(!(cond)) { printf(x); fflush(stdout); exit(1); } } while(0)

#define ALIGNDOWN(N, A) ((N) & ~((A)-1))
typedef at::BFloat16 bfloat16;
using namespace torch::autograd::profiler;

#define DECL_VLA_PTR(type, name, dims, ptr) type (*name)dims = (type (*)dims)ptr
#define DECL_VLA_PTR_PT(type, name, dims, t) type (*name)dims = (type (*)dims)(t.data_ptr<type>())

template<typename T> libxsmm_datatype getXsmmDtype();
template<> libxsmm_datatype getXsmmDtype<float>() { return LIBXSMM_DATATYPE_F32; }
template<> libxsmm_datatype getXsmmDtype<bfloat16>() { return LIBXSMM_DATATYPE_BF16; }

double get_time() {
  static bool init_done = false;
  static struct timespec stp = {0,0};
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tp);

  if(!init_done) {
    init_done = true;
    stp = tp;
  }
  double ret = (tp.tv_sec - stp.tv_sec) * 1e3 + (tp.tv_nsec - stp.tv_nsec)*1e-6;
  return ret;
}

#if 0 // defined in xsmm_functors.h
#ifdef __AVX512F__
inline __m512 _mm512_loadu_ps_auto (float const* mem_addr) { return _mm512_loadu_ps(mem_addr);}
inline __m512 _mm512_maskz_loadu_ps_auto (__mmask16 k, float const* mem_addr) { return _mm512_maskz_loadu_ps (k, mem_addr); }
inline void _mm512_storeu_ps_auto (float* mem_addr, __m512 a) { _mm512_storeu_ps (mem_addr, a); }
inline void _mm512_mask_storeu_ps_auto (float* mem_addr, __mmask16 k, __m512 a) { _mm512_mask_storeu_ps (mem_addr, k, a); }

inline __m512 _mm512_convert_bf_ps(__m256i a) { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(a),16)); }
inline __m256i _mm256_convert_ps_bf(__m512 a) { return _mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(a),16)); }

inline __m512 _mm512_loadu_ps_auto (bfloat16 const* mem_addr) { return _mm512_convert_bf_ps(_mm256_loadu_si256((__m256i*)mem_addr));}
inline __m512 _mm512_maskz_loadu_ps_auto (__mmask16 k, bfloat16 const* mem_addr) { return _mm512_convert_bf_ps(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr));}
inline void _mm512_storeu_ps_auto (bfloat16* mem_addr, __m512 a) { _mm256_storeu_si256 ((__m256i*)mem_addr, _mm256_convert_ps_bf(a)); }
inline void _mm512_mask_storeu_ps_auto (bfloat16* mem_addr, __mmask16 k, __m512 a) { _mm256_mask_storeu_epi16 ((__m256i*)mem_addr, k, _mm256_convert_ps_bf(a)); }

inline __m512 _mm512_split_loadu_ps(bfloat16 const* hi, bfloat16 const* lo) {
  auto yh = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)hi));
  auto yl = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)lo));
  return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_bslli_epi128(yh, 2), yl));
}
inline __m512 _mm512_maskz_split_loadu_ps(__mmask16 k, bfloat16 const* hi, bfloat16 const* lo) {
  auto yh = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, (__m256i*)hi));
  auto yl = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, (__m256i*)lo));
  return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_bslli_epi128(yh, 2), yl));
}
inline void _mm512_split_storeu_ps(bfloat16 *hi, bfloat16 *lo, __m512 a) {
  _mm256_storeu_si256((__m256i*)hi, _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
  _mm256_storeu_si256((__m256i*)lo, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
}
inline void _mm512_mask_split_storeu_ps(bfloat16 *hi, bfloat16 *lo, __mmask16 k, __m512 a) {
  _mm256_mask_storeu_epi16((__m256i*)hi, k, _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
  _mm256_mask_storeu_epi16((__m256i*)lo, k, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
}
#endif
#endif

inline void vec_split_buf_muladd(at::BFloat16 *inout_hi, at::BFloat16 *inout_lo, at::BFloat16 *in, int len, float alpha) {
#if defined(__AVX512F__)
  __m512 vAlpha = _mm512_set1_ps(alpha);
  int i = 0;
  for(; i < ALIGNDOWN(len, 16); i += 16) {
    auto y1 = _mm512_split_loadu_ps(inout_hi+i, inout_lo+i);
    auto y2 = _mm512_loadu_ps_auto(in+i);
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm512_split_storeu_ps(inout_hi+i, inout_lo+i, y1);
  }
  if(i < len) {
    int rem = len - i;
    __mmask16 mask = (1 << rem) - 1;
    auto y1 = _mm512_maskz_split_loadu_ps(mask, inout_hi+i, inout_lo+i);
    auto y2 = _mm512_maskz_loadu_ps_auto(mask, in+i);
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm512_mask_split_storeu_ps(inout_hi+i, inout_lo+i, mask, y1);
  }
#else
  for(int i = 0; i < len; i++) {
    union {
      at::BFloat16 i[2];
      float f;
    } s;
    s.i[0] = inout_lo[i];
    s.i[1] = inout_hi[i];
    s.f += in[i] * alpha;
    inout_lo[i] = s.i[0];
    inout_hi[i] = s.i[1];
  }
#endif
}

template<typename T1, typename T2>
inline void vec_muladd(T1 *inout, T2 *in, int len, float alpha) {
#ifdef __AVX512F__
  auto vAlpha = _mm512_set1_ps(alpha);
  long i;
  for (i = 0; i < ALIGNDOWN(len, 16); i += 16) {
    auto y1 = _mm512_loadu_ps_auto(inout+i);
    auto y2 = _mm512_loadu_ps_auto(in+i);
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm512_storeu_ps_auto(inout+i, y1);
  }
  if (i < len) {
    int rem = len - i;
    __mmask16 mask = (1 << rem) - 1;
    auto y1 = _mm512_maskz_loadu_ps_auto(mask, inout+i);
    auto y2 = _mm512_maskz_loadu_ps_auto(mask, in+i);
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm512_mask_storeu_ps_auto(inout+i, mask, y1);
  }
#else
#pragma omp simd
  for(long v = 0; v < len; v++) {
    inout[v] += in[v] * alpha;
  }
#endif
}

#ifdef FP32_OUTPUT
#define out_scalar_t float
#else
#define out_scalar_t scalar_t
#endif

template<typename scalar_t>
void pcl_embedding_bag_forward_tmpl(torch::Tensor t_weight, torch::Tensor t_input, torch::Tensor t_offsets, torch::Tensor& t_output)
{
  auto N = t_offsets.size(0);
  auto NS = t_input.size(0);
  //auto M = t_weight.size(0);
  auto E = t_weight.size(1);
  t_input = t_input.contiguous();
  t_offsets = t_offsets.contiguous();

  DECL_VLA_PTR_PT(scalar_t, weight, [E], t_weight);
  DECL_VLA_PTR_PT(out_scalar_t, output, [E], t_output);
  int64_t *input = t_input.data_ptr<int64_t>();
  int64_t *offsets = t_offsets.data_ptr<int64_t>();

#ifndef NO_TPP_FWD
  auto embbag = EmbBagFwdTPP<scalar_t, out_scalar_t, int64_t>(E);

#pragma omp parallel for
  for (int n = 0; n < N; n++)
  {
    auto start = offsets[n];
    auto end = (n < N - 1 ? offsets[n + 1] : NS);

    embbag(output[n], weight[0], &input[start], end-start);
  }
#else // NO_TPP
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
      auto ind = input[s];
      vec_muladd(output[n], weight[ind], E, 1.0f);
    }
  }
#endif // NO_TPP
}

// kHalf, kBFloat16, kFloat
at::Tensor pcl_embedding_bag_forward(torch::Tensor weight, torch::Tensor input, torch::Tensor offsets)
{
  auto N = offsets.size(0);
  //auto NS = input.size(0);
  auto E = weight.size(1);
#ifdef FP32_OUTPUT
  auto opts = weight.options().dtype(at::kFloat);
#else
  auto opts = weight.options();
#endif
  at::Tensor output = at::empty({N, E}, opts);
  if (weight.dtype() == at::kFloat) {
    pcl_embedding_bag_forward_tmpl<float>(weight, input, offsets, output);
  } else if (weight.dtype() == at::kBFloat16) {
    pcl_embedding_bag_forward_tmpl<bfloat16>(weight, input, offsets, output);
  } else {
    PCL_ASSERT(0, "This datatype is not supported\n");
  }
  return output;
}

template<typename scalar_t>
inline void pcl_embedding_bag_backward_tmpl(torch::Tensor t_gradout, torch::Tensor t_weight, torch::Tensor t_input, torch::Tensor t_offsets, torch::Tensor t_values)
{
  auto N = t_offsets.size(0);
  auto NS = t_input.size(0);
  auto E = t_gradout.size(1);

  //DECL_VLA_PTR_PT(scalar_t, weight, [E], t_weight);
  DECL_VLA_PTR_PT(scalar_t, values, [E], t_values);
  DECL_VLA_PTR_PT(out_scalar_t, gradout, [E], t_gradout);
  //int64_t *input = t_input.data_ptr<int64_t>();
  int64_t *offsets = t_offsets.data_ptr<int64_t>();
#ifndef NO_TPP_BWD
  auto embbag_bwd = EmbBagBwdTPP<out_scalar_t, scalar_t>(E);
#pragma omp parallel for
  for (int n = 0; n < N; n++) {
    auto start = offsets[n];
    auto end = (n < N - 1 ? offsets[n + 1] : NS);
    embbag_bwd(gradout[n], values[start], end-start);
  }
#else
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
#endif
}

at::Tensor pcl_embedding_bag_backward(torch::Tensor gradout, torch::Tensor weight, torch::Tensor input, torch::Tensor offsets)
{
  auto NS = input.size(0);
  auto E = gradout.size(1);
  auto values = at::empty({NS, E}, weight.options());
  auto indices = input.reshape({1, -1});
  if (weight.dtype() == at::kFloat) {
    pcl_embedding_bag_backward_tmpl<float>(gradout, weight, input, offsets, values);
  } else if (weight.dtype() == at::kBFloat16) {
    pcl_embedding_bag_backward_tmpl<bfloat16>(gradout, weight, input, offsets, values);
  } else {
    PCL_ASSERT(0, "This datatype is not supported\n");
  }

  auto grad_weight = at::_sparse_coo_tensor_unsafe(indices, values, weight.sizes());

  return grad_weight;
}

static int emb_update_use_lock_free() {
  int lock_free = 1;
  char *str = getenv("PCL_USE_RTM_UPDATE");
  if(str && atoi(str) > 0) {
    lock_free = 0;
    printf("PCL_EMBEDDING_BAG: Using RTM Based Update\n");
  } else {
    printf("PCL_EMBEDDING_BAG: Using Lock Free Update\n");
  }
  return lock_free;
}

static int use_lock_free = emb_update_use_lock_free();

template<typename scalar_t>
void pcl_dense_sparse_add_tmpl(torch::Tensor t_dense, torch::Tensor t_sparse, float alpha)
{
  auto NS = t_sparse._nnz();
  auto M = t_dense.size(0);
  auto E = t_dense.size(1);
  auto t_values = t_sparse._values();
  auto t_indices = t_sparse._indices();

  PCL_ASSERT(t_dense.is_contiguous(), "dense tensor muct be contiguous\n");
  // Not using below due to spurious compiler warnings
  //DECL_VLA_PTR_PT(scalar_t, dense, [E], t_dense);
  //DECL_VLA_PTR_PT(scalar_t, values, [E], t_values);
  auto dense = t_dense.data_ptr<scalar_t>();
  auto values = t_values.data_ptr<scalar_t>();
  auto indices = t_indices.data_ptr<long>();
  auto lr = alpha;
#ifndef NO_TPP_UPD
  auto embbag_upd = ScaleAddTPP<scalar_t, scalar_t>(E);
#endif

  int max_thr = omp_get_max_threads();
  if(use_lock_free) {
    int nthr = max_thr;
    if(M < nthr) nthr = M;
#pragma omp parallel num_threads(nthr)
    {
      int tid = omp_get_thread_num();
      long j_begin = (tid * M) / nthr;
      long j_end = ((tid+1) * M) / nthr;
      for(long i = 0; i < NS; i++) {
        auto ind = indices[i];
        if(ind >= j_begin && ind < j_end) {
          auto wa = &dense[ind * E];
          auto va = &values[i * E];
#ifndef NO_TPP_UPD
          embbag_upd(va, wa, lr);
#else
          vec_muladd(wa, va, E, lr);
#endif
        }
      }
    }
  } else {
    SimpleSpinLock fallBackLock;
#pragma omp parallel for
    for(int i = 0; i < NS; i++) {
      auto ind = indices[i];
      auto wa = &dense[ind * E];
      auto va = &values[i * E];
      {
        TransactionScope guard(fallBackLock, 100);
#ifndef NO_TPP_UPD
        embbag_upd(va, wa, lr);
#else
        vec_muladd(wa, va, E, lr);
#endif
      }
    }
  }
}

void pcl_dense_sparse_add(torch::Tensor dense, torch::Tensor sparse, /*torch::Scalar*/ float alpha)
{
  RECORD_FUNCTION("pcl_dense_sparse_add", std::vector<c10::IValue>({dense, sparse, alpha}));
  //auto t0 = get_time();
  if (dense.dtype() == at::kFloat) {
    pcl_dense_sparse_add_tmpl<float>(dense, sparse, alpha);
  //} else if (dense.dtype() == at::kBFloat16) {
  //  pcl_dense_sparse_add_tmpl<bfloat16>(dense, sparse, alpha);
  } else {
    PCL_ASSERT(0, "This datatype is not supported\n");
  }
}

void pcl_bf16_update(torch::Tensor hi_bits, torch::Tensor lo_bits, torch::Tensor grad, float lr)
{
  //auto t0 = get_time();
  //std::cout << "Calling pcl_bf16_update - " << hi_bits.sizes() << std::endl;
  if(grad.is_sparse()) {
    RECORD_FUNCTION("pcl_bf16_sparse_update", std::vector<c10::IValue>({hi_bits, lo_bits, grad, lr}));
    auto sparse = grad;
    auto NS = sparse._nnz();
    auto M = hi_bits.size(0);
    auto E = hi_bits.size(1);
    auto values_tensor = sparse._values();
    auto indices = sparse._indices();
    auto indices_data = indices.data_ptr<long>();
#ifndef NO_TPP_UPD
    auto split_sgd_kernel = SplitSGDTPP(E);
#endif

    if(hi_bits.is_contiguous() && values_tensor.is_contiguous() && indices.is_contiguous()) {
      auto hi_data = (unsigned short*)hi_bits.data_ptr();
      auto lo_data = (unsigned short*)lo_bits.data_ptr();
      auto values_data = values_tensor.data_ptr<at::BFloat16>();
      int max_thr = omp_get_max_threads();
      if(use_lock_free) {
        int nthr = max_thr;
        if(M < nthr) nthr = M;
#pragma omp parallel num_threads(nthr)
        {
          int tid = omp_get_thread_num();
          long j_begin = (tid * M) / nthr;
          long j_end = ((tid+1) * M) / nthr;
          for(long i = 0; i < NS; i++) {
            auto ind = indices_data[i];
            if(ind >= j_begin && ind < j_end) {
              auto ha = &hi_data[ind * E];
              auto la = &lo_data[ind * E];
              auto va = &values_data[i * E];
#ifndef NO_TPP_UPD
              split_sgd_kernel((at::BFloat16 *)ha, (at::BFloat16 *)la, va, lr);
#else
              vec_split_buf_muladd((at::BFloat16 *)ha, (at::BFloat16 *)la, va, E, lr);
#endif
            }
          }
        }
      } else {
        SimpleSpinLock fallBackLock;
#pragma omp parallel for
        for(long i = 0; i < NS; i++) {
          auto ind = indices_data[i];
          auto ha = &hi_data[ind * E];
          auto la = &lo_data[ind * E];
          auto va = &values_data[i * E];
          {
            TransactionScope guard(fallBackLock, 100);
#ifndef NO_TPP_UPD
            split_sgd_kernel((at::BFloat16 *)ha, (at::BFloat16 *)la, va, lr);
#else
            vec_split_buf_muladd((at::BFloat16 *)ha, (at::BFloat16 *)la, va, E, lr);
#endif
          }
        }
      }
    }
  } else {
    RECORD_FUNCTION("pcl_bf16_dense_update", std::vector<c10::IValue>({hi_bits, lo_bits, grad, lr}));
    MYASSERT(hi_bits.is_contiguous() && lo_bits.is_contiguous() && grad.is_contiguous());
    auto hi_ptr = (unsigned short *)hi_bits.data_ptr();
    auto lo_ptr = (unsigned short *)lo_bits.data_ptr();
    auto grad_ptr = grad.data_ptr<at::BFloat16>();
    long sz = hi_bits.numel();
    constexpr int block_size = 64;
#ifndef NO_TPP_UPD
    auto split_sgd_kernel = SplitSGDTPP(block_size);
#endif
    long i = 0;
#pragma omp parallel for lastprivate(i)
    for(i = 0; i < ALIGNDOWN(sz, block_size); i += block_size) {
#ifndef NO_TPP_UPD
      split_sgd_kernel((at::BFloat16 *)(hi_ptr+i), (at::BFloat16 *)(lo_ptr+i), grad_ptr+i, lr);
#else
      vec_split_buf_muladd((at::BFloat16 *)(hi_ptr+i), (at::BFloat16 *)(lo_ptr+i), grad_ptr+i, block_size, lr);
#endif
    }
    if (i < sz) {
#ifndef NO_TPP_UPD
      auto split_sgd_kernel = SplitSGDTPP(sz - i);
      split_sgd_kernel((at::BFloat16 *)(hi_ptr+i), (at::BFloat16 *)(lo_ptr+i), grad_ptr+i, lr);
#else
      vec_split_buf_muladd((at::BFloat16 *)(hi_ptr+i), (at::BFloat16 *)(lo_ptr+i), grad_ptr+i, sz - i, lr);
#endif
    }
  }
}

libxsmm_smmfunction get_smm_kernel(int M, int N, int K) {
  libxsmm_smmfunction mm_kernel;
  float alpha = 1.0;
  float beta = 0.0;
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  mm_kernel = libxsmm_smmdispatch( N, M, K, NULL, NULL, NULL, &alpha, &beta, &flags, NULL );
  MYASSERT(mm_kernel);
  return mm_kernel;
}

libxsmm_bmmfunction get_bmm_kernel(int M, int N, int K) {
  libxsmm_bmmfunction mm_kernel;
  float alpha = 1.0;
  float beta = 0.0;
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
  mm_kernel = libxsmm_bmmdispatch( N, M, K, NULL, NULL, NULL, &alpha, &beta, &flags, NULL );
  MYASSERT(mm_kernel);
  return mm_kernel;
}

libxsmm_xtransfunction get_tr_kernel(int typesize, int M, int N, int LDO) {
  libxsmm_xtransfunction tr_kernel;
  libxsmm_descriptor_blob blob;
  libxsmm_trans_descriptor* tr_desc;
  tr_desc = libxsmm_trans_descriptor_init(&blob, typesize, M, N, LDO);
  tr_kernel = libxsmm_dispatch_trans(tr_desc);
  MYASSERT(tr_kernel);
  return tr_kernel;
}

at::Tensor pcl_bdot_forward(at::Tensor &in)
{
  auto sizes = in.sizes();
  int MB = sizes[0];
  unsigned int M = sizes[1];
  unsigned int K = sizes[2];
  at::Tensor out = at::empty({MB, M, M}, in.options());

  if (in.scalar_type() == at::kFloat) {
    //std::cout << "pcl_dot: float " << in.scalar_type() << std::endl;
    libxsmm_smmfunction mm_kernel_f32 = get_smm_kernel( M, M, K );
#ifdef NO_TPP_DOT
    libxsmm_xtransfunction tr_kernel_f32 = get_tr_kernel( sizeof(float), K, M, M);
#else
    auto tr_kernel_tpp = XformExtTPP<float>(M, K, XformTPP::XFORM_XPOSE_TPP);
#endif
    float *input = in.data_ptr<float>();
    float *output = out.data_ptr<float>();

#pragma omp parallel for
    for (int i = 0; i < MB; ++i) {
      float tmpa[M*K];
#ifdef NO_TPP_DOT
      tr_kernel_f32( &input[i*M*K], &K, tmpa, &M );
#else
      tr_kernel_tpp(&input[i*M*K], tmpa);
#endif
      mm_kernel_f32( tmpa, &input[i*M*K], &output[i*M*M] );
    }
  } else if (in.scalar_type() == at::kBFloat16) {
    //std::cout << "pcl_dot: bfloat " << in.scalar_type() << std::endl;
    MYASSERT(K % 2 == 0);
    unsigned int K2 = K/2;
    libxsmm_bmmfunction mm_kernel_bf16 = get_bmm_kernel( M, M, K );
#ifdef NO_TPP_DOT
    libxsmm_xtransfunction tr_kernel_bf16 = get_tr_kernel( sizeof(float), K2, M, M);
#else
    auto tr_kernel_tpp = XformExtTPP<at::BFloat16>(M, K, XformTPP::XFORM_XPOSE_N2V_TPP);
#endif
    auto *input = in.data_ptr<at::BFloat16>();
    auto *output = out.data_ptr<at::BFloat16>();

#pragma omp parallel for
    for (int i = 0; i < MB; ++i) {
      at::BFloat16 tmpa[M*K];
      //at::BFloat16 tmpa[4096];
#ifdef NO_TPP_DOT
      tr_kernel_bf16( &input[i*M*K], &K2, tmpa, &M );
#else
      tr_kernel_tpp(&input[i*M*K], tmpa);
#endif
      mm_kernel_bf16( (libxsmm_bfloat16*)tmpa, (libxsmm_bfloat16*)&input[i*M*K], (libxsmm_bfloat16*)&output[i*M*M] );
    }
  } else {
    MYASSERT(0);
  }
  return out;
}

template<typename Tout, typename Tin>
void my_convert(Tout *out, Tin *in, long N)
{
#ifdef __AVX512F__
  int i;
  for (i = 0; i < ALIGNDOWN(N, 16); i+=16) {
    auto val = _mm512_loadu_ps_auto(in+i);
    _mm512_storeu_ps_auto(out+i, val);
  }
  if (i < N) {
    int rem = N - i;
    __mmask16 mask = (1 << rem) - 1;
    auto val = _mm512_maskz_loadu_ps_auto(mask, in+i);
    _mm512_mask_storeu_ps_auto(out+i, mask, val);
  }
#else
  for (int i = 0; i < N; i++) {
    out[i] = in[i];
  }
#endif
}

at::Tensor pcl_bdot_backward(at::Tensor &in1, at::Tensor &in2)
{
  auto sizes1 = in1.sizes();
  auto sizes2 = in2.sizes();
  int MB = sizes1[0];
  unsigned int M = sizes1[1];
  unsigned int K = sizes1[2];
  unsigned int N = sizes2[2];
  MYASSERT(M == K);
  at::Tensor out = at::empty({MB, M, N}, in1.options());
  //at::Tensor out = at::zeros({MB, M, N}, in1.options());
  libxsmm_smmfunction mm_kernel_f32 = get_smm_kernel( M, N, K );

  //std::cout << "in1: " << sizes1 << ", in2: " << sizes2 << ", out: " << out.sizes() << ", dt: " << in1.scalar_type() << std::endl;
  if (in1.scalar_type() == at::kFloat) {
    auto input1 = in1.data_ptr<float>();
    auto input2 = in2.data_ptr<float>();
    auto output = out.data_ptr<float>();
#pragma omp parallel for
    for (int i = 0; i < MB; ++i) {
      float tmpa[M*K];
      //float tmpa[4096];
      for(unsigned int j = 0; j < M; j++) {
        for(unsigned int k = 0; k < K; k++) {
          tmpa[j*K+k] = input1[i*M*K+j*K+k] + input1[i*M*K+k*M+j];
        }
      }
      mm_kernel_f32( &input2[i*K*N], tmpa, &output[i*M*N] );
    }
  } else if (in1.scalar_type() == at::kBFloat16) {
    auto input1 = in1.data_ptr<at::BFloat16>();
    auto input2 = in2.data_ptr<at::BFloat16>();
    auto output = out.data_ptr<at::BFloat16>();
#ifndef NO_TPP_DOT
    auto cvt_in1 = ConvertTPP<at::BFloat16, float>(M*K);
    auto cvt_in2 = ConvertTPP<at::BFloat16, float>(K*N);
    auto cvt_out = ConvertTPP<float, at::BFloat16>(M*N);
#endif
#pragma omp parallel for
    for (int i = 0; i < MB; ++i) {
      float tmpa[M*K], tmpb[K*N], tmpc[M*N];
      //float tmpa[4096], tmpb[4096], tmpc[4096];
#ifdef NO_TPP_DOT
      my_convert(tmpa, &input1[i*M*K], M*K);
      my_convert(tmpb, &input2[i*K*N], K*N);
#else
      cvt_in1(&input1[i*M*K], tmpa);
      cvt_in2(&input2[i*K*N], tmpb);
#endif
      for(unsigned int j = 0; j < M; j++) {
        for(unsigned int k = 0; k < K; k++) {
          tmpa[j*K+k] += tmpa[k*M+j];
        }
      }
      mm_kernel_f32( tmpb, tmpa, tmpc );
#ifdef NO_TPP_DOT
      my_convert(&output[i*M*N], tmpc, M*N);
#else
      cvt_out(tmpc, &output[i*M*N]);
#endif
    }
  } else {
    MYASSERT(0);
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pcl_embedding_bag_forward, "Pcl Embedding Bag forward");
  m.def("backward", &pcl_embedding_bag_backward, "Pcl Embedding Bag backward");
  m.def("dense_sparse_add", &pcl_dense_sparse_add, "Pcl pcl_dense_sparse_add");
  m.def("bf16_update", &pcl_bf16_update, "Pcl pcl_bf16_update");
  m.def("bdot_forward", &pcl_bdot_forward, "Pcl batch dot forward");
  m.def("bdot_backward", &pcl_bdot_backward, "Pcl batch dot backward");
}
