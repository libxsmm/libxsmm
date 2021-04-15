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
//#include <torch/csrc/autograd/record_function.h>
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
#include <libxsmm_intrinsics_x86.h>
#include <immintrin.h>

#include "xsmm_functors.h"
#include "timing.h"

using namespace pcl;

#define ALIGNDOWN(N, A) ((N) & ~((A)-1))

#define NO_TPP_AF
#define NO_TPP_OF
#define NO_TPP_IF
#define NO_TPP_EF
#define NO_TPP_AB
#define NO_TPP_OB
#define NO_TPP_IB
#define NO_TPP_EB
#define NO_TPP_ADAM

#define PCL_ASSERT(cond, x...) do { if(!(cond)) { printf(x); fflush(stdout); exit(1); } } while(0)

typedef at::BFloat16 bfloat16;

#define DECL_VLA_PTR(type, name, dims, ptr) type (*name)dims = (type (*)dims)ptr
#define DECL_VLA_PTR_PT(type, name, dims, t) type (*name)dims = (type (*)dims)(t.data_ptr<type>())
#define DTS sizeof(float)

#define AT_DISPATCH_FLOAT_AND_BFLOAT16_TYPES(TYPE, NAME, ...)                \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::BFloat16, bfloat16, __VA_ARGS__)      \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#ifdef __AVX512F__
inline __m512 _mm512_exp_ps(__m512 a) {
  float a16[16]; int i;
  _mm512_storeu_ps(a16, a);
  for (i = 0; i < 16; ++i) a16[i] = expf(a16[i]);
  return _mm512_loadu_ps(a16);
}
#endif
//#define NO_APPROX_EXP
#ifdef NO_APPROX_EXP
#define _MM512_EXP_PS _mm512_exp_ps
#else
#define _MM512_EXP_PS LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS
#endif
template<typename T> libxsmm_datatype getXsmmDtype();
template<> libxsmm_datatype getXsmmDtype<float>() { return LIBXSMM_DATATYPE_F32; }
template<> libxsmm_datatype getXsmmDtype<bfloat16>() { return LIBXSMM_DATATYPE_BF16; }

#if 1
#if 0
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
  //_mm512_storeu_ps_auto(hi, a);
  _mm256_storeu_si256((__m256i*)hi, _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
  _mm256_storeu_si256((__m256i*)lo, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
}
inline void _mm512_mask_split_storeu_ps(bfloat16 *hi, bfloat16 *lo, __mmask16 k, __m512 a) {
  //_mm512_mask_storeu_ps_auto(hi, k, a);
  _mm256_mask_storeu_epi16((__m256i*)hi, k, _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
  _mm256_mask_storeu_epi16((__m256i*)lo, k, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
}
#endif
#else
inline __m512 _mm512_convert_bf_ps(__m256i a) { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(a),16)); }
inline __m256i _mm256_convert_ps_bf(__m512 a) { return _mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(a),16)); }

template<typename T> __m512 _mm512_loadu_ps_auto (T * mem_addr) { }
template<> __m512 _mm512_loadu_ps_auto<float> (float * mem_addr) { return _mm512_loadu_ps(mem_addr);}
template<> __m512 _mm512_loadu_ps_auto<bfloat16> (bfloat16 * mem_addr) { return _mm512_convert_bf_ps(_mm256_loadu_si256((__m256i*)mem_addr));}

template<typename T> __m512 _mm512_maskz_loadu_ps_auto (__mmask16 k, T * mem_addr) { }
template<> __m512 _mm512_maskz_loadu_ps_auto<float> (__mmask16 k, float * mem_addr) { return _mm512_maskz_loadu_ps (k, mem_addr); }
template<> __m512 _mm512_maskz_loadu_ps_auto (__mmask16 k, bfloat16 * mem_addr) { return _mm512_convert_bf_ps(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr));}

template<typename T> void _mm512_storeu_ps_auto (T* mem_addr, __m512 a) { }
template<> void _mm512_storeu_ps_auto<float> (float* mem_addr, __m512 a) { _mm512_storeu_ps (mem_addr, a); }
template<> void _mm512_storeu_ps_auto<bfloat16> (bfloat16* mem_addr, __m512 a) { _mm256_storeu_si256 ((__m256i*)mem_addr, _mm256_convert_ps_bf(a)); }

template<typename T> void _mm512_mask_storeu_ps_auto (T* mem_addr, __mmask16 k, __m512 a) { }
template<> void _mm512_mask_storeu_ps_auto<float> (float* mem_addr, __mmask16 k, __m512 a) { _mm512_mask_storeu_ps (mem_addr, k, a); }
template<> void _mm512_mask_storeu_ps_auto<bfloat16> (bfloat16* mem_addr, __mmask16 k, __m512 a) { _mm256_mask_storeu_epi16 ((__m256i*)mem_addr, k, _mm256_convert_ps_bf(a)); }
#endif

thread_local unsigned int *rnd_state = NULL;
thread_local struct drand48_data drnd_state; // For non AVX512 version
void set_rnd_seed(unsigned int seed)
{
#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

    if(rnd_state) {
      libxsmm_rng_destroy_extstate(rnd_state);
      rnd_state = NULL;
    }
    rnd_state = libxsmm_rng_create_extstate(seed+tid);
    srand48_r(seed+tid, &drnd_state);
  }
}

double ifreq = 1.0/getFreq();
PassType globalPass = FWD;
ScopeType globalScope = st_other;
double debug_timers[MAX_THREADS][2][NUM_TIMERS];
double scope_timers[MAX_THREADS][NUM_SCOPES][NUM_TIMERS];
double master_scope_timers[NUM_SCOPES];
double scope_flops[MAX_THREADS][NUM_ALIGNED_SCOPES];
double master_debug_timers[2];

void reset_debug_timers() {
  master_debug_timers[0] = 0.0;
  master_debug_timers[1] = 0.0;
  for (int p = 0; p < NUM_SCOPES; p++) {
    master_scope_timers[p] = 0.0;
  }
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    for (int t = 0; t < NUM_TIMERS; t++) {
      debug_timers[tid][FWD][t] = 0.0;
      debug_timers[tid][BWD][t] = 0.0;
    }
    for (int p = 0; p < NUM_SCOPES; p++) {
      for (int t = 0; t < NUM_TIMERS; t++) {
        scope_timers[tid][p][t] = 0.0;
      }
      scope_flops[tid][p] = 0;
    }
  }
}

void print_debug_timers(int tid) {
  int max_threads = omp_get_max_threads();
  //printf("%-20s", "####");
  printf("### ##: %-11s: ", "#KEY#");
  for(int t = 0; t < LAST_TIMER; t++) {
    printf(" %7s", DebugTimerNames[t]);
  }
  printf(" %9s  %9s\n", "Total", "MTotal");
  for(int i = 0; i < max_threads; i++) {
    if (tid == -1 || tid == i) {
      for (int p = 0; p < 2; p++) {
        double total = 0.0;
        printf("TID %2d: %-11s: ", i, p == 1 ? "BWD" : "FWD");
        for (int t = 0; t < LAST_TIMER; t++) {
          printf(" %7.1f", debug_timers[i][p][t]*1e3);
          total += debug_timers[i][p][t];
        }
        printf(" %9.1f  %9.1f\n", total*1e3, master_debug_timers[p]*1e3);
      }
      for (int p = 0; p < NUM_SCOPES; p++) {
        double total = 0.0;
        printf("TID %2d: %-11s: ", i, ScopeNames[p]);
        for (int t = 0; t < LAST_TIMER; t++) {
          printf(" %7.1f", scope_timers[i][p][t]*1e3);
          total += scope_timers[i][p][t];
        }
        long t_flops = 0;
        for (int f = 0; f < max_threads; f++) t_flops += scope_flops[f][p];
        if (t_flops > 0.0) {
          printf(" %9.1f  %9.1f  %9.3f (%4.2f) %6.3f\n", total*1e3, master_scope_timers[p]*1e3, t_flops*1e-9, t_flops*100.0/(scope_flops[i][p]*max_threads), t_flops*1e-12/scope_timers[i][p][BRGEMM]);
        } else {
          printf(" %9.1f  %9.1f\n", total*1e3, master_scope_timers[p]*1e3);
        }
      }
    }
  }
}

#ifdef PURE_GEMM_TIME
  template<typename Tin, typename Tout>
    class TBrgemmExtTPP {
      public:
        TBrgemmExtTPP() { }
        TBrgemmExtTPP(long M, long N, long K, long str_a, long str_b, float beta = 1.0, XformTPP::XFORM_TYPE c_trans = XformTPP::XFORM_NONE_TPP, int a_trans=0) : M(M), N(N), K(K), beta(beta), c_trans(c_trans), brgemm(), xform(), add() {
          //auto dt_in = XsmmDtype<Tin>();
          auto dt_out = XsmmDtype<Tout>();
          if (dt_out == LIBXSMM_DATATYPE_F32 && c_trans == XformTPP::XFORM_N2V_TPP) c_trans = XformTPP::XFORM_NONE_TPP;
          auto beta_ = beta;

          if (c_trans != XformTPP::XFORM_NONE_TPP) {
            beta_ = 0.0;
            xform = XformExtTPP<Tout>(M, N, c_trans);
          }
          brgemm = BrgemmTPP<Tin, Tout>(M, N, K, str_a, str_b, beta_, a_trans);
          if (beta_ != beta) {
            add = AddTPP<Tout, Tout>(M, N);
          }
          xform_type = c_trans == XformTPP::XFORM_N2V_TPP ? VNNI : XPOSE;
        }

        void operator()(Tin *A, Tin *B, Tout *C, long count) {
          if (c_trans == XformTPP::XFORM_NONE_TPP) {
            ScopedTimer _t(BRGEMM, globalPass, 2*M*N*K*count);
            brgemm(A, B, C, count);
          } else {
            Tout tmp_C[M*N];
            {
              ScopedTimer _t(BRGEMM, globalPass, 2*M*N*K*count);
              brgemm(A, B, tmp_C, count);
            }
            if (beta == 0.0) {
              ScopedTimer _t(xform_type, globalPass);
              xform(tmp_C, C);
            } else {
              Tout tmp[M*N];
              {
                ScopedTimer _t(xform_type, globalPass);
                xform(tmp_C, tmp);
              }
              {
                ScopedTimer _t(EW_ADD, globalPass);
                add(C, tmp, C);
              }
            }
          }
        }

        void ref(Tin *A, Tin *B, Tout *C, long count) {
          if (c_trans == XformTPP::XFORM_NONE_TPP) {
            ScopedTimer _t(BRGEMM, globalPass, 2*M*N*K*count);
            brgemm.ref(A, B, C, count);
          } else {
            Tout tmp_C[M*N];
            {
              ScopedTimer _t(BRGEMM, globalPass, 2*M*N*K*count);
              brgemm(A, B, tmp_C, count);
            }
            if (beta == 0.0) {
              ScopedTimer _t(xform_type, globalPass);
              xform.ref(tmp_C, C);
            } else {
              Tout tmp[M*N];
              {
                ScopedTimer _t(xform_type, globalPass);
                xform.ref(tmp_C, tmp);
              }
              {
                ScopedTimer _t(EW_ADD, globalPass);
                add.ref(C, tmp, C);
              }
            }
          }
        }

      private:
        long M, N, K;
        float beta;
        XformTPP::XFORM_TYPE c_trans;
        BrgemmTPP<Tin, Tout> brgemm;
        XformExtTPP<Tout> xform;
        AddTPP<Tout, Tout> add;
        DebugTimer xform_type;
    };

#define BrgemmExtTPP TBrgemmExtTPP
#endif

#if 0
at::Tensor dense_fwd(at::Tensor t_in, at::Tensor t_wt, at::Tensor t_bias)
{
  auto in_sizes = t_in.sizes();
  auto wt_sizes = t_wt.sizes();
  auto B = in_sizes[0];
  auto S1 = in_sizes[1];
  auto Nc = in_sizes[2];
  auto S2 = in_sizes[3];
  auto Hc = in_sizes[4];

  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];

  auto t_gelu_out = t_in.new_empty({B,S1,Nk,S2,Hk});
  auto t_out     = t_in.new_empty({B,S1,Nk,S2,Hk});

  {
    RECORD_FUNCTION("dense_fwd", std::vector<c10::IValue>({t_in,t_wt}));
    DECL_VLA_PTR_PT(float, in, [S1][Nc][S2][Hc], t_in);
    DECL_VLA_PTR_PT(float, wt, [Nc][Hc][Hk], t_wt);
    DECL_VLA_PTR_PT(float, bias, [Hk], t_bias);
    DECL_VLA_PTR_PT(float, out, [S1][Nk][S2][Hk], t_out);
    auto Ncb = Nc;
    if (Nc > Nk && Nc % Nk == 0) {
      Ncb = Nk;
    }

    for(int nc = 0; nc < Nc; nc+=Ncb) {
#pragma omp parallel for collapse (3)
      for(int b = 0; b < B; b++) {
        for(int s1 = 0; s1 < S1; s1++) {
          for(int nk = 0; nk < Nk; nk++) {
            if (nc == 0) {
              for(int s2 = 0; s2 < S2; s2++) {
                for(int h = 0; h < Hk; h++) {
                  out[b][s1][nk][s2][h] = bias[nk][h];
                }
              }
            }
            brgemm_old(S2, Hk, Hc, S2*Hc*DTS, Hk*Hc*DTS, in[b][s1][nc][0], wt[nk][nc][0], out[b][s1][nk][0], Ncb);
          }
        }
      }
    }
  }
  return t_out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> dense_bwd(at::Tensor t_grad_out, at::Tensor t_in, at::Tensor t_wt)
{
  auto in_sizes = t_in.sizes();
  auto wt_sizes = t_wt.sizes();
  auto B = in_sizes[0];
  auto S1 = in_sizes[1];
  auto Nc = in_sizes[2];
  auto S2 = in_sizes[3];
  auto Hc = in_sizes[4];

  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];

  auto t_grad_in   = at::empty_like(t_in);
  auto t_grad_wt   = at::empty_like(t_wt);
  auto t_grad_bias  = t_wt.new_empty({Nk*Hk}); // [Nk][Hk]
  auto t_tr_wt   = t_wt.permute({0, 1, 3, 2}).contiguous();

  {
    RECORD_FUNCTION("dense_bwd", std::vector<c10::IValue>({t_in,t_wt}));
    DECL_VLA_PTR_PT(float, in, [S1][Nc][S2][Hc], t_in);
    DECL_VLA_PTR_PT(float, grad_in, [S1][Nc][S2][Hc], t_grad_in);
    DECL_VLA_PTR_PT(float, tr_wt, [Nc][Hk][Hc], t_tr_wt);
    DECL_VLA_PTR_PT(float, grad_wt, [Nc][Hc][Hk], t_grad_wt);
    DECL_VLA_PTR_PT(float, grad_bias, [Hk], t_grad_bias);
    DECL_VLA_PTR_PT(float, grad_out, [S1][Nk][S2][Hk], t_grad_out);
    {
      RECORD_FUNCTION("bias_bwd", std::vector<c10::IValue>());
      t_grad_bias.zero_();
#pragma omp parallel for collapse (3) reduction(+:grad_bias[:Nk][:Hk])
      for(int b = 0; b < B; b++) {
        for(int s1 = 0; s1 < S1; s1++) {
          for(int nk = 0; nk < Nk; nk++) {
            for(int s2 = 0; s2 < S2; s2++) {
              int h;
              for(h = 0; h < Hk-15; h+=16) {
                auto vgout = _mm512_loadu_ps(&grad_out[b][s1][nk][s2][h]);
                _mm512_storeu_ps(&grad_bias[nk][h], _mm512_add_ps(vgout, _mm512_loadu_ps(&grad_bias[nk][h])));
              }
              if(h < Hk) {
                int rem = Hk - h;
                __mmask16 mask = (1 << rem) - 1;
                auto vgout = _mm512_maskz_loadu_ps(mask, &grad_out[b][s1][nk][s2][h]);
                _mm512_mask_storeu_ps(&grad_bias[nk][h], mask, _mm512_add_ps(vgout, _mm512_maskz_loadu_ps(mask, &grad_bias[nk][h])));
              }
            }
          }
        }
      }
    }
    {
      RECORD_FUNCTION("gemm_in_bwd", std::vector<c10::IValue>());
      // This is just a hack for now
      auto Nkb = Nk;
      if (Nk > Nc && Nk % Nc == 0) {
        Nkb = Nc;
      }

      if(Nk != Nkb) t_grad_in.zero_();
      for(int nk = 0; nk < Nk; nk+=Nkb) {
#pragma omp parallel for collapse (3)
        for(int b = 0; b < B; b++) {
          for(int s1 = 0; s1 < S1; s1++) {
            for(int nc = 0; nc < Nc; nc++) {
              brgemm_old(S2, Hc, Hk, S2*Hk*DTS, Nc*Hk*Hc*DTS, grad_out[b][s1][nk][0], tr_wt[nk][nc][0], grad_in[b][s1][nc][0], Nkb, (Nk != Nkb ? 1.0 : 0.0));
            }
          }
        }
      }
    }
    {
      RECORD_FUNCTION("gemm_W_bwd", std::vector<c10::IValue>());
      t_grad_wt.zero_();
      for(int b = 0; b < B; b++) {
#pragma omp parallel for collapse (2)
        for(int nk = 0; nk < Nk; nk++) {
          for (int nc = 0; nc < Nc; nc++) {
            brgemm_old(Hc, Hk, S2, Nc*S2*Hc*DTS, Nk*S2*Hk*DTS, in[b][0][nc][0], grad_out[b][0][nk][0], grad_wt[nk][nc][0], S1, 1.0, 'T', 'N', 'N');
          }
        }
      }
    }
  }
  return std::make_tuple(t_grad_in, t_grad_wt, t_grad_bias);
}

at::Tensor gelu_fwd(at::Tensor t_input)
{
  auto sz = t_input.numel();
  auto t_output = at::empty_like(t_input);
  t_input = t_input.contiguous();
  float *input = t_input.data_ptr<float>();
  float *output = t_output.data_ptr<float>();
  long i = 0;
#pragma omp parallel for lastprivate(i)
  for(i = 0; i < sz-15; i+=16) {
    auto vin = _mm512_loadu_ps(&input[i]);
    //auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
    auto vout = LIBXSMM_INTRINSICS_MM512_GELU_FWD_PS_MINIMAX3(vin);
    _mm512_storeu_ps(&output[i], vout);
  }
  if(i < sz) {
    int rem = sz - i;
    __mmask16 mask = (1 << rem) - 1;
    auto vin = _mm512_maskz_loadu_ps(mask, &input[i]);
    //auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
    auto vout = LIBXSMM_INTRINSICS_MM512_GELU_FWD_PS_MINIMAX3(vin);
    _mm512_mask_storeu_ps(&output[i], mask, vout);
  }
  return t_output;
}

at::Tensor gelu_bwd(at::Tensor t_grad_out, at::Tensor t_input)
{
  auto sz = t_input.numel();
  auto t_output = at::empty_like(t_input);
  t_input = t_input.contiguous();
  float *input = t_input.data_ptr<float>();
  float *grad_out = t_grad_out.data_ptr<float>();
  float *output = t_output.data_ptr<float>();
  long i = 0;
#pragma omp parallel for lastprivate(i)
  for(i = 0; i < sz-15; i+=16) {
    auto vin = _mm512_loadu_ps(&input[i]);
    auto vgout = _mm512_loadu_ps(&grad_out[i]);
    //auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_BWD(vin);
    auto vout = LIBXSMM_INTRINSICS_MM512_GELU_BWD_PS_MINIMAX3(vin);
    _mm512_storeu_ps(&output[i], _mm512_mul_ps(vout, vgout));
  }
  if(i < sz) {
    int rem = sz - i;
    __mmask16 mask = (1 << rem) - 1;
    auto vin = _mm512_maskz_loadu_ps(mask, &input[i]);
    auto vgout = _mm512_maskz_loadu_ps(mask, &grad_out[i]);
    //auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_BWD(vin);
    auto vout = LIBXSMM_INTRINSICS_MM512_GELU_BWD_PS_MINIMAX3(vin);
    _mm512_mask_storeu_ps(&output[i], mask, _mm512_mul_ps(vout, vgout));
  }
  return t_output;
}

std::tuple<at::Tensor, at::Tensor> dropout_fwd(at::Tensor t_input, float p)
{
  auto sz = t_input.numel();
  auto t_output = at::empty_like(t_input);
  auto t_dp_mask = at::empty({(sz+15)/16}, at::kShort);
  t_input = t_input.contiguous();
  float *input = t_input.data_ptr<float>();
  float *output = t_output.data_ptr<float>();
  short *dp_mask = t_dp_mask.data_ptr<short>();
  long i = 0;
#pragma omp parallel for lastprivate(i)
  for(i = 0; i < sz-15; i+=16) {
    pcl_dropout_fwd(16, &input[i], &output[i], &dp_mask[i/16], p, true);
  }
  if(i < sz) {
    pcl_dropout_fwd(sz - i, &input[i], &output[i], &dp_mask[i/16], p, true);
  }
  return std::make_tuple(t_output, t_dp_mask);
}

at::Tensor dropout_bwd(at::Tensor t_grad_out, at::Tensor t_dp_mask, float p)
{
  auto sz = t_grad_out.numel();
  t_grad_out = t_grad_out.contiguous();
  auto t_grad_in = at::empty_like(t_grad_out);
  float *grad_out = t_grad_out.data_ptr<float>();
  float *grad_in = t_grad_in.data_ptr<float>();
  short *dp_mask = t_dp_mask.data_ptr<short>();
  long i = 0;
#pragma omp parallel for lastprivate(i)
  for(i = 0; i < sz-15; i+=16) {
    pcl_dropout_bwd(16, &grad_out[i], &grad_in[i], &dp_mask[i/16], p);
  }
  if(i < sz) {
    pcl_dropout_bwd(sz - i, &grad_out[i], &grad_in[i], &dp_mask[i/16], p);
  }
  return t_grad_in;
}

at::Tensor softmax_fwd(at::Tensor t_input)
{
  PCL_ASSERT(t_input.dim() > 3, "Softmax num dims less than 4");
  auto orig_shape = t_input.sizes();
  auto dims = t_input.dim();
  t_input = t_input.view({-1, orig_shape[dims-3], orig_shape[dims-2], orig_shape[dims-1]});
  auto S0 = t_input.size(0);
  auto S1 = orig_shape[dims-3];
  auto S2 = orig_shape[dims-2];
  auto S3 = orig_shape[dims-1];
  auto t_output = at::empty_like(t_input);

  DECL_VLA_PTR_PT(float, input, [S1][S2][S3], t_input);
  DECL_VLA_PTR_PT(float, output, [S1][S2][S3], t_output);
#pragma omp parallel for
  for(int s0 = 0; s0 < S0; s0++) {
    pcl_softmax_fwd(S1, S2, S3, input[s0][0][0], output[s0][0][0]);
  }

  t_output = t_output.view(orig_shape);
  return t_output;
}

at::Tensor softmax_bwd(at::Tensor t_grad_out, at::Tensor t_output)
{
  PCL_ASSERT(t_output.dim() > 3, "Softmax num dims less than 4");
  auto orig_shape = t_output.sizes();
  auto dims = t_output.dim();
  t_output = t_output.view({-1, orig_shape[dims-3], orig_shape[dims-2], orig_shape[dims-1]});
  t_grad_out = t_grad_out.view(t_output.sizes());
  auto S0 = t_output.size(0);
  auto S1 = orig_shape[dims-3];
  auto S2 = orig_shape[dims-2];
  auto S3 = orig_shape[dims-1];
  auto t_grad_in = at::empty_like(t_grad_out);

  DECL_VLA_PTR_PT(float, grad_out, [S1][S2][S3], t_grad_out);
  DECL_VLA_PTR_PT(float, grad_in, [S1][S2][S3], t_grad_in);
  DECL_VLA_PTR_PT(float, output, [S1][S2][S3], t_output);
#pragma omp parallel for
  for(int s0 = 0; s0 < S0; s0++) {
    pcl_softmax_bwd(S1, S2, S3, grad_in[s0][0][0], grad_out[s0][0][0], output[s0][0][0]);
  }

  t_grad_in = t_grad_in.view(orig_shape);
  return t_grad_in;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_fwd(at::Tensor t_input, at::Tensor t_gamma, at::Tensor t_beta, float eps)
{
  PCL_ASSERT(t_input.dim() > 3, "LayerNorm num dims less than 4");
  auto orig_shape = t_input.sizes();
  auto dims = t_input.dim();
  t_input = t_input.view({-1, orig_shape[dims-3], orig_shape[dims-2], orig_shape[dims-1]});
  auto S0 = t_input.size(0);
  auto S1 = orig_shape[dims-3];
  auto S2 = orig_shape[dims-2];
  auto S3 = orig_shape[dims-1];
  auto t_output = at::empty_like(t_input);
  auto t_mean = t_input.new_empty({S0, S2});
  auto t_var = t_input.new_empty({S0, S2});

  DECL_VLA_PTR_PT(float, input, [S1][S2][S3], t_input);
  DECL_VLA_PTR_PT(float, output, [S1][S2][S3], t_output);
  DECL_VLA_PTR_PT(float, gamma, [S3], t_gamma);
  DECL_VLA_PTR_PT(float, beta, [S3], t_beta);
  DECL_VLA_PTR_PT(float, mean, [S2], t_mean);
  DECL_VLA_PTR_PT(float, var, [S2], t_var);
#pragma omp parallel for
  for(int s0 = 0; s0 < S0; s0++) {
    pcl_layer_norm_fwd(S1, S2, S3, input[s0][0][0], gamma[0], beta[0], mean[s0], var[s0], output[s0][0][0], eps);
  }

  t_output = t_output.view(orig_shape);
  return std::make_tuple(t_output, t_mean, t_var);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_bwd(at::Tensor t_grad_out, at::Tensor t_input, at::Tensor t_mean, at::Tensor t_var, at::Tensor t_gamma)
{
  PCL_ASSERT(t_input.dim() > 3, "Softmax num dims less than 4");
  auto orig_shape = t_input.sizes();
  auto dims = t_input.dim();
  t_input = t_input.view({-1, orig_shape[dims-3], orig_shape[dims-2], orig_shape[dims-1]});
  t_grad_out = t_grad_out.view(t_input.sizes());
  auto S0 = t_input.size(0);
  auto S1 = orig_shape[dims-3];
  auto S2 = orig_shape[dims-2];
  auto S3 = orig_shape[dims-1];
  auto t_grad_in = at::empty_like(t_grad_out);
  auto t_grad_gamma = at::empty_like(t_gamma);
  auto t_grad_beta = at::empty_like(t_gamma);

  DECL_VLA_PTR_PT(float, grad_out, [S1][S2][S3], t_grad_out);
  DECL_VLA_PTR_PT(float, grad_in, [S1][S2][S3], t_grad_in);
  DECL_VLA_PTR_PT(float, input, [S1][S2][S3], t_input);
  DECL_VLA_PTR_PT(float, gamma, [S3], t_gamma);
  DECL_VLA_PTR_PT(float, grad_gamma, [S3], t_grad_gamma);
  DECL_VLA_PTR_PT(float, grad_beta, [S3], t_grad_beta);
  DECL_VLA_PTR_PT(float, mean, [S2], t_mean);
  DECL_VLA_PTR_PT(float, var, [S2], t_var);

  t_grad_beta.zero_();
  t_grad_gamma.zero_();
#pragma omp parallel for reduction(+:grad_beta[:S1][:S3], grad_gamma[:S1][:S3])
  for(int s0 = 0; s0 < S0; s0++) {
    pcl_layer_norm_bwd(S1, S2, S3, grad_out[s0][0][0], input[s0][0][0], mean[s0], var[s0], gamma[0], grad_in[s0][0][0], grad_gamma[0], grad_beta[0]);
  }

  t_grad_in = t_grad_in.view(orig_shape);
  return std::make_tuple(t_grad_in, t_grad_gamma, t_grad_beta);
}
#endif

void fused_adamw(at::Tensor &t_data, at::Tensor &t_grad, at::Tensor &t_exp_avg, at::Tensor &t_exp_avg_sq, float beta1, float beta2, float step_size, float lr, float weight_decay, float eps)
{
  RECORD_FUNCTION("fused_adamw", std::vector<c10::IValue>({t_data}));
  typedef float T;
  auto data = t_data.data_ptr<T>();
  auto grad = t_grad.data_ptr<T>();
  auto exp_avg = t_exp_avg.data_ptr<T>();
  auto exp_avg_sq = t_exp_avg_sq.data_ptr<T>();
  long sz = t_data.numel();
  constexpr int BS = 64;

  auto adamw_tpp = SCOPEIT(FusedAdamWTPP<T>(BS, beta1, beta2, weight_decay, eps), OPTIM);

#ifndef NO_TPP_ADAM
  long i;
#pragma omp parallel for lastprivate(i)
  for (i = 0; i < ALIGNDOWN(sz, BS); i+=BS) {
    adamw_tpp(&data[i], &grad[i], &exp_avg[i], &exp_avg_sq[i], step_size, lr);
  }
  if (i < sz) {
    auto adamw_tpp = SCOPEIT(FusedAdamWTPP<T>(sz - i, beta1, beta2, weight_decay, eps), OPTIM);
    adamw_tpp(&data[i], &grad[i], &exp_avg[i], &exp_avg_sq[i], step_size, lr);
  }
#else
  float beta1_1 = 1.0f - beta1;
  float beta2_1 = 1.0f - beta2;
#ifndef __AVX512F__
  if (weight_decay > 0.0) {
#pragma omp parallel for simd
    for (long i = 0; i < sz; i++) {
      auto avg_i = exp_avg[i];
      auto avg_sq_i = exp_avg_sq[i];
      auto grad_i = grad[i];
      auto data_i = data[i];
      avg_i = avg_i * beta1 + grad_i * beta1_1;
      avg_sq_i = avg_sq_i * beta2 + grad_i * grad_i * beta2_1;
      auto denom = sqrtf(avg_sq_i) + eps;
      data_i = data_i - step_size * (avg_i / denom);
      data_i = data_i - data_i * lr * weight_decay;
      exp_avg[i] = avg_i;
      exp_avg_sq[i] = avg_sq_i;
      data[i] = data_i;
    }
  } else {
#pragma omp parallel for simd
    for (long i = 0; i < sz; i++) {
      auto avg_i = exp_avg[i];
      auto avg_sq_i = exp_avg_sq[i];
      auto grad_i = grad[i];
      auto data_i = data[i];
      avg_i = avg_i * beta1 + grad_i * beta1_1;
      avg_sq_i = avg_sq_i * beta2 + grad_i * grad_i * beta2_1;
      auto denom = sqrtf(avg_sq_i) + eps;
      data_i = data_i - step_size * (avg_i / denom);
      exp_avg[i] = avg_i;
      exp_avg_sq[i] = avg_sq_i;
      data[i] = data_i;
    }
  }
#else
  auto vbeta1 = _mm512_set1_ps(beta1);
  auto vbeta1_1 = _mm512_set1_ps(beta1_1);
  auto vbeta2 = _mm512_set1_ps(beta2);
  auto vbeta2_1 = _mm512_set1_ps(beta2_1);
  auto veps = _mm512_set1_ps(eps);
  auto vstep_size = _mm512_set1_ps(step_size);
  auto vweight_decay = _mm512_set1_ps(lr * weight_decay);
  if (weight_decay > 0.0) {
    long i;
#pragma omp parallel for lastprivate(i)
    for (i = 0; i < ALIGNDOWN(sz, 16); i+=16) {
      auto avg_i = _mm512_loadu_ps(&exp_avg[i]);
      auto avg_sq_i = _mm512_loadu_ps(&exp_avg_sq[i]);
      auto grad_i = _mm512_loadu_ps(&grad[i]);
      auto data_i = _mm512_loadu_ps(&data[i]);
      avg_i = _mm512_add_ps(_mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(_mm512_mul_ps(avg_sq_i, vbeta2), _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
      _mm512_storeu_ps(&exp_avg[i], avg_i);
      _mm512_storeu_ps(&exp_avg_sq[i], avg_sq_i);
      _mm512_storeu_ps(&data[i], data_i);
    }
    if( i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto avg_i = _mm512_maskz_loadu_ps(mask, &exp_avg[i]);
      auto avg_sq_i = _mm512_maskz_loadu_ps(mask, &exp_avg_sq[i]);
      auto grad_i = _mm512_maskz_loadu_ps(mask, &grad[i]);
      auto data_i = _mm512_maskz_loadu_ps(mask, &data[i]);
      avg_i = _mm512_add_ps(_mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(_mm512_mul_ps(avg_sq_i, vbeta2), _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
      _mm512_mask_storeu_ps(&exp_avg[i], mask, avg_i);
      _mm512_mask_storeu_ps(&exp_avg_sq[i], mask, avg_sq_i);
      _mm512_mask_storeu_ps(&data[i], mask, data_i);
    }
  } else {
    long i;
#pragma omp parallel for lastprivate(i)
    for (i = 0; i < ALIGNDOWN(sz, 16); i+=16) {
      auto avg_i = _mm512_loadu_ps(&exp_avg[i]);
      auto avg_sq_i = _mm512_loadu_ps(&exp_avg_sq[i]);
      auto grad_i = _mm512_loadu_ps(&grad[i]);
      auto data_i = _mm512_loadu_ps(&data[i]);
      avg_i = _mm512_add_ps(_mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(_mm512_mul_ps(avg_sq_i, vbeta2), _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      _mm512_storeu_ps(&exp_avg[i], avg_i);
      _mm512_storeu_ps(&exp_avg_sq[i], avg_sq_i);
      _mm512_storeu_ps(&data[i], data_i);
    }
    if( i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto avg_i = _mm512_maskz_loadu_ps(mask, &exp_avg[i]);
      auto avg_sq_i = _mm512_maskz_loadu_ps(mask, &exp_avg_sq[i]);
      auto grad_i = _mm512_maskz_loadu_ps(mask, &grad[i]);
      auto data_i = _mm512_maskz_loadu_ps(mask, &data[i]);
      avg_i = _mm512_add_ps(_mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(_mm512_mul_ps(avg_sq_i, vbeta2), _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      _mm512_mask_storeu_ps(&exp_avg[i], mask, avg_i);
      _mm512_mask_storeu_ps(&exp_avg_sq[i], mask, avg_sq_i);
      _mm512_mask_storeu_ps(&data[i], mask, data_i);
    }
  }
#endif
#endif
}

void fused_split_adamw(at::Tensor &t_data_hi, at::Tensor &t_data_lo, at::Tensor &t_grad, at::Tensor &t_exp_avg, at::Tensor &t_exp_avg_sq, float beta1, float beta2, float step_size, float lr, float weight_decay, float eps)
{
  RECORD_FUNCTION("fused_split_adamw", std::vector<c10::IValue>({t_data_hi}));
  typedef bfloat16 T;
  auto data_hi = t_data_hi.data_ptr<T>();
  auto data_lo = t_data_lo.data_ptr<T>();
  auto grad = t_grad.data_ptr<T>();
  auto exp_avg = t_exp_avg.data_ptr<T>();
  auto exp_avg_sq = t_exp_avg_sq.data_ptr<T>();
  long sz = t_data_hi.numel();
  constexpr int BS = 64;

  auto split_adamw_tpp = SCOPEIT(FusedSplitAdamWTPP(BS, beta1, beta2, weight_decay, eps), OPTIM);

#ifndef NO_TPP_ADAM
  long i;
#pragma omp parallel for lastprivate(i)
  for (i = 0; i < ALIGNDOWN(sz, BS); i+=BS) {
    split_adamw_tpp(&data_hi[i], &data_lo[i], &grad[i], &exp_avg[i], &exp_avg_sq[i], step_size, lr);
  }
  if (i < sz) {
    auto split_adamw_tpp = SCOPEIT(FusedSplitAdamWTPP(sz - i, beta1, beta2, weight_decay, eps), OPTIM);
    split_adamw_tpp(&data_hi[i], &data_lo[i], &grad[i], &exp_avg[i], &exp_avg_sq[i], step_size, lr);
  }
#else
  float beta1_1 = 1.0f - beta1;
  float beta2_1 = 1.0f - beta2;
#ifdef __AVX512F__
  auto vbeta1 = _mm512_set1_ps(beta1);
  auto vbeta1_1 = _mm512_set1_ps(beta1_1);
  auto vbeta2 = _mm512_set1_ps(beta2);
  auto vbeta2_1 = _mm512_set1_ps(beta2_1);
  auto veps = _mm512_set1_ps(eps);
  auto vstep_size = _mm512_set1_ps(step_size);
  auto vweight_decay = _mm512_set1_ps(lr * weight_decay);
  if (weight_decay > 0.0) {
    long i;
#pragma omp parallel for lastprivate(i)
    for (i = 0; i < ALIGNDOWN(sz, 16); i+=16) {
      auto avg_i = _mm512_loadu_ps_auto(&exp_avg[i]);
      auto avg_sq_i = _mm512_loadu_ps_auto(&exp_avg_sq[i]);
      auto grad_i = _mm512_loadu_ps_auto(&grad[i]);
      auto data_i = _mm512_split_loadu_ps(&data_hi[i], &data_lo[i]);
      avg_i = _mm512_add_ps(_mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(_mm512_mul_ps(avg_sq_i, vbeta2), _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
      _mm512_storeu_ps_auto(&exp_avg[i], avg_i);
      _mm512_storeu_ps_auto(&exp_avg_sq[i], avg_sq_i);
      _mm512_split_storeu_ps(&data_hi[i], &data_lo[i], data_i);
    }
    if( i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto avg_i = _mm512_maskz_loadu_ps_auto(mask, &exp_avg[i]);
      auto avg_sq_i = _mm512_maskz_loadu_ps_auto(mask, &exp_avg_sq[i]);
      auto grad_i = _mm512_maskz_loadu_ps_auto(mask, &grad[i]);
      auto data_i = _mm512_maskz_split_loadu_ps(mask, &data_hi[i], &data_lo[i]);
      avg_i = _mm512_add_ps(_mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(_mm512_mul_ps(avg_sq_i, vbeta2), _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
      _mm512_mask_storeu_ps_auto(&exp_avg[i], mask, avg_i);
      _mm512_mask_storeu_ps_auto(&exp_avg_sq[i], mask, avg_sq_i);
      _mm512_mask_split_storeu_ps(&data_hi[i], &data_lo[i], mask, data_i);
    }
  } else {
    long i;
#pragma omp parallel for lastprivate(i)
    for (i = 0; i < ALIGNDOWN(sz, 16); i+=16) {
      auto avg_i = _mm512_loadu_ps_auto(&exp_avg[i]);
      auto avg_sq_i = _mm512_loadu_ps_auto(&exp_avg_sq[i]);
      auto grad_i = _mm512_loadu_ps_auto(&grad[i]);
      auto data_i = _mm512_split_loadu_ps(&data_hi[i], &data_lo[i]);
      avg_i = _mm512_add_ps(_mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(_mm512_mul_ps(avg_sq_i, vbeta2), _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      _mm512_storeu_ps_auto(&exp_avg[i], avg_i);
      _mm512_storeu_ps_auto(&exp_avg_sq[i], avg_sq_i);
      _mm512_split_storeu_ps(&data_hi[i], &data_lo[i], data_i);
    }
    if( i < sz) {
      int rem = sz - i;
      __mmask16 mask = (1 << rem) - 1;
      auto avg_i = _mm512_maskz_loadu_ps_auto(mask, &exp_avg[i]);
      auto avg_sq_i = _mm512_maskz_loadu_ps_auto(mask, &exp_avg_sq[i]);
      auto grad_i = _mm512_maskz_loadu_ps_auto(mask, &grad[i]);
      auto data_i = _mm512_maskz_split_loadu_ps(mask, &data_hi[i], &data_lo[i]);
      avg_i = _mm512_add_ps(_mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
      avg_sq_i = _mm512_add_ps(_mm512_mul_ps(avg_sq_i, vbeta2), _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
      auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
      data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
      _mm512_mask_storeu_ps_auto(&exp_avg[i], mask, avg_i);
      _mm512_mask_storeu_ps_auto(&exp_avg_sq[i], mask, avg_sq_i);
      _mm512_mask_split_storeu_ps(&data_hi[i], &data_lo[i], mask, data_i);
    }
  }
#else
  printf("split adam is not implemented without AVX512 support\n");
  exit(1);
#endif
#endif
}

void init_libxsmm()
{
  auto max_threads = omp_get_max_threads();
  PCL_ASSERT(max_threads <= MAX_THREADS, "Maximun %d threads supported, %d threads being used, please compile with increased  MAX_THREADS value\n", MAX_THREADS, max_threads);
  libxsmm_init();
  set_rnd_seed(0);
}

#if 0
#include <ATen/Config.h>
#include <ATen/core/op_registration/op_registration.h>

namespace Test {
  using namespace at;
  Tensor add(const Tensor & self, const Tensor & other, Scalar alpha) {
    const OptionalDeviceGuard device_guard(device_of(self));
    printf("Test_add called\n");
    return at::native::add(self, other, alpha);
  }
}
namespace {
  auto registerer = torch::RegisterOperators()
    .op(torch::RegisterOperators::options()
        .schema("aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
        .kernel<Tensor (const Tensor &, const Tensor &, Scalar)>(DispatchKey::CPUTensorId, &Test::add)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
}
#endif
inline void xsmm_xform(libxsmm_blasint rows_i, libxsmm_blasint rows_o, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_meltw_transform_flags trans_flags, libxsmm_datatype dtype, void *in, void *out)
{
  libxsmm_meltw_transform_param trans_param;
  trans_param.in_ptr  = in;
  trans_param.out_ptr = out;
  libxsmm_meltwfunction_transform trans_kernel = libxsmm_dispatch_meltw_transform(rows_o, rows_i, &ldi, &ldo, dtype, dtype, trans_flags);
  if ( trans_kernel == NULL ) {
    fprintf( stderr, "JIT for Transform TPP. Bailing...!\n");
    exit(-1);
  }
  trans_kernel( &trans_param );
}

inline void xsmm_eltwise_unary(libxsmm_blasint rows, libxsmm_blasint cols, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_meltw_unary_flags unary_flags, libxsmm_meltw_unary_type unary_type, libxsmm_datatype dtype_in, libxsmm_datatype dtype_out, void *in, void *out)
{
  libxsmm_meltw_unary_param unary_param;
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  libxsmm_datatype dtype_compute = LIBXSMM_DATATYPE_F32;
  // LIBXSMM_MELTW_TYPE_UNARY_IDENTITY - COPY
  // LIBXSMM_MELTW_TYPE_UNARY_XOR - ZERO

  if ((unary_type == LIBXSMM_MELTW_TYPE_UNARY_IDENTITY || unary_type == LIBXSMM_MELTW_TYPE_UNARY_XOR)
      && dtype_in == LIBXSMM_DATATYPE_BF16 && dtype_out == LIBXSMM_DATATYPE_BF16)
    dtype_compute = LIBXSMM_DATATYPE_BF16;

  auto unary_kernel = libxsmm_dispatch_meltw_unary(cols, rows, &ldi, &ldo, dtype_in, dtype_compute, dtype_out, unary_flags, unary_type);
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );
}

inline void xsmm_eltwise_binary(libxsmm_blasint rows, libxsmm_blasint cols, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_meltw_binary_flags binary_flags, libxsmm_meltw_binary_type binary_type, libxsmm_datatype dtype_in, libxsmm_datatype dtype_out, void *in, void *in2, void *out)
{
  libxsmm_meltw_binary_param binary_param;
  binary_param.in0.primary  = (void*)in;
  binary_param.in1.primary  = (void*)in2;
  binary_param.out.primary  = (void*)out;
  libxsmm_datatype dtype_compute = LIBXSMM_DATATYPE_F32;
  // LIBXSMM_MELTW_TYPE_BINARY_BCAST_ROW_IN_0
  // LIBXSMM_MELTW_TYPE_BINARY_BCAST_COL_IN_0
  // LIBXSMM_MELTW_TYPE_BINARY_BCAST_SCALAR_IN_0
  //
  // LIBXSMM_MELTW_TYPE_BINARY_ADD
  // LIBXSMM_MELTW_TYPE_BINARY_MUL
  // LIBXSMM_MELTW_TYPE_BINARY_MULADD

  auto binary_kernel = libxsmm_dispatch_meltw_binary(cols, rows, &ldi, &ldo, dtype_in, dtype_compute, dtype_out, binary_flags, binary_type);
  if ( binary_kernel == NULL ) {
    fprintf( stderr, "JIT for BINARY TPP. Bailing...!\n");
    exit(-1);
  }
  binary_kernel( &binary_param );
}

template<typename T>
inline void xsmm_xpose(unsigned int rows_i, unsigned int rows_o, T *in, T *out)
{
  ScopedTimer _t(XPOSE, globalPass);
  auto xsmm_dtype = getXsmmDtype<T>();
  xsmm_xform(rows_i, rows_o, rows_o, rows_i, LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_NORMT, xsmm_dtype, in, out);
}

inline void xsmm_n2v(unsigned int rows_i, unsigned int cols_i, bfloat16 *in, bfloat16 *out)
{
  ScopedTimer _t(VNNI, globalPass);
  PCL_ASSERT(rows_i % 2 == 0, "xsmm_n2v: uneven number of rows\n");
  xsmm_xform(rows_i, cols_i, cols_i, cols_i, LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_VNNI, LIBXSMM_DATATYPE_BF16, in, out);
}

inline void xsmm_n2v(unsigned int rows_i, unsigned int cols_i, float *in, bfloat16 *out)
{
  ScopedTimer _t(VNNI, globalPass);
  PCL_ASSERT(rows_i % 2 == 0, "xsmm_n2v: uneven number of rows\n");
  int sz = rows_i * cols_i;
  bfloat16 tmp[sz];
#ifdef __AVX512F__
  int i;
  for (i = 0; i < ALIGNDOWN(sz, 16); i+=16) {
    _mm512_storeu_ps_auto(&tmp[i], _mm512_loadu_ps_auto(&in[i]));
  }
  if (i < sz) {
    int rem = sz - i;
    __mmask16 mask = (1 << rem) - 1;
    _mm512_mask_storeu_ps_auto(&tmp[i], mask, _mm512_maskz_loadu_ps_auto(mask, &in[i]));
  }
#else
  for (long i = 0; i < sz; i++) {
    tmp[i] = in[i];
  }
#endif
  xsmm_xform(rows_i, cols_i, cols_i, cols_i, LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_VNNI, LIBXSMM_DATATYPE_BF16, tmp, out);
}

inline void xsmm_n2v(unsigned int rows_i, unsigned int cols_i, float *in, float *out)
{
  ScopedTimer _t(XPOSE, globalPass);
  PCL_ASSERT(in == out, "xsmm_n2v: Invalid pointers for float datatype\n");
}

inline void xsmm_xpose_n2v(unsigned int rows_i, unsigned int rows_o, bfloat16 *in, bfloat16 *out)
{
  ScopedTimer _t(XPOSE, globalPass);
  PCL_ASSERT(rows_o % 2 == 0, "xsmm_xpose_n2v: uneven number of rows\n");
  xsmm_xform(rows_i, rows_o/2, rows_o/2, rows_i, LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_NORMT, LIBXSMM_DATATYPE_F32, in, out);
}

inline void xsmm_xpose_n2v(unsigned int rows_i, unsigned int rows_o, float *in, float *out)
{
  xsmm_xpose(rows_i, rows_o, in, out);
}

inline void xsmm_xpose_v2v(unsigned int rows_i, unsigned int rows_o, bfloat16 *in, bfloat16 *out)
{
  ScopedTimer _t(XPOSE, globalPass);
  PCL_ASSERT(rows_i % 2 == 0, "xsmm_xpose_v2v: uneven number of rows\n");
  PCL_ASSERT(rows_o % 2 == 0, "xsmm_xpose_v2v: uneven number of cols\n");
  xsmm_xform(rows_i, rows_o, rows_o, rows_i, LIBXSMM_MELTW_FLAG_TRANSFORM_VNNI_TO_VNNIT, LIBXSMM_DATATYPE_BF16, in, out);
}

inline void xsmm_xpose_v2v(unsigned int rows_i, unsigned int rows_o, float *in, float *out)
{
  xsmm_xpose(rows_i, rows_o, in, out);
}

inline void xsmm_cpy(int rows, int cols, float *in, float *out)
{
  ScopedTimer _t(EW_COPY, globalPass);
#if 0
  for (int i = 0; i < rows*cols; i++) {
    out[i] = in[i];
  }
#else
  xsmm_eltwise_unary(1, rows*cols, rows*cols, rows*cols, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, in, out);
#endif
}

inline void xsmm_cpy(int rows, int cols, bfloat16 *in, bfloat16 *out)
{
  ScopedTimer _t(EW_COPY, globalPass);
#if 0
  for (int i = 0; i < rows*cols; i++) {
    out[i] = in[i];
  }
#else
  xsmm_eltwise_unary(1, rows*cols, rows*cols, rows*cols, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, in, out);
#endif
}

inline void xsmm_cpy(int rows, int cols, bfloat16 *in, float *out)
{
  ScopedTimer _t(EW_COPY, globalPass);
#if 0
  for (int i = 0; i < rows*cols; i++) {
    out[i] = in[i];
  }
#else
  xsmm_eltwise_unary(1, rows*cols, rows*cols, rows*cols, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, in, out);
#endif
}

inline void xsmm_cpy(int rows, int cols, float *in, bfloat16 *out)
{
  ScopedTimer _t(EW_COPY, globalPass);
#if 0
  for (int i = 0; i < rows*cols; i++) {
    out[i] = in[i];
  }
#else
  xsmm_eltwise_unary(1, rows*cols, rows*cols, rows*cols, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, in, out);
#endif
}

inline void xsmm_add(int N, float *in1, float *in2, float *out)
{
  ScopedTimer _t(EW_ADD, globalPass);
#if 0
  for (int i = 0; i < N; i++) {
    out[i] = in1[i] + in2[i];
  }
#else
  xsmm_eltwise_binary(1, N, N, N, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, in1, in2, out);
#endif
}

inline void xsmm_add(int N, bfloat16 *in1, bfloat16 *in2, bfloat16 *out)
{
  ScopedTimer _t(EW_ADD, globalPass);
#if 0
  for (int i = 0; i < N; i++) {
    out[i] = in1[i] + in2[i];
  }
#else
  xsmm_eltwise_binary(1, N, N, N, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, in1, in2, out);
#endif
}

template<typename T1, typename T2>
inline void xsmm_scale(int N, T1 *in, T2 *out, float scale)
{
  ScopedTimer _t(EW_SCL, globalPass);
#if 1
#pragma omp simd
  for (int i = 0; i < N; i++) {
    out[i] = in[i] * scale;
  }
#else
  //FIXME: datatype for in1 and in2 can be different
  // Maybe use eltwise_scale TPP
  auto dt_in = getXsmmDtype<T1>();
  auto dt_out = getXsmmDtype<T2>();
  xsmm_eltwise_binary(1, N, N, N, LIBXSMM_MELTW_TYPE_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_MELTW_TYPE_BINARY_MUL, dt_in, dt_out, in, &scale, out);
#endif
}

template<typename T>
inline void xsmm_set_zero(int N, T *buf)
{
  ScopedTimer _t(EW_ZERO, globalPass);
#if 0
  for (int i = 0; i < N; i++) {
    buf[i] = 0;
  }
#else
  //FIXME: what to set as input
  auto dt = getXsmmDtype<T>();
  xsmm_eltwise_unary(1, N, N, N, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR, dt, dt, buf, buf);
#endif
}

template<typename T>
inline void xsmm_reduce_buf(int num_threads, int N, float **ptrs, T *buf, bool accumulate = false)
{
  ScopedTimer _t(EW_RED, globalPass);
#pragma omp for
  for (int i = 0; i < N; i++) {
    float sum = 0.0;
    for (int j = 0; j < num_threads; j++) {
      sum += ptrs[j][i];
    }
    if (accumulate) {
      buf[i] += sum;
    } else {
      buf[i] = sum;
    }
  }
}

template<typename T>
inline void xsmm_grad_bias(int rows, int cols, T *in, float *out)
{
  ScopedTimer _t(BIAS, globalPass);
  for (int i = 0; i < rows; i++) {
#ifdef __AVX512F__
    int j;
    for (j = 0; j < ALIGNDOWN(cols, 16); j+=16) {
      //out[j] += in[i*cols+j];
      _mm512_storeu_ps_auto(&out[j], _mm512_add_ps(_mm512_loadu_ps_auto(&in[i*cols+j]), _mm512_loadu_ps_auto(&out[j])));
    }
    if (j < cols) {
      int rem = cols - j;
      __mmask16 mask = (1 << rem) - 1;
      _mm512_mask_storeu_ps_auto(&out[j], mask, _mm512_add_ps(_mm512_maskz_loadu_ps_auto(mask, &in[i*cols+j]), _mm512_maskz_loadu_ps_auto(mask, &out[j])));
    }
#else
    for (int j = 0; j < cols; j++) {
      out[j] += in[i*cols+j];
    }
#endif
  }
}

inline void xsmm_cpy_bias(int rows, int cols, float *in, float *out)
{
  ScopedTimer _t(BIAS, globalPass);
  for (int i = 0; i < rows; i++) {
#ifdef __AVX512F__
    int j;
    for (j = 0; j < ALIGNDOWN(cols, 16); j+=16) {
      //out[i*cols+j] = in[j];
      _mm512_storeu_ps(&out[i*cols+j], _mm512_loadu_ps(&in[j]));
    }
    if (j < cols) {
      int rem = cols - j;
      __mmask16 mask = (1 << rem) - 1;
      _mm512_mask_storeu_ps(&out[i*cols+j], mask, _mm512_maskz_loadu_ps(mask, &in[j]));
    }
#else
    for (int j = 0; j < cols; j++) {
      out[i*cols+j] = in[j];
    }
#endif
  }
}

inline void xsmm_cpy_bias(int rows, int cols, bfloat16 *in, bfloat16 *out)
{
  ScopedTimer _t(BIAS, globalPass);
  for (int i = 0; i < rows; i++) {
#ifdef __AVX512F__
    int j;
    for (j = 0; j < ALIGNDOWN(cols, 32); j+=32) {
      //out[i*cols+j] = in[j];
      //_mm512_storeu_ps_auto(&out[i*cols+j], _mm512_loadu_ps_auto(&in[j]));
      _mm512_storeu_si512(&out[i*cols+j], _mm512_loadu_si512(&in[j]));
    }
    if (j < cols) {
      int rem = cols - j;
      __mmask32 mask = (1 << rem) - 1;
      //_mm512_mask_storeu_ps_auto(&out[i*cols+j], mask, _mm512_maskz_loadu_ps_auto(mask, &in[j]));
      _mm512_mask_storeu_epi16(&out[i*cols+j], mask, _mm512_maskz_loadu_epi16(mask, &in[j]));
    }
#else
    for (int j = 0; j < cols; j++) {
      out[i*cols+j] = in[j];
    }
#endif
  }
}

template<typename T>
inline void xsmm_add_bias(int rows, int cols, T *in, float *out)
{
  ScopedTimer _t(EW_ADD, globalPass);
  for (int i = 0; i < rows; i++) {
#ifdef __AVX512F__
    int j;
    for (j = 0; j < ALIGNDOWN(cols, 16); j+=16) {
      // out[i*cols+j] += in[j];
      _mm512_storeu_ps_auto(&out[i*cols+j], _mm512_add_ps(_mm512_loadu_ps_auto(&out[i*cols+j]), _mm512_loadu_ps_auto(&in[j])));
    }
    if (j < cols) {
      int rem = cols - j;
      __mmask16 mask = (1 << rem) - 1;
      _mm512_mask_storeu_ps_auto(&out[i*cols+j], mask, _mm512_add_ps(_mm512_maskz_loadu_ps_auto(mask, &out[i*cols+j]), _mm512_maskz_loadu_ps_auto(mask, &in[j])));
    }
#else
    for (int j = 0; j < cols; j++) {
      out[i*cols+j] += in[j];
    }
#endif
  }
}

inline void xsmm_gelu_fwd(int N, float *in, float *out)
{
  ScopedTimer _t(GELU, globalPass);
#ifdef __AVX512F__
  int i;
  for(i = 0; i < ALIGNDOWN(N, 16); i+=16) {
    auto vin = _mm512_loadu_ps_auto(&in[i]);
    auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
    _mm512_storeu_ps_auto(&out[i], vout);
  }
  if(i < N) {
    int rem = N - i;
    __mmask16 mask = (1 << rem) - 1;
    auto vin = _mm512_maskz_loadu_ps_auto(mask, &in[i]);
    auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
    _mm512_mask_storeu_ps_auto(&out[i], mask, vout);
  }
#else
  for(int i = 0; i < N; i++) {
    float x = in[i];
    out[i] = (erff(x/sqrtf(2.0)) + 1.0)*0.5*x;
  }
#endif
}

inline void xsmm_gelu_fwd(int N, bfloat16 *in, bfloat16 *out)
{
  ScopedTimer _t(GELU, globalPass);
#ifdef __AVX512F__
  int i;
  for(i = 0; i < ALIGNDOWN(N, 16); i+=16) {
    auto vin = _mm512_loadu_ps_auto(&in[i]);
    auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
    _mm512_storeu_ps_auto(&out[i], vout);
  }
  if(i < N) {
    int rem = N - i;
    __mmask16 mask = (1 << rem) - 1;
    auto vin = _mm512_maskz_loadu_ps_auto(mask, &in[i]);
    auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
    _mm512_mask_storeu_ps_auto(&out[i], mask, vout);
  }
#else
  for(int i = 0; i < N; i++) {
    float x = in[i];
    out[i] = (erff(x/sqrtf(2.0)) + 1.0)*0.5*x;
  }
#endif
}

inline void xsmm_gelu_bwd(int N, float *gout, float *in, float *gin)
{
  ScopedTimer _t(GELU, globalPass);
#ifdef __AVX512F__
  int i;
  for(i = 0; i < ALIGNDOWN(N,16); i+=16) {
    auto vgout = _mm512_loadu_ps_auto(&gout[i]);
    auto vin_gelu = _mm512_loadu_ps_auto(&in[i]);
    auto vgin_gelu = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_BWD(vin_gelu);
    auto vout = _mm512_mul_ps(vgin_gelu, vgout);
    _mm512_storeu_ps_auto(&gin[i], vout);
  }
  if(i < N) {
    int rem = N - i;
    __mmask16 mask = (1 << rem) - 1;
    auto vgout = _mm512_maskz_loadu_ps_auto(mask, &gout[i]);
    auto vin_gelu = _mm512_maskz_loadu_ps_auto(mask, &in[i]);
    auto vgin_gelu = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_BWD(vin_gelu);
    auto vout = _mm512_mul_ps(vgin_gelu, vgout);
    _mm512_mask_storeu_ps_auto(&gin[i], mask, vout);
  }
#else
  constexpr float PI = 3.14159265358979323846;
  for (int i = 0; i < N; i++) {
    float x = in[i];
    gin[i] = (float)gout[i] * (0.5 + 0.5 * erff(x/sqrtf(2.0)) + x/(sqrtf(2.0*PI))*expf(-0.5*x*x));
  }
#endif
}

inline void xsmm_gelu_bwd(int N, bfloat16 *gout, bfloat16 *in, bfloat16 *gin)
{
  ScopedTimer _t(GELU, globalPass);
#ifdef __AVX512F__
  int i;
  for(i = 0; i < ALIGNDOWN(N,16); i+=16) {
    auto vgout = _mm512_loadu_ps_auto(&gout[i]);
    auto vin_gelu = _mm512_loadu_ps_auto(&in[i]);
    auto vgin_gelu = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_BWD(vin_gelu);
    auto vout = _mm512_mul_ps(vgin_gelu, vgout);
    _mm512_storeu_ps_auto(&gin[i], vout);
  }
  if(i < N) {
    int rem = N - i;
    __mmask16 mask = (1 << rem) - 1;
    auto vgout = _mm512_maskz_loadu_ps_auto(mask, &gout[i]);
    auto vin_gelu = _mm512_maskz_loadu_ps_auto(mask, &in[i]);
    auto vgin_gelu = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_BWD(vin_gelu);
    auto vout = _mm512_mul_ps(vgin_gelu, vgout);
    _mm512_mask_storeu_ps_auto(&gin[i], mask, vout);
  }
#else
  constexpr float PI = 3.14159265358979323846;
  for (int i = 0; i < N; i++) {
    float x = in[i];
    gin[i] = (float)gout[i] * (0.5 + 0.5 * erff(x/sqrtf(2.0)) + x/(sqrtf(2.0*PI))*expf(-0.5*x*x));
  }
#endif
}

#define XFORM_NONE  0
#define XFORM_XPOSE 1
#define XFORM_N2V   2
#define XFORM_XPOSE_N2V 3
#define XFORM_XPOSE_V2V 4

inline void xsmm_xform_ext(int count, unsigned int rows_o, unsigned int cols_o, int str_in, int str_out, int flag, float *in, float *out)
{
  if (flag == XFORM_XPOSE) {
    for (int i = 0; i < count; i++) {
      xsmm_xpose(cols_o, rows_o, &in[i*str_in], &out[i*str_out]);
    }
  } else if (flag == XFORM_N2V) {
    PCL_ASSERT(in == out, "xsmm_xform_input: F32 XFORM_N2V not have same pointers\n");
  } else if (flag == XFORM_XPOSE_N2V) {
    for (int i = 0; i < count; i++) {
      xsmm_xpose(cols_o, rows_o, &in[i*str_in], &out[i*str_out]);
    }
  } else if (flag == XFORM_XPOSE_V2V) {
    for (int i = 0; i < count; i++) {
      xsmm_xpose(cols_o, rows_o, &in[i*str_in], &out[i*str_out]);
    }
  } else {
    PCL_ASSERT(false, "Invalid flag value for xsmm_xform_ext\n");
  }
}

inline void xsmm_xform_ext(int count, unsigned int rows_o, unsigned int cols_o, int str_in, int str_out, int flag, bfloat16 *in, bfloat16 *out)
{
  if (flag == XFORM_XPOSE) {
    for (int i = 0; i < count; i++) {
      xsmm_xpose(cols_o, rows_o, &in[i*str_in], &out[i*str_out]);
    }
  } else if (flag == XFORM_N2V) {
    for (int i = 0; i < count; i++) {
      xsmm_n2v(rows_o, cols_o, &in[i*str_in], &out[i*str_out]);
    }
  } else if (flag == XFORM_XPOSE_N2V) {
    for (int i = 0; i < count; i++) {
      xsmm_xpose_n2v(cols_o, rows_o, &in[i*str_in], &out[i*str_out]);
    }
  } else if (flag == XFORM_XPOSE_V2V) {
    for (int i = 0; i < count; i++) {
      xsmm_xpose_v2v(cols_o, rows_o, &in[i*str_in], &out[i*str_out]);
    }
  } else {
    PCL_ASSERT(false, "Invalid flag value for xsmm_xform_ext\n");
  }
}

inline void xsmm_xform_input(int count, unsigned int rows_i, unsigned int cols_i, int str_in, int str_out, int flag, float *in, float *out)
{
  if (flag == XFORM_XPOSE) {
    for (int i = 0; i < count; i++) {
      xsmm_xpose(rows_i, cols_i, &in[i*str_in], &out[i*str_out]);
    }
  } else if (flag == XFORM_N2V) {
    PCL_ASSERT(in == out, "xsmm_xform_input: F32 XFORM_N2V not have same pointers\n");
  } else if (flag == XFORM_XPOSE_N2V) {
    for (int i = 0; i < count; i++) {
      xsmm_xpose(rows_i, cols_i, &in[i*str_in], &out[i*str_out]);
    }
  } else if (flag == XFORM_XPOSE_V2V) {
    for (int i = 0; i < count; i++) {
      xsmm_xpose(rows_i, cols_i, &in[i*str_in], &out[i*str_out]);
    }
  } else {
    PCL_ASSERT(false, "Invalid flag value for xsmm_xform_input\n");
  }
}

inline void xsmm_xform_input(int count, unsigned int rows_i, unsigned int cols_i, int str_in, int str_out, int flag, bfloat16 *in, bfloat16 *out)
{
  if (flag == XFORM_XPOSE) {
    for (int i = 0; i < count; i++) {
      xsmm_xpose(rows_i, cols_i, &in[i*str_in], &out[i*str_out]);
    }
  } else if (flag == XFORM_N2V) {
    for (int i = 0; i < count; i++) {
      xsmm_n2v(rows_i, cols_i, &in[i*str_in], &out[i*str_out]);
    }
  } else if (flag == XFORM_XPOSE_N2V) {
    for (int i = 0; i < count; i++) {
      xsmm_xpose_n2v(rows_i, cols_i, &in[i*str_in], &out[i*str_out]);
    }
  } else if (flag == XFORM_XPOSE_V2V) {
    for (int i = 0; i < count; i++) {
      xsmm_xpose_v2v(rows_i, cols_i, &in[i*str_in], &out[i*str_out]);
    }
  } else {
    PCL_ASSERT(false, "Invalid flag value for xsmm_xform_input\n");
  }
}

inline void brgemm(long m, long n, long k, long str_a, long str_b, float *A, float *B, float *C_, long count, const float beta_ = 1.0, int c_trans=0, int a_trans=0)
{
  const float alpha = 1.0;
  unsigned long long l_br = count;
  if (c_trans == XFORM_N2V) c_trans = XFORM_NONE;
  else if (c_trans == XFORM_XPOSE_N2V || c_trans == XFORM_XPOSE_V2V) c_trans = XFORM_XPOSE;
  PCL_ASSERT(a_trans == 0 || a_trans == XFORM_XPOSE, "Unsupported a_trans for FP32 BRGEMM\n");
  PCL_ASSERT(c_trans == 0 || c_trans == XFORM_XPOSE, "Unsupported c_trans for FP32 BRGEMM\n");
  int flags = LIBXSMM_GEMM_FLAGS( 'N', 'N' );
  if (a_trans == XFORM_XPOSE) flags = LIBXSMM_GEMM_FLAGS( 'N', 'T' );
  float* C = C_;
  float beta = beta_;
  float tmp_C[m*n];

  //printf("BRGEMM: %ld %ld %ld  %ld %ld  %p %p %p  %ld %.1f\n", n, m, k, str_b, str_a, A, B, C, count, beta); fflush(stdout);
  if (c_trans != 0) {
    beta = 0.0;
    C = tmp_C;
  }
  {
    ScopedTimer _t(BRGEMM, globalPass, 2*m*n*k*count);
#if 1
    //printf("BRGEMMDEBUG: before %g %g %g\n", beta, (float)C[0], (float)C_[0]);
    libxsmm_smmfunction_reducebatch_strd kernel = libxsmm_smmdispatch_reducebatch_strd(n, m, k, str_b*sizeof(float), str_a*sizeof(float), NULL, NULL, NULL, &alpha, &beta, &flags, NULL);
    PCL_ASSERT(kernel != NULL, "BRGEMM bailing out for mnk = %ld %ld %ld\n", m, n, k);
    kernel(B, A, C, &l_br);
    //printf("BRGEMMDEBUG: After %g %g\n", beta_, (float)C[0]);
#else
    for (int x = 0; x < m; x++) {
      for (int y = 0; y < n; y++) {
        float sum = beta * C[x*n+y];
        //C[x*n+y] = beta * C[x*n+y];
        for (int w = 0; w < count; w++) {
          for (int z = 0; z < k/2; z++) {
            //auto old_c = C[x*n+y];
            sum += (float)A[w*str_a + x*k+z*2+0] * (float)B[w*str_b + z*n*2+y*2+0];
            //if(x==0 && y == 0) printf("BRGEMMDEBUG: %g * %g = %g, O: %g N: %g\n", (float)A[w*str_a + x*k+z*2+0], (float)B[w*str_b + z*n*2+y*2+0], (float)A[w*str_a + x*k+z*2+0] * B[w*str_b + z*n*2+y*2+0], (float)old_c, (float)C[x*n+y]);
            sum += (float)A[w*str_a + x*k+z*2+1] * (float)B[w*str_b + z*n*2+y*2+1];
            //if(x==0 && y == 0) printf("BRGEMMDEBUG: %g * %g = %g, O: %g N: %g\n", (float)A[w*str_a + x*k+z*2+1], (float)B[w*str_b + z*n*2+y*2+1], (float)A[w*str_a + x*k+z*2+1] * B[w*str_b + z*n*2+y*2+1], (float)old_c, (float)C[x*n+y]);
          }
        }
        C[x*n+y] = sum;
      }
    }
#endif
  }
  if (c_trans != 0) {
    if (beta_ == 0.0) {
      xsmm_xpose(m, n, C, C_);
    } else {
      float tmp[m*n];
      xsmm_xpose(m, n, C, tmp);
#if 1
      xsmm_add(m*n, C_, tmp, C_);
#else
      for (int i = 0; i < m*n; i++) {
        C_[i] = beta_ * C_[i] + tmp[i];
      }
#endif
    }
  }
}

inline void brgemm(long m, long n, long k, long str_a, long str_b, bfloat16 *A, bfloat16 *B, bfloat16 *C_, long count, const float beta_ = 1.0, const int c_trans=0, const int a_trans=0)
{
  const float alpha = 1.0;
  unsigned long long l_br = count;
  //int flags = LIBXSMM_GEMM_FLAGS( 'N', 'N' );
  PCL_ASSERT(a_trans == 0, "BFloat16 BRGEMM doesn't support a_trans\n");
  int flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
  bfloat16* C = C_;
  float beta = beta_;
  bfloat16 tmp_C[m*n];

  //printf("BRGEMM: %ld %ld %ld  %ld %ld  %p %p %p  %ld %.1f\n", n, m, k, str_b, str_a, A, B, C, count, beta); fflush(stdout);
  if (c_trans != 0) {
    beta = 0.0;
    C = tmp_C;
  }
  {
    ScopedTimer _t(BRGEMM, globalPass, 2*m*n*k*count);
#if 1
    //printf("BRGEMMDEBUG: before %g %g %g\n", beta, (float)C[0], (float)C_[0]);
    libxsmm_bmmfunction_reducebatch_strd kernel = libxsmm_bmmdispatch_reducebatch_strd(n, m, k, str_b*sizeof(libxsmm_bfloat16), str_a*sizeof(libxsmm_bfloat16), NULL, NULL, NULL, &alpha, &beta, &flags, NULL);
    PCL_ASSERT(kernel != NULL, "BRGEMM bailing out for mnk = %ld %ld %ld\n", m, n, k);
    kernel((libxsmm_bfloat16*)B, (libxsmm_bfloat16*)A, (libxsmm_bfloat16*)C, &l_br);
    //printf("BRGEMMDEBUG: After %g %g\n", beta_, (float)C[0]);
#else
    for (int x = 0; x < m; x++) {
      for (int y = 0; y < n; y++) {
        float sum = beta * C[x*n+y];
        //C[x*n+y] = beta * C[x*n+y];
        for (int w = 0; w < count; w++) {
          for (int z = 0; z < k/2; z++) {
            //auto old_c = C[x*n+y];
            sum += (float)A[w*str_a + x*k+z*2+0] * (float)B[w*str_b + z*n*2+y*2+0];
            //if(x==0 && y == 0) printf("BRGEMMDEBUG: %g * %g = %g, O: %g N: %g\n", (float)A[w*str_a + x*k+z*2+0], (float)B[w*str_b + z*n*2+y*2+0], (float)A[w*str_a + x*k+z*2+0] * B[w*str_b + z*n*2+y*2+0], (float)old_c, (float)C[x*n+y]);
            sum += (float)A[w*str_a + x*k+z*2+1] * (float)B[w*str_b + z*n*2+y*2+1];
            //if(x==0 && y == 0) printf("BRGEMMDEBUG: %g * %g = %g, O: %g N: %g\n", (float)A[w*str_a + x*k+z*2+1], (float)B[w*str_b + z*n*2+y*2+1], (float)A[w*str_a + x*k+z*2+1] * B[w*str_b + z*n*2+y*2+1], (float)old_c, (float)C[x*n+y]);
          }
        }
        C[x*n+y] = sum;
      }
    }
#endif
  }
  if (c_trans != 0) {
    if (beta_ == 0.0) {
      xsmm_xform_input(1, m, n, m*n, m*n, c_trans, C, C_);
    } else {
      bfloat16 tmp[m*n];
      xsmm_xform_input(1, m, n, m*n, m*n, c_trans, C, tmp);
#if 1
      // beta_ is 1.0
      xsmm_add(m*n, C_, tmp, C_);
#else
      for (int i = 0; i < m*n; i++) {
        C_[i] = beta_ * C_[i] + tmp[i];
      }
#endif
    }
  }
}

inline void brgemm(long m, long n, long k, long str_a, long str_b, bfloat16 *A, bfloat16 *B, float *C, long count, const float beta = 1.0, const int c_trans=0)
{
  const float alpha = 1.0;
  unsigned long long l_br = count;
  //int flags = LIBXSMM_GEMM_FLAGS( 'N', 'N' );
  int flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');

  if (c_trans != 0) {
    PCL_ASSERT(false, "c_trans != 0 not supported\n");
  }
#if 1
  ScopedTimer _t(BRGEMM, globalPass, 2*m*n*k*count);
  libxsmm_bsmmfunction_reducebatch_strd kernel = libxsmm_bsmmdispatch_reducebatch_strd(n, m, k, str_b*sizeof(bfloat16), str_a*sizeof(bfloat16), NULL, NULL, NULL, &alpha, &beta, &flags, NULL);
  kernel((libxsmm_bfloat16*)B, (libxsmm_bfloat16*)A, (float*)C, &l_br);
#else
  for (int x = 0; x < m; x++) {
    for (int y = 0; y < n; y++) {
      float sum = beta * C[x*n+y];
      for (int w = 0; w < count; w++) {
        for (int z = 0; z < k/2; z++) {
          sum += (float)A[w*str_a + x*k+z*2+0] * (float)B[w*str_b + z*n*2+y*2+0];
          sum += (float)A[w*str_a + x*k+z*2+1] * (float)B[w*str_b + z*n*2+y*2+1];
        }
      }
      C[x*n+y] = sum;
    }
  }
#endif
}

template <typename T>
void pcl_dropout_fwd(long N, T *in, T *out, short *dropout_mask, float p)
{
  //RECORD_FUNCTION("xsmm_dropout", std::vector<c10::IValue>());
  ScopedTimer _t(DROPOUT, FWD);
#ifdef __AVX512F__
  long i = 0;
  long nnz = 0;
  p = 1 - p;
  __m512 vp = _mm512_set1_ps(p);
  __m512 vpi = _mm512_set1_ps(1.0/p);
  for (i = 0; i < N - 15; i+=16) {
    __m512 rnd = LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS(rnd_state);
    __m512 vin = _mm512_loadu_ps_auto(in+i);
   //__mmask16 dmsk = _mm512_cmplt_ps_mask(rnd, vp);
   __mmask16 dmsk = _mm512_cmp_ps_mask(rnd, vp, _CMP_LT_OS);
   nnz += __builtin_popcount((int)dmsk);
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_storeu_ps_auto(out+i, vout);
    dropout_mask[i/16] = dmsk;
  }
  if (i < N) {
    int rem = N - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 rnd = LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS(rnd_state);
    __m512 vin = _mm512_maskz_loadu_ps_auto(mask, in+i);
   //__mmask16 dmsk = _mm512_cmplt_ps_mask(rnd, vp);
   __mmask16 dmsk = _mm512_cmp_ps_mask(rnd, vp, _CMP_LT_OS);
   nnz += __builtin_popcount((int)dmsk);
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_mask_storeu_ps_auto(out+i, mask, vout);
    dropout_mask[i/16] = dmsk & mask;
  }
  //printf("%ld/%ld = %.3f\n", nnz, N, (float)nnz/N);
#else
  printf("Dropout FWD not implemented without AVX512 support\n");
  exit(1);
#endif
}

template<typename T>
void pcl_dropout_bwd(long N, T *in, T *out, short *dropout_mask, float p)
{
  //RECORD_FUNCTION("xsmm_dropout_bwd", std::vector<c10::IValue>());
  ScopedTimer _t(DROPOUT, BWD);
#ifdef __AVX512F__
  long i = 0;
  p = 1 - p;
  __m512 vpi = _mm512_set1_ps(1.0/p);
  for (i = 0; i < N - 15; i+=16) {
    __m512 vin = _mm512_loadu_ps_auto(in+i);
    __mmask16 dmsk = dropout_mask[i/16];
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_storeu_ps_auto(out+i, vout);
  }
  if (i < N) {
    int rem = N - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 vin = _mm512_maskz_loadu_ps_auto(mask, in+i);
   __mmask16 dmsk = dropout_mask[i/16];
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_mask_storeu_ps_auto(out+i, mask, vout);
  }
#else
  printf("Dropout BWD not implemented without AVX512 support\n");
  exit(1);
#endif
}

inline void pcl_softmax_fwd(long S1, long S2, long S3, float *pinp, float *pout)
{
  ScopedTimer _t(SOFTMAX, FWD);
  //RECORD_FUNCTION("pcl_softmax", std::vector<c10::IValue>());
  DECL_VLA_PTR(float, inp, [S2][S3], pinp);
  DECL_VLA_PTR(float, out, [S2][S3], pout);
#ifndef __AVX512F__
    //printf("Scalar Path: S1 = %ld  S2 = %ld  S3 = %ld\n", S1, S2, S3);
  for(int s2 = 0; s2 < S2; s2++) {
    float tmp[S1][S3];
    float max = inp[0][s2][0];
    float sum = 0.0;
    for(int s1 = 0; s1 < S1; s1++) {
      for(int s3 = 0; s3 < S3; s3++) {
        if(max < inp[s1][s2][s3]) max = inp[s1][s2][s3];
      }
    }
    for(int s1 = 0; s1 < S1; s1++) {
      for(int s3 = 0; s3 < S3; s3++) {
        float z = std::exp(inp[s1][s2][s3] - max);
        //out[s1][s2][s3] = z;
        tmp[s1][s3] = z;
        sum += z;
      }
    }
    sum = 1.0 / sum;
    for(int s1 = 0; s1 < S1; s1++) {
      for(int s3 = 0; s3 < S3; s3++) {
        //out[s1][s2][s3] *= sum;
        out[s1][s2][s3] = tmp[s1][s3] * sum;
      }
    }
  }
#else
  //printf("S1 = %d  S2 = %d  S3 = %d\n", S1, S2, S3);
  for(int s2 = 0; s2 < S2; s2++) {
    float tmp[S1][S3];
    float max = inp[0][s2][0];
    float sum = 0.0;
    __m512 vmax = _mm512_set1_ps(max);
    __m512 vsum = _mm512_setzero_ps();

    for(int s1 = 0; s1 < S1; s1++) {
      int s3;
      for(s3 = 0; s3 < ALIGNDOWN(S3, 16); s3+=16) {
        vmax = _mm512_max_ps(_mm512_loadu_ps_auto(&inp[s1][s2][s3]), vmax);
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        vmax = _mm512_mask_max_ps(vmax, mask, _mm512_maskz_loadu_ps_auto(mask, &inp[s1][s2][s3]), vmax);
      }
    }
    max = _mm512_reduce_max_ps(vmax);
    vmax = _mm512_set1_ps(max);
    for(int s1 = 0; s1 < S1; s1++) {
      int s3;
      for(s3 = 0; s3 < ALIGNDOWN(S3, 16); s3+=16) {
        __m512 vz = _MM512_EXP_PS(_mm512_sub_ps(_mm512_loadu_ps_auto(&inp[s1][s2][s3]), vmax));
        _mm512_storeu_ps_auto(&tmp[s1][s3], vz);
        vsum = _mm512_add_ps(vsum, vz);
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vz = _MM512_EXP_PS(_mm512_sub_ps(_mm512_maskz_loadu_ps_auto(mask, &inp[s1][s2][s3]), vmax));
        _mm512_mask_storeu_ps_auto(&tmp[s1][s3], mask, vz);
        vsum = _mm512_mask_add_ps(vsum, mask, vsum, vz);
      }
    }
    sum = _mm512_reduce_add_ps(vsum);
    sum = 1.0 / sum;
    vsum = _mm512_set1_ps(sum);
    for(int s1 = 0; s1 < S1; s1++) {
      int s3;
      for(s3 = 0; s3 < ALIGNDOWN(S3, 16); s3+=16) {
        _mm512_storeu_ps_auto(&out[s1][s2][s3], _mm512_mul_ps(vsum, _mm512_loadu_ps_auto(&tmp[s1][s3])));
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        _mm512_mask_storeu_ps_auto(&out[s1][s2][s3], mask, _mm512_mul_ps(vsum, _mm512_maskz_loadu_ps_auto(mask, &tmp[s1][s3])));
      }
    }
  }
#endif
}

inline void pcl_softmax_fwd(long S1, long S2, long S3, float *pinp, bfloat16 *pout)
{
  ScopedTimer _t(SOFTMAX, FWD);
  //RECORD_FUNCTION("pcl_softmax", std::vector<c10::IValue>());
  DECL_VLA_PTR(float, inp, [S2][S3], pinp);
  DECL_VLA_PTR(bfloat16, out, [S2][S3], pout);
#ifndef __AVX512F__
  //printf("Scalar Path: S1 = %ld  S2 = %ld  S3 = %ld\n", S1, S2, S3);
  for(int s2 = 0; s2 < S2; s2++) {
    float tmp[S1][S3];
    float max = inp[0][s2][0];
    float sum = 0.0;
    for(int s1 = 0; s1 < S1; s1++) {
      for(int s3 = 0; s3 < S3; s3++) {
        if(max < inp[s1][s2][s3]) max = inp[s1][s2][s3];
      }
    }
    for(int s1 = 0; s1 < S1; s1++) {
      for(int s3 = 0; s3 < S3; s3++) {
        float z = std::exp(inp[s1][s2][s3] - max);
        //out[s1][s2][s3] = z;
        tmp[s1][s3] = z;
        sum += z;
      }
    }
    sum = 1.0 / sum;
    for(int s1 = 0; s1 < S1; s1++) {
      for(int s3 = 0; s3 < S3; s3++) {
        //out[s1][s2][s3] *= sum;
        out[s1][s2][s3] = tmp[s1][s3] * sum;
      }
    }
  }
#else
  //printf("S1 = %d  S2 = %d  S3 = %d\n", S1, S2, S3);
  for(int s2 = 0; s2 < S2; s2++) {
    float tmp[S1][S3];
    float max = inp[0][s2][0];
    float sum = 0.0;
    __m512 vmax = _mm512_set1_ps(max);
    __m512 vsum = _mm512_setzero_ps();

    for(int s1 = 0; s1 < S1; s1++) {
      int s3;
      for(s3 = 0; s3 < ALIGNDOWN(S3, 16); s3+=16) {
        vmax = _mm512_max_ps(_mm512_loadu_ps_auto(&inp[s1][s2][s3]), vmax);
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        vmax = _mm512_mask_max_ps(vmax, mask, _mm512_maskz_loadu_ps_auto(mask, &inp[s1][s2][s3]), vmax);
      }
    }
    max = _mm512_reduce_max_ps(vmax);
    vmax = _mm512_set1_ps(max);
    for(int s1 = 0; s1 < S1; s1++) {
      int s3;
      for(s3 = 0; s3 < ALIGNDOWN(S3, 16); s3+=16) {
        __m512 vz = _MM512_EXP_PS(_mm512_sub_ps(_mm512_loadu_ps_auto(&inp[s1][s2][s3]), vmax));
        _mm512_storeu_ps_auto(&tmp[s1][s3], vz);
        vsum = _mm512_add_ps(vsum, vz);
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vz = _MM512_EXP_PS(_mm512_sub_ps(_mm512_maskz_loadu_ps_auto(mask, &inp[s1][s2][s3]), vmax));
        _mm512_mask_storeu_ps_auto(&tmp[s1][s3], mask, vz);
        vsum = _mm512_mask_add_ps(vsum, mask, vsum, vz);
      }
    }
    sum = _mm512_reduce_add_ps(vsum);
    sum = 1.0 / sum;
    vsum = _mm512_set1_ps(sum);
    for(int s1 = 0; s1 < S1; s1++) {
      int s3;
      for(s3 = 0; s3 < ALIGNDOWN(S3, 16); s3+=16) {
        _mm512_storeu_ps_auto(&out[s1][s2][s3], _mm512_mul_ps(vsum, _mm512_loadu_ps_auto(&tmp[s1][s3])));
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        _mm512_mask_storeu_ps_auto(&out[s1][s2][s3], mask, _mm512_mul_ps(vsum, _mm512_maskz_loadu_ps_auto(mask, &tmp[s1][s3])));
      }
    }
  }
#endif
}

inline void pcl_softmax_bwd(long S1, long S2, long S3, float *pgradinp, float *pgradout, float *pout)
{
  ScopedTimer _t(SOFTMAX, BWD);
  //RECORD_FUNCTION("pcl_softmax_bwd", std::vector<c10::IValue>());
  DECL_VLA_PTR(float, ginp, [S2][S3], pgradinp);
  DECL_VLA_PTR(float, gout, [S2][S3], pgradout);
  DECL_VLA_PTR(float, out, [S2][S3], pout);
#ifndef __AVX512F__
  //printf("Scalar Path: S1 = %ld  S2 = %ld  S3 = %ld\n", S1, S2, S3);
  for(int s2 = 0; s2 < S2; s2++) {
    float sum = 0.0;
    for(int s1 = 0; s1 < S1; s1++) {
      for(int s3 = 0; s3 < S3; s3++) {
        sum += gout[s1][s2][s3] * out[s1][s2][s3];
      }
    }
    for(int s1 = 0; s1 < S1; s1++) {
      for(int s3 = 0; s3 < S3; s3++) {
        ginp[s1][s2][s3] = out[s1][s2][s3] * (gout[s1][s2][s3] - sum);
      }
    }
  }
#else
  //printf("S1 = %d  S2 = %d  S3 = %d\n", S1, S2, S3);
  for(int s2 = 0; s2 < S2; s2++) {
    float sum = 0.0;
    __m512 vsum = _mm512_setzero_ps();

    for(int s1 = 0; s1 < S1; s1++) {
      int s3;
      for(s3 = 0; s3 < ALIGNDOWN(S3, 16); s3+=16) {
        __m512 vgo = _mm512_loadu_ps_auto(&gout[s1][s2][s3]);
        __m512 vo = _mm512_loadu_ps_auto(&out[s1][s2][s3]);
        vsum = _mm512_fmadd_ps(vgo, vo, vsum);
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vgo = _mm512_maskz_loadu_ps_auto(mask, &gout[s1][s2][s3]);
        __m512 vo = _mm512_maskz_loadu_ps_auto(mask, &out[s1][s2][s3]);
        vsum = _mm512_fmadd_ps(vgo, vo, vsum);
      }
    }
    sum = _mm512_reduce_add_ps(vsum);
    vsum = _mm512_set1_ps(sum);
    for(int s1 = 0; s1 < S1; s1++) {
      int s3;
      for(s3 = 0; s3 < ALIGNDOWN(S3, 16); s3+=16) {
        __m512 tmp = _mm512_sub_ps(_mm512_loadu_ps_auto(&gout[s1][s2][s3]), vsum);
        _mm512_storeu_ps_auto(&ginp[s1][s2][s3], _mm512_mul_ps(_mm512_loadu_ps_auto(&out[s1][s2][s3]), tmp));
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 tmp = _mm512_sub_ps(_mm512_maskz_loadu_ps_auto(mask, &gout[s1][s2][s3]), vsum);
        _mm512_mask_storeu_ps_auto(&ginp[s1][s2][s3], mask, _mm512_mul_ps(_mm512_maskz_loadu_ps_auto(mask, &out[s1][s2][s3]), tmp));
      }
    }
  }
#endif
}

inline void pcl_softmax_bwd(long S1, long S2, long S3, float *pgradinp, float *pgradout, bfloat16 *pout)
{
  ScopedTimer _t(SOFTMAX, BWD);
  //RECORD_FUNCTION("pcl_softmax_bwd", std::vector<c10::IValue>());
  DECL_VLA_PTR(float, ginp, [S2][S3], pgradinp);
  DECL_VLA_PTR(float, gout, [S2][S3], pgradout);
  DECL_VLA_PTR(bfloat16, out, [S2][S3], pout);
#ifndef __AVX512F__
  //printf("Scalar Path: S1 = %ld  S2 = %ld  S3 = %ld\n", S1, S2, S3);
  for(int s2 = 0; s2 < S2; s2++) {
    float sum = 0.0;
    for(int s1 = 0; s1 < S1; s1++) {
      for(int s3 = 0; s3 < S3; s3++) {
        sum += gout[s1][s2][s3] * out[s1][s2][s3];
      }
    }
    for(int s1 = 0; s1 < S1; s1++) {
      for(int s3 = 0; s3 < S3; s3++) {
        ginp[s1][s2][s3] = out[s1][s2][s3] * (gout[s1][s2][s3] - sum);
      }
    }
  }
#else
  //printf("S1 = %d  S2 = %d  S3 = %d\n", S1, S2, S3);
  for(int s2 = 0; s2 < S2; s2++) {
    float sum = 0.0;
    __m512 vsum = _mm512_setzero_ps();

    for(int s1 = 0; s1 < S1; s1++) {
      int s3;
      for(s3 = 0; s3 < ALIGNDOWN(S3, 16); s3+=16) {
        __m512 vgo = _mm512_loadu_ps_auto(&gout[s1][s2][s3]);
        __m512 vo = _mm512_loadu_ps_auto(&out[s1][s2][s3]);
        vsum = _mm512_fmadd_ps(vgo, vo, vsum);
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vgo = _mm512_maskz_loadu_ps_auto(mask, &gout[s1][s2][s3]);
        __m512 vo = _mm512_maskz_loadu_ps_auto(mask, &out[s1][s2][s3]);
        vsum = _mm512_fmadd_ps(vgo, vo, vsum);
      }
    }
    sum = _mm512_reduce_add_ps(vsum);
    vsum = _mm512_set1_ps(sum);
    for(int s1 = 0; s1 < S1; s1++) {
      int s3;
      for(s3 = 0; s3 < ALIGNDOWN(S3, 16); s3+=16) {
        __m512 tmp = _mm512_sub_ps(_mm512_loadu_ps_auto(&gout[s1][s2][s3]), vsum);
        _mm512_storeu_ps_auto(&ginp[s1][s2][s3], _mm512_mul_ps(_mm512_loadu_ps_auto(&out[s1][s2][s3]), tmp));
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 tmp = _mm512_sub_ps(_mm512_maskz_loadu_ps_auto(mask, &gout[s1][s2][s3]), vsum);
        _mm512_mask_storeu_ps_auto(&ginp[s1][s2][s3], mask, _mm512_mul_ps(_mm512_maskz_loadu_ps_auto(mask, &out[s1][s2][s3]), tmp));
      }
    }
  }
#endif
}

template<typename T>
void pcl_layer_norm_fwd(long S1, long S2, long S3, T *pinp, T *pgamma, T *pbeta, float *mean, float *var, T *pout, float eps)
{
  ScopedTimer _t(LAYER_NORM, FWD);
  DECL_VLA_PTR(T, inp, [S2][S3], pinp);
  DECL_VLA_PTR(T, out, [S2][S3], pout);
  DECL_VLA_PTR(T, gamma, [S3], pgamma);
  DECL_VLA_PTR(T, beta, [S3], pbeta);

#ifdef __AVX512F__
  for(int s2 = 0; s2 < S2; s2++) {
    __m512 vm = _mm512_setzero_ps();
    __m512 vv = _mm512_setzero_ps();
    for(int s1 = 0; s1 < S1; s1++) {
      int s3;
      for( s3 = 0; s3 < S3-15; s3+=16) {
        auto vin = _mm512_loadu_ps_auto(&inp[s1][s2][s3]);
        vm = _mm512_add_ps(vm, vin);
        vv = _mm512_add_ps(vv, _mm512_mul_ps(vin, vin));
      }
      if(s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        //printf("s1,s2,s3,addr = %d %d %d  %p,  S2,S3=%ld %ld diff = %lu\n", s1, s2, s3, &inp[s1][s2][s3], S2, S3, &inp[s1][s2][s3] - &inp[0][0][0]);
        auto vin = _mm512_maskz_loadu_ps_auto(mask, &inp[s1][s2][s3]);
        vm = _mm512_add_ps(vm, vin);
        vv = _mm512_add_ps(vv, _mm512_mul_ps(vin, vin));
      }
    }
    float c = 1.0 / (S1*S3);
    float m = _mm512_reduce_add_ps(vm) * c;
    float v = _mm512_reduce_add_ps(vv) * c;
    //std::cout << "m: " << m << ", v: " << v << std::endl;
    v = std::max(v - m * m, 0.0f);
    //std::cout << "m: " << m << ", v: " << v << std::endl;
    v = 1.0f / ((float)sqrt(v+eps));
    mean[s2] = m;
    var[s2] = v;
    //std::cout << "m: " << m << ", v: " << v << std::endl;
    float s = v;
    float b = -1.0 * v * m;
    __m512 vs = _mm512_set1_ps(s);
    __m512 vb = _mm512_set1_ps(b);
    for (int s1 = 0; s1 < S1; s1++) {
      int s3;
      for (s3 = 0; s3 < S3-15; s3+=16) {
        //out[s1][s2][s3] = (inp[s1][s2][s3] * s + b) * gamma[s1][s3] + beta[s1][s3];
        __m512 vin = _mm512_loadu_ps_auto(&inp[s1][s2][s3]);
        __m512 vg = _mm512_loadu_ps_auto(&gamma[s1][s3]);
        __m512 vbt = _mm512_loadu_ps_auto(&beta[s1][s3]);
        __m512 vout = _mm512_add_ps(_mm512_mul_ps(vin, vs), vb);
        vout = _mm512_add_ps(_mm512_mul_ps(vout, vg), vbt);
        _mm512_storeu_ps_auto(&out[s1][s2][s3], vout);
      }
      /*for (; s3 < S3; s3++) {
        std::cout << "s: " << s << ", b: " << b << std::endl;
        out[s1][s2][s3] = (inp[s1][s2][s3] * s + b) * gamma[s1][s3] + beta[s1][s3];
      }*/
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vin = _mm512_maskz_loadu_ps_auto(mask, &inp[s1][s2][s3]);
        __m512 vg = _mm512_maskz_loadu_ps_auto(mask, &gamma[s1][s3]);
        __m512 vbt = _mm512_maskz_loadu_ps_auto(mask, &beta[s1][s3]);
        __m512 vout = _mm512_add_ps(_mm512_mul_ps(vin, vs), vb);
        vout = _mm512_add_ps(_mm512_mul_ps(vout, vg), vbt);
        _mm512_mask_storeu_ps_auto(&out[s1][s2][s3], mask, vout);
      }
    }
  }
#else
  int s1, s2, s3;
  for (s2 = 0; s2 < S2; s2++) {
    float m = 0;
    float v = 0;
    float c = 1.0 / (S1*S3);
    for (s1 = 0; s1 < S1; s1++) {
      for( s3 = 0; s3 < S3; s3++) {
        m += inp[s1][s2][s3];
        v += inp[s1][s2][s3] * inp[s1][s2][s3];
      }
    }
    m = m * c;
    v = v * c;
    v = LIBXSMM_MAX(v - m * m, 0.0f);
    v = 1.0f / ((float)sqrt(v+eps));
    mean[s2] = m;
    var[s2] = v;
    float s = v;
    float b = -1.0 * v * m;
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        out[s1][s2][s3] = (inp[s1][s2][s3] * s + b) * gamma[s1][s3] + beta[s1][s3];
      }
    }
  }
#endif
}

//template<typename T>
void pcl_layer_norm_bwd(long S1, long S2, long S3, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta)
{
  typedef float T;
  ScopedTimer _t(LAYER_NORM, BWD);
  DECL_VLA_PTR(T, din, [S2][S3], pdin);
  DECL_VLA_PTR(T, inp, [S2][S3], pinp);
  DECL_VLA_PTR(T, dout, [S2][S3], pdout);
  DECL_VLA_PTR(T, gamma, [S3], pgamma);
  DECL_VLA_PTR(float, dgamma, [S3], pdgamma);
  DECL_VLA_PTR(float, dbeta, [S3], pdbeta);
#ifdef __AVX512F__
  for(int s2 = 0; s2 < S2; s2++) {
    float a = var[s2];
    float b = -a*mean[s2];
    __m512 va = _mm512_set1_ps(a);
    __m512 vb = _mm512_set1_ps(b);
    __m512 vds = _mm512_setzero_ps();
    __m512 vdb = _mm512_setzero_ps();
    float ds = 0.0f;
    float db = 0.0f;
    for(int s1 = 0; s1 < S1; s1++) {
      int s3;
      for( s3 = 0; s3 < S3-15; s3+=16) {
        auto vdout = _mm512_loadu_ps_auto(&dout[s1][s2][s3]);
        auto vin = _mm512_loadu_ps_auto(&inp[s1][s2][s3]);
        auto vdg = _mm512_mul_ps(vdout, _mm512_add_ps(_mm512_mul_ps(va, vin), vb));
        _mm512_storeu_ps_auto(&dgamma[s1][s3], _mm512_add_ps(vdg, _mm512_loadu_ps_auto(&dgamma[s1][s3])));
        _mm512_storeu_ps_auto(&dbeta[s1][s3], _mm512_add_ps(vdout, _mm512_loadu_ps_auto(&dbeta[s1][s3])));
        auto vtmp = _mm512_mul_ps(vdout, _mm512_loadu_ps_auto(&gamma[s1][s3]));
        vds = _mm512_add_ps(vds, _mm512_mul_ps(vtmp, vin));
        vdb = _mm512_add_ps(vdb, vtmp);
      }
#if 0
      for(; s3 < S3; s3++) {
        dgamma[s1][s3] += (a * inp[s1][s2][s3] + b) * dout[s1][s2][s3];
        dbeta[s1][s3] += dout[s1][s2][s3];
        ds += (dout[s1][s2][s3] * gamma[s1][s3]) * inp[s1][s2][s3];
        db += dout[s1][s2][s3] * gamma[s1][s3];
        //printf("s1: %d s2: %d s3: %d  dbeta: %14.8g   %14.8g\n", s1, s2, s3, dbeta[s1][s3], dout[s1][s2][s3]);
      }
#endif
      if(s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        auto vdout = _mm512_maskz_loadu_ps_auto(mask, &dout[s1][s2][s3]);
        auto vin = _mm512_maskz_loadu_ps_auto(mask, &inp[s1][s2][s3]);
        auto vdg = _mm512_mul_ps(vdout, _mm512_add_ps(_mm512_mul_ps(va, vin), vb));
        _mm512_mask_storeu_ps_auto(&dgamma[s1][s3], mask, _mm512_add_ps(vdg, _mm512_maskz_loadu_ps_auto(mask, &dgamma[s1][s3])));
        _mm512_mask_storeu_ps_auto(&dbeta[s1][s3], mask, _mm512_add_ps(vdout, _mm512_maskz_loadu_ps_auto(mask, &dbeta[s1][s3])));
        auto vtmp = _mm512_mul_ps(vdout, _mm512_maskz_loadu_ps_auto(mask, &gamma[s1][s3]));
        vds = _mm512_add_ps(vds, _mm512_mul_ps(vtmp, vin));
        vdb = _mm512_add_ps(vdb, vtmp);
      }
    }
    ds += _mm512_reduce_add_ps(vds);
    db += _mm512_reduce_add_ps(vdb);
    float scale = 1.0f / (S1 * S3);
    b = (db * mean[s2] - ds) * a * a * a * scale;
    float c = -b * mean[s2] - db * a * scale;

    vb = _mm512_set1_ps(b);
    __m512 vc = _mm512_set1_ps(c);
    for (int s1 = 0; s1 < S1; s1++) {
      int s3;
      for (s3 = 0; s3 < S3-15; s3+=16) {
        auto vdout = _mm512_loadu_ps_auto(&dout[s1][s2][s3]);
        auto vin = _mm512_loadu_ps_auto(&inp[s1][s2][s3]);
        __m512 vg = _mm512_loadu_ps_auto(&gamma[s1][s3]);
        auto vtmp1 = _mm512_mul_ps(_mm512_mul_ps(va, vdout), vg);
        auto vtmp2 = _mm512_add_ps(vtmp1, _mm512_mul_ps(vb, vin));
        auto vdin = _mm512_add_ps(vtmp2, vc);
        _mm512_storeu_ps_auto(&din[s1][s2][s3], vdin);
      }
#if 0
      for (; s3 < S3; s3++) {
        din[s1][s2][s3] = (double)dout[s1][s2][s3] * a * gamma[s1][s3] + b * inp[s1][s2][s3] + c;
        printf("s3: %d  in: %18.10g dout: %18.10g a: %18.10g g: %18.10g b: %18.10g c: %18.10g  din: %18.10g\n", s3, inp[s1][s2][s3], dout[s1][s2][s3], a, gamma[s1][s3], b, c, din[s1][s2][s3]);
      }
#endif
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        auto vdout = _mm512_maskz_loadu_ps_auto(mask, &dout[s1][s2][s3]);
        auto vin = _mm512_maskz_loadu_ps_auto(mask, &inp[s1][s2][s3]);
        __m512 vg = _mm512_maskz_loadu_ps_auto(mask, &gamma[s1][s3]);
        auto vtmp1 = _mm512_mul_ps(_mm512_mul_ps(va, vdout), vg);
        auto vtmp2 = _mm512_add_ps(vtmp1, _mm512_mul_ps(vb, vin));
        auto vdin = _mm512_add_ps(vtmp2, vc);
        _mm512_mask_storeu_ps_auto(&din[s1][s2][s3], mask, vdin);
      }
    }
  }
#else
  int s1, s2, s3;
  for (s2 = 0; s2 < S2; s2++) {
    float a = var[s2], c;
    float b = -a*mean[s2];
    float ds = 0.0f;
    float db = 0.0f;
    float scale = 1.0f / (S1 * S3);
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        dgamma[s1][s3] += (a * inp[s1][s2][s3] + b) * dout[s1][s2][s3];
        dbeta[s1][s3] +=  dout[s1][s2][s3];
        ds += dout[s1][s2][s3] * gamma[s1][s3] * inp[s1][s2][s3];
        db += dout[s1][s2][s3] * gamma[s1][s3];
      }
    }
    b = (db * mean[s2] - ds) * a * a * a * scale;
    c = -b * mean[s2] - db * a * scale;
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        din[s1][s2][s3] = dout[s1][s2][s3] * a * gamma[s1][s3] + b * inp[s1][s2][s3] + c;
      }
    }
  }
#endif
}

void pcl_layer_norm_bwd(long S1, long S2, long S3, bfloat16 *pdout, bfloat16 *pinp, float *mean, float *var, bfloat16 *pgamma, bfloat16 *pdin, float *pdgamma, float *pdbeta)
{
  typedef bfloat16 T;
  ScopedTimer _t(LAYER_NORM, BWD);
  DECL_VLA_PTR(T, din, [S2][S3], pdin);
  DECL_VLA_PTR(T, inp, [S2][S3], pinp);
  DECL_VLA_PTR(T, dout, [S2][S3], pdout);
  DECL_VLA_PTR(T, gamma, [S3], pgamma);
  DECL_VLA_PTR(float, dgamma, [S3], pdgamma);
  DECL_VLA_PTR(float, dbeta, [S3], pdbeta);
#ifdef __AVX512F__
  for(int s2 = 0; s2 < S2; s2++) {
    float a = var[s2];
    float b = -a*mean[s2];
    __m512 va = _mm512_set1_ps(a);
    __m512 vb = _mm512_set1_ps(b);
    __m512 vds = _mm512_setzero_ps();
    __m512 vdb = _mm512_setzero_ps();
    float ds = 0.0f;
    float db = 0.0f;
    for(int s1 = 0; s1 < S1; s1++) {
      int s3;
      for( s3 = 0; s3 < S3-15; s3+=16) {
        auto vdout = _mm512_loadu_ps_auto(&dout[s1][s2][s3]);
        auto vin = _mm512_loadu_ps_auto(&inp[s1][s2][s3]);
        auto vdg = _mm512_mul_ps(vdout, _mm512_add_ps(_mm512_mul_ps(va, vin), vb));
        _mm512_storeu_ps_auto(&dgamma[s1][s3], _mm512_add_ps(vdg, _mm512_loadu_ps_auto(&dgamma[s1][s3])));
        _mm512_storeu_ps_auto(&dbeta[s1][s3], _mm512_add_ps(vdout, _mm512_loadu_ps_auto(&dbeta[s1][s3])));
        auto vtmp = _mm512_mul_ps(vdout, _mm512_loadu_ps_auto(&gamma[s1][s3]));
        vds = _mm512_add_ps(vds, _mm512_mul_ps(vtmp, vin));
        vdb = _mm512_add_ps(vdb, vtmp);
      }
#if 0
      for(; s3 < S3; s3++) {
        dgamma[s1][s3] += (a * inp[s1][s2][s3] + b) * dout[s1][s2][s3];
        dbeta[s1][s3] += dout[s1][s2][s3];
        ds += (dout[s1][s2][s3] * gamma[s1][s3]) * inp[s1][s2][s3];
        db += dout[s1][s2][s3] * gamma[s1][s3];
        //printf("s1: %d s2: %d s3: %d  dbeta: %14.8g   %14.8g\n", s1, s2, s3, dbeta[s1][s3], dout[s1][s2][s3]);
      }
#endif
      if(s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        auto vdout = _mm512_maskz_loadu_ps_auto(mask, &dout[s1][s2][s3]);
        auto vin = _mm512_maskz_loadu_ps_auto(mask, &inp[s1][s2][s3]);
        auto vdg = _mm512_mul_ps(vdout, _mm512_add_ps(_mm512_mul_ps(va, vin), vb));
        _mm512_mask_storeu_ps_auto(&dgamma[s1][s3], mask, _mm512_add_ps(vdg, _mm512_maskz_loadu_ps_auto(mask, &dgamma[s1][s3])));
        _mm512_mask_storeu_ps_auto(&dbeta[s1][s3], mask, _mm512_add_ps(vdout, _mm512_maskz_loadu_ps_auto(mask, &dbeta[s1][s3])));
        auto vtmp = _mm512_mul_ps(vdout, _mm512_maskz_loadu_ps_auto(mask, &gamma[s1][s3]));
        vds = _mm512_add_ps(vds, _mm512_mul_ps(vtmp, vin));
        vdb = _mm512_add_ps(vdb, vtmp);
      }
    }
    ds += _mm512_reduce_add_ps(vds);
    db += _mm512_reduce_add_ps(vdb);
    float scale = 1.0f / (S1 * S3);
    b = (db * mean[s2] - ds) * a * a * a * scale;
    float c = -b * mean[s2] - db * a * scale;

    vb = _mm512_set1_ps(b);
    __m512 vc = _mm512_set1_ps(c);
    for (int s1 = 0; s1 < S1; s1++) {
      int s3;
      for (s3 = 0; s3 < S3-15; s3+=16) {
        auto vdout = _mm512_loadu_ps_auto(&dout[s1][s2][s3]);
        auto vin = _mm512_loadu_ps_auto(&inp[s1][s2][s3]);
        __m512 vg = _mm512_loadu_ps_auto(&gamma[s1][s3]);
        auto vtmp1 = _mm512_mul_ps(_mm512_mul_ps(va, vdout), vg);
        auto vtmp2 = _mm512_add_ps(vtmp1, _mm512_mul_ps(vb, vin));
        auto vdin = _mm512_add_ps(vtmp2, vc);
        _mm512_storeu_ps_auto(&din[s1][s2][s3], vdin);
      }
#if 0
      for (; s3 < S3; s3++) {
        din[s1][s2][s3] = (double)dout[s1][s2][s3] * a * gamma[s1][s3] + b * inp[s1][s2][s3] + c;
        printf("s3: %d  in: %18.10g dout: %18.10g a: %18.10g g: %18.10g b: %18.10g c: %18.10g  din: %18.10g\n", s3, inp[s1][s2][s3], dout[s1][s2][s3], a, gamma[s1][s3], b, c, din[s1][s2][s3]);
      }
#endif
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        auto vdout = _mm512_maskz_loadu_ps_auto(mask, &dout[s1][s2][s3]);
        auto vin = _mm512_maskz_loadu_ps_auto(mask, &inp[s1][s2][s3]);
        __m512 vg = _mm512_maskz_loadu_ps_auto(mask, &gamma[s1][s3]);
        auto vtmp1 = _mm512_mul_ps(_mm512_mul_ps(va, vdout), vg);
        auto vtmp2 = _mm512_add_ps(vtmp1, _mm512_mul_ps(vb, vin));
        auto vdin = _mm512_add_ps(vtmp2, vc);
        _mm512_mask_storeu_ps_auto(&din[s1][s2][s3], mask, vdin);
      }
    }
  }
#else
  int s1, s2, s3;
  for (s2 = 0; s2 < S2; s2++) {
    float a = var[s2], c;
    float b = -a*mean[s2];
    float ds = 0.0f;
    float db = 0.0f;
    float scale = 1.0f / (S1 * S3);
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        dgamma[s1][s3] += (a * inp[s1][s2][s3] + b) * dout[s1][s2][s3];
        dbeta[s1][s3] +=  dout[s1][s2][s3];
        ds += dout[s1][s2][s3] * gamma[s1][s3] * inp[s1][s2][s3];
        db += dout[s1][s2][s3] * gamma[s1][s3];
      }
    }
    b = (db * mean[s2] - ds) * a * a * a * scale;
    c = -b * mean[s2] - db * a * scale;
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        din[s1][s2][s3] = dout[s1][s2][s3] * a * gamma[s1][s3] + b * inp[s1][s2][s3] + c;
      }
    }
  }
#endif
}

inline at::Tensor wt_tensor_n2v(long Nk, long Hk, long Nc, long Hc, at::Tensor &input) {
#if 0
  return input.view({Nk, Nc, Hc/2, 2, Hk}).permute({0, 1, 2, 4, 3}).contiguous();
#else
  auto output = input.new_empty({Nk, Nc, Hc/2, Hk, 2});
  DECL_VLA_PTR_PT(bfloat16, out, [Hc*Hk], output);
  DECL_VLA_PTR_PT(bfloat16,  in, [Hc*Hk], input);
  auto n2v_tpp = SCOPEIT(XformExtTPP<bfloat16>(Hc, Hk, XformTPP::XFORM_N2V_TPP), VNNI);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
  for(int n = 0; n < Nk*Nc; n++) {
#ifdef NO_TPP_PT
    xsmm_n2v(Hc, Hk, in[n], out[n]);
#else
    n2v_tpp(in[n], out[n]);
#endif
  }
  return output;
#endif
}

inline at::Tensor wt_tensor_trans_n2v(long Nk, long Hk, long Nc, long Hc, at::Tensor &input) {
#if 0
  return input.view({Nk, Nc, Hc, Hk/2, 2}).permute({0, 1, 3, 2, 4}).contiguous();
#else
  auto output = input.new_empty({Nk, Nc, Hk/2, Hc, 2});
  DECL_VLA_PTR_PT(bfloat16, out, [Hk*Hc], output);
  DECL_VLA_PTR_PT(bfloat16,  in, [Hc*Hk], input);
  auto trans_n2v_tpp = SCOPEIT(XformExtTPP<bfloat16>(Hc, Hk, XformTPP::XFORM_XPOSE_N2V_TPP), XPOSE);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
  for(int n = 0; n < Nk*Nc; n++) {
#ifdef NO_TPP_PT
    xsmm_xpose_n2v(Hc, Hk, in[n], out[n]);
#else
    trans_n2v_tpp(in[n], out[n]);
#endif
  }
  return output;
#endif
}

inline at::Tensor wt_tensor_trans_v2v(long Nk, long Hk, long Nc, long Hc, at::Tensor &input) {
#if 0
  return input.view({Nk, Nc, Hc/2, Hk/2, 2, 2}).permute({0, 1, 3, 2, 5, 4}).contiguous().view({Nk, Nc, Hk/2, Hc, 2});
#else
  auto output = input.new_empty({Nk, Nc, Hk/2, Hc, 2});
  DECL_VLA_PTR_PT(bfloat16, out, [Hk*Hc], output);
  DECL_VLA_PTR_PT(bfloat16,  in, [Hc*Hk], input);
  auto trans_v2v_tpp = SCOPEIT(XformExtTPP<bfloat16>(Hc, Hk, XformTPP::XFORM_XPOSE_V2V_TPP), XPOSE);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
  for(int n = 0; n < Nk*Nc; n++) {
#ifdef NO_TPP_PT
    xsmm_xpose_v2v(Hc, Hk, in[n], out[n]);
#else
    trans_v2v_tpp(in[n], out[n]);
#endif
  }
  return output;
#endif
}

inline at::Tensor wt_tensor_for_fwd(long Nk, long Hk, long Nc, long Hc, at::Tensor &input)
{
  RECORD_SCOPE(w_vnni, {input});
  if (input.dtype() == at::kBFloat16) {
    if (input.dim() == 5) {
      return input;
    } else {
      return wt_tensor_n2v(Nk, Hk, Nc, Hc, input);
    }
  } else {
    return input;
  }
}

inline at::Tensor wt_tensor_for_bwd(long Nk, long Hk, long Nc, long Hc, at::Tensor &input)
{
  RECORD_SCOPE(w_xpose, {input});
  if (input.dtype() == at::kBFloat16) {
    if (input.dim() == 5) {
      return wt_tensor_trans_v2v(Nk, Hk, Nc, Hc, input);
    } else {
      return wt_tensor_trans_n2v(Nk, Hk, Nc, Hc, input);
    }
  } else {
#if 0
    return input.permute({0, 1, 3, 2}).contiguous();
#else
    auto output = input.new_empty({Nk, Nc, Hk, Hc});
    DECL_VLA_PTR_PT(float, out, [Hk*Hc], output);
    DECL_VLA_PTR_PT(float,  in, [Hc*Hk], input);
    auto trans_tpp = SCOPEIT(XformExtTPP<float>(Hc, Hk, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for(int n = 0; n < Nk*Nc; n++) {
#ifdef NO_TPP_PT
      xsmm_xpose(Hc, Hk, in[n], out[n]);
#else
      trans_tpp(in[n], out[n]);
#endif
    }
    return output;
#endif
  }
}

inline at::Tensor act_tensor_trans(long B, long S1, long N, long S2, long H, at::Tensor &input) {
  RECORD_SCOPE(a_xpose, {input});
#if 0
  return input.permute({0, 1, 2, 4, 3}).contiguous();
#else
  auto output = input.new_empty({B, S1, N, H, S2});
  DECL_VLA_PTR_PT(bfloat16, out, [H*S2], output);
  DECL_VLA_PTR_PT(bfloat16,  in, [H*S2], input);
  auto trans_tpp = SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for(int n = 0; n < B*S1*N; n++) {
#ifdef NO_TPP_PT
      xsmm_xpose(S2, H, in[n], out[n]);
#else
      trans_tpp(in[n], out[n]);
#endif
    }
  }
  return output;
#endif
}

inline at::Tensor act_tensor_n2v(long B, long S1, long N, long S2, long H, at::Tensor &input) {
  RECORD_SCOPE(a_vnni, {input});
#if 0
  return input.view({B, S1, N, S2/2, 2, H}).permute({0,1,2,3,5,4}).contiguous();
#else
  auto output = input.new_empty({B, S1, N, S2/2, H, 2});
  DECL_VLA_PTR_PT(bfloat16, out, [H*S2], output);
  DECL_VLA_PTR_PT(bfloat16,  in, [H*S2], input);
  auto n2v_tpp = SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_N2V_TPP), VNNI);
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for(int n = 0; n < B*S1*N; n++) {
#ifdef NO_TPP_PT
      xsmm_n2v(S2, H, in[n], out[n]);
#else
      n2v_tpp(in[n], out[n]);
#endif
    }
  }
  return output;
#endif
}

inline void tensor_set_zero(long N, long sz, at::Tensor &input) {
#if 0
  input.zero_();
#else
  RECORD_FUNCTION("zero_", std::vector<c10::IValue>({input}));
  if (input.dtype() == at::kFloat) {
    DECL_VLA_PTR_PT(float, in, [sz], input);
    auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(sz), EW_ZERO);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for(int n = 0; n < N; n++) {
#ifdef NO_TPP_PT
      xsmm_set_zero(sz, in[n]);
#else
      set_zero_tpp(in[n]);
#endif
    }
  } else {
    DECL_VLA_PTR_PT(bfloat16, in, [sz], input);
    auto set_zero_tpp = SCOPEIT(SetZeroTPP<bfloat16>(sz), EW_ZERO);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for(int n = 0; n < N; n++) {
#ifdef NO_TPP_PT
      xsmm_set_zero(sz, in[n]);
#else
      set_zero_tpp(in[n]);
#endif
    }
  }
#endif
}

std::vector<at::Tensor> fused_self_attention_fwd(float p, std::vector<at::Tensor> inputs, bool training)
{
  if (inputs[6].dtype() == at::kFloat) {
    typedef float T;
#include "fused_self_attention_fwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "fused_self_attention_fwd_tmpl.h"
  }
}

std::vector<at::Tensor> fused_self_attention_bwd(float p, std::vector<at::Tensor> inputs)
{
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_self_attention_bwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "fused_self_attention_bwd_tmpl.h"
  }
}

std::vector<at::Tensor> fused_dense_dropout_layernorm_fwd(float p, float eps, std::vector<at::Tensor> inputs, bool training)
{
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_dense_dropout_layernorm_fwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "fused_dense_dropout_layernorm_fwd_tmpl.h"
  }
}

std::vector<at::Tensor> fused_dense_dropout_layernorm_bwd(float p, std::vector<at::Tensor> inputs)
{
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_dense_dropout_layernorm_bwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "fused_dense_dropout_layernorm_bwd_tmpl.h"
  }
}

std::vector<at::Tensor> fused_dense_gelu_fwd(at::Tensor t_in, at::Tensor t_wt, at::Tensor t_bias, bool training)
{
  if (t_in.dtype() == at::kFloat) {
    typedef float T;
#include "fused_dense_gelu_fwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "fused_dense_gelu_fwd_tmpl.h"
  }
}

std::vector<at::Tensor> fused_dense_gelu_bwd(at::Tensor t_grad_out, at::Tensor t_gelu_in, at::Tensor t_in, at::Tensor t_wt)
{
  if (t_grad_out.dtype() == at::kFloat) {
    typedef float T;
#include "fused_dense_gelu_bwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "fused_dense_gelu_bwd_tmpl.h"
  }
}

std::vector<at::Tensor> fused_embedding_layernorm_dropout_fwd(float p, float eps, long H,long pad_id,  std::vector<at::Tensor> &inputs, bool training)
{
  if (inputs[4].dtype() == at::kFloat && inputs[6].dtype() == at::kFloat) {
    typedef float T;
    typedef float ET;
#include "fused_embedding_layernorm_dropout_fwd_tmpl.h"
  } else if (inputs[4].dtype() == at::kBFloat16 && inputs[6].dtype() == at::kFloat) {
    typedef bfloat16 T;
    typedef float ET;
#include "fused_embedding_layernorm_dropout_fwd_tmpl.h"
  } else if (inputs[4].dtype() == at::kFloat && inputs[6].dtype() == at::kBFloat16) {
    typedef float T;
    typedef bfloat16 ET;
#include "fused_embedding_layernorm_dropout_fwd_tmpl.h"
  } else if (inputs[4].dtype() == at::kBFloat16 && inputs[6].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
    typedef bfloat16 ET;
#include "fused_embedding_layernorm_dropout_fwd_tmpl.h"
  } else {
    PCL_ASSERT(0, "Should not come here\n");
  }
}

std::vector<at::Tensor> fused_embedding_layernorm_dropout_bwd(float p, long pad_id, std::vector<at::Tensor> &inputs)
{
  if (inputs[0].dtype() == at::kFloat && inputs[6].dtype() == at::kFloat) {
    typedef float T;
    typedef float ET;
#include "fused_embedding_layernorm_dropout_bwd_tmpl.h"
  } else if (inputs[0].dtype() == at::kBFloat16 && inputs[6].dtype() == at::kFloat) {
    typedef bfloat16 T;
    typedef float ET;
#include "fused_embedding_layernorm_dropout_bwd_tmpl.h"
  } else if (inputs[0].dtype() == at::kFloat && inputs[6].dtype() == at::kBFloat16) {
    typedef float T;
    typedef bfloat16 ET;
#include "fused_embedding_layernorm_dropout_bwd_tmpl.h"
  } else if (inputs[0].dtype() == at::kBFloat16 && inputs[6].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
    typedef bfloat16 ET;
#include "fused_embedding_layernorm_dropout_bwd_tmpl.h"
  } else {
    PCL_ASSERT(0, "Should not come here\n");
  }
}
#ifdef NO_TPP_ADAM
float norm2(float *ptr, long N)
{
  float sum = 0.0f;
#pragma omp parallel for reduction(+:sum)
  for (long i = 0; i < N; i++) {
    sum += ptr[i] * ptr[i];
  }
  return sum;
}

float norm2(bfloat16 *ptr, long N)
{
#ifdef __AVX512F__
  float sum = 0.0f;
#pragma omp parallel reduction(+:sum)
  {
    int tid = omp_get_thread_num();
    int nThr = omp_get_num_threads();
    long NA = (N + 15) / 16;
    long s = (NA * tid) / nThr;
    long e = (NA * (tid+1)) / nThr;
    s *= 16;
    e *= 16;
    if (e > N) e = N;
    long len = e - s;

    auto lptr = &ptr[s];
    auto v = _mm512_setzero_ps();
    long i = 0;
    for (i = 0; i < ALIGNDOWN(len, 16); i+=16) {
      auto a = _mm512_loadu_ps_auto(&lptr[i]);
      v = _mm512_add_ps(v, _mm512_mul_ps(a, a));
    }
    if (i < len) {
      int rem = len - i;
      __mmask16 mask = (1 << rem) - 1;
      auto a = _mm512_maskz_loadu_ps_auto(mask, &lptr[i]);
      v = _mm512_add_ps(v, _mm512_mul_ps(a, a));
    }
    sum += _mm512_reduce_add_ps(v);
  }
  return sum;
#else
  printf("norm2 not supported for bfloat16 without AVX512 support\n");
  exit(1);
#endif
}
#else
template<typename T>
float norm2(T *ptr, long N)
{
  constexpr int BS = 256;
  auto norm_tpp = SCOPEIT(Norm2TPP<T>(BS), OPTIM);
  float sum = 0.0f;
  long i;
#pragma omp parallel for reduction(+:sum) lastprivate(i)
  for (i = 0; i < ALIGNDOWN(N, BS); i+=BS) {
    norm_tpp(&ptr[i], &sum);
  }
  if (i < N) {
    auto norm_tpp = SCOPEIT(Norm2TPP<T>(N-i), OPTIM);
    norm_tpp(&ptr[i], &sum);
  }
  return sum;
}
#endif

#ifdef NO_TPP_ADAM
template<typename T>
void tensor_scale(T *ptr, long N, float scale)
{
  constexpr int BS = 256;
  long i = 0;
#pragma omp parallel for lastprivate(i)
  for (i = 0; i < ALIGNDOWN(N, BS); i+=BS) {
    xsmm_scale(BS, &ptr[i], &ptr[i], scale);
  }
  if (i < N) {
    xsmm_scale(N-i, &ptr[i], &ptr[i], scale);
  }
}
#else
template<typename T>
void tensor_scale(T *ptr, long N, float scale)
{
  constexpr int BS = 256;
  auto scale_tpp = SCOPEIT((ScaleTPP<T,T>(BS)), EW_SCL);
  long i = 0;
#pragma omp parallel for lastprivate(i)
  for (i = 0; i < ALIGNDOWN(N, BS); i+=BS) {
    scale_tpp(&ptr[i], &ptr[i], scale);
  }
  if (i < N) {
    auto scale_tpp = SCOPEIT((ScaleTPP<T,T>(N-i)), EW_SCL);
    scale_tpp(&ptr[i], &ptr[i], scale);
  }
}
#endif

at::Tensor clip_grad_norm(std::vector<at::Tensor> &grads, float max_norm)
{
  RECORD_FUNCTION("clip_grad_norm", std::vector<c10::IValue>());
  float total_norm = 0.0;
  int N = grads.size();

  for (int i = 0; i < N; i++) {
    if (grads[i].dtype() == at::kFloat) {
      total_norm += norm2(grads[i].data_ptr<float>(), grads[i].numel());
    } else if (grads[i].dtype() == at::kBFloat16) {
      total_norm += norm2(grads[i].data_ptr<bfloat16>(), grads[i].numel());
    } else {
      PCL_ASSERT(0, "Unsupported data type");
    }
  }

  total_norm = sqrtf(total_norm);
  float clip_coef = max_norm / (total_norm + 1e-6);
  if (clip_coef < 1.0) {
    for (int i = 0; i < N; i++) {
      if (grads[i].dtype() == at::kFloat) {
        tensor_scale(grads[i].data_ptr<float>(), grads[i].numel(), clip_coef);
      } else if (grads[i].dtype() == at::kBFloat16) {
        tensor_scale(grads[i].data_ptr<bfloat16>(), grads[i].numel(), clip_coef);
      } else {
        PCL_ASSERT(0, "Unsupported data type");
      }
    }
  }
  //printf("total_norm = %g\n", total_norm);
  return at::tensor(total_norm);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("gelu_fwd", &gelu_fwd, "Pcl BERT forward");
  // m.def("gelu_bwd", &gelu_bwd, "Pcl BERT backward");
  // m.def("dropout_fwd", &dropout_fwd, "Pcl BERT forward");
  // m.def("dropout_bwd", &dropout_bwd, "Pcl BERT forward");
  // m.def("softmax_fwd", &softmax_fwd, "Pcl BERT forward");
  // m.def("softmax_bwd", &softmax_bwd, "Pcl BERT forward");
  // m.def("layer_norm_fwd", &layer_norm_fwd, "Pcl BERT forward");
  // m.def("layer_norm_bwd", &layer_norm_bwd, "Pcl BERT forward");
  m.def("set_rnd_seed", &set_rnd_seed, "Set libxsmm random seed");
  m.def("init_libxsmm", &init_libxsmm, "Initialize libxsmm");
  m.def("print_debug_timers", &print_debug_timers, "print_debug_timers");
  m.def("reset_debug_timers", &reset_debug_timers, "reset_debug_timers");
  m.def("fused_adamw", &fused_adamw, "Fused AdamW optimizer");
  m.def("fused_split_adamw", &fused_split_adamw, "Fused AdamW optimizer for BF16");

  m.def("fused_self_attention_fwd", &fused_self_attention_fwd, "Pcl BERT forward");
  m.def("fused_self_attention_bwd", &fused_self_attention_bwd, "Pcl BERT backward");
  m.def("fused_dense_dropout_layernorm_fwd", &fused_dense_dropout_layernorm_fwd, "Pcl BERT forward");
  m.def("fused_dense_dropout_layernorm_bwd", &fused_dense_dropout_layernorm_bwd, "Pcl BERT forward");
  m.def("fused_dense_gelu_fwd", &fused_dense_gelu_fwd, "Pcl BERT forward");
  m.def("fused_dense_gelu_bwd", &fused_dense_gelu_bwd, "Pcl BERT forward");
  m.def("fused_embedding_layernorm_dropout_fwd", &fused_embedding_layernorm_dropout_fwd, "Pcl BERT forward");
  m.def("fused_embedding_layernorm_dropout_bwd", &fused_embedding_layernorm_dropout_bwd, "Pcl BERT backward");
  m.def("clip_grad_norm", &clip_grad_norm, "Pcl BERT clip_grad_norm");
}
