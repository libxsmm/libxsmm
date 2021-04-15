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

  RECORD_FUNCTION("bert_bwd", std::vector<c10::IValue>());
  globalPass = BWD;
  MasterScopedTimer _mt(globalPass);
  auto in_sizes = t_in.sizes();
  auto wt_sizes = t_wt.sizes();
  auto B = in_sizes[0];
  auto S1 = in_sizes[1];
  auto Nc = in_sizes[2];
  auto S2 = in_sizes[3];
  auto Hc = in_sizes[4];

  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];

  const int grad_wt_flag = (t_wt.dim() == 5 ? XFORM_N2V : XFORM_NONE);
  const int input_trans_flag = (t_in.dtype() == at::kFloat ? XFORM_XPOSE : XFORM_NONE);
  auto t_wt_TV = wt_tensor_for_bwd(Nk, Hk, Nc, Hc, t_wt);

  auto t_in_T = t_in;
  if (input_trans_flag == XFORM_NONE) {
    t_in_T = act_tensor_trans(B, S1, Nc, S2, Hc, t_in);
  }

  auto t_grad_in   = at::empty_like(t_in);
  auto t_grad_gelu = at::empty_like(t_grad_out);
  auto t_grad_wt   = at::empty_like(t_wt);
  auto t_grad_bias  = t_wt.new_empty({Nk*Hk}); // [Nk][Hk]
  auto t_grad_gelu_V = t_grad_gelu;
  if (t_grad_gelu.dtype() == at::kBFloat16) {
    t_grad_gelu_V = t_grad_out.new_empty({B, S1, Nk, S2/2, Hk, 2});
  }

  DECL_VLA_PTR_PT(T, in_T, [S1][Nc][Hc*S2], t_in_T);
  DECL_VLA_PTR_PT(T, gelu_in, [S1][Nk][S2*Hk], t_gelu_in);
  DECL_VLA_PTR_PT(T, grad_in, [S1][Nc][S2*Hc], t_grad_in);
  DECL_VLA_PTR_PT(T, wt_TV, [Nc][Hk*Hc], t_wt_TV);
  DECL_VLA_PTR_PT(T, grad_wt, [Nc][Hc*Hk], t_grad_wt);
  DECL_VLA_PTR_PT(T, grad_bias, [Hk], t_grad_bias);
  DECL_VLA_PTR_PT(T, grad_gelu, [S1][Nk][S2*Hk], t_grad_gelu);
  DECL_VLA_PTR_PT(T, grad_out, [S1][Nk][S2*Hk], t_grad_out);
  DECL_VLA_PTR_PT(T, grad_gelu_V, [S1][Nk][S2*Hk], t_grad_gelu_V);
  auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(Nk*Hk), EW_ZERO);
  auto gelu_bwd_tpp = SCOPEIT(GeluBwdTPP<T>(S2*Hk), GELU);
  auto grad_bias_tpp = SCOPEIT(GradBiasTPP<T>(S2, Hk), BIAS);
  auto n2v_tpp = SCOPEIT(XformExtTPP<T>(S2, Hk, XformTPP::XFORM_N2V_TPP), VNNI);
  auto di_gemm_b0_tpp = SCOPEITGEMM((BrgemmExtTPP<T,T>(S2, Hc, Hk, S2*Hk, Nc*Hk*Hc, 0.0)), BRGEMM, S2*Hc*Hk);
  auto di_gemm_b1_tpp = SCOPEITGEMM((BrgemmExtTPP<T,T>(S2, Hc, Hk, S2*Hk, Nc*Hk*Hc, 1.0)), BRGEMM, S2*Hc*Hk);
  auto dw_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T,T>(Hc, Hk, S2, Nc*S2*Hc, Nk*S2*Hk, 1.0, (XformTPP::XFORM_TYPE)grad_wt_flag, input_trans_flag)), BRGEMM, Hc*Hk*S2);
  {
    RECORD_SCOPE(di_bias, {t_grad_out});
    //t_grad_bias.zero_();
    tensor_set_zero(Nk, Hk, t_grad_bias);
    int num_threads = omp_get_max_threads();
    float *bias_ptrs[num_threads];
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        float prv_grad_bias[Nk][Hk];
        bias_ptrs[tid] = prv_grad_bias[0];
#ifdef NO_TPP_IB
        xsmm_set_zero(Nk*Hk, prv_grad_bias[0]);
#else
        set_zero_tpp(prv_grad_bias[0]);
#endif
#pragma omp for collapse (3) //reduction(+:grad_bias[:Nk][:Hk])
        for(int b = 0; b < B; b++) {
          for(int s1 = 0; s1 < S1; s1++) {
            for(int nk = 0; nk < Nk; nk++) {
#ifdef NO_TPP_IB
              xsmm_gelu_bwd(S2*Hk, grad_out[b][s1][nk], gelu_in[b][s1][nk], grad_gelu[b][s1][nk]);
              xsmm_grad_bias(S2, Hk, grad_gelu[b][s1][nk], prv_grad_bias[nk]);
              xsmm_n2v(S2, Hk, grad_gelu[b][s1][nk], grad_gelu_V[b][s1][nk]);
#else
              gelu_bwd_tpp(grad_out[b][s1][nk], gelu_in[b][s1][nk], grad_gelu[b][s1][nk]);
              grad_bias_tpp(grad_gelu[b][s1][nk], prv_grad_bias[nk]);
              n2v_tpp(grad_gelu[b][s1][nk], grad_gelu_V[b][s1][nk]);
#endif
            }
          }
        }
#pragma omp barrier
        xsmm_reduce_buf(num_threads, Nk*Hk, bias_ptrs, grad_bias[0]);
      }
    }
  }
  {
    RECORD_SCOPE(dii_gemm, {t_grad_gelu, t_wt_TV});
    auto Nkb = Nk;
    if (Nk > Nc && Nk % Nc == 0) {
      Nkb = Nc;
    }

    //if(Nk != Nkb) t_grad_in.zero_();
    if(Nk != Nkb) tensor_set_zero(B*S1*Nc, S2*Hc, t_grad_in);
    for(int nk = 0; nk < Nk; nk+=Nkb) {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse (3)
      for(int b = 0; b < B; b++) {
        for(int s1 = 0; s1 < S1; s1++) {
          for(int nc = 0; nc < Nc; nc++) {
#ifdef NO_TPP_IB
            brgemm(S2, Hc, Hk, S2*Hk, Nc*Hk*Hc, grad_gelu[b][s1][nk], wt_TV[nk][nc], grad_in[b][s1][nc], Nkb, (Nk != Nkb ? 1.0 : 0.0));
#else
            if (Nk != Nkb)
              di_gemm_b1_tpp(grad_gelu[b][s1][nk], wt_TV[nk][nc], grad_in[b][s1][nc], Nkb);
            else
              di_gemm_b0_tpp(grad_gelu[b][s1][nk], wt_TV[nk][nc], grad_in[b][s1][nc], Nkb);
#endif
          }
        }
      }
    }
  }
  {
    RECORD_SCOPE(dwi_gemm, {t_in_T, t_grad_gelu_V});
    //t_grad_wt.zero_();
    tensor_set_zero(Nk*Nc, Hk*Hc, t_grad_wt);
    for(int b = 0; b < B; b++) {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse (2)
      for(int nk = 0; nk < Nk; nk++) {
        for (int nc = 0; nc < Nc; nc++) {
#ifdef NO_TPP_IB
          brgemm(Hc, Hk, S2, Nc*S2*Hc, Nk*S2*Hk, in_T[b][0][nc], grad_gelu_V[b][0][nk], grad_wt[nk][nc], S1, 1.0, grad_wt_flag, input_trans_flag);
#else
          dw_gemm_tpp(in_T[b][0][nc], grad_gelu_V[b][0][nk], grad_wt[nk][nc], S1);
#endif
        }
      }
    }
  }
  return std::vector<at::Tensor>({t_grad_in, t_grad_wt, t_grad_bias});

