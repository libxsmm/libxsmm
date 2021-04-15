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

  RECORD_FUNCTION("bert_fwd", std::vector<c10::IValue>());
  globalPass = FWD;
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

  auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto t_gelu_out = t_in.new_empty({B,S1,Nk,S2,Hk});
  auto t_out = t_gelu_out;
  if (training) {
    t_out = t_in.new_empty({B,S1,Nk,S2,Hk});
  }

  DECL_VLA_PTR_PT(T, in, [S1][Nc][S2*Hc], t_in);
  DECL_VLA_PTR_PT(T, wt_V, [Nc][Hc*Hk], t_wt_V);
  DECL_VLA_PTR_PT(T, bias, [Hk], t_bias);
  DECL_VLA_PTR_PT(T, out, [S1][Nk][S2*Hk], t_out);
  DECL_VLA_PTR_PT(T, gelu_out, [S1][Nk][S2*Hk], t_gelu_out);

  // Create TPPs
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(S2, Hk), BIAS);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T,T>(S2, Hk, Hc, S2*Hc, Hk*Hc)), BRGEMM, S2*Hk*Hc);
  auto gelu_fwd_tpp = SCOPEIT(GeluFwdTPP<T>(S2*Hk), GELU);

  auto Ncb = Nc;
  if (Nc > Nk && Nc % Nk == 0) {
    Ncb = Nk;
  }

  {
    RECORD_SCOPE(i_gemm, {t_in, t_wt_V});
    for(int nc = 0; nc < Nc; nc+=Ncb) {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse (3)
      for(int b = 0; b < B; b++) {
        for(int s1 = 0; s1 < S1; s1++) {
          for(int nk = 0; nk < Nk; nk++) {
            if (nc == 0) {
#ifdef NO_TPP_IF
              xsmm_cpy_bias(S2, Hk, bias[nk], out[b][s1][nk]);
#else
              copy_bias_tpp(bias[nk], out[b][s1][nk]);
#endif
            }
#ifdef NO_TPP_IF
            brgemm(S2, Hk, Hc, S2*Hc, Hk*Hc, in[b][s1][nc], wt_V[nk][nc], out[b][s1][nk], Ncb);
#else
            brgemm_tpp(in[b][s1][nc], wt_V[nk][nc], out[b][s1][nk], Ncb);
#endif
            if (nc == Nc - Ncb) { // last iter
#ifdef NO_TPP_IF
              xsmm_gelu_fwd(S2*Hk, out[b][s1][nk], gelu_out[b][s1][nk]);
#else
              gelu_fwd_tpp(out[b][s1][nk], gelu_out[b][s1][nk]);
#endif
            }
          }
        }
      }
    }
  }
  //if (at::isnan(t_out).any().item<bool>()) std::cout << "t_out has NaN" << std::endl;
  //if (at::isnan(t_gelu_out).any().item<bool>()) std::cout << "t_gelu_out has NaN" << std::endl;
  return std::vector<at::Tensor>({t_out, t_gelu_out});
