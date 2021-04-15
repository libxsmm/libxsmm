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
  // B - Batch size
  // S - Max seq len
  // N - Number of attention heads
  // H - Head size
  auto t_Wq = inputs[0];  // [HS][NH] --> [N1][N2][H2][H1]
  auto t_Bq = inputs[1];  // [HS]
  auto t_Wk = inputs[2];  // [HS][NH] --> [N1][N2][H2][H1]
  auto t_Bk = inputs[3];  // [HS]
  auto t_Wv = inputs[4];  // [HS][NH] --> [N1][N2][H2][H1]
  auto t_Bv = inputs[5];  // [HS]
  auto t_HS = inputs[6];  // [B][S][HS] --> [B][S1][N][S2][H]
  auto t_AM = inputs[7]; // Optional [B][S]
  auto t_HM = inputs[8];      // Optional [B][N][S][S]
  auto t_EHS = inputs[9]; // [B][S][HS] --> [B][S1][N][S2][H]
  auto t_EAM = inputs[10]; // Optional [B][S]

  auto sizes = t_HS.sizes();
  long B = sizes[0];
  long S1 = sizes[1];
  long N = sizes[2];
  long S2 = sizes[3];
  long H = sizes[4];
  //long NH = N*H;
  float one_by_sqrt_H = 1.0/sqrt(H);
  bool null_EHS = false;
  bool dt_bf16 = (t_HS.dtype() == at::kBFloat16);
  bool bf16_training = (training && dt_bf16);
  auto t_EHS_orig = t_EHS;

  //std::cout << "B: " << B << " S1: " << S1 << " S2: " << S2 << " N: " << N << " H: " << H << std::endl;
  if(t_EHS.numel() == 0) {
    null_EHS = true;
    t_EHS = t_HS;
  } else {
    t_AM = t_EAM;
  }

  //#define PRINT_T(x) std::cout << #x << ": " << x << std::endl
  auto t_HS_T = t_HS;
  auto t_EHS_T = t_EHS;

  auto t_Wq_V = wt_tensor_for_fwd(N, H, N, H, t_Wq);
  auto t_Wk_V = wt_tensor_for_fwd(N, H, N, H, t_Wk);
  auto t_Wv_V = wt_tensor_for_fwd(N, H, N, H, t_Wv);

  auto t_QL = t_HS.new_empty({B, S1, N, S2, H});
  auto t_QL_T = t_QL;
  auto t_KL_TV = t_EHS.new_empty({B, S1, N, H, S2});
  if (dt_bf16) t_KL_TV = t_KL_TV.view({B, S1, N, H/2, S2, 2});
  auto t_KL_V = t_KL_TV;
  auto t_VL_V = t_EHS.new_empty({B, S1, N, S2, H});
  if (dt_bf16) t_VL_V = t_VL_V.view({B, S1, N, S2/2, H, 2});
  auto t_VL_TV = t_VL_V;
  auto t_AP = t_QL.new_empty({B, S1, N, S1, S2, S2});
  auto t_CL = t_AP.new_empty({B, S1, N, S2, H});

  auto t_APD = t_AP;
  auto t_APD_mask = at::empty({B, S1, N, (S1*S2*S2+15)/16}, at::kShort);
  if (p > 0 || t_HM.numel() != 0) {
    t_APD = at::empty_like(t_AP);
  }

  auto t_APD_T = t_APD;

  if (bf16_training) {
    t_HS_T = t_HS.new_empty({B, S1, N, H, S2}); // For BWD only
    t_EHS_T = null_EHS ? t_HS_T : t_HS.new_empty({B, S1, N, H, S2}); // For BWD only

    t_QL_T = t_HS.new_empty({B, S1, N, H, S2}); // For BWD only
    t_APD_T = t_QL.new_empty({B, S1, N, S1, S2, S2}); // For BWD only
  }
  if (training) {
    if (dt_bf16) {
      t_KL_V = t_EHS.new_empty({B, S1, N, S2/2, H, 2}); // Saved For BWD
      t_VL_TV = t_EHS.new_empty({B, S1, N, H/2, S2, 2}); // For BWD only
    } else {
      t_KL_V = t_EHS.new_empty({B, S1, N, S2, H}); // Saved For BWD
      t_VL_TV = t_EHS.new_empty({B, S1, N, H, S2}); // For BWD only
    }
  }

  {
    //float (*QL)[S1][N][S2][H] = (float (*)[S1][N][S2][H])t_QL.data_ptr<float>();
    DECL_VLA_PTR_PT(T, Wq_V, [N][H*H], t_Wq_V);
    DECL_VLA_PTR_PT(T, Wk_V, [N][H*H], t_Wk_V);
    DECL_VLA_PTR_PT(T, Wv_V, [N][H*H], t_Wv_V);
    DECL_VLA_PTR_PT(T, Bq, [H], t_Bq);
    DECL_VLA_PTR_PT(T, Bk, [H], t_Bk);
    DECL_VLA_PTR_PT(T, Bv, [H], t_Bv);
    DECL_VLA_PTR_PT(T, QL, [S1][N][S2*H], t_QL);
    DECL_VLA_PTR_PT(T, QL_T, [S1][N][H*S2], t_QL_T); // For BWD only
    DECL_VLA_PTR_PT(T, KL_V, [S1][N][S2*H], t_KL_V);
    DECL_VLA_PTR_PT(T, KL_TV, [S1][N][H*S2], t_KL_TV);
    DECL_VLA_PTR_PT(T, VL_V, [S1][N][S2*H], t_VL_V);
    DECL_VLA_PTR_PT(T, VL_TV, [S1][N][H*S2], t_VL_TV);
    DECL_VLA_PTR_PT(T, AP, [S1][N][S1][S2*S2], t_AP);
    DECL_VLA_PTR_PT(T, APD, [S1][N][S1][S2*S2], t_APD);
    DECL_VLA_PTR_PT(T, APD_T, [S1][N][S1][S2*S2], t_APD_T); // For BWD only
    DECL_VLA_PTR_PT(short, APD_mask, [S1][N][(S1*S2*S2+15)/16], t_APD_mask);
    DECL_VLA_PTR_PT(T, CL, [S1][N][S2*H], t_CL);
    DECL_VLA_PTR_PT(T, HS, [S1][N][S2*H], t_HS);
    DECL_VLA_PTR_PT(T, HS_T, [S1][N][H*S2], t_HS_T); // for BWD only
    DECL_VLA_PTR_PT(T, EHS, [S1][N][S2*H], t_EHS);
    DECL_VLA_PTR_PT(T, EHS_T, [S1][N][H*S2], t_EHS_T); // for BWD only
    DECL_VLA_PTR_PT(T, AM, [S1][S2], t_AM);

    auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(S2, H), BIAS);
    auto qkv_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T,T>(S2, H, H, S2*H, H*H, 1.0)), BRGEMM, S2*H*H);
    auto xpose_tpp = SCOPEIT(XformExtTPP<T>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    auto k_xpose_tpp_1 = SCOPEIT(XformExtTPP<T>(S2, H, training ? XformTPP::XFORM_N2V_TPP : XformTPP::XFORM_XPOSE_N2V_TPP), XPOSE);
    auto kv_xpose_tpp_2 = SCOPEIT(XformExtTPP<T>(S2, H, XformTPP::XFORM_XPOSE_N2V_TPP), VNNI);
    auto v_xpose_tpp_1 = SCOPEIT(XformExtTPP<T>(S2, H, XformTPP::XFORM_N2V_TPP), VNNI);
    auto a_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T,float>(S2, S2, H, S2*H, H*S2, 0.0)), BRGEMM, S2*S2*H);
    auto scale_tpp = SCOPEIT((ScaleTPP<float,float>(S2*S2)), EW_SCL);
    auto add_mask_tpp = SCOPEIT(AddBiasTPP<T>(S2, S2), EW_ADD);
    auto softmax_fwd_tpp = SCOPEIT((SoftMaxFwdTPP<float,T>(S1, S2, S2)), SOFTMAX);
    auto dropout_fwd_tpp = SCOPEIT(DropOutFwdTPP<T>(S1*S2*S2, p), DROPOUT);
    auto a_xpose_tpp = SCOPEIT(XformExtTPP<T>(S2, S2, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    auto c_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T,T>(S2, H, S2, S2*S2, N*S2*H, 0.0)), BRGEMM, S2*H*S2);

    {
      RECORD_SCOPE(q_gemm, {t_HS, t_Wq_V});
      {
        RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse (3)
        for(int b = 0; b < B; b++) {
          for(int s1 = 0; s1 < S1; s1++) {
            for(int nk = 0; nk < N; nk++) {
#ifdef NO_TPP_AF
              if (bf16_training && nk == 0) xsmm_xform_ext(N, S2, H, S2*H, S2*H, XFORM_XPOSE, HS[b][s1][0], HS_T[b][s1][0]);
              xsmm_cpy_bias(S2, H, Bq[nk], QL[b][s1][nk]);
              brgemm(S2, H, H, S2*H, H*H, HS[b][s1][0], Wq_V[nk][0], QL[b][s1][nk], N, 1.0, XFORM_NONE);
              if (bf16_training) xsmm_xform_input(1, S2, H, H*S2, H*S2, XFORM_XPOSE, QL[b][s1][nk], QL_T[b][s1][nk]);
#else
              if (bf16_training && nk == 0) xpose_tpp(N, S2*H, S2*H, HS[b][s1][0], HS_T[b][s1][0]);
              copy_bias_tpp(Bq[nk], QL[b][s1][nk]);
              qkv_gemm_tpp(HS[b][s1][0], Wq_V[nk][0], QL[b][s1][nk], N);
              if (bf16_training) xpose_tpp(QL[b][s1][nk], QL_T[b][s1][nk]);
#endif
            }
          }
        }
      }
    }

    //PRINT_T(t_QL.permute({0,1,3,2,4}).contiguous().view({B,S1*S2,N*H}));

    {
      RECORD_SCOPE(k_gemm, {t_EHS, t_Wk_V});
      {
        RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse (3)
        for(int b = 0; b < B; b++) {
          for(int s1 = 0; s1 < S1; s1++) {
            for(int nk = 0; nk < N; nk++) {
              T tmp[S2*H];
              T *tmpp = (training && !bf16_training) ? KL_V[b][s1][nk] : tmp;
#ifdef NO_TPP_AF
              if (!null_EHS && bf16_training && nk == 0) xsmm_xform_ext(N, S2, H, S2*H, S2*H, XFORM_XPOSE, EHS[b][s1][0], EHS_T[b][s1][0]);
              xsmm_cpy_bias(S2, H, Bk[nk], tmpp);
              brgemm(S2, H, H, S2*H, H*H, EHS[b][s1][0], Wk_V[nk][0], tmpp, N, 1.0, XFORM_NONE);
              xsmm_xform_input(1, S2, H, H*S2, H*S2, training ? XFORM_N2V : XFORM_XPOSE_N2V, tmpp, KL_V[b][s1][nk]); // KL_V = KL_VT if not training
              if (training) xsmm_xform_input(1, S2, H, H*S2, H*S2, XFORM_XPOSE_N2V, tmpp, KL_TV[b][s1][nk]);
#else
              if (!null_EHS && bf16_training && nk == 0) xpose_tpp(N, S2*H, S2*H, EHS[b][s1][0], EHS_T[b][s1][0]);
              copy_bias_tpp(Bk[nk], tmpp);
              qkv_gemm_tpp(EHS[b][s1][0], Wk_V[nk][0], tmpp, N);
              k_xpose_tpp_1(tmpp, KL_V[b][s1][nk]); // KL_V = KL_VT if not training
              if (training) kv_xpose_tpp_2(tmpp, KL_TV[b][s1][nk]);
#endif
            }
          }
        }
      }
    }
    //PRINT_T(t_EHS);
    //PRINT_T(t_Wk_V.permute({0,1,2,4,3}).contiguous().view({N,N,H,H}));
    //PRINT_T(t_Wk_V);
    //PRINT_T(t_Bk);
    //PRINT_T(t_KL_V.permute({0,1,3,5,2,4}).contiguous().view({B,S1*S2,N*H}));

    {
      RECORD_SCOPE(v_gemm, {t_EHS, t_Wv_V});
      {
        RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse (3)
        for(int b = 0; b < B; b++) {
          for(int s1 = 0; s1 < S1; s1++) {
            for(int nk = 0; nk < N; nk++) {
              T tmp[S2*H];
              T *tmpp = (!dt_bf16) ? VL_V[b][s1][nk] : tmp;
#ifdef NO_TPP_AF
              xsmm_cpy_bias(S2, H, Bv[nk], tmpp);
              brgemm(S2, H, H, S2*H, H*H, EHS[b][s1][0], Wv_V[nk][0], tmpp, N, 1.0, XFORM_NONE);
              xsmm_xform_input(1, S2, H, H*S2, H*S2, XFORM_N2V, tmpp, VL_V[b][s1][nk]);
              if (training) xsmm_xform_input(1, S2, H, H*S2, H*S2, XFORM_XPOSE_N2V, tmpp, VL_TV[b][s1][nk]);
#else
              copy_bias_tpp(Bv[nk], tmpp);
              qkv_gemm_tpp(EHS[b][s1][0], Wv_V[nk][0], tmpp, N);
              v_xpose_tpp_1(tmpp, VL_V[b][s1][nk]);
              if (training) kv_xpose_tpp_2(tmpp, VL_TV[b][s1][nk]);
#endif
            }
          }
        }
      }
    }
    // Take the dot product between "query" and "key" to get the raw attention scores.
    {
      RECORD_SCOPE(a_gemm, {t_QL, t_KL_TV});
      {
        RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse (3)
        for (int b = 0; b < B; b++) {
          for (int n = 0; n < N; n++) {
            for (int s11 = 0; s11 < S1; s11++) {
              float AS[S1][S2][S2];
              for (int s21 = 0; s21 < S1; s21++) {
#ifdef NO_TPP_AF
                brgemm(S2, S2, H, S2*H, H*S2, QL[b][s11][n], KL_TV[b][s21][n], AS[s21][0], 1, 0.0, XFORM_NONE);
                xsmm_scale(S2*S2, AS[s21][0], AS[s21][0], one_by_sqrt_H);
                if (t_AM.numel() != 0) xsmm_add_bias(S2, S2, AM[b][s21], AS[s21][0]);
#else
                a_gemm_tpp(QL[b][s11][n], KL_TV[b][s21][n], AS[s21][0], 1);
                scale_tpp(AS[s21][0], AS[s21][0], one_by_sqrt_H);
                if (t_AM.numel() != 0) add_mask_tpp(AM[b][s21], AS[s21][0]);
#endif
              }
#ifdef NO_TPP_AF
              pcl_softmax_fwd(S1, S2, S2, AS[0][0], AP[b][s11][n][0]);
#else
              softmax_fwd_tpp(AS[0][0], AP[b][s11][n][0]);
#endif
              if(p > 0) {
#ifdef NO_TPP_AF
                pcl_dropout_fwd(S1*S2*S2, AP[b][s11][n][0], APD[b][s11][n][0], APD_mask[b][s11][n], p);
#else
                dropout_fwd_tpp(AP[b][s11][n][0], rnd_state, APD[b][s11][n][0], APD_mask[b][s11][n]);
#endif
              }
              if (t_HM.numel() != 0) {
                // FIXME: shape of head mask is not correct here yet
                PCL_ASSERT(0, "t_HM used");
                //t_APD[b][s11][n] *= t_HM[b][s11][n];
              }
#ifdef NO_TPP_AF
              if (bf16_training) xsmm_xform_input(S1, S2, S2, S2*S2, S2*S2, XFORM_XPOSE, APD[b][s11][n][0], APD_T[b][s11][n][0]);
#else
              if (bf16_training) a_xpose_tpp(S1, S2*S2, S2*S2, APD[b][s11][n][0], APD_T[b][s11][n][0]);
#endif
            }
          }
        }
      }
    }

    {
      RECORD_SCOPE(c_gemm, {t_APD, t_VL_V});
      {
        RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse (3)
        for (int b = 0; b < B; b++) {
          for (int n = 0; n < N; n++) {
            for (int s11 = 0; s11 < S1; s11++) {
#ifdef NO_TPP_AF
              brgemm(S2, H, S2, S2*S2, N*S2*H, APD[b][s11][n][0], VL_V[b][0][n], CL[b][s11][n], S1, 0.0, XFORM_NONE);
#else
              c_gemm_tpp(APD[b][s11][n][0], VL_V[b][0][n], CL[b][s11][n], S1);
#endif
            }
          }
        }
      }
    }
  }
  //auto t_APO = t_APD.permute({0, 2, 1, 4, 3, 5}).contiguous().view({B, N, S, S});
  auto t_APO = t_APD;
  return std::vector<at::Tensor>({t_CL, t_APO,
      t_HS_T, null_EHS ? t_EHS_orig : t_EHS_T, t_QL_T, t_KL_V, t_VL_TV, t_AP, t_APD_T, t_APD_mask});
