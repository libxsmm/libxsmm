/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
void my_cnn_bwd_exec( my_cnn_config cfg, const float* wt_ptr, const float* tr_wt_ptr,  const float* dout_act_ptr, float* din_act_ptr,
    unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch ) {

  if (cfg.use_fallback_bwd_loops == 0) {
    int img, ofm1, ofm2, ifm1, ifm2, oj, oi, kj, ki, oi_use, oj_use, ii_use, ij_use, ofmb, ifmb, ojb, myIfmId, nIfmBlocks, ind, task, ifm1ofm1;
    /* computing first logical thread */
    const int ltid = my_tid - start_tid;
    int imgpt = LIBXSMM_UPDIV(cfg.N, cfg.threads);
    int threads_per_image = cfg.threads / cfg.N;
    int my_img_start = LIBXSMM_MIN(ltid * imgpt, cfg.N);
    int my_img_end = LIBXSMM_MIN((ltid+1) * imgpt, cfg.N);
    int my_ifm_start = 0;
    int my_ifm_end = cfg.blocksifm;

    /* Batch reduce related variables */
    const float *A_ptrs[1024];
    const float  *B_ptrs[1024];
    unsigned long long n_blocks;

    /* number of tasks for transpose that could be run in parallel */
    int transpose_work = cfg.blocksifm * cfg.blocksofm * cfg.R * cfg.S;
    /* compute chunk size */
    int transpose_chunksize = (transpose_work % cfg.threads == 0) ? (transpose_work / cfg.threads) : ((transpose_work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
    int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;
    /* offset output pointer in case of physical  padding */
    const int IFW = (cfg.pack_input_bwd == 1) ? cfg.ofw : cfg.ifwp;
    const int IFH = (cfg.pack_input_bwd == 1) ? cfg.ofh : cfg.ifhp;
    float *input_ptr = (cfg.pack_input_bwd == 1) ? (float*)((char*)scratch + cfg.bwd_packing_padding_scratch_offset) : (float*)din_act_ptr + ((size_t)cfg.pad_h_in * cfg.ifwp + cfg.pad_w_in) * cfg.ifmblock;
    LIBXSMM_VLA_DECL(5, float, del_input, input_ptr, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
    float *const out = (float*)dout_act_ptr;
    LIBXSMM_VLA_DECL(5, const float, output, out, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);

    /* Weight and transpose_weight tensor declaration */
    LIBXSMM_VLA_DECL(6, float, wt, (float*)wt_ptr, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
    LIBXSMM_VLA_DECL(6, float, tr_wt, (float*)((char*)scratch + cfg.bwd_filter_trans_scratch_offset), cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
    /* define weight pointer which has the correct format */
    float* weight_base = ((cfg.options & LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE) > 0 ) ? (float*)tr_wt_ptr : (float*)((char*)scratch + cfg.bwd_filter_trans_scratch_offset);
    LIBXSMM_VLA_DECL(6, const float, weight, weight_base, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);

    /* lazy barrier init */
    libxsmm_barrier_init(cfg.barrier, ltid);

    /* transpose filters, if requested */
    if ( (cfg.options & LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE) == 0 ) {
      /* Special case of 64x64 transpose with JITed transpose */
      if (cfg.ifmblock == 64 && cfg.ofmblock == 64) {
        libxsmm_meltwfunction_unary tr_kernel = cfg.tr_kernel;
        libxsmm_meltw_unary_param trans_param;
        for (task = transpose_thr_begin; task < transpose_thr_end; ++task) {
          ifm1 = task/(cfg.blocksofm * cfg.R * cfg.S);
          ofm1 = (task%(cfg.blocksofm * cfg.R * cfg.S))/(cfg.R * cfg.S);
          kj =   ((task%(cfg.blocksofm * cfg.R * cfg.S))%(cfg.R * cfg.S))/cfg.S;
          ki =   ((task%(cfg.blocksofm * cfg.R * cfg.S))%(cfg.R * cfg.S))%cfg.S;
          trans_param.in.primary  = &LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
          trans_param.out.primary = &LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, cfg.R-1-kj, cfg.S-1-ki, 0, 0, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
          tr_kernel( &trans_param );
          trans_param.in.primary  = &LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, 16, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
          trans_param.out.primary = &LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, cfg.R-1-kj, cfg.S-1-ki, 0, 16, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
          tr_kernel( &trans_param );
          trans_param.in.primary  = &LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, 32, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
          trans_param.out.primary = &LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, cfg.R-1-kj, cfg.S-1-ki, 0, 32, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
          tr_kernel( &trans_param );
          trans_param.in.primary  = &LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, 48, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
          trans_param.out.primary = &LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, cfg.R-1-kj, cfg.S-1-ki, 0, 48, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
          tr_kernel( &trans_param );
        }
      } else {
        /* number of tasks for transpose that could be run in parallel */
        transpose_work = cfg.blocksifm * cfg.blocksofm;
        /* compute chunk size */
        transpose_chunksize = (transpose_work % cfg.threads == 0) ? (transpose_work / cfg.threads) : ((transpose_work / cfg.threads) + 1);
        /* compute thr_begin and thr_end */
        transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
        transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;
        for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
          ofm1 = ifm1ofm1 / cfg.blocksifm;
          ifm1 = ifm1ofm1 % cfg.blocksifm;
          for (kj=0; kj < cfg.R; kj++) {
            for (ki=0; ki < cfg.S; ki++) {
              for (ofm2 = 0; ofm2 < cfg.ofmblock; ++ofm2) {
                for (ifm2 = 0; ifm2 < cfg.ifmblock; ++ifm2) {
                  LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, cfg.R-1-kj , cfg.S-1-ki, ofm2, ifm2, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock) =
                    LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                }
              }
            }
          }
        }
      }
      /* wait for transpose to finish */
      libxsmm_barrier_wait(cfg.barrier, ltid);
    }

    if ( imgpt <= 1 ) {
      my_img_start = LIBXSMM_MIN(ltid / threads_per_image, cfg.N);
      my_img_end = LIBXSMM_MIN(my_img_start + 1, cfg.N);
      myIfmId = ltid % threads_per_image;
      nIfmBlocks = LIBXSMM_UPDIV(cfg.blocksifm, threads_per_image);
      my_ifm_start = LIBXSMM_MIN(myIfmId * nIfmBlocks, cfg.blocksifm);
      my_ifm_end = LIBXSMM_MIN((myIfmId+1) * nIfmBlocks, cfg.blocksifm);
    }

    if ( cfg.use_ifm_parallelization == 1 ) {
      int spread_out = 0;
      if ( cfg.N % 8 == 0) {
        spread_out = 8;
      } else if ( cfg.N % 4 == 0) {
        spread_out = 4;
      } else if (cfg.N % 3 == 0) {
        spread_out = 3;
      } else if (cfg.N % 2 == 0) {
        spread_out = 2;
      } else {
        spread_out = 1;
      }
      if ((spread_out > 1) && (cfg.threads % spread_out == 0)) {
        int tile_id = ltid / spread_out;
        int ifmpt = LIBXSMM_UPDIV(cfg.blocksifm, spread_out);
        int ifm_id = ltid % spread_out;
        imgpt = LIBXSMM_UPDIV(cfg.N, cfg.threads) * spread_out;
        my_img_start = LIBXSMM_MIN(tile_id * imgpt, cfg.N);
        my_img_end = LIBXSMM_MIN((tile_id+1) * imgpt, cfg.N);
        my_ifm_start = LIBXSMM_MIN(ifm_id * ifmpt, cfg.blocksifm);
        my_ifm_end = LIBXSMM_MIN((ifm_id+1) * ifmpt, cfg.blocksifm);
      }
    }

    if (cfg.loop_order == 0) { /* (loop_order == N_Kb_Cb_Hb_k_c_h_w) {*/
      if ( cfg.avoid_fmas_in_rim == 1) {
        for (img = my_img_start; img < my_img_end; img++) {
          for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += cfg.block_bwd_ifm) {
            for (ofmb = 0; ofmb < cfg.blocksofm; ofmb += cfg.block_bwd_ofm) {
              for (ojb = 0; ojb < cfg.ofh; ojb += cfg.block_bwd_oj) {
                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+cfg.block_bwd_ifm, my_ifm_end); ifm1++ ) {

                  if ( (ofmb == 0) && ((cfg.options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && cfg.avoid_acc_load_bwd == 0 && ojb == 0) {
                    /* set output feature map to zero */
                    for (oj = 0; oj < cfg.ofh; ++oj) {
                      float* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, del_input, img, ifm1, oj, 0, 0,  cfg.blocksifm, IFH, IFW, cfg.ifmblock));
                      for (oi = 0; oi < cfg.ofw; ++oi) {
                        LIBXSMM_PRAGMA_SIMD
                          for (ifm2 = 0; ifm2 < cfg.ifmblock; ++ifm2) {
                            temp_ptr[ifm2] = (float)0;
                          }
                        temp_ptr += cfg.ifmblock;
                      }
                    }
                  }

                  for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+cfg.block_bwd_ofm, cfg.blocksofm); ofm1 += cfg.blocksofm_blocking) {
                    for (oj = ojb; oj < LIBXSMM_MIN(ojb+cfg.block_bwd_oj,cfg.ofh); oj += cfg.bwd_ofh_rb) {
                      for (oi = 0; oi < cfg.ofw; oi += cfg.bwd_ofw_rb) {
                        for (kj = 0; kj < cfg.R; kj++) {
                          for (ki = 0; ki < cfg.S; ki++) {
                            /* Prepare batch-reduce kernel arguments */
                            ij_use = oj;
                            ii_use = oi;
                            oj_use = oj - (1-cfg.pad_h_out);
                            oi_use = oi - (1-cfg.pad_w_out);

                            if (kj == 0 && oj == 0) {
                              /* Do no FLOPS  */
                            } else if (kj == cfg.R-1 && oj == cfg.ofh-1 ) {
                              /* Do no FLOPS  */
                            } else if ( oi == 0 && ki == 0 ) {
                              ind = 0;
                              for (ofm2 = ofm1; ofm2 < ofm1 + cfg.blocksofm_blocking; ofm2++) {
                                A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm2, kj, ki, 0, 0, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
                                B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  output,  img, ofm2, oj_use + kj, oi_use + ki + 1, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                                ind++;
                              }
                              n_blocks = ind;
                              br_gemm_kernel2(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use + 1, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock), &n_blocks);
                            } else if (oi == cfg.ofw-cfg.bwd_ofw_rb  && ki == cfg.S-1) {
                              ind = 0;
                              for (ofm2 = ofm1; ofm2 < ofm1 + cfg.blocksofm_blocking; ofm2++) {
                                A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm2, kj, ki, 0, 0, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
                                B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  output,  img, ofm2, oj_use + kj, oi_use + ki, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                                ind++;
                              }
                              n_blocks = ind;
                              br_gemm_kernel2(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock), &n_blocks);
                            } else {
                              ind = 0;
                              for (ofm2 = ofm1; ofm2 < ofm1 + cfg.blocksofm_blocking; ofm2++) {
                                A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm2, kj, ki, 0, 0, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
                                B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  output,  img, ofm2, oj_use + kj, oi_use + ki, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                                ind++;
                              }
                              n_blocks = ind;
                              br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock), &n_blocks);
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        for (img = my_img_start; img < my_img_end; img++) {
          for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += cfg.block_bwd_ifm) {
            for (ofmb = 0; ofmb < cfg.blocksofm; ofmb += cfg.block_bwd_ofm) {
              for (ojb = 0; ojb < cfg.ofh; ojb += cfg.block_bwd_oj) {
                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+cfg.block_bwd_ifm, my_ifm_end); ifm1++ ) {

                  if ( (ofmb == 0) && ((cfg.options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && cfg.avoid_acc_load_bwd == 0 && ojb == 0) {
                    /* set output feature map to zero */
                    for (oj = 0; oj < cfg.ofh; ++oj) {
                      float* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, del_input, img, ifm1, oj, 0, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock));
                      for (oi = 0; oi < cfg.ofw; ++oi) {
                        LIBXSMM_PRAGMA_SIMD
                          for (ifm2 = 0; ifm2 < cfg.ifmblock; ++ifm2) {
                            temp_ptr[ifm2] = (float)0;
                          }
                        temp_ptr += cfg.ifmblock;
                      }
                    }
                  }

                  for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+cfg.block_bwd_ofm, cfg.blocksofm); ofm1 += cfg.blocksofm_blocking) {
                    for (oj = ojb; oj < LIBXSMM_MIN(ojb+cfg.block_bwd_oj,cfg.ofh); oj += cfg.bwd_ofh_rb) {
                      for (oi = 0; oi < cfg.ofw; oi += cfg.bwd_ofw_rb) {
                        /* Prepare batch-reduce kernel arguments */
                        ij_use = (cfg.spread_input_bwd == 1) ? oj * cfg.u : oj;
                        ii_use = (cfg.spread_input_bwd == 1) ? oi * cfg.v : oi;
                        oi_use = oi;
                        oj_use = oj;
                        ind = 0;
                        for (ofm2 = ofm1; ofm2 < ofm1 + cfg.blocksofm_blocking; ofm2++) {
                          for (kj = 0; kj < cfg.R; kj++) {
                            for (ki = 0; ki < cfg.S; ki++) {
                              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm2, kj, ki, 0, 0, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
                              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  output,  img, ofm2, oj_use + kj, oi_use + ki, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                              ind++;
                            }
                          }
                        }
                        n_blocks = ind;
                        br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock), &n_blocks);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    if (cfg.loop_order == 1) { /* (loop_order == N_Kb_Cb_Hb_k_c_h_w) { */
      for (img = my_img_start; img < my_img_end; img++) {
        for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += cfg.block_bwd_ifm) {
          for (ojb = 0; ojb < cfg.ofh; ojb += cfg.block_bwd_oj) {
            for (oj = ojb; oj < LIBXSMM_MIN(ojb+cfg.block_bwd_oj,cfg.ofh); oj += cfg.bwd_ofh_rb) {
              for (oi = 0; oi < cfg.ofw; oi += cfg.bwd_ofw_rb) {
                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+cfg.block_bwd_ifm, my_ifm_end); ifm1++ ) {
                  for (ofmb = 0; ofmb < cfg.blocksofm; ofmb += cfg.block_bwd_ofm) {
                    if ( (ofmb == 0) && ((cfg.options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && cfg.avoid_acc_load_bwd == 0 && ojb == 0 && oj == 0 && oi == 0) {
                      /* set output feature map to zero */
                      for (oj = 0; oj < cfg.ofh; ++oj) {
                        float* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, del_input, img, ifm1, oj, 0, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock));
                        for (oi = 0; oi < cfg.ofw; ++oi) {
                          LIBXSMM_PRAGMA_SIMD
                            for (ifm2 = 0; ifm2 < cfg.ifmblock; ++ifm2) {
                              temp_ptr[ifm2] = (float)0;
                            }
                          temp_ptr += cfg.ifmblock;
                        }
                      }
                    }
                    for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+cfg.block_bwd_ofm, cfg.blocksofm); ofm1 += cfg.blocksofm_blocking) {
                      /* Prepare batch-reduce kernel arguments */
                      ij_use = (cfg.spread_input_bwd == 1) ? oj * cfg.u : oj;
                      ii_use = (cfg.spread_input_bwd == 1) ? oi * cfg.v : oi;
                      oi_use = oi;
                      oj_use = oj;
                      ind = 0;
                      for (ofm2 = ofm1; ofm2 < ofm1 + cfg.blocksofm_blocking; ofm2++) {
                        for (kj = 0; kj < cfg.R; kj++) {
                          for (ki = 0; ki < cfg.S; ki++) {
                            A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm2, kj, ki, 0, 0, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
                            B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  output,  img, ofm2, oj_use + kj, oi_use + ki, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                            ind++;
                          }
                        }
                      }
                      n_blocks = ind;
                      br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock), &n_blocks);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    if (cfg.pack_input_bwd == 1) {
      LIBXSMM_VLA_DECL(5, float, del_input_full, (float*)din_act_ptr + ((size_t)cfg.pad_h_in * cfg.ifwp + cfg.pad_w_in) * cfg.ifmblock, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
      for (img = my_img_start; img < my_img_end; img++) {
        for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
          for (oj = 0; oj < cfg.ifhp; oj++) {
            for (oi = 0; oi < cfg.ifwp; oi++) {
              if (oi % cfg.v != 0 || oj % cfg.u != 0) {
                LIBXSMM_PRAGMA_SIMD
                  for (ifm2 = 0; ifm2 < cfg.ifmblock; ifm2++) {
                    LIBXSMM_VLA_ACCESS(5,  del_input_full, img, ifm1, oj, oi, ifm2, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock) = (float)0;
                  }
              } else {
                LIBXSMM_PRAGMA_SIMD
                  for (ifm2 = 0; ifm2 < cfg.ifmblock; ifm2++) {
                    LIBXSMM_VLA_ACCESS(5,  del_input_full, img, ifm1, oj, oi, ifm2, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock) = LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, oj/cfg.u, oi/cfg.v, ifm2, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                  }
              }
            }
          }
        }
      }
    } else if (cfg.spread_input_bwd == 1) {
      LIBXSMM_VLA_DECL(5, float, del_input_full, (float*)din_act_ptr + ((size_t)cfg.pad_h_in * cfg.ifwp + cfg.pad_w_in) * cfg.ifmblock, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
      for (img = my_img_start; img < my_img_end; img++) {
        for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
          for (oj = 0; oj < cfg.ifhp; oj++) {
            for (oi = 0; oi < cfg.ifwp; oi++) {
              if (oi % cfg.v != 0 || oj % cfg.u != 0) {
                LIBXSMM_PRAGMA_SIMD
                  for (ifm2 = 0; ifm2 < cfg.ifmblock; ifm2++) {
                    LIBXSMM_VLA_ACCESS(5,  del_input_full, img, ifm1, oj, oi, ifm2, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock) = (float)0;
                  }
              }
            }
          }
        }
      }
    }
  } else {
    int imgifm1, img, ofm1, ifm1, oj, ij, oi, ii, kj, ki, ifm2, ofm2, ifm1ofm1;
    /* computing first logical thread */
    const int ltid = my_tid - start_tid;

    /* number of tasks that could be run in parallel */
    const int work = cfg.N * cfg.blocksifm;
    /* compute chunk size */
    const int chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

    /* number of tasks for transpose that could be run in parallel */
    int transpose_work = cfg.blocksifm * cfg.blocksofm;
    /* compute chunk size */
    const int transpose_chunksize = (transpose_work % cfg.threads == 0) ? (transpose_work / cfg.threads) : ((transpose_work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
    const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

    /* offset pointer in case of physical padding */
    float *const out = (float*)dout_act_ptr + ((size_t)cfg.pad_h_out * cfg.ofwp + cfg.pad_w_out) * cfg.ofmblock;

    /* Weight and transpose_weight tensor declaration */
    LIBXSMM_VLA_DECL(6, float, wt, (float*)wt_ptr, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
    LIBXSMM_VLA_DECL(6, float, tr_wt, (float*)((char*)scratch + cfg.bwd_filter_trans_scratch_offset), cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
    /* define weight pointer which has the correct format */
    float* weight_base = 0;

    /* padding via stack allocated buffers */
    const int padded_w = cfg.W + (2 * cfg.pad_w);
    const int padded_h = cfg.H + (2 * cfg.pad_h);
    const int size_tls1 = padded_h * padded_w * cfg.ifmblock;
    float *const del_input_scratch_padding = (float*)((char*)scratch + cfg.bwd_packing_padding_scratch_offset) + ltid * size_tls1;
    for ( ii = 0; ii < size_tls1; ++ii ) { del_input_scratch_padding[ii] = (float)0; }

    /* lazy barrier init */
    libxsmm_barrier_init(cfg.barrier, ltid);

    /* transpose filters, if requested */
    if ( (cfg.options & LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE) > 0 ) {
      weight_base = (float*)tr_wt_ptr;
    } else {
      for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
        ofm1 = ifm1ofm1 / cfg.blocksifm;
        ifm1 = ifm1ofm1 % cfg.blocksifm;
        for (kj=0; kj < cfg.R; kj++) {
          for (ki=0; ki < cfg.S; ki++) {
            for (ofm2 = 0; ofm2 < cfg.ofmblock; ++ofm2) {
              for (ifm2 = 0; ifm2 < cfg.ifmblock; ++ifm2) {
                LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, cfg.R-1-kj , cfg.S-1-ki, ofm2, ifm2, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock) =
                  LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
              }
            }
          }
        }
      }
      weight_base = (float*)((char*)scratch + cfg.bwd_filter_trans_scratch_offset);

      /* wait for transpose to finish */
      libxsmm_barrier_wait(cfg.barrier, ltid);
    }

    {/* open new scope for additional variable declarations (C89) */
    LIBXSMM_VLA_DECL(5, float, del_input, (float*)din_act_ptr, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
    LIBXSMM_VLA_DECL(3, float, del_input_padded, del_input_scratch_padding, padded_w, cfg.ifmblock);
    LIBXSMM_VLA_DECL(5, const float, output, out, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
    LIBXSMM_VLA_DECL(6, const float, weight, weight_base, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);

    for (imgifm1 = thr_begin; imgifm1 < thr_end; ++imgifm1) {
      img = imgifm1 / cfg.blocksifm;
      ifm1 = imgifm1 % cfg.blocksifm;

      /* check if we need padding, for now we do physical padding on the fly, however we can play with N parameter of the GEMM */
      /* @TODO: add variant which deals with multiple GEMMS by varying N to deal with padding */
      if ( (cfg.pad_h == cfg.pad_h_in) && (cfg.pad_w == cfg.pad_w_in) ) {

        /* reset result buffer to zero when intent is to overwrite when first block
           of input channels should be convoluted */
        if ( ((cfg.options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
          float* temp_ptr = &(LIBXSMM_VLA_ACCESS(  5, del_input, img, ifm1, 0, 0, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock));
          LIBXSMM_PRAGMA_SIMD
          for (ij = 0; ij < cfg.ifhp*cfg.ifwp*cfg.ifmblock; ij++) {
            temp_ptr[ij] = (float)0;
          }
        }

        /* run convolution */
        for (ofm1 = 0; ofm1 < cfg.blocksofm; ++ofm1) {
          for ( oj = 0; oj < cfg.ofh; ++oj) {
            ij = oj * cfg.u;
            oi = 0; ii = 0;
            for (kj = 0; kj < cfg.R; ++kj) {
              for (ki = 0; ki < cfg.S; ++ki) {
                gemm_kernel( &LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm1, cfg.R-1-kj, cfg.S-1-ki, 0, 0,        cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock),
                             &LIBXSMM_VLA_ACCESS(5, output,  img, ofm1, oj, oi, 0,           cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock),
                             &LIBXSMM_VLA_ACCESS(5, del_input,  img, ifm1, ij + kj, ii + ki, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock) );
              }
            }
          }
        }

        /* zero rim in case of physical padding.... this code is extremely stupid and crappy as it requires a complicated if... */
        if (cfg.pad_h_in > 0 || cfg.pad_w_in > 0) {
          for ( ij = 0; ij < cfg.ifhp; ij++ ) {
            for ( ii = 0; ii < cfg.ifwp; ii++ ) {
              if ( (ij < cfg.pad_h_in) || (ij >= (cfg.H+cfg.pad_h_in)) ||
                   (ii < cfg.pad_w_in) || (ii >= (cfg.W+cfg.pad_w_in)) ) {
                for (ifm2 = 0; ifm2 < cfg.ifmblock; ++ifm2) {
                  LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, ii, ifm2, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock) = (float)0;
                }
              }
            }
          }
        }
      } else {
        /* reset result buffer to zero when intent is to overwrite when first block
           of input channels should be convoluted */
        if ( ((cfg.options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
          LIBXSMM_PRAGMA_SIMD
          for (ij = 0; ij < size_tls1; ++ij) {
            del_input_scratch_padding[ij] = (float)0;
          }
        } else {
          for (ij = 0; ij < cfg.H; ij++) {
            for (ii = 0; ii < cfg.W; ii++) {
              LIBXSMM_PRAGMA_SIMD
              for (ifm2 = 0; ifm2 < cfg.ifmblock; ifm2++) {
                  LIBXSMM_VLA_ACCESS(3, del_input_padded, ij + cfg.pad_h, ii + cfg.pad_w, ifm2, padded_w, cfg.ifmblock) =
                    LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, ii, ifm2, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
              }
            }
          }
        }

        /* run convolution */
        for (ofm1 = 0; ofm1 < cfg.blocksofm; ++ofm1) {
          for ( oj = 0; oj < cfg.ofh; ++oj) {
            ij = oj * cfg.u;
            oi = 0; ii = 0;
            for (kj = 0; kj < cfg.R; ++kj) {
              for (ki = 0; ki < cfg.S; ++ki) {
                gemm_kernel( &LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm1, cfg.R-1-kj, cfg.S-1-ki, 0, 0,        cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock),
                             &LIBXSMM_VLA_ACCESS(5, output,  img, ofm1, oj, oi, 0,           cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock),
                             &LIBXSMM_VLA_ACCESS(3, del_input_padded, ij + kj, ii + ki, 0, padded_w, cfg.ifmblock) );
              }
            }
          }
        }

        /* input padding copy back */
        for (ij = 0; ij < cfg.H; ij++) {
          for (ii = 0; ii < cfg.W; ii++) {
            LIBXSMM_PRAGMA_SIMD
            for (ifm2 = 0; ifm2 < cfg.ifmblock; ifm2++) {
              LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, ii, ifm2, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock) =
                LIBXSMM_VLA_ACCESS(3, del_input_padded, ij + cfg.pad_h, ii + cfg.pad_w, ifm2, padded_w, cfg.ifmblock);
            }
          }
        }
      }
    } /* end of imgifm1 loop */

    } /* end of new scope for additional variable declarations (C89) */
  }
  libxsmm_barrier_wait(cfg.barrier, ltid);
}

