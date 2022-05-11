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

#include <libxsmm_dnn_conv.h>

LIBXSMM_API void libxsmm_dnn_conv_bwd_exec_bf16( libxsmm_dnn_conv_config cfg, const libxsmm_bfloat16* wt_ptr, const libxsmm_bfloat16* tr_wt_ptr,  const libxsmm_bfloat16* dout_act_ptr, libxsmm_bfloat16* din_act_ptr,
    unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch ) {
  const int ltid = my_tid - start_tid;
  libxsmm_gemm_param        gemm_param;
  libxsmm_meltw_unary_param unary_param;

  if (cfg.use_fallback_bwd_loops == 0) {
    int img, ofm1, ifm1, oj, oi, kj, ki, oi_use, oj_use, ii_use, ij_use, ofmb, ifmb, ojb, myIfmId, nIfmBlocks, task;
    /* computing first logical thread */
    int imgpt = LIBXSMM_UPDIV(cfg.N, cfg.threads);
    int threads_per_image = cfg.threads / cfg.N;
    int my_img_start = LIBXSMM_MIN(ltid * imgpt, cfg.N);
    int my_img_end = LIBXSMM_MIN((ltid+1) * imgpt, cfg.N);
    int my_ifm_start = 0;
    int my_ifm_end = cfg.blocksifm;

    /* Batch reduce related variables */
    unsigned long long        n_blocks;

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
    libxsmm_bfloat16 *input_ptr = (cfg.pack_input_bwd == 1) ? (libxsmm_bfloat16*)((char*)scratch + cfg.bwd_packing_padding_scratch_offset) : (libxsmm_bfloat16*)din_act_ptr + ((size_t)cfg.pad_h_in * cfg.ifwp + cfg.pad_w_in) * cfg.ifmblock;
    LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, del_input, input_ptr, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
    float *del_inp_fp32 = (float*)((char*)scratch + cfg.bwd_lp_input_full_scratch_offset) + ((size_t)cfg.pad_h_in * cfg.ifwp + cfg.pad_w_in) * cfg.ifmblock;
    LIBXSMM_VLA_DECL(5, float, del_input_fp32, del_inp_fp32, cfg.blocksifm, IFH, IFW, cfg.ifmblock);

    libxsmm_bfloat16 *const out = (libxsmm_bfloat16*)dout_act_ptr;
    LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, output, out, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);

    /* Weight and transpose_weight tensor declaration */
    LIBXSMM_VLA_DECL(6, libxsmm_bfloat16, wt, (libxsmm_bfloat16*)wt_ptr, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
    LIBXSMM_VLA_DECL(6, libxsmm_bfloat16, tr_wt, (libxsmm_bfloat16*)((char*)scratch + cfg.bwd_filter_trans_scratch_offset), cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
    /* define weight pointer which has the correct format */
    libxsmm_bfloat16* weight_base = (cfg.avoid_bwd_wt_trans > 0 ) ? (libxsmm_bfloat16*)tr_wt_ptr : (libxsmm_bfloat16*)((char*)scratch + cfg.bwd_filter_trans_scratch_offset);
    LIBXSMM_VLA_DECL(6, const libxsmm_bfloat16, weight, weight_base, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);

    /* lazy barrier init */
    libxsmm_barrier_init(cfg.barrier, ltid);

    gemm_param.a.secondary = (void*)cfg.A_offsets_bwd;
    gemm_param.b.secondary = (void*)cfg.B_offsets_bwd;

    /* transpose filters, if requested */
    if ( cfg.avoid_bwd_wt_trans == 0 ) {
      for (task = transpose_thr_begin; task < transpose_thr_end; ++task) {
        ifm1 = task/(cfg.blocksofm * cfg.R * cfg.S);
        ofm1 = (task%(cfg.blocksofm * cfg.R * cfg.S))/(cfg.R * cfg.S);
        kj =   ((task%(cfg.blocksofm * cfg.R * cfg.S))%(cfg.R * cfg.S))/cfg.S;
        ki =   ((task%(cfg.blocksofm * cfg.R * cfg.S))%(cfg.R * cfg.S))%cfg.S;
        unary_param.in.primary  = &LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
        unary_param.out.primary = &LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, cfg.R-1-kj, cfg.S-1-ki, 0, 0, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
        cfg.tr_kernel( &unary_param );
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
                  if ( (ofmb == 0) && (cfg.overwrite_output > 0) && (cfg.avoid_acc_load_bwd == 0) && (ojb == 0) ) {
                    /* set output feature map to zero */
                    unary_param.out.primary = (void*)&(LIBXSMM_VLA_ACCESS(  5, del_input_fp32, img, ifm1, 0, 0, 0,  cfg.blocksifm, IFH, IFW, cfg.ifmblock));
                    cfg.ofh_x_ofw_x_ifmblock_zero_kernel_f32( &unary_param );
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
                              n_blocks = cfg.blocksofm_blocking;
                              gemm_param.op.tertiary = &n_blocks;
                              gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm1, kj, ki, 0, 0, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
                              gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  output,  img, ofm1, oj_use + kj, oi_use + ki + 1, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                              gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, ij_use, ii_use + 1, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                              cfg.bwd_compute_kernel2_strd_bf16f32.gemm( &gemm_param );
                            } else if (oi == cfg.ofw-cfg.bwd_ofw_rb  && ki == cfg.S-1) {
                              n_blocks = cfg.blocksofm_blocking;
                              gemm_param.op.tertiary = &n_blocks;
                              gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm1, kj, ki, 0, 0, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
                              gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  output,  img, ofm1, oj_use + kj, oi_use + ki, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                              gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, ij_use, ii_use, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                              cfg.bwd_compute_kernel2_strd_bf16f32.gemm( &gemm_param );
                            } else {
                              n_blocks = cfg.blocksofm_blocking;
                              gemm_param.op.tertiary = &n_blocks;
                              gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm1, kj, ki, 0, 0, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
                              gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  output,  img, ofm1, oj_use + kj, oi_use + ki, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                              gemm_param.c.primary = (void*) &LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, ij_use, ii_use, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                              cfg.bwd_compute_kernel_strd_bf16f32.gemm( &gemm_param );
                            }

                            if ((kj == cfg.R-1) && (ki == cfg.S-1) && (ofm1 + cfg.blocksofm_blocking >= cfg.blocksofm) && (oi + cfg.bwd_ofw_rb >= cfg.ofw)) {
                              unary_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, ij_use, 0, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                              unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, 0, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                              cfg.cvt_kernel_bwd_fp32bf16 ( &unary_param );
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
                  if ( (ofmb == 0) && (cfg.overwrite_output > 0) && (cfg.avoid_acc_load_bwd == 0) && (ojb == 0) ) {
                    /* set output feature map to zero */
                    unary_param.out.primary = (void*)&(LIBXSMM_VLA_ACCESS(  5, del_input, img, ifm1, 0, 0, 0,  cfg.blocksifm, IFH, IFW, cfg.ifmblock));
                    cfg.ofh_x_ofw_x_ifmblock_zero_kernel_bf16( &unary_param );
                  }
                  for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+cfg.block_bwd_ofm, cfg.blocksofm); ofm1 += cfg.blocksofm_blocking) {
                    for (oj = ojb; oj < LIBXSMM_MIN(ojb+cfg.block_bwd_oj,cfg.ofh); oj += cfg.bwd_ofh_rb) {
                      for (oi = 0; oi < cfg.ofw; oi += cfg.bwd_ofw_rb) {
                        /* Prepare batch-reduce kernel arguments */
                        ij_use = (cfg.spread_input_bwd == 1) ? oj * cfg.u : oj;
                        ii_use = (cfg.spread_input_bwd == 1) ? oi * cfg.v : oi;
                        oi_use = oi;
                        oj_use = oj;
                        n_blocks = cfg.blocksofm_blocking * cfg.R * cfg.S;
                        gemm_param.op.tertiary = &n_blocks;
                        gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm1, 0, 0, 0, 0, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
                        gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  output,  img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                        gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                        if (cfg.R == 1 && cfg.S == 1) {
                          cfg.bwd_compute_kernel_strd_bf16.gemm( &gemm_param );
                        } else {
                          cfg.bwd_compute_kernel_offs_bf16.gemm( &gemm_param );
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

    if (cfg.loop_order == 1) { /* (loop_order == N_Kb_Cb_Hb_k_c_h_w) { */
      for (img = my_img_start; img < my_img_end; img++) {
        for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += cfg.block_bwd_ifm) {
          for (ojb = 0; ojb < cfg.ofh; ojb += cfg.block_bwd_oj) {
            for (oj = ojb; oj < LIBXSMM_MIN(ojb+cfg.block_bwd_oj,cfg.ofh); oj += cfg.bwd_ofh_rb) {
              for (oi = 0; oi < cfg.ofw; oi += cfg.bwd_ofw_rb) {
                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+cfg.block_bwd_ifm, my_ifm_end); ifm1++ ) {
                  for (ofmb = 0; ofmb < cfg.blocksofm; ofmb += cfg.block_bwd_ofm) {
                    if ( (ofmb == 0) && (cfg.overwrite_output > 0) && (cfg.avoid_acc_load_bwd == 0) && (ojb == 0) && (oj == 0) && (oi == 0) ) {
                      /* set output feature map to zero */
                      unary_param.out.primary = (void*)&(LIBXSMM_VLA_ACCESS(  5, del_input, img, ifm1, 0, 0, 0,  cfg.blocksifm, IFH, IFW, cfg.ifmblock));
                      cfg.ofh_x_ofw_x_ifmblock_zero_kernel_bf16( &unary_param );
                    }
                    for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+cfg.block_bwd_ofm, cfg.blocksofm); ofm1 += cfg.blocksofm_blocking) {
                      /* Prepare batch-reduce kernel arguments */
                      ij_use = (cfg.spread_input_bwd == 1) ? oj * cfg.u : oj;
                      ii_use = (cfg.spread_input_bwd == 1) ? oi * cfg.v : oi;
                      oi_use = oi;
                      oj_use = oj;

                      n_blocks = cfg.blocksofm_blocking * cfg.R * cfg.S;
                      gemm_param.op.tertiary = &n_blocks;
                      gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm1, 0, 0, 0, 0, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
                      gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  output,  img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                      gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                      if (cfg.R == 1 && cfg.S == 1) {
                        cfg.bwd_compute_kernel_strd_bf16.gemm( &gemm_param );
                      } else {
                        cfg.bwd_compute_kernel_offs_bf16.gemm( &gemm_param );
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

    if (cfg.pack_input_bwd == 1) {
      LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, del_input_full, (libxsmm_bfloat16*)din_act_ptr + ((size_t)cfg.pad_h_in * cfg.ifwp + cfg.pad_w_in) * cfg.ifmblock, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
      for (img = my_img_start; img < my_img_end; img++) {
        for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
          for (oj = 0; oj < cfg.ifhp; oj++) {
            for (oi = 0; oi < cfg.ifwp; oi++) {
              if (oi % cfg.v != 0 || oj % cfg.u != 0) {
                unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  del_input_full, img, ifm1, oj, oi, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
                cfg.ifmblock_zero_kernel_bf16( &unary_param );
              } else {
                unary_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, oj/cfg.u, oi/cfg.v, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  del_input_full, img, ifm1, oj, oi, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
                cfg.ifmblock_copy_kernel_bf16( &unary_param );
              }
            }
          }
        }
      }
    } else if (cfg.spread_input_bwd == 1) {
      LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, del_input_full, (libxsmm_bfloat16*)din_act_ptr + ((size_t)cfg.pad_h_in * cfg.ifwp + cfg.pad_w_in) * cfg.ifmblock, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
      for (img = my_img_start; img < my_img_end; img++) {
        for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
          for (oj = 0; oj < cfg.ifhp; oj++) {
            for (oi = 0; oi < cfg.ifwp; oi++) {
              if (oi % cfg.v != 0 || oj % cfg.u != 0) {
                unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  del_input_full, img, ifm1, oj, oi, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
                cfg.ifmblock_zero_kernel_bf16( &unary_param );
              }
            }
          }
        }
      }
    }
  } else {
    int imgifm1, img, ofm1, ifm1, oj, ij, oi, ii, kj, ki, task;
    /* number of tasks that could be run in parallel */
    const int work = cfg.N * cfg.blocksifm;
    /* compute chunk size */
    const int chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

    /* number of tasks for transpose that could be run in parallel */
    int transpose_work = cfg.blocksifm * cfg.blocksofm * cfg.R * cfg.S;
    /* compute chunk size */
    const int transpose_chunksize = (transpose_work % cfg.threads == 0) ? (transpose_work / cfg.threads) : ((transpose_work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
    const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

    /* offset pointer in case of physical padding */
    libxsmm_bfloat16 *const out = (libxsmm_bfloat16*)dout_act_ptr + ((size_t)cfg.pad_h_out * cfg.ofwp + cfg.pad_w_out) * cfg.ofmblock;

    /* Weight and transpose_weight tensor declaration */
    LIBXSMM_VLA_DECL(6, libxsmm_bfloat16, wt, (libxsmm_bfloat16*)wt_ptr, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
    LIBXSMM_VLA_DECL(6, libxsmm_bfloat16, tr_wt, (libxsmm_bfloat16*)((char*)scratch + cfg.bwd_filter_trans_scratch_offset), cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
    /* define weight pointer which has the correct format */
    libxsmm_bfloat16* weight_base = (cfg.avoid_bwd_wt_trans > 0 ) ? (libxsmm_bfloat16*)tr_wt_ptr : (libxsmm_bfloat16*)((char*)scratch + cfg.bwd_filter_trans_scratch_offset);

    /* padding via stack allocated buffers */
    const int padded_w = cfg.W + (2 * cfg.pad_w);
    const int padded_h = cfg.H + (2 * cfg.pad_h);
    const int size_tls1 = padded_h * padded_w * cfg.ifmblock;
    libxsmm_bfloat16 *const del_input_scratch_padding = (libxsmm_bfloat16*)((char*)scratch + cfg.bwd_packing_padding_scratch_offset) + ltid * size_tls1;

    LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, del_input, (libxsmm_bfloat16*)din_act_ptr, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
    LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, del_input_padded, del_input_scratch_padding, padded_w, cfg.ifmblock);
    LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, output, out, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
    LIBXSMM_VLA_DECL(6, const libxsmm_bfloat16, weight, weight_base, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);

    /* Zero tls scratch */
    unary_param.out.primary = (void*)del_input_scratch_padding;
    cfg.paddedH_x_paddedW_x_ifmblock_zero_kernel_bf16( &unary_param );

    /* lazy barrier init */
    libxsmm_barrier_init(cfg.barrier, ltid);

    /* transpose filters, if requested */
    if ( cfg.avoid_bwd_wt_trans == 0 ) {
      for (task = transpose_thr_begin; task < transpose_thr_end; ++task) {
        ifm1 = task/(cfg.blocksofm * cfg.R * cfg.S);
        ofm1 = (task%(cfg.blocksofm * cfg.R * cfg.S))/(cfg.R * cfg.S);
        kj =   ((task%(cfg.blocksofm * cfg.R * cfg.S))%(cfg.R * cfg.S))/cfg.S;
        ki =   ((task%(cfg.blocksofm * cfg.R * cfg.S))%(cfg.R * cfg.S))%cfg.S;
        unary_param.in.primary  = &LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
        unary_param.out.primary = &LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, cfg.R-1-kj, cfg.S-1-ki, 0, 0, cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
        cfg.tr_kernel( &unary_param );
      }
      weight_base = (libxsmm_bfloat16*)((char*)scratch + cfg.bwd_filter_trans_scratch_offset);
      /* wait for transpose to finish */
      libxsmm_barrier_wait(cfg.barrier, ltid);
    }

    for (imgifm1 = thr_begin; imgifm1 < thr_end; ++imgifm1) {
      img = imgifm1 / cfg.blocksifm;
      ifm1 = imgifm1 % cfg.blocksifm;

      /* check if we need padding, for now we do physical padding on the fly, however we can play with N parameter of the GEMM */
      /* @TODO: add variant which deals with multiple GEMMS by varying N to deal with padding */
      if ( (cfg.pad_h == cfg.pad_h_in) && (cfg.pad_w == cfg.pad_w_in) ) {
        /* reset result buffer to zero when intent is to overwrite when first block of input channels should be convoluted */
        if ( cfg.overwrite_output > 0 ) {
          unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(  5, del_input, img, ifm1, 0, 0, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
          cfg.ifhp_x_ifwp_x_ifmblock_zero_kernel_bf16 ( &unary_param );
        }
        /* run convolution */
        for (ofm1 = 0; ofm1 < cfg.blocksofm; ++ofm1) {
          for ( oj = 0; oj < cfg.ofh; ++oj) {
            ij = oj * cfg.u;
            oi = 0; ii = 0;
            for (kj = 0; kj < cfg.R; ++kj) {
              for (ki = 0; ki < cfg.S; ++ki) {
                gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm1, cfg.R-1-kj, cfg.S-1-ki, 0, 0,        cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
                gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output,  img, ofm1, oj, oi, 0,           cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, del_input,  img, ifm1, ij + kj, ii + ki, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
                cfg.bwd_compute_kernel_fallback_bf16.gemm( &gemm_param );
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
                unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, ii, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
                cfg.ifmblock_zero_kernel_bf16( &unary_param );
              }
            }
          }
        }
      } else {
        /* reset result buffer to zero when intent is to overwrite when first block
           of input channels should be convoluted */
        if ( cfg.overwrite_output > 0 ) {
          unary_param.out.primary = (void*)del_input_scratch_padding;
          cfg.paddedH_x_paddedW_x_ifmblock_zero_kernel_bf16( &unary_param );
        } else {
          for (ij = 0; ij < cfg.H; ij++) {
            for (ii = 0; ii < cfg.W; ii++) {
              unary_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, ii, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
              unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(3, del_input_padded, ij + cfg.pad_h, ii + cfg.pad_w, 0, padded_w, cfg.ifmblock);
              cfg.ifmblock_copy_kernel_bf16( &unary_param );
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
                gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm1, cfg.R-1-kj, cfg.S-1-ki, 0, 0,        cfg.blocksofm, cfg.R, cfg.S, cfg.ofmblock, cfg.ifmblock);
                gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output,  img, ofm1, oj, oi, 0,           cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(3, del_input_padded, ij + kj, ii + ki, 0, padded_w, cfg.ifmblock);
                cfg.bwd_compute_kernel_fallback_bf16.gemm( &gemm_param );
              }
            }
          }
        }
        /* input padding copy back */
        for (ij = 0; ij < cfg.H; ij++) {
          for (ii = 0; ii < cfg.W; ii++) {
            unary_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(3, del_input_padded, ij + cfg.pad_h, ii + cfg.pad_w, 0, padded_w, cfg.ifmblock);
            unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, ii, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
            cfg.ifmblock_copy_kernel_bf16( &unary_param );
          }
        }
      }
    }
  }
  libxsmm_barrier_wait(cfg.barrier, ltid);
}

