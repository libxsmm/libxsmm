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

LIBXSMM_API void libxsmm_dnn_conv_fwd_exec_bf16( libxsmm_dnn_conv_config cfg, const libxsmm_bfloat16* wt_ptr, const libxsmm_bfloat16* in_act_ptr, libxsmm_bfloat16* out_act_ptr,
    const libxsmm_bfloat16* bias_ptr, unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch ) {

  int img, ofm1, ifm1, oj, oi, kj, ki, oi_use, oj_use, ii_use, ij_use, ofmb, ifmb, ojb, myOfmId, nOfmBlocks, ofm11, ki1, kj1, ii, ij, spread_out = 1;

  /* computing first logical thread */
  const int ltid = my_tid - start_tid;
  int imgpt = LIBXSMM_UPDIV(cfg.N, cfg.threads);
  int threads_per_image = cfg.threads / cfg.N;
  int my_img_start = LIBXSMM_MIN(ltid * imgpt, cfg.N);
  int my_img_end = LIBXSMM_MIN((ltid+1) * imgpt, cfg.N);
  int my_ofm_start = 0;
  int my_ofm_end = cfg.blocksofm;
  int fuse_colbias  = ((cfg.fuse_type & LIBXSMM_DNN_CONV_ELTWISE_FUSE_BIAS) > 0) ? 1 : 0;
  int fuse_relu     = ((cfg.fuse_type & LIBXSMM_DNN_CONV_ELTWISE_FUSE_RELU) > 0) ? 1 : 0;

  /* Batch reduce related variables */
  unsigned long long n_blocks;
  libxsmm_gemm_param      gemm_param;
  libxsmm_gemm_ext_param  gemm_param_ext;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_binary_param binary_param;

  /* offset output pointer in case of physical output padding */
  libxsmm_bfloat16* out = (libxsmm_bfloat16*)out_act_ptr + ((size_t)cfg.pad_h_out * cfg.ofwp + cfg.pad_w_out) * cfg.ofmblock;
  float* out_fp32 = (float*)((char*)scratch + cfg.fwd_lp_output_full_scratch_offset) + ((size_t)cfg.pad_h_out * cfg.ofwp + cfg.pad_w_out) * cfg.ofmblock;
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, output, out, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
  LIBXSMM_VLA_DECL(5, float, output_fp32, out_fp32, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
  libxsmm_bfloat16 *input_ptr = ( (cfg.pack_input == 1) || (cfg.fwd_padding_copy == 1) ) ? (libxsmm_bfloat16*)((char*)scratch + cfg.fwd_packing_padding_scratch_offset) : (libxsmm_bfloat16*)in_act_ptr;
  const int IFW = (cfg.fwd_padding_copy == 1) ? cfg.ifwp + 2*cfg.pad_w : ( (cfg.pack_input == 1) ? cfg.ofwp : cfg.ifwp );
  const int IFH = (cfg.fwd_padding_copy == 1) ? cfg.ifhp + 2*cfg.pad_h : ( (cfg.pack_input == 1) ? cfg.ofhp : cfg.ifhp );
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, input, input_ptr, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
  LIBXSMM_VLA_DECL(6, const libxsmm_bfloat16, weight, (libxsmm_bfloat16*)wt_ptr, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, colbias, (libxsmm_bfloat16*)bias_ptr, cfg.ofmblock);

  gemm_param.a.secondary = (void*)cfg.A_offsets;
  gemm_param.b.secondary = (void*)cfg.B_offsets;
  gemm_param_ext.a.secondary = (void*)cfg.A_offsets;
  gemm_param_ext.b.secondary = (void*)cfg.B_offsets;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  if ( imgpt <= 1 ) {
    my_img_start = LIBXSMM_MIN(ltid / threads_per_image, cfg.N);
    my_img_end = LIBXSMM_MIN(my_img_start + 1, cfg.N);
    myOfmId = ltid % threads_per_image;
    nOfmBlocks = LIBXSMM_UPDIV(cfg.blocksofm, threads_per_image);
    my_ofm_start = LIBXSMM_MIN(myOfmId * nOfmBlocks, cfg.blocksofm);
    my_ofm_end = LIBXSMM_MIN((myOfmId+1) * nOfmBlocks, cfg.blocksofm);
  }

  if ( cfg.use_ofm_parallelization == 1 ) {
    if ( cfg.N % 8 == 0) {
      spread_out = 8;
    } else if ( cfg.N % 4 == 0) {
      spread_out = 4;
    } else if (cfg.N % 2 == 0) {
      spread_out = 2;
    } else if (cfg.N % 3 == 0) {
      spread_out = 3;
    } else {
      spread_out = 1;
    }
    if ((spread_out > 1) && (cfg.threads % spread_out == 0)) {
      int tile_id = ltid / spread_out;
      int ofmpt = LIBXSMM_UPDIV(cfg.blocksofm, spread_out);
      int ofm_id = ltid % spread_out;
      imgpt = LIBXSMM_UPDIV(cfg.N, cfg.threads) * spread_out;
      my_img_start = LIBXSMM_MIN(tile_id * imgpt, cfg.N);
      my_img_end = LIBXSMM_MIN((tile_id+1) * imgpt, cfg.N);
      my_ofm_start = LIBXSMM_MIN(ofm_id * ofmpt, cfg.blocksofm);
      my_ofm_end = LIBXSMM_MIN((ofm_id+1) * ofmpt, cfg.blocksofm);
    }
  }

  /* remove stride from input */
  if (cfg.pack_input == 1) {
    int ifmpt = LIBXSMM_UPDIV(cfg.blocksifm, spread_out);
    int ifm_id = ltid % spread_out;
    int my_ifm_start = LIBXSMM_MIN(ifm_id * ifmpt, cfg.blocksifm);
    int my_ifm_end = LIBXSMM_MIN((ifm_id+1) * ifmpt, cfg.blocksifm);
    LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, input_src, (libxsmm_bfloat16*)in_act_ptr, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
    for (img = my_img_start; img < my_img_end; img++) {
      for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
        for (oj = 0; oj < cfg.ofh; oj++) {
          ij_use = oj * cfg.u;
          unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5,  input_src,  img, ifm1, ij_use, 0, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
          unary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, oj, 0, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
          cfg.strided_copy_kernel_bf16( &unary_param );
        }
      }
    }
    if ( cfg.use_ofm_parallelization == 1 || cfg.N % cfg.threads != 0) {
      libxsmm_barrier_wait(cfg.barrier, ltid);
    }
  }

  /* physical pad input */
  if (cfg.fwd_padding_copy == 1) {
    int ifmpt = LIBXSMM_UPDIV(cfg.blocksifm, spread_out);
    int ifm_id = ltid % spread_out;
    int my_ifm_start = LIBXSMM_MIN(ifm_id * ifmpt, cfg.blocksifm);
    int my_ifm_end = LIBXSMM_MIN((ifm_id+1) * ifmpt, cfg.blocksifm);
    LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, input_src, (libxsmm_bfloat16*)in_act_ptr, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
    for (img = my_img_start; img < my_img_end; img++) {
      for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
        /* copy the inner part */
        for (ij = 0; ij < cfg.ifhp+(2*cfg.pad_h); ij++) {
          for (ii = 0; ii < cfg.ifwp+(2*cfg.pad_w); ii++) {
            if ( (ij >= cfg.pad_h) && (ii >= cfg.pad_w) && (ij < cfg.ifhp+cfg.pad_h) && (ii < cfg.ifwp+cfg.pad_w) ) {
              unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5,  input_src,  img, ifm1, ij-cfg.pad_h, ii-cfg.pad_w, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
              unary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, ij, ii, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
              cfg.ifmblock_copy_kernel_bf16( &unary_param );
            } else {
              unary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, ij, ii, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
              cfg.ifmblock_zero_kernel_bf16( &unary_param );
            }
          }
        }
      }
    }
    if ( cfg.use_ofm_parallelization == 1 || cfg.N % cfg.threads != 0 ) {
      libxsmm_barrier_wait(cfg.barrier, ltid);
    }
  }

  if (cfg.use_fallback_fwd_loops == 1) {
    /* number of tasks that could be run in parallel */
    const int work = cfg.N * cfg.blocksofm * cfg.ofh;
    /* compute chunk size */
    const int chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
    int imgofm1ofh;

    if ( cfg.avoid_fmas_in_rim == 1) {
      for (imgofm1ofh = thr_begin; imgofm1ofh < thr_end; ++imgofm1ofh) {
        img = imgofm1ofh / (cfg.blocksofm*cfg.ofh);
        if (cfg.N > 1) {
          oj = (imgofm1ofh % (cfg.blocksofm*cfg.ofh))/cfg.blocksofm;
          ofm1 = (imgofm1ofh % (cfg.blocksofm*cfg.ofh))%cfg.blocksofm;
        } else {
          oj = (imgofm1ofh % (cfg.blocksofm*cfg.ofh))%cfg.ofh;
          ofm1 = (imgofm1ofh % (cfg.blocksofm*cfg.ofh))/cfg.ofh;
        }
        if ( (cfg.avoid_acc_load == 0) && (cfg.overwrite_output > 0) ) {
          /* set output feature map to zero */
          unary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(  5, output_fp32, img, ofm1, oj, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
          cfg.ofw_x_ofmblock_zero_kernel_f32( &unary_param );
        }
        for (ifmb = 0; ifmb < cfg.blocksifm; ifmb += cfg.block_fwd_ifm) {
          for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+cfg.block_fwd_ifm, cfg.blocksifm); ifm1 += cfg.blocksifm_blocking) {
            for (oi = 0; oi < cfg.ofw; oi += cfg.fwd_ofw_rb) {
              for (kj = 0; kj < cfg.R; kj++) {
                for (ki = 0; ki < cfg.S; ki++) {
                  /* Prepare batch-reduce kernel arguments */
                  if (cfg.pack_input == 1) {
                    ij_use = oj;
                    ii_use = oi;
                  } else {
                    ij_use = oj * cfg.u - (1-cfg.pad_h_in);
                    ii_use = oi * cfg.v - (1-cfg.pad_w_in);
                  }
                  oi_use = oi;
                  oj_use = oj;

                  if (kj == 0 && oj == 0) {
                    /* Do no FLOPS  */
                  } else if (kj == cfg.R-1 && oj == cfg.ofh-1 ) {
                    /* Do no FLOPS  */
                  } else if ( oi == 0 && ki == 0 ) {
                    n_blocks = cfg.blocksifm_blocking;
                    gemm_param.op.tertiary = &n_blocks;
                    gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                    gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use + kj, ii_use + ki + 1, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                    gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, oi_use + 1, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                    cfg.fwd_compute_kernel2_strd_bf16f32.gemm( &gemm_param );
                  } else if (oi == cfg.ofw-cfg.fwd_ofw_rb  && ki == cfg.S-1) {
                    n_blocks = cfg.blocksifm_blocking;
                    gemm_param.op.tertiary = &n_blocks;
                    gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                    gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use + kj, ii_use + ki, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                    gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                    cfg.fwd_compute_kernel2_strd_bf16f32.gemm( &gemm_param );
                  } else {
                    n_blocks = cfg.blocksifm_blocking;
                    gemm_param.op.tertiary = &n_blocks;
                    gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                    gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use + kj, ii_use + ki, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                    gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                    cfg.fwd_compute_kernel_strd_bf16f32.gemm( &gemm_param );
                  }

                  if ( (kj == cfg.R-1) && (ki == cfg.S-1) && (ifm1 + cfg.blocksifm_blocking >= cfg.blocksifm) && (oi + cfg.fwd_ofw_rb >= cfg.ofw) ) {
                    unary_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                    unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output     , img, ofm1, oj_use, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                    cfg.cvt_kernel_fp32bf16 ( &unary_param );
                  }

                  if (fuse_relu || fuse_colbias) {
                    if ( (kj == cfg.R-1) && (ki == cfg.S-1) && (ifm1 + cfg.blocksifm_blocking >= cfg.blocksifm) ) {
                      /* Apply b2b kernels */
                      if (fuse_colbias) {
                        binary_param.in0.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                        binary_param.in1.primary = (void*)&LIBXSMM_VLA_ACCESS(2, colbias, ofm1, 0, cfg.ofmblock);
                        binary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                        cfg.colbias_add_kernel_bf16( &binary_param );
                      }
                      if (fuse_relu) {
                        unary_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                        unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                        cfg.relu_kernel_bf16( &unary_param );
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
      for (imgofm1ofh = thr_begin; imgofm1ofh < thr_end; ++imgofm1ofh) {
        img = imgofm1ofh / (cfg.blocksofm*cfg.ofh);
        ofm1 = (imgofm1ofh % (cfg.blocksofm*cfg.ofh))/cfg.ofh;
        oj = (imgofm1ofh % (cfg.blocksofm*cfg.ofh))%cfg.ofh;
        if ((cfg.overwrite_output > 0) && (cfg.avoid_acc_load == 0)) {
          /* set output feature map to zero */
          unary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, oj, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
          cfg.ofw_x_ofmblock_zero_kernel_bf16( &unary_param );
        }
        for (ifmb = 0; ifmb < cfg.blocksifm; ifmb += cfg.block_fwd_ifm) {
          for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+cfg.block_fwd_ifm, cfg.blocksifm); ifm1 += cfg.blocksifm_blocking) {
            for (oi = 0; oi < cfg.ofw; oi += cfg.fwd_ofw_rb) {
              /* Prepare batch-reduce kernel arguments */
              if (cfg.pack_input == 1) {
                ij_use = oj;
                ii_use = oi;
              } else {
                ij_use = oj * cfg.u;
                ii_use = oi * cfg.v;
              }
              oi_use = oi;
              oj_use = oj;

              n_blocks = cfg.blocksifm_blocking * cfg.R * cfg.S;
              if ((fuse_relu || fuse_colbias) && (ifm1 + cfg.blocksifm_blocking >= cfg.blocksifm)) {
                gemm_param_ext.op.tertiary = &n_blocks;
                gemm_param_ext.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                gemm_param_ext.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                gemm_param_ext.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                if (fuse_colbias) {
                  gemm_param_ext.d.primary = (void*)&LIBXSMM_VLA_ACCESS(2, colbias, ofm1, 0, cfg.ofmblock);
                }
                if (cfg.R == 1 && cfg.S == 1) {
                  cfg.fwd_compute_kernel_strd_fused_bf16.gemm_ext( &gemm_param_ext );
                } else {
                  cfg.fwd_compute_kernel_offs_fused_bf16.gemm_ext( &gemm_param_ext );
                }
              } else {
                gemm_param.op.tertiary = &n_blocks;
                gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                if (cfg.R == 1 && cfg.S == 1) {
                  cfg.fwd_compute_kernel_strd_bf16.gemm( &gemm_param );
                } else {
                  cfg.fwd_compute_kernel_offs_bf16.gemm( &gemm_param );
                }
              }
            }
          }
        }
      }
    }
  } else {
    if (cfg.loop_order == 0) {
      if ( cfg.avoid_fmas_in_rim == 1) {
        for (img = my_img_start; img < my_img_end; img++) {
          for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += cfg.block_fwd_ofm) {
            for (ifmb = 0; ifmb < cfg.blocksifm; ifmb += cfg.block_fwd_ifm) {
              for (ojb = 0; ojb < cfg.ofh; ojb += cfg.block_fwd_oj) {
                for (ofm11 = ofmb; ofm11 < LIBXSMM_MIN(ofmb+cfg.block_fwd_ofm, my_ofm_end); ofm11++ ) {
                  ofm1 = (cfg.shuffle_filter_accesses == 1) ? (ofm11+ltid)%cfg.blocksofm : ofm11;
                  if ( (ifmb == 0) && (cfg.overwrite_output > 0) && (cfg.avoid_acc_load == 0) && (ojb == 0)) {
                    /* set output feature map to zero */
                    unary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(  5, output_fp32, img, ofm1, 0, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                    cfg.ofh_x_ofw_x_ofmblock_zero_kernel_f32( &unary_param );
                  }
                  for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+cfg.block_fwd_ifm, cfg.blocksifm); ifm1 += cfg.blocksifm_blocking) {
                    for (oj = ojb; oj < LIBXSMM_MIN(ojb+cfg.block_fwd_oj,cfg.ofh); oj += cfg.fwd_ofh_rb) {
                      for (oi = 0; oi < cfg.ofw; oi += cfg.fwd_ofw_rb) {
                        for (kj1 = 0; kj1 < cfg.R; kj1++) {
                          for (ki1 = 0; ki1 < cfg.S; ki1++) {
                            /* Prepare batch-reduce kernel arguments */
                            if (cfg.pack_input == 1) {
                              ij_use = oj;
                              ii_use = oi;
                            } else {
                              ij_use = oj * cfg.u - (1-cfg.pad_h_in);
                              ii_use = oi * cfg.v - (1-cfg.pad_w_in);
                            }
                            oi_use = oi;
                            oj_use = oj;

                            ki = (cfg.shuffle_filter_accesses == 1) ?  (ki1+ltid)%cfg.S : ki1;
                            kj = (cfg.shuffle_filter_accesses == 1) ?  (kj1+ltid)%cfg.R : kj1;

                            if (kj == 0 && oj == 0) {
                              /* Do no FLOPS  */
                            } else if (kj == cfg.R-1 && oj == cfg.ofh-1 ) {
                              /* Do no FLOPS  */
                            } else if ( oi == 0 && ki == 0 ) {
                              n_blocks = cfg.blocksifm_blocking;
                              gemm_param.op.tertiary = &n_blocks;
                              gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                              gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use + kj, ii_use + ki + 1, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                              gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, oi_use + 1, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                              cfg.fwd_compute_kernel2_strd_bf16f32.gemm( &gemm_param );
                            } else if (oi == cfg.ofw-cfg.fwd_ofw_rb  && ki == cfg.S-1) {
                              n_blocks = cfg.blocksifm_blocking;
                              gemm_param.op.tertiary = &n_blocks;
                              gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                              gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use + kj, ii_use + ki, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                              gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                              cfg.fwd_compute_kernel2_strd_bf16f32.gemm( &gemm_param );
                            } else {
                              n_blocks = cfg.blocksifm_blocking;
                              gemm_param.op.tertiary = &n_blocks;
                              gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                              gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use + kj, ii_use + ki, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                              gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                              cfg.fwd_compute_kernel_strd_bf16f32.gemm( &gemm_param );
                            }

                            if ( (kj1 == cfg.R-1) && (ki1 == cfg.S-1) && (ifm1 + cfg.blocksifm_blocking >= cfg.blocksifm) && (oi + cfg.fwd_ofw_rb >= cfg.ofw) ) {
                              unary_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                              unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output     , img, ofm1, oj_use, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                              cfg.cvt_kernel_fp32bf16 ( &unary_param );
                            }

                            if (fuse_relu || fuse_colbias) {
                              if ( (kj1 == cfg.R-1) && (ki1 == cfg.S-1) && (ifm1 + cfg.blocksifm_blocking >= cfg.blocksifm) ) {
                                /* Apply b2b kernels */
                                if (fuse_colbias) {
                                  binary_param.in0.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                                  binary_param.in1.primary = (void*)&LIBXSMM_VLA_ACCESS(2, colbias, ofm1, 0, cfg.ofmblock);
                                  binary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                                  cfg.colbias_add_kernel_bf16( &binary_param );
                                }
                                if (fuse_relu) {
                                  unary_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                                  unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                                  cfg.relu_kernel_bf16 ( &unary_param );
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
          }
        }
      } else {
        for (img = my_img_start; img < my_img_end; img++) {
          for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += cfg.block_fwd_ofm) {
            for (ifmb = 0; ifmb < cfg.blocksifm; ifmb += cfg.block_fwd_ifm) {
              for (ojb = 0; ojb < cfg.ofh; ojb += cfg.block_fwd_oj) {
                for (ofm11 = ofmb; ofm11 < LIBXSMM_MIN(ofmb+cfg.block_fwd_ofm, my_ofm_end); ofm11++ ) {
                  ofm1 = (cfg.shuffle_filter_accesses == 1) ? (ofm11+ltid)%cfg.blocksofm : ofm11;
                  if ( (ifmb == 0) && (cfg.overwrite_output > 0) && (cfg.avoid_acc_load == 0) && (ojb == 0)) {
                    /* set output feature map to zero */
                    unary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, 0, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                    cfg.ofh_x_ofw_x_ofmblock_zero_kernel_bf16( &unary_param );
                  }
                  for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+cfg.block_fwd_ifm, cfg.blocksifm); ifm1 += cfg.blocksifm_blocking) {
                    for (oj = ojb; oj < LIBXSMM_MIN(ojb+cfg.block_fwd_oj,cfg.ofh); oj += cfg.fwd_ofh_rb) {
                      for (oi = 0; oi < cfg.ofw; oi += cfg.fwd_ofw_rb) {
                        /* Prepare batch-reduce kernel arguments */
                        if (cfg.pack_input == 1) {
                          ij_use = oj;
                          ii_use = oi;
                        } else {
                          ij_use = oj * cfg.u;
                          ii_use = oi * cfg.v;
                        }
                        oi_use = oi;
                        oj_use = oj;

                        n_blocks = cfg.blocksifm_blocking * cfg.R * cfg.S;
                        if ((fuse_relu || fuse_colbias) && (ifm1 + cfg.blocksifm_blocking >= cfg.blocksifm) ) {
                          gemm_param_ext.op.tertiary = &n_blocks;
                          gemm_param_ext.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                          gemm_param_ext.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                          gemm_param_ext.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                          if (fuse_colbias) {
                            gemm_param_ext.d.primary = (void*)&LIBXSMM_VLA_ACCESS(2, colbias, ofm1, 0, cfg.ofmblock);
                          }
                          if (cfg.R == 1 && cfg.S == 1) {
                            cfg.fwd_compute_kernel_strd_fused_bf16.gemm_ext( &gemm_param_ext );
                          } else {
                            cfg.fwd_compute_kernel_offs_fused_bf16.gemm_ext( &gemm_param_ext );
                          }
                        } else {
                          gemm_param.op.tertiary = &n_blocks;
                          gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                          gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                          gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                          if (cfg.R == 1 && cfg.S == 1) {
                            cfg.fwd_compute_kernel_strd_bf16.gemm( &gemm_param );
                          } else {
                            cfg.fwd_compute_kernel_offs_bf16.gemm( &gemm_param );
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
    }

    if (cfg.loop_order == 1) {
      for (img = my_img_start; img < my_img_end; img++) {
        for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += cfg.block_fwd_ofm) {
          for (ojb = 0; ojb < cfg.ofh; ojb += cfg.block_fwd_oj) {
            for (oj = ojb; oj < LIBXSMM_MIN(ojb+cfg.block_fwd_oj,cfg.ofh); oj += cfg.fwd_ofh_rb) {
              for (oi = 0; oi < cfg.ofw; oi += cfg.fwd_ofw_rb) {
                for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+cfg.block_fwd_ofm, my_ofm_end); ofm1++ ) {
                  if ((cfg.overwrite_output > 0) && (cfg.avoid_acc_load == 0) && (oj == 0) && (oi == 0)) {
                    /* set output feature map to zero */
                    unary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, 0, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                    cfg.ofh_x_ofw_x_ofmblock_zero_kernel_bf16( &unary_param );
                  }
                  for (ifmb = 0; ifmb < cfg.blocksifm; ifmb += cfg.block_fwd_ifm) {
                    for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+cfg.block_fwd_ifm, cfg.blocksifm); ifm1 += cfg.blocksifm_blocking) {
                      /* Prepare batch-reduce kernel arguments */
                      if (cfg.pack_input == 1) {
                        ij_use = oj;
                        ii_use = oi;
                      } else {
                        ij_use = oj * cfg.u;
                        ii_use = oi * cfg.v;
                      }
                      oi_use = oi;
                      oj_use = oj;

                      n_blocks = cfg.blocksifm_blocking * cfg.R * cfg.S;
                      if ((fuse_relu || fuse_colbias) && (ifm1 + cfg.blocksifm_blocking >= cfg.blocksifm) ) {
                        gemm_param_ext.op.tertiary = &n_blocks;
                        gemm_param_ext.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                        gemm_param_ext.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                        gemm_param_ext.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                        if (fuse_colbias) {
                          gemm_param_ext.d.primary = (void*)&LIBXSMM_VLA_ACCESS(2, colbias, ofm1, 0, cfg.ofmblock);
                        }
                        if (cfg.R == 1 && cfg.S == 1) {
                          cfg.fwd_compute_kernel_strd_fused_bf16.gemm_ext( &gemm_param_ext );
                        } else {
                          cfg.fwd_compute_kernel_offs_fused_bf16.gemm_ext( &gemm_param_ext );
                        }
                      } else {
                        gemm_param.op.tertiary = &n_blocks;
                        gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                        gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                        gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                        if (cfg.R == 1 && cfg.S == 1) {
                          cfg.fwd_compute_kernel_strd_bf16.gemm( &gemm_param );
                        } else {
                          cfg.fwd_compute_kernel_offs_bf16.gemm( &gemm_param );
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
  }

  if (cfg.zero_fwd_output_rim > 0) {
    /* number of tasks that could be run in parallel */
    const int work = cfg.N * cfg.blocksofm * cfg.ofhp;
    /* compute chunk size */
    const int chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
    int imgofm1ofh;
    LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, full_output, (libxsmm_bfloat16*)out_act_ptr, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);

    for (imgofm1ofh = thr_begin; imgofm1ofh < thr_end; ++imgofm1ofh) {
      img = imgofm1ofh / (cfg.blocksofm*cfg.ofhp);
      if (cfg.N > 1) {
        oj = (imgofm1ofh % (cfg.blocksofm*cfg.ofhp))/cfg.blocksofm;
        ofm1 = (imgofm1ofh % (cfg.blocksofm*cfg.ofhp))%cfg.blocksofm;
      } else {
        oj = (imgofm1ofh % (cfg.blocksofm*cfg.ofhp))%cfg.ofhp;
        ofm1 = (imgofm1ofh % (cfg.blocksofm*cfg.ofhp))/cfg.ofhp;
      }

      if ( (oj < cfg.pad_h_out) || (oj >= (cfg.H+cfg.pad_h_out))) {
        for (oi = 0; oi < cfg.ofwp; oi++) {
          unary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(  5, full_output, img, ofm1, oj, oi, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
          cfg.ofmblock_zero_kernel_bf16( &unary_param );
        }
      } else {
        for (oi = 0; oi < cfg.pad_w_out; oi++) {
          unary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(  5, full_output, img, ofm1, oj, oi, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
          cfg.ofmblock_zero_kernel_bf16( &unary_param );
        }
        for (oi = cfg.W + cfg.pad_w_out; oi < cfg.W + 2*cfg.pad_w_out; oi++) {
          unary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(  5, full_output, img, ofm1, oj, oi, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
          cfg.ofmblock_zero_kernel_bf16( &unary_param );
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

