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

LIBXSMM_API void libxsmm_dnn_conv_upd_exec( libxsmm_dnn_conv_config cfg, const float* in_act_ptr, const float* dout_act_ptr, float* dfilter_ptr,
    unsigned char* bias_ptr, int start_tid, int my_tid, void* scratch ) {
  int img, my_img_start, my_img_end, ofmb, ifmb, ojb, ofm1, ifm1,  oj, oi, ii, ij, kj, ki, img_block_size = 1, my_ofm_start, my_ofm_end, my_ifm_start, my_ifm_end, block_ofm, block_ifm;
  /* computing first logical thread */
  const int ltid = my_tid - start_tid;
  float *const out = (float*)dout_act_ptr + ((size_t)cfg.pad_h_out * cfg.ofwp + cfg.pad_w_out) * cfg.ofmblock;
  LIBXSMM_VLA_DECL(5, const float, output, (const float*)out, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
  const int IFWP = (cfg.upd_padding_copy == 1) ? cfg.ifwp + 2*cfg.pad_w :  cfg.ifwp;
  const int IFHP = (cfg.upd_padding_copy == 1) ? cfg.ifhp + 2*cfg.pad_h :  cfg.ifhp;
  float *input_ptr_to_use = (cfg.upd_padding_copy == 1) ? (float*) ((char*)scratch + cfg.upd_packing_padding_scratch_offset) : (float*)in_act_ptr;
  LIBXSMM_VLA_DECL(5, float, input, (float*) input_ptr_to_use, cfg.blocksifm, IFHP, IFWP, cfg.ifmblock);
  LIBXSMM_VLA_DECL(6, float, weight_global, (float*)dfilter_ptr, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
  float *weight_ptr = (cfg.weight_copies == 1) ? (float*)dfilter_ptr : (float*) ((char*)scratch + cfg.upd_filter_scratch_offset) + ltid * cfg.C * cfg.K * cfg.R * cfg.S;
  LIBXSMM_VLA_DECL(6, float, weight_private, (float*)weight_ptr, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);

  /* Batch reduce related variables */
  libxsmm_gemm_param        gemm_param;
  libxsmm_meltw_unary_param unary_param;
  unsigned long long n_blocks;
  gemm_param.op.tertiary = &n_blocks;
  libxsmm_barrier_init(cfg.barrier, ltid);

  /* physical pad input */
  if (cfg.upd_padding_copy == 1) {
    LIBXSMM_VLA_DECL(5, float, input_src, (float*)in_act_ptr, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
    int imgpt = LIBXSMM_UPDIV(cfg.N, cfg.threads);
    my_img_start = LIBXSMM_MIN(ltid * imgpt, cfg.N);
    my_img_end = LIBXSMM_MIN((ltid+1) * imgpt, cfg.N);
    my_ifm_start = 0;
    my_ifm_end = cfg.blocksifm;
    for (img = my_img_start; img < my_img_end; img++) {
      for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
        /* copy the inner part */
        for (ij = 0; ij < cfg.ifhp+(2*cfg.pad_h); ij++) {
          for (ii = 0; ii < cfg.ifwp+(2*cfg.pad_w); ii++) {
            if ( (ij >= cfg.pad_h) && (ii >= cfg.pad_w) && (ij < cfg.ifhp+cfg.pad_h) && (ii < cfg.ifwp+cfg.pad_w) ) {
              unary_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  input_src,  img, ifm1, ij-cfg.pad_h, ii-cfg.pad_w, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
              unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, ij, ii, 0, cfg.blocksifm, IFHP, IFWP, cfg.ifmblock);
              cfg.ifmblock_copy_kernel_f32( &unary_param );
            } else {
              unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, ij, ii, 0, cfg.blocksifm, IFHP, IFWP, cfg.ifmblock);
              cfg.ifmblock_zero_kernel_f32( &unary_param );
            }
          }
        }
      }
    }
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }

  if (cfg.upd_use_batchreduce == 0 && cfg.upd_linearized_tasklist == 0) {
    /* Parallelize over minibatch */
    const int img_work = cfg.N;
    const int img_chunksize = (img_work % cfg.threads == 0) ? (img_work / cfg.threads) : (img_work / cfg.threads) + 1;
    my_img_start = (ltid * img_chunksize < img_work) ? (ltid * img_chunksize) : img_work;
    my_img_end = ((ltid + 1) * img_chunksize < img_work) ? ((ltid + 1) * img_chunksize) : img_work;

    if (!((img_chunksize == 1) && (cfg.upd_ofh_rb == cfg.ofh) && (cfg.upd_ofw_rb == cfg.ofw))) {
      unary_param.out.primary = (void*)weight_ptr;
      cfg.zero_weights_kernel_f32( &unary_param );
    }

    if (cfg.upd_loop_order == 0) {
      for (img = my_img_start; img < my_img_end; img++) {
        for (ofmb = 0; ofmb < cfg.blocksofm; ofmb += cfg.block_upd_ofm) {
          for (ifmb = 0; ifmb < cfg.blocksifm; ifmb += cfg.block_upd_ifm) {
            for (ojb = 0; ojb < cfg.ofh; ojb += cfg.upd_ofh_rb) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+cfg.block_upd_ofm, cfg.blocksofm); ofm1++ ) {
                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+cfg.block_upd_ifm, cfg.blocksifm); ifm1++) {
                  for (oj = ojb; oj < LIBXSMM_MIN(ojb+cfg.upd_ofh_rb,cfg.ofh); oj+= cfg.upd_ofh_rb) {
                    for (oi = 0; oi < cfg.ofw; oi += cfg.upd_ofw_rb) {
                      for (kj = 0; kj < cfg.R; ++kj) {
                        for (ki = 0; ki < cfg.S; ++ki) {
                          ii = oi * cfg.u + ki;
                          ij = oj * cfg.v + kj;
                          gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                          gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, ii, 0, cfg.blocksifm, IFHP, IFWP, cfg.ifmblock);
                          gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight_private, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                          cfg.upd_compute_kernel_no_linearized_tasklist_f32.gemm( &gemm_param );
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
    if (cfg.upd_loop_order == 1) {
      for (img = my_img_start; img < my_img_end; img++) {
        for (ifmb = 0; ifmb < cfg.blocksifm; ifmb += cfg.block_upd_ifm) {
          for (ofmb = 0; ofmb < cfg.blocksofm; ofmb += cfg.block_upd_ofm) {
            for (ojb = 0; ojb < cfg.ofh; ojb += cfg.upd_ofh_rb) {
              for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+cfg.block_upd_ifm, cfg.blocksifm); ifm1++) {
                for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+cfg.block_upd_ofm, cfg.blocksofm); ofm1++ ) {
                  for (oj = ojb; oj < LIBXSMM_MIN(ojb+cfg.upd_ofh_rb,cfg.ofh); oj+= cfg.upd_ofh_rb) {
                    for (oi = 0; oi < cfg.ofw; oi += cfg.upd_ofw_rb) {
                      for (kj = 0; kj < cfg.R; ++kj) {
                        for (ki = 0; ki < cfg.S; ++ki) {
                          ii = oi * cfg.u + ki;
                          ij = oj * cfg.v + kj;
                          gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                          gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, ii, 0, cfg.blocksifm, IFHP, IFWP, cfg.ifmblock);
                          gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight_private, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                          cfg.upd_compute_kernel_no_linearized_tasklist_f32.gemm( &gemm_param );
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
    if (cfg.upd_linearized_tasklist == 1) {
      /* Amount of work when using linearized view of tasks */
      const int work = cfg.R * cfg.S * cfg.blocksofm * cfg.blocksifm;
      const int chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : (work / cfg.threads) + 1;
      const int work_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
      const int work_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
      int work_item;
      int Cb = cfg.blocksifm;
      int R = cfg.R;
      int S = cfg.S;

      if (cfg.upd_avoid_rim_fmas == 0) {
        const int IFH = (cfg.upd_pack_input == 1) ? cfg.ifhp/cfg.u : IFHP;
        const int IFW = (cfg.upd_pack_input == 1) ? cfg.ifwp/cfg.v : IFWP;
        float *input_ptr_base = (cfg.upd_pack_input == 1) ? (float*) ((char*)scratch + cfg.upd_packing_padding_scratch_offset) : (float*)input_ptr_to_use;
        LIBXSMM_VLA_DECL(5, float, input_use, (float*)input_ptr_base, cfg.blocksifm, IFH, IFW, cfg.ifmblock);

        /* If requested, pack input to avoid strided accesses */
        if (cfg.upd_pack_input == 1) {
          LIBXSMM_VLA_DECL(5, float, input_src, (float*)in_act_ptr, cfg.blocksifm, IFHP, IFWP, cfg.ifmblock);
          const int img_chunk = (cfg.N % cfg.threads == 0) ? cfg.N/cfg.threads : (cfg.N/cfg.threads) + 1;
          const int img_copy_start = LIBXSMM_MIN(ltid*img_chunk, cfg.N);
          const int img_copy_end = LIBXSMM_MIN((ltid+1)*img_chunk, cfg.N);

          for (img = img_copy_start; img < img_copy_end; img++) {
            for (ifm1 = 0; ifm1 < cfg.blocksifm; ifm1++) {
              for (oj = 0; oj < cfg.ofh; oj++) {
                ij = oj * cfg.u;
                unary_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(5, input_src, img, ifm1, ij, 0, 0, cfg.blocksifm, IFHP, IFWP, cfg.ifmblock);
                unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, input_use, img, ifm1, oj, 0, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                cfg.strided_copy_kernel_f32( &unary_param );
              }
            }
          }
          libxsmm_barrier_wait(cfg.barrier, ltid);
        }

        /* Initialize weights to zero */
        if (!((cfg.N == 1) && (cfg.upd_ofh_rb == cfg.ofh) && (cfg.upd_ofw_rb == cfg.ofw))) {
          for (work_item = work_begin; work_item < work_end; work_item++) {
            ofm1 = work_item/(Cb*R*S);
            ifm1 = (work_item%(Cb*R*S))/(R*S);
            kj = ((work_item%(Cb*R*S))%(R*S))/S;
            ki = ((work_item%(Cb*R*S))%(R*S))%S;
            unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
            cfg.zero_ifmblock_x_ofmblock_kernel_f32( &unary_param );
          }
        }

        for (img = 0; img < cfg.N; img++) {
          for (work_item = work_begin; work_item < work_end; work_item++) {
            ofm1 = work_item/(Cb*R*S);
            ifm1 = (work_item%(Cb*R*S))/(R*S);
            kj = ((work_item%(Cb*R*S))%(R*S))/S;
            ki = ((work_item%(Cb*R*S))%(R*S))%S;
            oi = 0;
            ii = ki;
            for (oj = 0; oj < cfg.ofh; oj += cfg.upd_ofh_rb) {
              ij = oj * cfg.u + kj;
              gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
              gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(5, input_use, img, ifm1, ij, ii, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
              gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
              cfg.upd_compute_kernel_linearized_tasklist_f32.gemm( &gemm_param );
            }
          }
        }
      } else {
        for (work_item = work_begin; work_item < work_end; work_item++) {
          ofm1 = work_item/(Cb*R*S);
          ifm1 = (work_item%(Cb*R*S))/(R*S);
          kj = ((work_item%(Cb*R*S))%(R*S))/S;
          ki = ((work_item%(Cb*R*S))%(R*S))%S;
          oi = 0;
          oj = 0;
          ii = oi * cfg.u + ki;
          ij = oj * cfg.v + kj;
          img = 0;
          img_block_size = cfg.N;

          if (kj == 0) {
            n_blocks = img_block_size * (cfg.upd_ofh_rb-1);
            gemm_param.a.primary    = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
            gemm_param.b.primary    = (void*)&LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, ij, ii, 0, cfg.blocksifm, IFHP, IFWP, cfg.ifmblock);
            gemm_param.c.primary    = (void*)&LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
            gemm_param.a.secondary  = (void*)cfg.A_offsets2_upd;
            gemm_param.b.secondary  = (void*)cfg.B_offsets2_upd;
            cfg.upd_compute_kernel_linearized_tasklist_offs_f32.gemm( &gemm_param );
          } else if (ki == 0) {
            n_blocks = img_block_size * cfg.upd_ofh_rb;
            gemm_param.a.primary    = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi + 1, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
            gemm_param.b.primary    = (void*)&LIBXSMM_VLA_ACCESS(5, input,  img, ifm1, ij, ii + 1, 0, cfg.blocksifm, IFHP, IFWP, cfg.ifmblock);
            gemm_param.c.primary    = (void*)&LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
            gemm_param.a.secondary  = (void*)cfg.A_offsets_upd;
            gemm_param.b.secondary  = (void*)cfg.B_offsets_upd;
            cfg.upd_compute_kernel2_linearized_tasklist_offs_f32.gemm( &gemm_param );
          } else if (oi == cfg.ofw-cfg.fwd_ofw_rb  && ki == cfg.S-1) {
            n_blocks = img_block_size * cfg.upd_ofh_rb;
            gemm_param.a.primary    = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
            gemm_param.b.primary    = (void*)&LIBXSMM_VLA_ACCESS(5, input,  img, ifm1, ij, ii, 0, cfg.blocksifm, IFHP, IFWP, cfg.ifmblock);
            gemm_param.c.primary    = (void*)&LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
            gemm_param.a.secondary  = (void*)cfg.A_offsets_upd;
            gemm_param.b.secondary  = (void*)cfg.B_offsets_upd;
            cfg.upd_compute_kernel2_linearized_tasklist_offs_f32.gemm( &gemm_param );
          } else {
            if (kj == cfg.R-1) {
              n_blocks = img_block_size * (cfg.upd_ofh_rb-1);
              gemm_param.a.primary    = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
              gemm_param.b.primary    = (void*)&LIBXSMM_VLA_ACCESS(5, input,  img, ifm1, ij, ii, 0, cfg.blocksifm, IFHP, IFWP, cfg.ifmblock);
              gemm_param.c.primary    = (void*)&LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
              gemm_param.a.secondary  = (void*)cfg.A_offsets3_upd;
              gemm_param.b.secondary  = (void*)cfg.B_offsets3_upd;
              cfg.upd_compute_kernel_linearized_tasklist_offs_f32.gemm( &gemm_param );
            } else {
              n_blocks = img_block_size * cfg.upd_ofh_rb;
              gemm_param.a.primary    = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
              gemm_param.b.primary    = (void*)&LIBXSMM_VLA_ACCESS(5, input,  img, ifm1, ij, ii, 0, cfg.blocksifm, IFHP, IFWP, cfg.ifmblock);
              gemm_param.c.primary    = (void*)&LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
              gemm_param.a.secondary  = (void*)cfg.A_offsets_upd;
              gemm_param.b.secondary  = (void*)cfg.B_offsets_upd;
              cfg.upd_compute_kernel_linearized_tasklist_offs_f32.gemm( &gemm_param );
            }
          }
        }
      }
    } else {
      /* Here we are using batch-reduce kernel and hybrid minibatch/FM parallelization */
      /* FIXME: Hardcoed logic for N=27  */
      int group_size = (cfg.threads == 27 && cfg.N == 27 && cfg.ofw == 14 && cfg.R == 1 && cfg.u == 1 && ltid >= 24) ? 3 : LIBXSMM_UPDIV(cfg.threads, cfg.weight_copies);
      int tile_id = ltid / LIBXSMM_UPDIV(cfg.threads, cfg.weight_copies);
      int tiles = cfg.weight_copies;
      int img_per_tile = LIBXSMM_UPDIV(cfg.N, tiles);
      int my_in_tile_id = ltid % group_size;
      int ifms_per_thread = LIBXSMM_UPDIV(cfg.blocksifm, group_size);
      int ofms_per_thread = LIBXSMM_UPDIV(cfg.blocksofm, group_size);
      int my_R_start = 0;
      int my_R_end = cfg.R;
      float *weight_ptr_group = (cfg.weight_copies > 1) ? (float*) ((char*)scratch + cfg.upd_filter_scratch_offset) + tile_id * cfg.C * cfg.K * cfg.R * cfg.S : (float*)dfilter_ptr;
      LIBXSMM_VLA_DECL(6, float, weight_private_group, (float*)weight_ptr_group, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
      my_img_start = LIBXSMM_MIN(tile_id * img_per_tile, cfg.N);
      my_img_end = LIBXSMM_MIN((tile_id+1) * img_per_tile, cfg.N);
      my_ifm_start = LIBXSMM_MIN(my_in_tile_id * ifms_per_thread, cfg.blocksifm  );
      my_ifm_end = LIBXSMM_MIN((my_in_tile_id+1) * ifms_per_thread, cfg.blocksifm  );
      my_ofm_start = 0;
      my_ofm_end = cfg.blocksofm;
      /* FIXME: Hardcoed logic for N=27  */
      if (cfg.threads == 27 && cfg.N == 27 && cfg.C == 256 && cfg.K == 1024 && cfg.ofh == 14 && cfg.u == 1) {
        my_ofm_start = LIBXSMM_MIN(my_in_tile_id * ofms_per_thread, cfg.blocksofm);
        my_ofm_end = LIBXSMM_MIN((my_in_tile_id+1) * ofms_per_thread, cfg.blocksofm);
        my_ifm_start = 0;
        my_ifm_end = cfg.blocksifm;
      }
      if (cfg.threads == 27 && cfg.N == 27 && cfg.R == 3 && cfg.S == 3 && cfg.ofh == 14) {
        int r_per_tile = LIBXSMM_UPDIV(cfg.R, group_size);
        my_ifm_start = 0;
        my_ifm_end = cfg.blocksifm;
        my_ofm_start = 0;
        my_ofm_end = cfg.blocksofm;
        my_R_start = LIBXSMM_MIN(my_in_tile_id * r_per_tile, cfg.R);
        my_R_end = LIBXSMM_MIN((my_in_tile_id+1) * r_per_tile, cfg.R);
      }
      if (cfg.threads == 92 && cfg.N == 92 && cfg.C == 512 && cfg.K == 512 && cfg.ofh == 7 && cfg.u == 1 && cfg.R == 3) {
        my_ofm_start = LIBXSMM_MIN(my_in_tile_id * ofms_per_thread, cfg.blocksofm);
        my_ofm_end = LIBXSMM_MIN((my_in_tile_id+1) * ofms_per_thread, cfg.blocksofm);
        my_ifm_start = 0;
        my_ifm_end = cfg.blocksifm;
      }
      block_ofm = my_ofm_end-my_ofm_start+1;
      block_ifm = my_ifm_end-my_ifm_start+1;
      img_block_size = my_img_end - my_img_start;

      if (cfg.N != cfg.threads) {
        /* Use "flat" parallelism + reduction */
        const int work = cfg.R * cfg.S * cfg.blocksofm * cfg.blocksifm * cfg.N;
        const int chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : (work / cfg.threads) + 1;
        const int work_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
        const int work_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
        int work_item;
        int Cb = cfg.blocksifm;
        int Kb = cfg.blocksofm;
        int R = cfg.R;
        int S = cfg.S;
        const int IFH = (cfg.upd_pack_input == 1) ? cfg.ifhp/cfg.u : IFHP;
        const int IFW = (cfg.upd_pack_input == 1) ? cfg.ifwp/cfg.v : IFWP;
        float *input_ptr_base = (cfg.upd_pack_input == 1) ? (float*) ((char*)scratch + cfg.upd_packing_padding_scratch_offset) : (float*)input_ptr_to_use;
        LIBXSMM_VLA_DECL(5, float, input_use, (float*)input_ptr_base, cfg.blocksifm, IFH, IFW, cfg.ifmblock);

        /* If requested, pack input to avoid strided accesses */
        if (cfg.upd_pack_input == 1) {
          LIBXSMM_VLA_DECL(5, float, input_src, (float*)in_act_ptr, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
          const int img_chunk = (cfg.N % cfg.threads == 0) ? cfg.N/cfg.threads : (cfg.N/cfg.threads) + 1;
          const int img_copy_start = LIBXSMM_MIN(ltid*img_chunk, cfg.N);
          const int img_copy_end = LIBXSMM_MIN((ltid+1)*img_chunk, cfg.N);
          for (img = img_copy_start; img < img_copy_end; img++) {
            for (ifm1 = 0; ifm1 < cfg.blocksifm; ifm1++) {
              for (oj = 0; oj < cfg.ofh; oj++) {
                ij = oj * cfg.u;
                unary_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(5, input_src, img, ifm1, ij, 0, 0, cfg.blocksifm, IFHP, IFWP, cfg.ifmblock);
                unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, input_use, img, ifm1, oj, 0, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
                cfg.strided_copy_kernel_f32( &unary_param );
              }
            }
          }
          libxsmm_barrier_wait(cfg.barrier, ltid);
        }

        /* Initialize weights to zero */
        if (cfg.upd_ofw_rb != cfg.ofw) {
          for (work_item = work_begin; work_item < work_end; work_item++) {
            img = work_item/(Cb*Kb*R*S);
            ofm1 = (work_item%(Cb*Kb*R*S))/(Cb*R*S);
            ifm1 = ((work_item%(Cb*Kb*R*S))%(Cb*R*S))/(R*S);
            kj = (((work_item%(Cb*Kb*R*S))%(Cb*R*S))%(R*S))/S;
            ki = (((work_item%(Cb*Kb*R*S))%(Cb*R*S))%(R*S))%S;
            {
              float *weight_ptr_current = (cfg.weight_copies > 1) ? (float*) ((char*)scratch + cfg.upd_filter_scratch_offset) + img * cfg.C * cfg.K * cfg.R * cfg.S : (float*)dfilter_ptr;
              LIBXSMM_VLA_DECL(6, float, weight_current, (float*)weight_ptr_current, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
              unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight_current, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
              cfg.zero_ifmblock_x_ofmblock_kernel_f32( &unary_param );
            }
          }
        }

        for (work_item = work_begin; work_item < work_end; work_item++) {
          img = work_item/(Cb*Kb*R*S);
          ofm1 = (work_item%(Cb*Kb*R*S))/(Cb*R*S);
          ifm1 = ((work_item%(Cb*Kb*R*S))%(Cb*R*S))/(R*S);
          kj = (((work_item%(Cb*Kb*R*S))%(Cb*R*S))%(R*S))/S;
          ki = (((work_item%(Cb*Kb*R*S))%(Cb*R*S))%(R*S))%S;
          ii = 0 + ki;
          ij = 0 + kj;
          oj = 0;
          oi = 0;
          {
            float *weight_ptr_current = (cfg.weight_copies > 1) ? (float*) ((char*)scratch + cfg.upd_filter_scratch_offset) + img * cfg.C * cfg.K * cfg.R * cfg.S : (float*)dfilter_ptr;
            LIBXSMM_VLA_DECL(6, float, weight_current, (float*)weight_ptr_current, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
            n_blocks = cfg.ofh;
            gemm_param.a.primary    = (void*)&LIBXSMM_VLA_ACCESS(5, output,   img , ofm1, oj, oi, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
            gemm_param.b.primary    = (void*)&LIBXSMM_VLA_ACCESS(5, input_use, img, ifm1, ij, ii, 0, cfg.blocksifm, IFH, IFW, cfg.ifmblock);
            gemm_param.c.primary    = (void*)&LIBXSMM_VLA_ACCESS(6, weight_current, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
            gemm_param.a.secondary  = (void*)cfg.A_offsets_upd;
            gemm_param.b.secondary  = (void*)cfg.B_offsets_upd;
            cfg.upd_compute_kernel_flat_linearized_tasklist_offs_f32.gemm( &gemm_param );
          }
        }
      } else {
        /* May need to initialized private weights to zero  */
        if (!((cfg.upd_ofh_rb == cfg.ofh) && (cfg.upd_ofw_rb == cfg.ofw))) {
          for (ofm1 = my_ofm_start; ofm1 < my_ofm_end; ofm1++ ) {
            for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
              for (kj = my_R_start; kj < my_R_end; ++kj) {
                for (ki = 0; ki < cfg.S; ++ki) {
                  unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(6, weight_private_group, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                  cfg.zero_ifmblock_x_ofmblock_kernel_f32( &unary_param );
                }
              }
            }
          }
        }
        if (cfg.upd_loop_order == 0) {
          for (img = my_img_start; img < my_img_end; img += img_block_size) {
            for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += block_ofm) {
              for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += block_ifm) {
                for (ojb = 0; ojb < cfg.ofh; ojb += cfg.upd_ofh_rb) {
                  for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+block_ofm, my_ofm_end); ofm1++ ) {
                    for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+block_ifm, my_ifm_end); ifm1++) {
                      for (oj = ojb; oj < LIBXSMM_MIN(ojb+cfg.upd_ofh_rb,cfg.ofh); oj+= cfg.upd_ofh_rb) {
                        for (oi = 0; oi < cfg.ofw; oi += cfg.upd_ofw_rb) {
                          for (kj = my_R_start; kj < my_R_end; ++kj) {
                            for (ki = 0; ki < cfg.S; ++ki) {
                              ii = oi * cfg.u + ki;
                              ij = oj * cfg.v + kj;
                              n_blocks = img_block_size * cfg.upd_ofh_rb;
                              gemm_param.a.primary    = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                              gemm_param.b.primary    = (void*)&LIBXSMM_VLA_ACCESS(5, input,  img, ifm1, ij, ii, 0, cfg.blocksifm, IFHP, IFWP, cfg.ifmblock);
                              gemm_param.c.primary    = (void*)&LIBXSMM_VLA_ACCESS(6, weight_private_group, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                              gemm_param.a.secondary  = (void*)cfg.A_offsets_upd;
                              gemm_param.b.secondary  = (void*)cfg.B_offsets_upd;
                              cfg.upd_compute_kernel_hybrid_linearized_tasklist_offs_f32.gemm( &gemm_param );
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
          for (img = my_img_start; img < my_img_end; img += img_block_size) {
            for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += block_ifm) {
              for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += block_ofm) {
                for (ojb = 0; ojb < cfg.ofh; ojb += cfg.upd_ofh_rb) {
                  for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+block_ifm, my_ifm_end); ifm1++) {
                    for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+block_ofm, my_ofm_end); ofm1++ ) {
                      for (oj = ojb; oj < LIBXSMM_MIN(ojb+cfg.upd_ofh_rb,cfg.ofh); oj+= cfg.upd_ofh_rb) {
                        for (oi = 0; oi < cfg.ofw; oi += cfg.upd_ofw_rb) {
                          for (kj = my_R_start; kj < my_R_end; ++kj) {
                            for (ki = 0; ki < cfg.S; ++ki) {
                              ii = oi * cfg.u + ki;
                              ij = oj * cfg.v + kj;
                              n_blocks = img_block_size * cfg.upd_ofh_rb;
                              gemm_param.a.primary    = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                              gemm_param.b.primary    = (void*)&LIBXSMM_VLA_ACCESS(5, input,  img, ifm1, ij, ii, 0, cfg.blocksifm, IFHP, IFWP, cfg.ifmblock);
                              gemm_param.c.primary    = (void*)&LIBXSMM_VLA_ACCESS(6, weight_private_group, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                              gemm_param.a.secondary  = (void*)cfg.A_offsets_upd;
                              gemm_param.b.secondary  = (void*)cfg.B_offsets_upd;
                              cfg.upd_compute_kernel_hybrid_linearized_tasklist_offs_f32.gemm( &gemm_param );
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
  }

  if (cfg.weight_copies > 1) {
    /* reduce work-related variables  */
    const int fm_blocking = (cfg.ofmblock % 16 == 0) ? 16 : cfg.ofmblock;
    const int reduce_work = cfg.blocksofm * cfg.blocksifm * cfg.R * cfg.S * (cfg.ofmblock/fm_blocking) * cfg.ifmblock;
    const int reduce_chunksize = (reduce_work % cfg.threads == 0) ? (reduce_work / cfg.threads) : (reduce_work / cfg.threads) + 1;
    const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
    const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;
    /* Perform reduction here  */
    libxsmm_barrier_wait(cfg.barrier, ltid);
    unary_param.in.primary  = (void*)((float*) ((char*)scratch + cfg.upd_filter_scratch_offset) + reduce_thr_begin * fm_blocking);
    unary_param.out.primary = (void*)((float*) dfilter_ptr + reduce_thr_begin * fm_blocking);
    if ((reduce_thr_end - reduce_thr_begin) == reduce_chunksize) {
      cfg.wt_reduce_kernel0_f32( &unary_param );
    } else {
      if ((reduce_thr_end - reduce_thr_begin) > 0) {
        cfg.wt_reduce_kernel1_f32( &unary_param );
      }
    }
  }
  libxsmm_barrier_wait(cfg.barrier, ltid);
}

