/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/

#include <libxsmm_dnn_conv.h>

LIBXSMM_API void libxsmm_dnn_conv_upd_exec_bf16( libxsmm_dnn_conv_config cfg, const libxsmm_bfloat16* in_act_ptr, const libxsmm_bfloat16* dout_act_ptr, libxsmm_bfloat16* dfilter_ptr,
    unsigned char* bias_ptr, int start_tid, int my_tid, void* scratch ) {
  int img, my_img_start, my_img_end, ofmb, ifmb, ofm1, ifm1, ifm2, ofm2, oj, oi, ii, ij, kj, ki, j, img_block_size = 1, my_ofm_start, my_ofm_end, my_ifm_start, my_ifm_end, block_ofm, block_ifm, pix;
  /* computing first logical thread */
  const int ltid = my_tid - start_tid;
  libxsmm_gemm_param        gemm_param;
  libxsmm_meltw_unary_param unary_param;

  const int IFWP = (cfg.upd_padding_copy == 1) ? cfg.ifwp + 2*cfg.pad_w :  cfg.ifwp;
  const int IFHP = (cfg.upd_padding_copy == 1) ? cfg.ifhp + 2*cfg.pad_h :  cfg.ifhp;
  const int OFWP = (cfg.upd_padding_copy == 1) ? cfg.ofwp + 2*cfg.pad_w :  cfg.ofwp;
  const int OFHP = (cfg.upd_padding_copy == 1) ? cfg.ofhp + 2*cfg.pad_h :  cfg.ofhp;

  libxsmm_bfloat16 *const out = (libxsmm_bfloat16*)dout_act_ptr + ((size_t)cfg.pad_h_out * cfg.ofwp + cfg.pad_w_out) * cfg.ofmblock;
  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, output, (const libxsmm_bfloat16*)out, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, input, (const libxsmm_bfloat16*)in_act_ptr, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);

  libxsmm_bfloat16 *weight_ptr = (libxsmm_bfloat16*)((char*)scratch + cfg.upd_filter_scratch_offset) + ltid * cfg.C * cfg.K * cfg.R * cfg.S;

  libxsmm_bfloat16 *filter_dst_ptr = (cfg.weight_copies > 1) ? (libxsmm_bfloat16*)weight_ptr : (libxsmm_bfloat16*)dfilter_ptr;
  LIBXSMM_VLA_DECL(7, libxsmm_bfloat16, weight_dst, (libxsmm_bfloat16*)filter_dst_ptr, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock/2, cfg.ofmblock, 2);

  /* This intermediate tensor is used when pixels are NOT fully accumulated  */
  float *weight_ptr_f32 = (float*) ((char*)scratch + cfg.upd_lp_filter_full_scratch_offset) + ltid * cfg.C * cfg.K * cfg.R * cfg.S;

  LIBXSMM_VLA_DECL(6, float, weight_private_f32, (float*)weight_ptr_f32, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
  /* Accumulation scratch is used when pixels are ully accumulated  */
  libxsmm_bfloat16 *filter_scratch = (libxsmm_bfloat16*)((char*)scratch + cfg.upd_lp_filter_full_scratch_offset) + ltid * cfg.ofmblock * cfg.ifmblock * 2;

  LIBXSMM_VLA_DECL(2, float, filter_tmp, (float*)filter_scratch, cfg.ofmblock);

  libxsmm_bfloat16 *scratch_tr_input = (libxsmm_bfloat16*)((char*)scratch + cfg.upd_lp_input_full_scratch_offset);
  libxsmm_bfloat16 *zero_ptr_in;
  libxsmm_bfloat16 *zero_ptr_out;
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, tr_input, (libxsmm_bfloat16*) scratch_tr_input, cfg.blocksifm, cfg.ifmblock, cfg.input_pixels);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, tr_input_2, (libxsmm_bfloat16*) scratch_tr_input, cfg.blocksifm, cfg.ifmblock, IFHP, cfg.ifwp_extended);

  libxsmm_bfloat16 *scratch_tr_output = (libxsmm_bfloat16*)((char*)scratch + cfg.upd_lp_output_full_scratch_offset);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, tr_output, (libxsmm_bfloat16*) scratch_tr_output, cfg.blocksofm, cfg.output_pixels/2, cfg.ofmblock, 2);
  LIBXSMM_VLA_DECL(6, libxsmm_bfloat16, tr_output_2, (libxsmm_bfloat16*) scratch_tr_output, cfg.blocksofm, OFHP, cfg.ofwp_extended/2, cfg.ofmblock, 2);

  /* transpose, copy and reduce work-related variables  */
  float *dst_ptr;

  /* Batch reduce related variables */
  unsigned long long n_blocks;

  const int img_work = cfg.N;
  const int img_chunksize = (img_work % cfg.threads == 0) ? (img_work / cfg.threads) : (img_work / cfg.threads) + 1;
  my_img_start = (ltid * img_chunksize < img_work) ? (ltid * img_chunksize) : img_work;
  my_img_end = ((ltid + 1) * img_chunksize < img_work) ? ((ltid + 1) * img_chunksize) : img_work;

  libxsmm_barrier_init(cfg.barrier, ltid);

  if (cfg.upd_linearized_pixels == 1) {
    /* First transpose input and output */
    if (cfg.use_hybrid_imgofm_parallelization == 1) {
      if (cfg.upd_pack_input_upfront == 0) {
        for (img = my_img_start; img < my_img_end; img++) {
          if (cfg.upd_padding_copy == 1) {
            for (ifm1 = 0; ifm1 < cfg.blocksifm; ifm1++) {
              zero_ptr_in = (libxsmm_bfloat16*) &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, 0, cfg.blocksifm, cfg.ifmblock, cfg.input_pixels);
              unary_param.out.primary = (void*) zero_ptr_in;
              cfg.zero_ifmblock_input_pixels_bf16( &unary_param );
              for (ij = 0; ij < cfg.ifhp; ij++) {
                unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, 0, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
                unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, (ij + cfg.pad_h) * IFWP + (0 + cfg.pad_w), cfg.blocksifm, cfg.ifmblock, cfg.input_pixels);
                cfg.transpose_input_pixels_bf16( &unary_param );
              }
            }
          } else {
            for (ifm1 = 0; ifm1 < cfg.blocksifm; ifm1++) {
              for (ij = 0; ij < cfg.ifhp; ij++) {
                unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, 0, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
                unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, ij * cfg.ifwp, cfg.blocksifm, cfg.ifmblock, cfg.input_pixels);
                cfg.transpose_input_pixels_bf16( &unary_param );
              }
            }
          }
        }
      } else {
        for (img = my_img_start; img < my_img_end; img++) {
          for (ifm1 = 0; ifm1 < cfg.blocksifm; ifm1++) {
            for (ij = 0; ij < cfg.ifhp/cfg.u; ij++) {
              unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij*cfg.u, 0, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
              unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, ij * (cfg.ifwp/cfg.v), cfg.blocksifm, cfg.ifmblock, cfg.input_pixels);
              cfg.transposeNpack_input_pixels_bf16( &unary_param );
            }
          }
        }
      }

      if (cfg.upd_padding_copy == 1) {
        for (img = my_img_start; img < my_img_end; img++) {
          for (ofm1 = 0; ofm1 < cfg.blocksofm; ofm1++) {
            zero_ptr_out = (libxsmm_bfloat16*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, 0, 0, 0, cfg.blocksofm, cfg.output_pixels/2, cfg.ofmblock, 2);
            unary_param.out.primary = (void*) zero_ptr_out;
            cfg.zero_ofmblock_output_pixels_bf16( &unary_param );
            if (cfg.ofwp % 2 == 0) {
              for (oj = 0; oj < cfg.ofhp; oj++) {
                unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, (oj*OFWP)/2, 0, 0, cfg.blocksofm, cfg.output_pixels/2, cfg.ofmblock, 2);
                cfg.vnni_output_w_pixels_bf16( &unary_param );
              }
            } else {
#if 0
              for (oj = 0; oj < cfg.ofhp; oj++) {
                unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, (oj*OFWP)/2, 0, 0, cfg.blocksofm, cfg.output_pixels/2, cfg.ofmblock, 2);
                cfg.vnni_output_w2_pixels_bf16( &unary_param );
                for (ofm2 = 0; ofm2 < cfg.ofmblock; ofm2++) {
                  LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, (oj*OFWP+cfg.ofwp-1)/2, ofm2, (oj*OFWP+cfg.ofwp-1)%2, cfg.blocksofm, cfg.output_pixels/2, cfg.ofmblock, 2) =
                    LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, cfg.ofwp-1, ofm2, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                }
              }
#else
              for (oj = 0; oj < cfg.ofhp; oj++) {
                for (oi = 0; oi < cfg.ofwp; oi++) {
                  for (ofm2 = 0; ofm2 < cfg.ofmblock; ofm2++) {
                    LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, (oj*OFWP+oi)/2, ofm2, (oj*OFWP+oi)%2, cfg.blocksofm, cfg.output_pixels/2, cfg.ofmblock, 2) =
                      LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, ofm2, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                  }
                }
              }
#endif
            }
          }
        }
      } else {
        for (img = my_img_start; img < my_img_end; img++) {
          for (ofm1 = 0; ofm1 < cfg.blocksofm; ofm1++) {
            unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
            unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, 0, 0, 0, cfg.blocksofm, cfg.output_pixels/2, cfg.ofmblock, 2);
            cfg.vnni_output_compute_pixels_bf16( &unary_param );
            if (cfg.upd_remaining_pixels > 0) {
              unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, (cfg.compute_pixels+1)/2, 0, 0, cfg.blocksofm, cfg.output_pixels/2, cfg.ofmblock, 2);
              cfg.vnni_output_zero_remaining_pixels_bf16( &unary_param );
            }
          }
        }
      }
    }
  } else {
    if (cfg.upd_trans_w_only == 0) {
      if (cfg.on_the_fly_input_packing == 0) {
        for (img = my_img_start; img < my_img_end; img++) {
          zero_ptr_in = (libxsmm_bfloat16*) &LIBXSMM_VLA_ACCESS(5, tr_input_2, img, 0, 0, 0, 0, cfg.blocksifm, cfg.ifmblock, IFHP, cfg.ifwp_extended);
          unary_param.out.primary = (void*) zero_ptr_in;
          cfg.zero_ifmblock_input_pixels_extended_bf16( &unary_param );
          for (ifm1 = 0; ifm1 < cfg.blocksifm; ifm1++) {
            for (ij = 0; ij < cfg.ifhp; ij++) {
              unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, 0, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
              unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(5, tr_input_2, img, ifm1, 0, ij, 0, cfg.blocksifm, cfg.ifmblock, IFHP, cfg.ifwp_extended);
              cfg.transpose_input_pixels_ifwp_extended2_bf16( &unary_param );
            }
          }
        }
      }
      for (img = my_img_start; img < my_img_end; img++) {
        for (ofm1 = 0; ofm1 < cfg.blocksofm; ofm1++) {
          for (oj = 0; oj < cfg.ofh; oj++) {
            unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
            unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(6, tr_output_2, img, ofm1, oj, 0, 0, 0, cfg.blocksofm, OFHP, cfg.ofwp_extended/2, cfg.ofmblock, 2);
            cfg.vnni_output_w_pixels_bf16( &unary_param );
          }
        }
      }
    }
  }

  /* Make sure we initialize intermediate weights to zero */
  if (cfg.use_intermediate_f32_wt_tensor == 1 && cfg.use_hybrid_imgofm_parallelization == 0) {
    unary_param.out.primary = (void*) weight_ptr_f32;
    cfg.zero_full_weights_f32( &unary_param );
  }

  if (cfg.upd_linearized_pixels == 0) {
    if (cfg.upd_trans_w_only == 1) {
      n_blocks = cfg.batchreduce_h_pixels;
      for (img = my_img_start; img < my_img_end; img++) {
        for (ofmb = 0; ofmb < cfg.blocksofm; ofmb += cfg.block_upd_ofm) {
          for (oj = 0; oj < cfg.ofh; oj += cfg.batchreduce_h_pixels){
            for (ifmb = 0; ifmb < cfg.blocksifm; ifmb += cfg.block_upd_ifm) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+cfg.block_upd_ofm, cfg.blocksofm); ofm1++ ) {
                /* Transpose output block */
                for (j=0; j < cfg.batchreduce_h_pixels; j++) {
                  unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj+j, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                  unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(6, tr_output_2, img, 0, j, 0, 0, 0, cfg.blocksofm, OFHP, cfg.ofwp_extended/2, cfg.ofmblock, 2);
                  cfg.vnni_output_w_pixels_bf16( &unary_param );
                }
                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+cfg.block_upd_ifm, cfg.blocksifm); ifm1++) {
                  /* Transpose input block */
                  for (j=0; j < cfg.batchreduce_h_pixels; j++) {
                    unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, oj+j, 0, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
                    unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(5, tr_input_2, img, 0, 0, j, 0, cfg.blocksifm, cfg.ifmblock, IFHP, cfg.ifwp_extended);
                    cfg.transpose_input_pixels_ifwp_extended_bf16( &unary_param );
                  }
                  for (kj = 0; kj < cfg.R; ++kj) {
                    for (ki = 0; ki < cfg.S; ++ki) {
                      /* Determine if destination is the accumulation scratch or the intermediate fp32 weight tensor */
                      if (cfg.use_intermediate_f32_wt_tensor == 1) {
                        dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(6, weight_private_f32, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                      } else {
                        dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(2, filter_tmp, 0, 0, cfg.ofmblock);
                      }
                      gemm_param.op.tertiary  = (void*) &n_blocks;
                      gemm_param.a.primary    = (void*) &LIBXSMM_VLA_ACCESS(6, tr_output_2, img, 0, 0, 0, 0, 0, cfg.blocksofm, OFHP, cfg.ofwp_extended/2, cfg.ofmblock, 2);
                      gemm_param.b.primary    = (void*) &LIBXSMM_VLA_ACCESS(5, tr_input_2, img, 0, 0, 0, 0, cfg.blocksifm, cfg.ifmblock, IFHP, cfg.ifwp_extended);
                      gemm_param.c.primary    = (void*) dst_ptr;
                      cfg.upd_compute_kernel1_bf16f32.gemm( &gemm_param );

                      /* Convert fully accumulated buffer to bf16 weight buffer in case of full accumulation has happened */
                      if ((oj + cfg.batchreduce_h_pixels >= cfg.ofh) && (img == my_img_end - 1)) {
                        unary_param.in.primary = (void*) dst_ptr;
                        unary_param.out.primary= (void*) dst_ptr;
                        cfg.upd_weight_cvt_f32bf16( &unary_param );
                        unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(7, weight_dst, ofm1, ifm1, kj, ki, 0, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock/2, cfg.ofmblock, 2);
                        cfg.upd_weight_vnni_format_bf16( &unary_param );
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
      int fast_trans = (cfg.ofw == 112 && cfg.v == 2 && cfg.ifmblock == 4 && cfg.batchreduce_h_pixels == 1) ? 1 : 0;
      n_blocks = cfg.batchreduce_h_pixels;
      for (img = my_img_start; img < my_img_end; img++) {
        for (ofmb = 0; ofmb < cfg.blocksofm; ofmb += cfg.block_upd_ofm) {
          for (oj = 0; oj < cfg.ofh; oj += cfg.batchreduce_h_pixels){
            for (ifmb = 0; ifmb < cfg.blocksifm; ifmb += cfg.block_upd_ifm) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+cfg.block_upd_ofm, cfg.blocksofm); ofm1++ ) {
                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+cfg.block_upd_ifm, cfg.blocksifm); ifm1++) {
                  for (kj = 0; kj < cfg.R; ++kj) {
                    for (ki = 0; ki < cfg.S; ++ki) {
                      /* Determine if destination is the accumulation scratch or the intermediate fp32 weight tensor */
                      if (cfg.use_intermediate_f32_wt_tensor == 1) {
                        dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(6, weight_private_f32, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                      } else {
                        dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(2, filter_tmp, 0, 0, cfg.ofmblock);
                      }

                      /* Copy the input in such a way that we ignore "w-pixels" based on ki value  */
                      if (cfg.on_the_fly_input_packing == 1) {
                        if ((fast_trans == 1) && (cfg.upd_padding_copy == 0)) {
                          for (ij = 0; ij < cfg.batchreduce_h_pixels; ij++) {
                            unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, (oj+ij)*cfg.u+kj, ki, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
                            unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(5, tr_input_2, img, 0, 0, 0, 0, cfg.blocksifm, cfg.ifmblock, IFHP, cfg.ifwp_extended);
                            cfg.transpose_input_pixels_ifwp_strided_extended_bf16( &unary_param );
                          }
                        } else {
                          if (cfg.upd_padding_copy == 1) {
                            for (ij = 0; ij < cfg.batchreduce_h_pixels; ij++) {
                              for (ii = 0; ii < cfg.ofw; ii++) {
                                int j_pixel = (oj+ij)*cfg.u+kj;
                                int i_pixel = ii*cfg.v+ki;
                                if ( (j_pixel >= cfg.pad_h) && (i_pixel >= cfg.pad_w) && (j_pixel < cfg.ifhp+cfg.pad_h) && (i_pixel < cfg.ifwp+cfg.pad_w) ) {
                                  for (ifm2 = 0; ifm2 < cfg.ifmblock; ifm2++) {
                                    LIBXSMM_VLA_ACCESS(5, tr_input_2, img, 0, ifm2, ij, ii, cfg.blocksifm, cfg.ifmblock, IFHP, cfg.ifwp_extended) =
                                      LIBXSMM_VLA_ACCESS(5, input, img, ifm1, (oj+ij)*cfg.u+kj-cfg.pad_h, ii*cfg.v+ki-cfg.pad_w, ifm2, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
                                  }
                                } else {
                                  for (ifm2 = 0; ifm2 < cfg.ifmblock; ifm2++) {
                                    LIBXSMM_VLA_ACCESS(5, tr_input_2, img, 0, ifm2, ij, ii, cfg.blocksifm, cfg.ifmblock, IFHP, cfg.ifwp_extended) = (libxsmm_bfloat16)0;
                                  }
                                }
                              }
                            }
                          } else {
                            for (ij = 0; ij < cfg.batchreduce_h_pixels; ij++) {
                              unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, (oj+ij)*cfg.u+kj, ki, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
                              unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(5, tr_input_2, img, 0, 0, 0, 0, cfg.blocksifm, cfg.ifmblock, IFHP, cfg.ifwp_extended);
                              cfg.transpose_input_pixels_ifwp_strided_extended_bf16( &unary_param );
                            }
                          }
                        }
                      }

                      gemm_param.op.tertiary  = (void*) &n_blocks;
                      gemm_param.a.primary    = (void*) &LIBXSMM_VLA_ACCESS(6, tr_output_2, img, ofm1, oj, 0, 0, 0, cfg.blocksofm, OFHP, cfg.ofwp_extended/2, cfg.ofmblock, 2);
                      if (cfg.on_the_fly_input_packing == 1) {
                        gemm_param.b.primary    = (void*) &LIBXSMM_VLA_ACCESS(5, tr_input_2, img, 0, 0, 0, 0, cfg.blocksifm, cfg.ifmblock, IFHP, cfg.ifwp_extended);
                      } else {
                        gemm_param.b.primary    = (void*) &LIBXSMM_VLA_ACCESS(5, tr_input_2, img, ifm1, 0, oj+kj, ki, cfg.blocksifm, cfg.ifmblock, IFHP, cfg.ifwp_extended);
                      }
                      gemm_param.c.primary    = (void*) dst_ptr;
                      cfg.upd_compute_kernel2_bf16f32.gemm( &gemm_param );

                      /* Convert fully accumulated buffer to bf16 weight buffer in case of full accumulation has happened */
                      if ((oj + cfg.batchreduce_h_pixels >= cfg.ofh) && (img == my_img_end - 1)) {
                        unary_param.in.primary = (void*) dst_ptr;
                        unary_param.out.primary= (void*) dst_ptr;
                        cfg.upd_weight_cvt_f32bf16( &unary_param );
                        unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(7, weight_dst, ofm1, ifm1, kj, ki, 0, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock/2, cfg.ofmblock, 2);
                        cfg.upd_weight_vnni_format_bf16( &unary_param );
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
    if (cfg.use_hybrid_imgofm_parallelization == 1) {
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
      libxsmm_bfloat16 *weight_ptr_group = (cfg.weight_copies > 1) ? (libxsmm_bfloat16*)((char*)scratch + cfg.upd_filter_scratch_offset) + tile_id * cfg.C * cfg.K * cfg.R * cfg.S : (libxsmm_bfloat16*)dfilter_ptr;
      LIBXSMM_VLA_DECL(7, libxsmm_bfloat16, weight_private_group, (libxsmm_bfloat16*)weight_ptr_group, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock/2, cfg.ofmblock, 2);
      /* This intermediate tensor is used when pixels are NOT fully accumulated  */
      float *weight_tile_ptr_f32 = (float*)((char*)scratch + cfg.upd_lp_filter_full_scratch_offset) + tile_id * cfg.C * cfg.K * cfg.R * cfg.S;
      LIBXSMM_VLA_DECL(6, float, weight_private_tile_f32, (float*)weight_tile_ptr_f32, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);

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
      block_ofm = my_ofm_end-my_ofm_start+1;
      block_ifm = my_ifm_end-my_ifm_start+1;
      img_block_size = my_img_end - my_img_start;
      n_blocks = img_block_size;

      /* Make sure we initialize intermediate weights to zero */
      if (cfg.use_intermediate_f32_wt_tensor == 1) {
        for (ofm1 = my_ofm_start; ofm1 < my_ofm_end; ofm1++ ) {
          for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
            for (kj = my_R_start; kj < my_R_end; ++kj) {
              unary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(6, weight_private_tile_f32, ofm1, ifm1, kj, 0, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
              cfg.zero_partial_weights_f32( &unary_param );
            }
          }
        }
      }

      libxsmm_barrier_wait(cfg.barrier, ltid);
      for (img = my_img_start; img < my_img_end; img += img_block_size) {
        for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += block_ofm) {
          for (pix = 0; pix < cfg.n_used_pixels; pix += cfg.pixel_blocking){
            for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += block_ifm) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+block_ofm, my_ofm_end); ofm1++ ) {
                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+block_ifm, my_ifm_end); ifm1++) {
                  for (kj = my_R_start; kj < my_R_end; ++kj) {
                    for (ki = 0; ki < cfg.S; ++ki) {
                      /* Determine if destination is the accumulation scratch or the intermediate fp32 weight tensor */
                      if (cfg.use_intermediate_f32_wt_tensor == 1) {
                        dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(6, weight_private_tile_f32, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                      } else {
                        dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(2, filter_tmp, 0, 0, cfg.ofmblock);
                      }
                      gemm_param.op.tertiary  = (void*) &n_blocks;
                      gemm_param.a.primary    = (void*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, pix/2, 0, 0, cfg.blocksofm, cfg.output_pixels/2, cfg.ofmblock, 2);
                      gemm_param.b.primary    = (void*) &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, pix + kj * IFWP + ki, cfg.blocksifm, cfg.ifmblock, cfg.input_pixels);
                      gemm_param.c.primary    = (void*) dst_ptr;
                      cfg.upd_compute_kernel3_bf16f32.gemm( &gemm_param );

                      /* Convert fully caccumulated buffer to bf16 weight buffer in case of full accumulation has happened */
                      if ((pix + cfg.pixel_blocking >= cfg.n_used_pixels) && (img == my_img_end - img_block_size)) {
                        unary_param.in.primary = (void*) dst_ptr;
                        unary_param.out.primary= (void*) dst_ptr;
                        cfg.upd_weight_cvt_f32bf16( &unary_param );
                        unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(7, weight_private_group, ofm1, ifm1, kj, ki, 0, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock/2, cfg.ofmblock, 2);
                        cfg.upd_weight_vnni_format_bf16( &unary_param );
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
        for (ofmb = 0; ofmb < cfg.blocksofm; ofmb += cfg.block_upd_ofm) {
          for (pix = 0; pix < cfg.n_used_pixels; pix += cfg.pixel_blocking){
            for (ifmb = 0; ifmb < cfg.blocksifm; ifmb += cfg.block_upd_ifm) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+cfg.block_upd_ofm, cfg.blocksofm); ofm1++ ) {
                /* Transpose output block  */
                if (pix == 0 && ifmb == 0) {
                  if (cfg.upd_padding_copy == 1) {
                    zero_ptr_out = (libxsmm_bfloat16*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, 0, 0, 0, cfg.blocksofm, cfg.output_pixels/2, cfg.ofmblock, 2);
                    unary_param.out.primary = (void*) zero_ptr_out;
                    cfg.zero_ofmblock_output_pixels_bf16( &unary_param );
                    if (OFWP % 2 == 1) {
                      for (oj = 0; oj < cfg.ofhp; oj++) {
                        for (oi = 0; oi < cfg.ofwp; oi++) {
                          for (ofm2 = 0; ofm2 < cfg.ofmblock; ofm2++) {
                            LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, (oj*OFWP+oi)/2, ofm2, (oj*OFWP+oi)%2, cfg.blocksofm, cfg.output_pixels/2, cfg.ofmblock, 2) =
                              LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, ofm2, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                          }
                        }
                      }
                    } else {
                      for (oj = 0; oj < cfg.ofhp; oj++) {
                        unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                        unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, (oj*OFWP)/2, 0, 0, cfg.blocksofm, cfg.output_pixels/2, cfg.ofmblock, 2);
                        cfg.vnni_output_w_pixels_bf16( &unary_param );
                      }
                    }
                  } else {
                    unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0, cfg.blocksofm, cfg.ofhp, cfg.ofwp, cfg.ofmblock);
                    unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, 0, 0, 0, cfg.blocksofm, cfg.output_pixels/2, cfg.ofmblock, 2);
                    cfg.vnni_output_compute_pixels_bf16( &unary_param );
                    if (cfg.upd_remaining_pixels > 0) {
                      unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, (cfg.compute_pixels+1)/2, 0, 0, cfg.blocksofm, cfg.output_pixels/2, cfg.ofmblock, 2);
                      cfg.vnni_output_zero_remaining_pixels_bf16( &unary_param );
                    }
                  }
                }
                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+cfg.block_upd_ifm, cfg.blocksifm); ifm1++) {
                  /* Transpose input block */
                  if (pix == 0 && ofmb == 0 && ofm1 == 0) {
                    if (cfg.upd_padding_copy == 1) {
                      zero_ptr_in = (libxsmm_bfloat16*) &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, 0, cfg.blocksifm, cfg.ifmblock, cfg.input_pixels);
                      unary_param.out.primary = (void*) zero_ptr_in;
                      cfg.zero_ifmblock_input_pixels_bf16( &unary_param );
                      for (ij = 0; ij < cfg.ifhp; ij++) {
                        unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, 0, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
                        unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, (ij + cfg.pad_h) * IFWP + (0 + cfg.pad_w), cfg.blocksifm, cfg.ifmblock, cfg.input_pixels);
                        cfg.transpose_input_pixels_bf16( &unary_param );
                      }
                    } else {
                      if (cfg.upd_pack_input_upfront == 0) {
                        for (ij = 0; ij < cfg.ifhp; ij++) {
                          unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, 0, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
                          unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, ij * cfg.ifwp, cfg.blocksifm, cfg.ifmblock, cfg.input_pixels);
                          cfg.transpose_input_pixels_bf16( &unary_param );
                        }
                      } else {
                        for (ij = 0; ij < cfg.ifhp/cfg.u; ij++) {
                          unary_param.in.primary = (void*) &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij*cfg.u, 0, 0, cfg.blocksifm, cfg.ifhp, cfg.ifwp, cfg.ifmblock);
                          unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, ij * (cfg.ifwp/cfg.v), cfg.blocksifm, cfg.ifmblock, cfg.input_pixels);
                          cfg.transposeNpack_input_pixels_bf16( &unary_param );
                        }
                      }
                    }
                  }
                  for (kj = 0; kj < cfg.R; ++kj) {
                    for (ki = 0; ki < cfg.S; ++ki) {
                      /* Determine if destination is the accumulation scratch or the intermediate fp32 weight tensor */
                      if (cfg.use_intermediate_f32_wt_tensor == 1) {
                        dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(6, weight_private_f32, ofm1, ifm1, kj, ki, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock, cfg.ofmblock);
                      } else {
                        dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(2, filter_tmp, 0, 0, cfg.ofmblock);
                      }
                      gemm_param.a.primary    = (void*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, pix/2, 0, 0, cfg.blocksofm, cfg.output_pixels/2, cfg.ofmblock, 2);
                      gemm_param.b.primary    = (void*) &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, pix + kj * IFWP + ki, cfg.blocksifm, cfg.ifmblock, cfg.input_pixels);
                      gemm_param.c.primary    = (void*) dst_ptr;
                      cfg.upd_compute_kernel4_bf16f32.gemm( &gemm_param );

                      /* Convert fully accumulated buffer to bf16 weight buffer in case of full accumulation has happened */
                      if ((pix + cfg.pixel_blocking >= cfg.n_used_pixels) && (img == my_img_end - 1)) {
                        unary_param.in.primary = (void*) dst_ptr;
                        unary_param.out.primary= (void*) dst_ptr;
                        cfg.upd_weight_cvt_f32bf16( &unary_param );
                        unary_param.out.primary= (void*) &LIBXSMM_VLA_ACCESS(7, weight_dst, ofm1, ifm1, kj, ki, 0, 0, 0, cfg.blocksifm, cfg.R, cfg.S, cfg.ifmblock/2, cfg.ofmblock, 2);
                        cfg.upd_weight_vnni_format_bf16( &unary_param );
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

  libxsmm_barrier_wait(cfg.barrier, ltid);
  if (cfg.weight_copies > 1) {
    /* reduce work-related variables  */
    const int fm_blocking = (cfg.ofmblock % 16 == 0) ? 16 : cfg.ofmblock;
    const int reduce_work = cfg.blocksofm * cfg.blocksifm * cfg.R * cfg.S * (cfg.ofmblock/fm_blocking) * cfg.ifmblock;
    const int reduce_chunksize = (reduce_work % cfg.threads == 0) ? (reduce_work / cfg.threads) : (reduce_work / cfg.threads) + 1;
    const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
    const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;
    /* Perform reduction here  */
    libxsmm_barrier_wait(cfg.barrier, ltid);
    unary_param.in.primary  = (void*)((libxsmm_bfloat16*) ((char*)scratch + cfg.upd_filter_scratch_offset) + reduce_thr_begin * fm_blocking);
    unary_param.out.primary = (void*)((libxsmm_bfloat16*) dfilter_ptr + reduce_thr_begin * fm_blocking);
    if ((reduce_thr_end - reduce_thr_begin) == reduce_chunksize) {
      cfg.wt_reduce_kernel0_bf16( &unary_param );
    } else {
      if ((reduce_thr_end - reduce_thr_begin) > 0) {
        cfg.wt_reduce_kernel1_bf16( &unary_param );
      }
    }
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }
}
