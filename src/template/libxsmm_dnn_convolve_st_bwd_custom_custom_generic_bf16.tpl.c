/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/
int img, ofm1, ofm2, ifm1, ifm2, oj, ojj, oi, kj, ki, oi_use, oj_use, ii_use, ij_use, ofmb, ifmb, ojb, myIfmId, nIfmBlocks, ind, task;
int last_ki, last_kj, next_kj;
/* computing first logical thread */
const int ltid = tid - start_thread;
int imgpt = LIBXSMM_UPDIV(handle->desc.N, handle->desc.threads);
int threads_per_image = handle->desc.threads / handle->desc.N;
int my_img_start = LIBXSMM_MIN(ltid * imgpt, handle->desc.N);
int my_img_end = LIBXSMM_MIN((ltid+1) * imgpt, handle->desc.N);
int my_ifm_start = 0;
int my_ifm_end = handle->blocksifm;
int ofmblock_lp = handle->ofmblock/handle->fm_lp_block;
int ifmblock_lp = handle->ifmblock/handle->fm_lp_block;
int lpb = handle->fm_lp_block;

/* Batch reduce related variables */
const element_filter_type *A_ptrs[1024];
const element_input_type  *B_ptrs[1024];
unsigned long long n_blocks;

/* number of tasks for transpose that could be run in parallel */
int transpose_work = handle->blocksifm * handle->blocksofm * handle->desc.R * handle->desc.S;
/* compute chunk size */
int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : ((transpose_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;
/* offset output pointer in case of physical  padding */
const int IFW = (handle->pack_input_bwd == 1) ? handle->ofw : handle->ifwp;
const int IFH = (handle->pack_input_bwd == 1) ? handle->ofh : handle->ifhp;
const int ifwp_scratch = (handle->spread_input_bwd == 1) ? handle->desc.v * handle->bwd_ofw_rb : handle->bwd_ofw_rb;

/* Auxiliary fp32 accumulators */
float *del_inp_ptr;
float *del_inp_fp32 = (float*)((char*)handle->scratch + handle->bwd_lp_input_full_scratch_offset) + ((size_t)handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * handle->ifmblock;
LIBXSMM_VLA_DECL(5, float, del_input_fp32, del_inp_fp32, handle->blocksifm, IFH, IFW, handle->ifmblock);

element_input_type *input_ptr = (handle->pack_input_bwd == 1) ? (element_input_type*)((char*)handle->scratch + handle->bwd_packing_padding_scratch_offset) : (element_input_type*)handle->grad_input->data + ((size_t)handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * handle->ifmblock;
LIBXSMM_VLA_DECL(5, element_input_type, del_input, input_ptr, handle->blocksifm, IFH, IFW, handle->ifmblock);
element_output_type *const out = (element_output_type*)handle->grad_output->data;
LIBXSMM_VLA_DECL(5, const element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);

/* Weight and transpose_weight tensor declaration */
LIBXSMM_VLA_DECL(7, element_filter_type, wt, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, lpb);
LIBXSMM_VLA_DECL(7, element_filter_type, tr_wt, (element_filter_type*)((char*)handle->scratch + handle->bwd_filter_trans_scratch_offset), handle->blocksofm, handle->desc.R, handle->desc.S, ofmblock_lp, handle->ifmblock, lpb);

/* define weight pointer which has the correct format */
element_filter_type* weight_base = ((handle->options & LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE) > 0 ) ? (element_filter_type*)handle->reg_filter_tr->data : (element_filter_type*)((char*)handle->scratch + handle->bwd_filter_trans_scratch_offset);
LIBXSMM_VLA_DECL(7, const element_filter_type, weight, weight_base, handle->blocksofm, handle->desc.R, handle->desc.S, ofmblock_lp, handle->ifmblock, lpb);

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

/* transpose filters, if requested */
if ( (handle->options & LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE) == 0 ) {
  for (task = transpose_thr_begin; task < transpose_thr_end; ++task) {
    ifm1 = task/(handle->blocksofm * handle->desc.R * handle->desc.S);
    ofm1 = (task%(handle->blocksofm * handle->desc.R * handle->desc.S))/(handle->desc.R * handle->desc.S);
    kj =   ((task%(handle->blocksofm * handle->desc.R * handle->desc.S))%(handle->desc.R * handle->desc.S))/handle->desc.S;
    ki =   ((task%(handle->blocksofm * handle->desc.R * handle->desc.S))%(handle->desc.R * handle->desc.S))%handle->desc.S;
    for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
      for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
        LIBXSMM_VLA_ACCESS(7, tr_wt, ifm1, ofm1, handle->desc.R-1-kj , handle->desc.S-1-ki, ofm2/lpb, ifm2, ofm2%lpb, handle->blocksofm, handle->desc.R, handle->desc.S, ofmblock_lp, handle->ifmblock, lpb) =
          LIBXSMM_VLA_ACCESS(7, wt, ofm1, ifm1, kj, ki, ifm2/lpb, ofm2, ifm2%lpb, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, lpb);
      }
    }
  }
  /* wait for transpose to finish */
  libxsmm_barrier_wait(handle->barrier, ltid);
}

if ( imgpt <= 1 ) {
  my_img_start = LIBXSMM_MIN(ltid / threads_per_image, handle->desc.N);
  my_img_end = LIBXSMM_MIN(my_img_start + 1, handle->desc.N);
  myIfmId = ltid % threads_per_image;
  nIfmBlocks = LIBXSMM_UPDIV(handle->blocksifm, threads_per_image);
  my_ifm_start = LIBXSMM_MIN(myIfmId * nIfmBlocks, handle->blocksifm);
  my_ifm_end = LIBXSMM_MIN((myIfmId+1) * nIfmBlocks, handle->blocksifm);
}

if ( handle->use_ifm_parallelization == 1 ) {
  int spread_out = 0;
  if ( handle->desc.N % 8 == 0) {
    spread_out = 8;
  } else if ( handle->desc.N % 4 == 0) {
    spread_out = 4;
  } else if (handle->desc.N % 3 == 0) {
    spread_out = 3;
  } else if (handle->desc.N % 2 == 0) {
    spread_out = 2;
  } else {
    spread_out = 1;
  }
  if ((spread_out > 1) && (handle->desc.threads % spread_out == 0)) {
    int tile_id = ltid / spread_out;
    int ifmpt = LIBXSMM_UPDIV(handle->blocksifm, spread_out);
    int ifm_id = ltid % spread_out;
    imgpt = LIBXSMM_UPDIV(handle->desc.N, handle->desc.threads) * spread_out;
    my_img_start = LIBXSMM_MIN(tile_id * imgpt, handle->desc.N);
    my_img_end = LIBXSMM_MIN((tile_id+1) * imgpt, handle->desc.N);
    my_ifm_start = LIBXSMM_MIN(ifm_id * ifmpt, handle->blocksifm);
    my_ifm_end = LIBXSMM_MIN((ifm_id+1) * ifmpt, handle->blocksifm);
  }
}

if (handle->loop_order == 0) { /* (loop_order == N_Kb_Cb_Hb_k_c_h_w) {*/
  if ( handle->avoid_fmas_in_rim == 1) {
    for (img = my_img_start; img < my_img_end; img++) {
      for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += handle->block_bwd_ifm) {
        for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_bwd_ofm) {
          for (ojb = 0; ojb < handle->ofh; ojb += handle->block_bwd_oj) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_bwd_ifm, my_ifm_end); ifm1++ ) {

              if ( (ofmb == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && handle->avoid_acc_load_bwd == 0 && ojb == 0) {
                /* set output feature map to zero */
                for (oj = 0; oj < handle->ofh; ++oj) {
                  float *temp_ptr = (float*)&LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, oj, 0, 0,  handle->blocksifm, IFH, IFW, handle->ifmblock);
                  for (oi = 0; oi < handle->ofw; ++oi) {
                    LIBXSMM_PRAGMA_SIMD
                      for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                        temp_ptr[ifm2] = (float)0;
                      }
                    temp_ptr += handle->ifmblock;
                  }
                }
              }

              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_bwd_ofm, handle->blocksofm); ofm1 += handle->blocksofm_blocking) {
                for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_bwd_oj,handle->ofh); oj += handle->bwd_ofh_rb) {
                  for (oi = 0; oi < handle->ofw; oi += handle->bwd_ofw_rb) {
                    for (kj = 0; kj < handle->desc.R; kj++) {
                      for (ki = 0; ki < handle->desc.S; ki++) {
                        /* Prepare batch-reduce kernel arguments */
                        ij_use = oj;
                        ii_use = oi;
                        oj_use = oj - (1-handle->desc.pad_h_out);
                        oi_use = oi - (1-handle->desc.pad_w_out);
                        last_kj = handle->desc.R-1;
                        last_ki = handle->desc.S-1;
                        next_kj = kj+1;

                        if (kj == 0 && oj == 0) {
                          /* Do no FLOPS  */
                        } else if (kj == handle->desc.R-1 && oj == handle->ofh-1 ) {
                          /* Do no FLOPS  */
                        } else if ( oi == 0 && ki == 0 ) {
                          ind = 0;
                          for (ofm2 = ofm1; ofm2 < ofm1 + handle->blocksofm_blocking; ofm2++) {
                            A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(7, weight, ifm1, ofm2, kj, ki, 0, 0, 0, handle->blocksofm, handle->desc.R, handle->desc.S, ofmblock_lp, handle->ifmblock, lpb);
                            B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  output,  img, ofm2, oj_use + kj, oi_use + ki + 1, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                            ind++;
                          }
                          n_blocks = ind;
                          if (handle->avoid_acc_load_bwd == 1) {
                            br_gemm_kernel2_bf16bf16(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use + 1, 0, handle->blocksifm, IFH, IFW, handle->ifmblock), &n_blocks);
                          } else {
                            del_inp_ptr = &LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, ij_use, ii_use + 1, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                            br_gemm_kernel2(A_ptrs, B_ptrs, del_inp_ptr, &n_blocks);
                            if (ofm2 == handle->blocksofm &&
                                ((kj == last_kj && ki == last_ki) ||
                                 (next_kj == 0 && next_kj == last_kj && oj == 0) ||
                                 (next_kj == handle->desc.R-1 && next_kj == last_kj && oj == handle->ofh-1))) {
                              for (ojj = 0; ojj < handle->bwd_ofh_rb; ojj++) {
                                LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, ij_use+ojj, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                                    &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use+ojj, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                                    handle->bwd_ofw_rb * handle->ifmblock);
                              }
                            }
                          }
                        } else if (oi == handle->ofw-handle->bwd_ofw_rb  && ki == handle->desc.S-1) {
                          ind = 0;
                          for (ofm2 = ofm1; ofm2 < ofm1 + handle->blocksofm_blocking; ofm2++) {
                            A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(7, weight, ifm1, ofm2, kj, ki, 0, 0, 0, handle->blocksofm, handle->desc.R, handle->desc.S, ofmblock_lp, handle->ifmblock, lpb);
                            B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  output,  img, ofm2, oj_use + kj, oi_use + ki, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                            ind++;
                          }
                          n_blocks = ind;
                          if (handle->avoid_acc_load_bwd == 1) {
                            br_gemm_kernel2_bf16bf16(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock), &n_blocks);
                          } else {
                            del_inp_ptr = &LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                            br_gemm_kernel2(A_ptrs, B_ptrs, del_inp_ptr, &n_blocks);
                            if (ofm2 == handle->blocksofm &&
                                ((kj == last_kj && ki == last_ki) ||
                                 (next_kj == 0 && next_kj == last_kj && oj == 0) ||
                                 (next_kj == handle->desc.R-1 && next_kj == last_kj && oj == handle->ofh-1))) {
                              for (ojj = 0; ojj < handle->bwd_ofh_rb; ojj++) {
                                LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, ij_use+ojj, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                                    &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use+ojj, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                                    handle->bwd_ofw_rb * handle->ifmblock);
                              }
                            }
                          }
                        } else {
                          ind = 0;
                          for (ofm2 = ofm1; ofm2 < ofm1 + handle->blocksofm_blocking; ofm2++) {
                            A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(7, weight, ifm1, ofm2, kj, ki, 0, 0, 0, handle->blocksofm, handle->desc.R, handle->desc.S, ofmblock_lp, handle->ifmblock, lpb);
                            B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  output,  img, ofm2, oj_use + kj, oi_use + ki, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                            ind++;
                          }
                          n_blocks = ind;
                          if (handle->avoid_acc_load_bwd == 1) {
                            br_gemm_kernel_bf16bf16(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock), &n_blocks);
                          } else {
                            del_inp_ptr = &LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                            br_gemm_kernel(A_ptrs, B_ptrs, del_inp_ptr, &n_blocks);
                            if (ofm2 == handle->blocksofm &&
                                ((kj == last_kj && ki == last_ki) ||
                                 (next_kj == 0 && next_kj == last_kj && oj == 0) ||
                                 (next_kj == handle->desc.R-1 && next_kj == last_kj && oj == handle->ofh-1))) {
                              for (ojj = 0; ojj < handle->bwd_ofh_rb; ojj++) {
                                LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, ij_use+ojj, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                                    &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use+ojj, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                                    handle->bwd_ofw_rb * handle->ifmblock);
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
  } else {
    for (img = my_img_start; img < my_img_end; img++) {
      for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += handle->block_bwd_ifm) {
        for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_bwd_ofm) {
          for (ojb = 0; ojb < handle->ofh; ojb += handle->block_bwd_oj) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_bwd_ifm, my_ifm_end); ifm1++ ) {

              if ( (ofmb == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && handle->avoid_acc_load_bwd == 0 && ojb == 0) {
                /* set output feature map to zero */
                for (oj = 0; oj < handle->ofh; ++oj) {
                  float *temp_ptr = (float*)&LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, oj, 0, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                  for (oi = 0; oi < handle->ofw; ++oi) {
                    LIBXSMM_PRAGMA_SIMD
                      for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                        temp_ptr[ifm2] = (float)0;
                      }
                    temp_ptr += handle->ifmblock;
                  }
                }
              }

              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_bwd_ofm, handle->blocksofm); ofm1 += handle->blocksofm_blocking) {
                for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_bwd_oj,handle->ofh); oj += handle->bwd_ofh_rb) {
                  for (oi = 0; oi < handle->ofw; oi += handle->bwd_ofw_rb) {
                    /* Prepare batch-reduce kernel arguments */
                    ij_use = (handle->spread_input_bwd == 1) ? oj * handle->desc.u : oj;
                    ii_use = (handle->spread_input_bwd == 1) ? oi * handle->desc.v : oi;
                    oi_use = oi;
                    oj_use = oj;
                    ind = 0;
                    kj = 0;
                    ki = 0;
                    for (ofm2 = ofm1; ofm2 < ofm1 + handle->blocksofm_blocking; ofm2++) {
                      for (kj = 0; kj < handle->desc.R; kj++) {
                        for (ki = 0; ki < handle->desc.S; ki++) {
                          A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(7, weight, ifm1, ofm2, kj, ki, 0, 0, 0, handle->blocksofm, handle->desc.R, handle->desc.S, ofmblock_lp, handle->ifmblock, lpb);
                          B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  output,  img, ofm2, oj_use + kj, oi_use + ki, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                          ind++;
                        }
                      }
                    }
                    n_blocks = ind;
                    if (handle->avoid_acc_load_bwd == 1) {
                      br_gemm_kernel_bf16bf16(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock), &n_blocks);
                    } else {
                      del_inp_ptr = &LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                      br_gemm_kernel(A_ptrs, B_ptrs, del_inp_ptr, &n_blocks);
                      if (ofm2 == handle->blocksofm && kj == handle->desc.R && ki == handle->desc.S) {
                        for (ojj = 0; ojj < handle->bwd_ofh_rb; ojj++) {
                          LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, ij_use+ojj, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                              &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use+ojj, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                              ifwp_scratch * handle->ifmblock);
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

if (handle->loop_order == 1) { /* (loop_order == N_Kb_Cb_Hb_k_c_h_w) { */
  for (img = my_img_start; img < my_img_end; img++) {
    for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += handle->block_bwd_ifm) {
      for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
        for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_bwd_oj,handle->ofh); oj += handle->bwd_ofh_rb) {
          for (oi = 0; oi < handle->ofw; oi += handle->bwd_ofw_rb) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_bwd_ifm, my_ifm_end); ifm1++ ) {
              for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_bwd_ofm) {
                if ( (ofmb == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && handle->avoid_acc_load_bwd == 0 && ojb == 0 && oj == 0 && oi == 0) {
                  /* set output feature map to zero */
                  for (oj = 0; oj < handle->ofh; ++oj) {
                    float *temp_ptr = (float*)&LIBXSMM_VLA_ACCESS(  5, del_input_fp32, img, ifm1, oj, 0, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                    for (oi = 0; oi < handle->ofw; ++oi) {
                      LIBXSMM_PRAGMA_SIMD
                        for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                          temp_ptr[ifm2] = (float)0;
                        }
                      temp_ptr += handle->ifmblock;
                    }
                  }
                }
                for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_bwd_ofm, handle->blocksofm); ofm1 += handle->blocksofm_blocking) {
                  /* Prepare batch-reduce kernel arguments */
                  ij_use = (handle->spread_input_bwd == 1) ? oj * handle->desc.u : oj;
                  ii_use = (handle->spread_input_bwd == 1) ? oi * handle->desc.v : oi;
                  oi_use = oi;
                  oj_use = oj;
                  ind = 0;
                  kj = 0;
                  ki = 0;
                  for (ofm2 = ofm1; ofm2 < ofm1 + handle->blocksofm_blocking; ofm2++) {
                    for (kj = 0; kj < handle->desc.R; kj++) {
                      for (ki = 0; ki < handle->desc.S; ki++) {
                        A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(7, weight, ifm1, ofm2, kj, ki, 0, 0, 0, handle->blocksofm, handle->desc.R, handle->desc.S, ofmblock_lp, handle->ifmblock, lpb);
                        B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  output,  img, ofm2, oj_use + kj, oi_use + ki, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                        ind++;
                      }
                    }
                  }
                  n_blocks = ind;
                  if (handle->avoid_acc_load_bwd == 1) {
                    br_gemm_kernel_bf16bf16(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock), &n_blocks);
                  } else {
                    del_inp_ptr = &LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                    br_gemm_kernel(A_ptrs, B_ptrs, del_inp_ptr, &n_blocks);
                    if (ofm2 == handle->blocksofm && kj == handle->desc.R && ki == handle->desc.S) {
                      for (ojj = 0; ojj < handle->bwd_ofh_rb; ojj++) {
                        LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, ij_use+ojj, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use+ojj, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                            ifwp_scratch * handle->ifmblock);
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

if (handle->pack_input_bwd == 1) {
  LIBXSMM_VLA_DECL(5, element_input_type, del_input_full, (element_input_type*)handle->grad_input->data + ((size_t)handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * handle->ifmblock, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
  for (img = my_img_start; img < my_img_end; img++) {
    for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
      for (oj = 0; oj < handle->ifhp; oj++) {
        for (oi = 0; oi < handle->ifwp; oi++) {
          if (oi % handle->desc.v != 0 || oj % handle->desc.u != 0) {
            LIBXSMM_PRAGMA_SIMD
              for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
                LIBXSMM_VLA_ACCESS(5,  del_input_full, img, ifm1, oj, oi, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock) = (element_input_type)0;
              }
          } else {
            LIBXSMM_PRAGMA_SIMD
              for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
                LIBXSMM_VLA_ACCESS(5,  del_input_full, img, ifm1, oj, oi, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock) = LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, oj/handle->desc.u, oi/handle->desc.v, ifm2, handle->blocksifm, IFH, IFW, handle->ifmblock);
              }
          }
        }
      }
    }
  }
} else if (handle->spread_input_bwd == 1) {
  LIBXSMM_VLA_DECL(5, element_input_type, del_input_full, (element_input_type*)handle->grad_input->data + ((size_t)handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * handle->ifmblock, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
  for (img = my_img_start; img < my_img_end; img++) {
    for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
      for (oj = 0; oj < handle->ifhp; oj++) {
        for (oi = 0; oi < handle->ifwp; oi++) {
          if (oi % handle->desc.v != 0 || oj % handle->desc.u != 0) {
            LIBXSMM_PRAGMA_SIMD
              for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
                LIBXSMM_VLA_ACCESS(5,  del_input_full, img, ifm1, oj, oi, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock) = (element_input_type)0;
              }
          }
        }
      }
    }
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

