/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/
int img, ofm1, ofm2, ifm1, ifm2, oj, oi, kj, ki, oi_use, oj_use, ii_use, ij_use, ofmb, ifmb, ojb, myIfmId, nIfmBlocks, ind, task, ifm1ofm1;
/* computing first logical thread */
const int ltid = tid - start_thread;
int imgpt = LIBXSMM_UPDIV(handle->desc.N, handle->desc.threads);
int threads_per_image = handle->desc.threads / handle->desc.N;
int my_img_start = LIBXSMM_MIN(ltid * imgpt, handle->desc.N);
int my_img_end = LIBXSMM_MIN((ltid+1) * imgpt, handle->desc.N);
int my_ifm_start = 0;
int my_ifm_end = handle->blocksifm;

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
element_input_type *input_ptr = (handle->pack_input_bwd == 1) ? (element_input_type*)((char*)handle->scratch + handle->bwd_packing_padding_scratch_offset) : (element_input_type*)handle->grad_input->data + ((size_t)handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * handle->ifmblock;
LIBXSMM_VLA_DECL(5, element_input_type, del_input, input_ptr, handle->blocksifm, IFH, IFW, handle->ifmblock);
element_output_type *const out = (element_output_type*)handle->grad_output->data;
LIBXSMM_VLA_DECL(5, const element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);

/* Weight and transpose_weight tensor declaration */
LIBXSMM_VLA_DECL(6, element_filter_type, wt, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
LIBXSMM_VLA_DECL(6, element_filter_type, tr_wt, (element_filter_type*)((char*)handle->scratch + handle->bwd_filter_trans_scratch_offset), handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
/* define weight pointer which has the correct format */
element_filter_type* weight_base = ((handle->options & LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE) > 0 ) ? (element_filter_type*)handle->reg_filter_tr->data : (element_filter_type*)((char*)handle->scratch + handle->bwd_filter_trans_scratch_offset);
LIBXSMM_VLA_DECL(6, const element_filter_type, weight, weight_base, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

/* transpose filters, if requested */
if ( (handle->options & LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE) == 0 ) {
  /* Special case of 64x64 transpose with JITed transpose */
  if (handle->ifmblock == 64 && handle->ofmblock == 64) {
    libxsmm_meltwfunction_unary tr_kernel = handle->tr_kernel;
    libxsmm_meltw_unary_param trans_param;
    for (task = transpose_thr_begin; task < transpose_thr_end; ++task) {
      ifm1 = task/(handle->blocksofm * handle->desc.R * handle->desc.S);
      ofm1 = (task%(handle->blocksofm * handle->desc.R * handle->desc.S))/(handle->desc.R * handle->desc.S);
      kj =   ((task%(handle->blocksofm * handle->desc.R * handle->desc.S))%(handle->desc.R * handle->desc.S))/handle->desc.S;
      ki =   ((task%(handle->blocksofm * handle->desc.R * handle->desc.S))%(handle->desc.R * handle->desc.S))%handle->desc.S;
      trans_param.in.primary  = &LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
      trans_param.out.primary = &LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, handle->desc.R-1-kj, handle->desc.S-1-ki, 0, 0, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      tr_kernel( &trans_param );
      trans_param.in.primary  = &LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, 16, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
      trans_param.out.primary = &LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, handle->desc.R-1-kj, handle->desc.S-1-ki, 0, 16, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      tr_kernel( &trans_param );
      trans_param.in.primary  = &LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, 32, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
      trans_param.out.primary = &LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, handle->desc.R-1-kj, handle->desc.S-1-ki, 0, 32, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      tr_kernel( &trans_param );
      trans_param.in.primary  = &LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, 48, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
      trans_param.out.primary = &LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, handle->desc.R-1-kj, handle->desc.S-1-ki, 0, 48, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      tr_kernel( &trans_param );
    }
  } else {
    /* number of tasks for transpose that could be run in parallel */
    transpose_work = handle->blocksifm * handle->blocksofm;
    /* compute chunk size */
    transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : ((transpose_work / handle->desc.threads) + 1);
    /* compute thr_begin and thr_end */
    transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
    transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;
    for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
      ofm1 = ifm1ofm1 / handle->blocksifm;
      ifm1 = ifm1ofm1 % handle->blocksifm;
      for (kj=0; kj < handle->desc.R; kj++) {
        for (ki=0; ki < handle->desc.S; ki++) {
          for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, handle->desc.R-1-kj , handle->desc.S-1-ki, ofm2, ifm2, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock) =
                LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
            }
          }
        }
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
                  element_input_type* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, del_input, img, ifm1, oj, 0, 0,  handle->blocksifm, IFH, IFW, handle->ifmblock));
                  for (oi = 0; oi < handle->ofw; ++oi) {
                    LIBXSMM_PRAGMA_SIMD
                      for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                        temp_ptr[ifm2] = (element_input_type)0;
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

                        if (kj == 0 && oj == 0) {
                          /* Do no FLOPS  */
                        } else if (kj == handle->desc.R-1 && oj == handle->ofh-1 ) {
                          /* Do no FLOPS  */
                        } else if ( oi == 0 && ki == 0 ) {
                          ind = 0;
                          for (ofm2 = ofm1; ofm2 < ofm1 + handle->blocksofm_blocking; ofm2++) {
                            A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm2, kj, ki, 0, 0, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
                            B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  output,  img, ofm2, oj_use + kj, oi_use + ki + 1, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                            ind++;
                          }
                          n_blocks = ind;
                          br_gemm_kernel2(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use + 1, 0, handle->blocksifm, IFH, IFW, handle->ifmblock), &n_blocks);
                        } else if (oi == handle->ofw-handle->bwd_ofw_rb  && ki == handle->desc.S-1) {
                          ind = 0;
                          for (ofm2 = ofm1; ofm2 < ofm1 + handle->blocksofm_blocking; ofm2++) {
                            A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm2, kj, ki, 0, 0, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
                            B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  output,  img, ofm2, oj_use + kj, oi_use + ki, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                            ind++;
                          }
                          n_blocks = ind;
                          br_gemm_kernel2(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock), &n_blocks);
                        } else {
                          ind = 0;
                          for (ofm2 = ofm1; ofm2 < ofm1 + handle->blocksofm_blocking; ofm2++) {
                            A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm2, kj, ki, 0, 0, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
                            B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  output,  img, ofm2, oj_use + kj, oi_use + ki, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                            ind++;
                          }
                          n_blocks = ind;
                          br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock), &n_blocks);
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
                  element_input_type* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, del_input, img, ifm1, oj, 0, 0, handle->blocksifm, IFH, IFW, handle->ifmblock));
                  for (oi = 0; oi < handle->ofw; ++oi) {
                    LIBXSMM_PRAGMA_SIMD
                      for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                        temp_ptr[ifm2] = (element_input_type)0;
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
                    for (ofm2 = ofm1; ofm2 < ofm1 + handle->blocksofm_blocking; ofm2++) {
                      for (kj = 0; kj < handle->desc.R; kj++) {
                        for (ki = 0; ki < handle->desc.S; ki++) {
                          A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm2, kj, ki, 0, 0, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
                          B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  output,  img, ofm2, oj_use + kj, oi_use + ki, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                          ind++;
                        }
                      }
                    }
                    n_blocks = ind;
                    br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock), &n_blocks);
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
      for (ojb = 0; ojb < handle->ofh; ojb += handle->block_bwd_oj) {
        for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_bwd_oj,handle->ofh); oj += handle->bwd_ofh_rb) {
          for (oi = 0; oi < handle->ofw; oi += handle->bwd_ofw_rb) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_bwd_ifm, my_ifm_end); ifm1++ ) {
              for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_bwd_ofm) {
                if ( (ofmb == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && handle->avoid_acc_load_bwd == 0 && ojb == 0 && oj == 0 && oi == 0) {
                  /* set output feature map to zero */
                  for (oj = 0; oj < handle->ofh; ++oj) {
                    element_input_type* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, del_input, img, ifm1, oj, 0, 0, handle->blocksifm, IFH, IFW, handle->ifmblock));
                    for (oi = 0; oi < handle->ofw; ++oi) {
                      LIBXSMM_PRAGMA_SIMD
                        for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                          temp_ptr[ifm2] = (element_input_type)0;
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
                  for (ofm2 = ofm1; ofm2 < ofm1 + handle->blocksofm_blocking; ofm2++) {
                    for (kj = 0; kj < handle->desc.R; kj++) {
                      for (ki = 0; ki < handle->desc.S; ki++) {
                        A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm2, kj, ki, 0, 0, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
                        B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  output,  img, ofm2, oj_use + kj, oi_use + ki, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                        ind++;
                      }
                    }
                  }
                  n_blocks = ind;
                  br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock), &n_blocks);
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

