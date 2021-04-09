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

int img, ofm1, ofm2 = 0, ifm1, ifm2 = 0, oj, oi, kj, ki, oi_use, oj_use, ii_use, ij_use, ofmb, ifmb, ojb, myOfmId, nOfmBlocks, ind, ofm11, ki1, kj1, ojj, oii, ii, ij, spread_out = 1;
/* computing first logical thread */
const int ltid = tid - start_thread;
int imgpt = LIBXSMM_UPDIV(handle->desc.N, handle->desc.threads);
int threads_per_image = handle->desc.threads / handle->desc.N;
int my_img_start = LIBXSMM_MIN(ltid * imgpt, handle->desc.N);
int my_img_end = LIBXSMM_MIN((ltid+1) * imgpt, handle->desc.N);
int my_ofm_start = 0;
int my_ofm_end = handle->blocksofm;

/* Batch reduce related variables */
const element_filter_type *A_ptrs[1024];
const element_input_type  *B_ptrs[1024];
unsigned long long n_blocks;

/* offset output pointer in case of physical output padding */
element_output_type* out = (element_output_type*)handle->reg_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
element_input_type *input_ptr = ( (handle->pack_input == 1) || (handle->fwd_padding_copy == 1) ) ? (element_input_type*)((char*)handle->scratch + handle->fwd_packing_padding_scratch_offset) : (element_input_type*)handle->reg_input->data;
const int IFW = (handle->fwd_padding_copy == 1) ? handle->ifwp + 2*handle->desc.pad_w : ( (handle->pack_input == 1) ? handle->ofwp : handle->ifwp );
const int IFH = (handle->fwd_padding_copy == 1) ? handle->ifhp + 2*handle->desc.pad_h : ( (handle->pack_input == 1) ? handle->ofhp : handle->ifhp );
LIBXSMM_VLA_DECL(5, element_input_type, input, input_ptr, handle->blocksifm, IFH, IFW, handle->ifmblock);
LIBXSMM_VLA_DECL(6, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

if ( imgpt <= 1 ) {
  my_img_start = LIBXSMM_MIN(ltid / threads_per_image, handle->desc.N);
  my_img_end = LIBXSMM_MIN(my_img_start + 1, handle->desc.N);
  myOfmId = ltid % threads_per_image;
  nOfmBlocks = LIBXSMM_UPDIV(handle->blocksofm, threads_per_image);
  my_ofm_start = LIBXSMM_MIN(myOfmId * nOfmBlocks, handle->blocksofm);
  my_ofm_end = LIBXSMM_MIN((myOfmId+1) * nOfmBlocks, handle->blocksofm);
}

if ( handle->use_ofm_parallelization == 1 ) {
  if ( handle->desc.N % 8 == 0) {
    spread_out = 8;
  } else if ( handle->desc.N % 4 == 0) {
    spread_out = 4;
  } else if (handle->desc.N % 2 == 0) {
    spread_out = 2;
  } else if (handle->desc.N % 3 == 0) {
    spread_out = 3;
  } else {
    spread_out = 1;
  }
  if ((spread_out > 1) && (handle->desc.threads % spread_out == 0)) {
    int tile_id = ltid / spread_out;
    int ofmpt = LIBXSMM_UPDIV(handle->blocksofm, spread_out);
    int ofm_id = ltid % spread_out;
    imgpt = LIBXSMM_UPDIV(handle->desc.N, handle->desc.threads) * spread_out;
    my_img_start = LIBXSMM_MIN(tile_id * imgpt, handle->desc.N);
    my_img_end = LIBXSMM_MIN((tile_id+1) * imgpt, handle->desc.N);
    my_ofm_start = LIBXSMM_MIN(ofm_id * ofmpt, handle->blocksofm);
    my_ofm_end = LIBXSMM_MIN((ofm_id+1) * ofmpt, handle->blocksofm);
  }
}

/* remove stride from input */
if (handle->pack_input == 1) {
  int ifmpt = LIBXSMM_UPDIV(handle->blocksifm, spread_out);
  int ifm_id = ltid % spread_out;
  int my_ifm_start = LIBXSMM_MIN(ifm_id * ifmpt, handle->blocksifm);
  int my_ifm_end = LIBXSMM_MIN((ifm_id+1) * ifmpt, handle->blocksifm);
  LIBXSMM_VLA_DECL(5, element_input_type, input_src, (element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
  for (img = my_img_start; img < my_img_end; img++) {
    for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
      for (oj = 0; oj < handle->ofh; oj++) {
        for (oi = 0; oi < handle->ofw; oi++) {
          ij_use = oj * handle->desc.u;
          ii_use = oi * handle->desc.v;
          LIBXSMM_PRAGMA_SIMD
            for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
              LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, oj, oi, ifm2, handle->blocksifm, IFH, IFW, handle->ifmblock) = LIBXSMM_VLA_ACCESS(5,  input_src,  img, ifm1, ij_use, ii_use, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
            }
        }
      }
    }
  }
  if ( handle->use_ofm_parallelization == 1 || handle->desc.N % handle->desc.threads != 0) {
    libxsmm_barrier_wait(handle->barrier, ltid);
  }
}

/* physical pad input */
if (handle->fwd_padding_copy == 1) {
  int ifmpt = LIBXSMM_UPDIV(handle->blocksifm, spread_out);
  int ifm_id = ltid % spread_out;
  int my_ifm_start = LIBXSMM_MIN(ifm_id * ifmpt, handle->blocksifm);
  int my_ifm_end = LIBXSMM_MIN((ifm_id+1) * ifmpt, handle->blocksifm);
  LIBXSMM_VLA_DECL(5, element_input_type, input_src, (element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
  for (img = my_img_start; img < my_img_end; img++) {
    for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
      /* copy the inner part */
      for (ij = 0; ij < handle->ifhp+(2*handle->desc.pad_h); ij++) {
        for (ii = 0; ii < handle->ifwp+(2*handle->desc.pad_w); ii++) {
          if ( (ij >= handle->desc.pad_h) && (ii >= handle->desc.pad_w) && (ij < handle->ifhp+handle->desc.pad_h) && (ii < handle->ifwp+handle->desc.pad_w) ) {
            LIBXSMM_PRAGMA_SIMD
            for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
              LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, ij, ii, ifm2, handle->blocksifm, IFH, IFW, handle->ifmblock) =
                LIBXSMM_VLA_ACCESS(5,  input_src,  img, ifm1, ij-handle->desc.pad_h, ii-handle->desc.pad_w, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
            }
          } else {
            LIBXSMM_PRAGMA_SIMD
            for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
              LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, ij, ii, ifm2, handle->blocksifm, IFH, IFW, handle->ifmblock) = (element_input_type)0;
            }
          }
        }
      }
    }
  }
  if ( handle->use_ofm_parallelization == 1 || handle->desc.N % handle->desc.threads != 0 ) {
    libxsmm_barrier_wait(handle->barrier, ltid);
  }
}

if (handle->use_fallback_fwd_loops == 1) {
  /* number of tasks that could be run in parallel */
  const int work = handle->desc.N * handle->blocksofm * handle->ofh;
  /* compute chunk size */
  const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
  /* compute thr_begin and thr_end */
  const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
  int imgofm1ofh;

  if ( handle->avoid_fmas_in_rim == 1) {
    for (imgofm1ofh = thr_begin; imgofm1ofh < thr_end; ++imgofm1ofh) {
      img = imgofm1ofh / (handle->blocksofm*handle->ofh);
#if 1
      ofm1 = (imgofm1ofh % (handle->blocksofm*handle->ofh))/handle->ofh;
      oj = (imgofm1ofh % (handle->blocksofm*handle->ofh))%handle->ofh;
#else
      oj = (imgofm1ofh % (handle->blocksofm*handle->ofh))/handle->blocksofm;
      ofm1 = (imgofm1ofh % (handle->blocksofm*handle->ofh))%handle->blocksofm;
#endif

      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {
        if ( (ifmb == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && handle->avoid_acc_load == 0) {
          /* set output feature map to zero */
          element_output_type* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
          for (oi = 0; oi < handle->ofw; ++oi) {
            LIBXSMM_PRAGMA_SIMD
              for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                temp_ptr[ofm2] = (element_output_type)0;
              }
            temp_ptr += handle->ofmblock;
          }
        }
        for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, handle->blocksifm); ifm1 += handle->blocksifm_blocking) {
          for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
            for (kj = 0; kj < handle->desc.R; kj++) {
              for (ki = 0; ki < handle->desc.S; ki++) {
                /* Prepare batch-reduce kernel arguments */
                if (handle->pack_input == 1) {
                  ij_use = oj;
                  ii_use = oi;
                } else {
                  ij_use = oj * handle->desc.u - (1-handle->desc.pad_h_in);
                  ii_use = oi * handle->desc.v - (1-handle->desc.pad_w_in);
                }
                oi_use = oi;
                oj_use = oj;

                if (kj == 0 && oj == 0) {
                  /* Do no FLOPS  */
                } else if (kj == handle->desc.R-1 && oj == handle->ofh-1 ) {
                  /* Do no FLOPS  */
                } else if ( oi == 0 && ki == 0 ) {
                  ind = 0;
                  for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                    A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm2, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                    B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki + 1, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                    ind++;
                  }
                  n_blocks = ind;
                  br_gemm_kernel_b_addr(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use + 1, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks);
                } else if (oi == handle->ofw-handle->fwd_ofw_rb  && ki == handle->desc.S-1) {
                  ind = 0;
                  for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                    A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm2, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                    B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                    ind++;
                  }
                  n_blocks = ind;
                  br_gemm_kernel_b_addr(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks);
                } else {
                  ind = 0;
                  for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                    A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm2, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                    B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                    ind++;
                  }
                  n_blocks = ind;
                  br_gemm_kernel_a_addr(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks);
                }
              }
            }
          }
        }
      }
    }
  } else {
    for (imgofm1ofh = thr_begin; imgofm1ofh < thr_end; ++imgofm1ofh) {
      img = imgofm1ofh / (handle->blocksofm*handle->ofh);
#if 1
      ofm1 = (imgofm1ofh % (handle->blocksofm*handle->ofh))/handle->ofh;
      oj = (imgofm1ofh % (handle->blocksofm*handle->ofh))%handle->ofh;
#else
      oj = (imgofm1ofh % (handle->blocksofm*handle->ofh))/handle->blocksofm;
      ofm1 = (imgofm1ofh % (handle->blocksofm*handle->ofh))%handle->blocksofm;
#endif

      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {

        if ( (ifmb == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && handle->avoid_acc_load == 0) {
          /* set output feature map to zero */
          element_output_type* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
          for (oi = 0; oi < handle->ofw; ++oi) {
            LIBXSMM_PRAGMA_SIMD
              for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                temp_ptr[ofm2] = (element_output_type)0;
              }
            temp_ptr += handle->ofmblock;
          }
        }

        for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, handle->blocksifm); ifm1 += handle->blocksifm_blocking) {
          for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
            /* Prepare batch-reduce kernel arguments */
            if (handle->pack_input == 1) {
              ij_use = oj;
              ii_use = oi;
            } else {
              ij_use = oj * handle->desc.u;
              ii_use = oi * handle->desc.v;
            }
            oi_use = oi;
            oj_use = oj;
#if 1
            ind = 0;
            for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
              for (kj = 0; kj < handle->desc.R; kj++) {
                for (ki = 0; ki < handle->desc.S; ki++) {
                  A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm2, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                  B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                  ind++;
                }
              }
            }
            n_blocks = ind;
            br_gemm_kernel_a_addr(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks);
#else
            LIBXSMM_UNUSED( ifm2 );
            LIBXSMM_UNUSED( kj );
            LIBXSMM_UNUSED( ki );
            n_blocks = handle->blocksifm_blocking * handle->desc.R * handle->desc.S;
            if (handle->desc.R == 1 && handle->desc.S == 1) {
              br_gemm_kernel_strd( &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock),
                                   &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                                   &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks );
            } else {
              br_gemm_kernel_offs( &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock),
                                   &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                                   &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks, handle->A_offsets, handle->B_offsets );
            }
#endif
          }
        }
      }
    }
  }

} else {
  if (handle->loop_order == 0) {
    if ( handle->avoid_fmas_in_rim == 1) {
      for (img = my_img_start; img < my_img_end; img++) {
        for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
          for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {
            for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
              for (ofm11 = ofmb; ofm11 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm11++ ) {
                ofm1 = (handle->shuffle_filter_accesses == 1) ? (ofm11+ltid)%handle->blocksofm : ofm11;
                if ( (ifmb == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && handle->avoid_acc_load == 0 && ojb == 0) {
                  /* set output feature map to zero */
                  for (oj = 0; oj < handle->ofh; ++oj) {
                    element_output_type* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
                    for (oi = 0; oi < handle->ofw; ++oi) {
                      LIBXSMM_PRAGMA_SIMD
                        for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                          temp_ptr[ofm2] = (element_output_type)0;
                        }
                      temp_ptr += handle->ofmblock;
                    }
                  }
                }

                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, handle->blocksifm); ifm1 += handle->blocksifm_blocking) {
                  for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
                    for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
                      for (kj1 = 0; kj1 < handle->desc.R; kj1++) {
                        for (ki1 = 0; ki1 < handle->desc.S; ki1++) {
                          /* Prepare batch-reduce kernel arguments */
                          if (handle->pack_input == 1) {
                            ij_use = oj;
                            ii_use = oi;
                          } else {
                            ij_use = oj * handle->desc.u - (1-handle->desc.pad_h_in);
                            ii_use = oi * handle->desc.v - (1-handle->desc.pad_w_in);
                          }
                          oi_use = oi;
                          oj_use = oj;

                          ki = (handle->shuffle_filter_accesses == 1) ?  (ki1+ltid)%handle->desc.S : ki1;
                          kj = (handle->shuffle_filter_accesses == 1) ?  (kj1+ltid)%handle->desc.R : kj1;

                          if (kj == 0 && oj == 0) {
                            /* Do no FLOPS  */
                          } else if (kj == handle->desc.R-1 && oj == handle->ofh-1 ) {
                            /* Do no FLOPS  */
                          } else if ( oi == 0 && ki == 0 ) {
                            ind = 0;
                            for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm2, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki + 1, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                              ind++;
                            }
                            n_blocks = ind;
                            br_gemm_kernel_b_addr(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use + 1, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks);
                          } else if (oi == handle->ofw-handle->fwd_ofw_rb  && ki == handle->desc.S-1) {
                            ind = 0;
                            for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm2, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                              ind++;
                            }
                            n_blocks = ind;
                            br_gemm_kernel_b_addr(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks);
                          } else {
                            ind = 0;
                            for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm2, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                              ind++;
                            }
                            n_blocks = ind;
                            br_gemm_kernel_a_addr(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks);
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
        for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
          for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {
            for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
              for (ofm11 = ofmb; ofm11 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm11++ ) {
                ofm1 = (handle->shuffle_filter_accesses == 1) ? (ofm11+ltid)%handle->blocksofm : ofm11;
                if ( (ifmb == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && handle->avoid_acc_load == 0 && ojb == 0) {
                  /* set output feature map to zero */
                  for (oj = 0; oj < handle->ofh; ++oj) {
                    element_output_type* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
                    for (oi = 0; oi < handle->ofw; ++oi) {
                      LIBXSMM_PRAGMA_SIMD
                        for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                          temp_ptr[ofm2] = (element_output_type)0;
                        }
                      temp_ptr += handle->ofmblock;
                    }
                  }
                }

                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, handle->blocksifm); ifm1 += handle->blocksifm_blocking) {
                  for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
                    for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
                      /* Prepare batch-reduce kernel arguments */
                      if (handle->pack_input == 1) {
                        ij_use = oj;
                        ii_use = oi;
                      } else {
                        ij_use = oj * handle->desc.u;
                        ii_use = oi * handle->desc.v;
                      }
                      oi_use = oi;
                      oj_use = oj;
#if 1
                      ind = 0;
                      for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                        for (kj1 = 0; kj1 < handle->desc.R; kj1++) {
                          for (ki1 = 0; ki1 < handle->desc.S; ki1++) {
                            ki = (handle->shuffle_filter_accesses == 1) ?  (ki1+ltid)%handle->desc.S : ki1;
                            kj = (handle->shuffle_filter_accesses == 1) ?  (kj1+ltid)%handle->desc.R : kj1;
                            A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm2, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                            B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                            ind++;
                          }
                        }
                      }
                      n_blocks = ind;
                      br_gemm_kernel_a_addr(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks);
#else
                      LIBXSMM_UNUSED( ifm2 );
                      LIBXSMM_UNUSED( kj );
                      LIBXSMM_UNUSED( ki );
                      n_blocks = handle->blocksifm_blocking * handle->desc.R * handle->desc.S;
                      if (handle->desc.R == 1 && handle->desc.S == 1) {
                        br_gemm_kernel_strd( &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock),
                                             &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                                             &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks );
                      } else {
                        br_gemm_kernel_offs( &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock),
                                             &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                                             &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks, handle->A_offsets, handle->B_offsets );
                      }
#endif
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

  if (handle->loop_order == 1) {
    for (img = my_img_start; img < my_img_end; img++) {
      for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
        for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
          for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
            for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
                if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && handle->avoid_acc_load == 0 && oj == 0 && oi == 0) {
                  /* set output feature map to zero */
                  for (ojj = 0; ojj < handle->ofh; ++ojj) {
                    element_output_type* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, ojj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
                    for (oii = 0; oii < handle->ofw; ++oii) {
                      LIBXSMM_PRAGMA_SIMD
                        for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                          temp_ptr[ofm2] = (element_output_type)0;
                        }
                      temp_ptr += handle->ofmblock;
                    }
                  }
                }
                for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {
                  for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, handle->blocksifm); ifm1 += handle->blocksifm_blocking) {
                    /* Prepare batch-reduce kernel arguments */
                    if (handle->pack_input == 1) {
                      ij_use = oj;
                      ii_use = oi;
                    } else {
                      ij_use = oj * handle->desc.u;
                      ii_use = oi * handle->desc.v;
                    }
                    oi_use = oi;
                    oj_use = oj;
#if 1
                    ind = 0;
                    for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                      for (kj = 0; kj < handle->desc.R; kj++) {
                        for (ki = 0; ki < handle->desc.S; ki++) {
                          A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm2, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                          B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                          ind++;
                        }
                      }
                    }
                    n_blocks = ind;
                    br_gemm_kernel_a_addr(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks);
#else
                    LIBXSMM_UNUSED( ifm2 );
                    LIBXSMM_UNUSED( kj );
                    LIBXSMM_UNUSED( ki );
                    n_blocks = handle->blocksifm_blocking * handle->desc.R * handle->desc.S;
                    if (handle->desc.R == 1 && handle->desc.S == 1) {
                      br_gemm_kernel_strd( &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock),
                                           &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                                           &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks );
                    } else {
                      br_gemm_kernel_offs( &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock),
                                           &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                                           &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks, handle->A_offsets, handle->B_offsets );
                    }
#endif
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

libxsmm_barrier_wait(handle->barrier, ltid);

