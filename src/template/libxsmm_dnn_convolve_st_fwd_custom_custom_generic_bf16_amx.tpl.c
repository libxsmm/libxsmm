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
int img, ofm1, ifm1, ifm2, /*ofm2, ifm1, ifm2,*/ oj, oi, ij, ii, /*kj, ki, oi_use, oj_use, */ii_use, ij_use, ofmb,/* ifmb,*/ ojb, myOfmId, nOfmBlocks, /*ind, ofm11, ki1, kj1,*/ ojj, /*oii,*/ spread_out = 1;
/*int last_ki, last_kj, next_kj;*/
/* computing first logical thread */
const int ltid = tid - start_thread;
int imgpt = (handle->desc.N + handle->desc.threads - 1)/handle->desc.threads;
int threads_per_image = handle->desc.threads / handle->desc.N;
int my_img_start = LIBXSMM_MIN( ltid * imgpt, handle->desc.N);
int my_img_end = LIBXSMM_MIN( (ltid+1) * imgpt, handle->desc.N);
int my_ofm_start = 0;
int my_ofm_end = handle->blocksofm;
int ifmblock_lp =  handle->ifmblock/handle->fm_lp_block;
/* Batch reduce related variables */
/*const element_filter_type *A_ptrs[1024];*/
/*const element_input_type  *B_ptrs[1024];*/
unsigned long long n_blocks;

/* offset output pointer in case of physical output padding */
element_output_type* out = (element_output_type*)handle->reg_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
/*float* out_fp32 = (float*)handle->scratch6 + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;*/
float* out_ptr;
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
/*LIBXSMM_VLA_DECL(5, float, output_fp32, out_fp32, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);*/
int scratch_ofwp = (handle->fwd_gemm_pixels == (handle->fwd_ofw_rb * handle->fwd_ofh_rb)) ? handle->fwd_ofw_rb : ((handle->fwd_padding_copy == 1) ? handle->ofwp + 2 * handle->desc.pad_w : handle->ofwp);
/*float scratch_stack_fp32[8*16*16];*/
float *out_scratch = (float*)((char*)handle->scratch + handle->fwd_lp_output_full_scratch_offset) + ltid * handle->fwd_gemm_pixels * handle->ofmblock;
LIBXSMM_VLA_DECL(3, float, scratch_fp32, out_scratch, scratch_ofwp, handle->ofmblock);
element_input_type *input_ptr = ((handle->pack_input == 1) || (handle->fwd_padding_copy == 1)) ?(element_input_type*)((char*)handle->scratch + handle->fwd_packing_padding_scratch_offset) : (element_input_type*)handle->reg_input->data;
const int IFW = (handle->fwd_padding_copy == 1) ? handle->ifwp + 2*handle->desc.pad_w : ( (handle->pack_input == 1) ? handle->ofwp : handle->ifwp );
const int IFH = (handle->fwd_padding_copy == 1) ? handle->ifhp + 2*handle->desc.pad_h : ( (handle->pack_input == 1) ? handle->ofhp : handle->ifhp );
LIBXSMM_VLA_DECL(5, element_input_type, input, input_ptr, handle->blocksifm, IFH, IFW, handle->ifmblock);
LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block);

if ( imgpt <= 1 ) {
  my_img_start = LIBXSMM_MIN( ltid / threads_per_image, handle->desc.N);
  my_img_end = LIBXSMM_MIN( my_img_start + 1, handle->desc.N);
  myOfmId = ltid % threads_per_image;
  nOfmBlocks = (handle->blocksofm + threads_per_image - 1) / threads_per_image;
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
    int ofmpt = (handle->blocksofm+spread_out-1)/spread_out;
    int ofm_id = ltid % spread_out;
    imgpt = ((handle->desc.N + handle->desc.threads - 1)/handle->desc.threads) * spread_out;
    my_img_start = LIBXSMM_MIN( tile_id * imgpt, handle->desc.N);
    my_img_end = LIBXSMM_MIN( (tile_id+1) * imgpt, handle->desc.N);
    my_ofm_start = LIBXSMM_MIN( ofm_id * ofmpt, handle->blocksofm);
    my_ofm_end = LIBXSMM_MIN( (ofm_id+1) * ofmpt, handle->blocksofm);
  }
}

n_blocks = (unsigned long long)handle->blocksifm_blocking * handle->desc.R * handle->desc.S;
out_ptr = (float*) &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, 0, 0, 0, scratch_ofwp, handle->ofmblock);

libxsmm_barrier_init(handle->barrier, ltid);

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
  if ( handle->use_ofm_parallelization == 1 ) {
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

/* Execute the tileconfig kernel */
tile_config_kernel(NULL, NULL, NULL);

#if 1
if (handle->desc.R == 1 && handle->desc.S == 1) {
  for (img = my_img_start; img < my_img_end; img++) {
    for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
      for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
        for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
          for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
            ij_use = (handle->pack_input == 1) ? oj : oj * handle->desc.u;
            for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
              ii_use = (handle->pack_input == 1) ? oi : oi * handle->desc.v;
              /* Batch-reduce GEMM call  */
              br_gemm_kernel_strd( &LIBXSMM_VLA_ACCESS(7, weight, ofm1, 0, 0, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block),
                                   &LIBXSMM_VLA_ACCESS(5,  input,  img, 0, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                                   &LIBXSMM_VLA_ACCESS(5, output,  img, ofm1, oj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks);
            }
          }
        }
      }
    }
  }
}
/* @TODO this needs a reasonable fix */
else if ( handle->fwd_ofw_rb*handle->fwd_ofh_rb == handle->fwd_gemm_pixels ) {
  for (img = my_img_start; img < my_img_end; img++) {
    for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
      for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
        for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
          for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
            for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
              /* Batch-reduce GEMM call  */
              br_gemm_kernel_offs_a( &LIBXSMM_VLA_ACCESS(7, weight, ofm1, 0, 0, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block),
                                     &LIBXSMM_VLA_ACCESS(5,  input,  img, 0, oj*handle->desc.u, oi*handle->desc.v, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                                     &LIBXSMM_VLA_ACCESS(5, output,  img, ofm1, oj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                                     &n_blocks, handle->A_offsets, handle->B_offsets);
            }
          }
        }
      }
    }
  }
} else {
  for (img = my_img_start; img < my_img_end; img++) {
    for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
      for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
        for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
          for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
            for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
              /* Batch-reduce GEMM call  */
              br_gemm_kernel_offs_b( &LIBXSMM_VLA_ACCESS(7, weight, ofm1, 0, 0, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block),
                                     &LIBXSMM_VLA_ACCESS(5,  input,  img, 0, oj, oi, 0, handle->blocksifm, IFH, IFW, handle->ifmblock), out_ptr, &n_blocks, handle->A_offsets, handle->B_offsets);
              /* Downconvert accumulated tiles to BF16  */
              for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, ojj, 0, 0, scratch_ofwp, handle->ofmblock), &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj+ojj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), handle->fwd_ofw_rb * handle->ofmblock);
              }
            }
          }
        }
      }
    }
  }
}
#else
if (handle->pack_input == 1) {
  int ifmpt = (handle->blocksifm+spread_out-1)/spread_out;
  int ifm_id = ltid % spread_out;
  int my_ifm_start = LIBXSMM_MIN( ifm_id * ifmpt, handle->blocksifm);
  int my_ifm_end = LIBXSMM_MIN( (ifm_id+1) * ifmpt, handle->blocksifm);
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
  if ( handle->use_ofm_parallelization == 1 ) {
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
      ofm1 = (imgofm1ofh % (handle->blocksofm*handle->ofh))/handle->ofh;
      oj = (imgofm1ofh % (handle->blocksofm*handle->ofh))%handle->ofh;
      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {
        if ( (ifmb == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && handle->avoid_acc_load == 0) {
          /* set output feature map to zero */
          float* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output_fp32, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
          for (oi = 0; oi < handle->ofw; ++oi) {
            LIBXSMM_PRAGMA_SIMD
              for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                temp_ptr[ofm2] = (float)0;
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
                last_kj = handle->desc.R-1;
                last_ki = handle->desc.S-1;
                next_kj = kj+1;

                if (kj == 0 && oj == 0) {
                  /* Do no FLOPS  */
                } else if (kj == handle->desc.R-1 && oj == handle->ofh-1 ) {
                  /* Do no FLOPS  */
                } else if ( oi == 0 && ki == 0 ) {
                  ind = 0;
                  for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                    A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm2, kj, ki, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block);
                    B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki + 1, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                    ind++;
                  }
                  n_blocks = ind;
                  out_ptr = (handle->avoid_acc_load == 1) ? &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, 0, 0, 0, scratch_ofwp, handle->ofmblock) : &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, oi_use + 1, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                  br_gemm_kernel2(A_ptrs, B_ptrs, out_ptr, &n_blocks);
                  if (handle->avoid_acc_load == 1) {
                    for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                      LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, ojj, 0, 0, handle->fwd_ofw_rb, handle->ofmblock),
                          &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use+1, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                          (handle->fwd_ofw_rb-1) * handle->ofmblock);
                    }
                  } else if (ifm2 == handle->blocksifm &&
                      ((kj == last_kj && ki == last_ki) ||
                       (next_kj == 0 && next_kj == last_kj && oj == 0) ||
                       (next_kj == handle->desc.R-1 && next_kj == last_kj && oj == handle->ofh-1))) {
                    for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                      LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16(  &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                          &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                          handle->fwd_ofw_rb * handle->ofmblock);
                    }
                  }
                } else if (oi == handle->ofw-handle->fwd_ofw_rb  && ki == handle->desc.S-1) {
                  ind = 0;
                  for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                    A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm2, kj, ki, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block);
                    B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                    ind++;
                  }
                  n_blocks = ind;
                  out_ptr = (handle->avoid_acc_load == 1) ? &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, 0, 0, 0, scratch_ofwp, handle->ofmblock) : &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                  br_gemm_kernel2(A_ptrs, B_ptrs, out_ptr, &n_blocks);
                  if (handle->avoid_acc_load == 1) {
                    for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                      LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, ojj, 0, 0, handle->fwd_ofw_rb, handle->ofmblock),
                          &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                          (handle->fwd_ofw_rb-1) * handle->ofmblock);
                    }
                  } else if (ifm2 == handle->blocksifm &&
                      ((kj == last_kj && ki == last_ki) ||
                       (next_kj == 0 && next_kj == last_kj && oj == 0) ||
                       (next_kj == handle->desc.R-1 && next_kj == last_kj && oj == handle->ofh-1))) {
                    for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                      LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                          &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                          handle->fwd_ofw_rb * handle->ofmblock);
                    }
                  }
                } else {
                  ind = 0;
                  for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                    A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm2, kj, ki, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block);
                    B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                    ind++;
                  }
                  n_blocks = ind;
                  out_ptr = (handle->avoid_acc_load == 1) ? &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, 0, 0, 0, scratch_ofwp, handle->ofmblock) : &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                  br_gemm_kernel(A_ptrs, B_ptrs, out_ptr, &n_blocks);
                  if (handle->avoid_acc_load == 1) {
                    for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                      LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, ojj, 0, 0, handle->fwd_ofw_rb, handle->ofmblock),
                          &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                          handle->fwd_ofw_rb * handle->ofmblock);
                    }
                  } else if (ifm2 == handle->blocksifm &&
                      ((kj == last_kj && ki == last_ki) ||
                       (next_kj == 0 && next_kj == last_kj && oj == 0) ||
                       (next_kj == handle->desc.R-1 && next_kj == last_kj && oj == handle->ofh-1))) {
                    for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                      LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                          &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                          handle->fwd_ofw_rb * handle->ofmblock);
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
      img = imgofm1ofh / (handle->blocksofm*handle->ofh);
      ofm1 = (imgofm1ofh % (handle->blocksofm*handle->ofh))/handle->ofh;
      oj = (imgofm1ofh % (handle->blocksofm*handle->ofh))%handle->ofh;

      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {

        if ( (ifmb == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && handle->avoid_acc_load == 0) {
          /* set output feature map to zero */
          float* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output_fp32, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
          for (oi = 0; oi < handle->ofw; ++oi) {
            LIBXSMM_PRAGMA_SIMD
              for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                temp_ptr[ofm2] = (float)0;
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
            ind = 0;
            kj = 0;
            ki = 0;
            for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
              for (kj = 0; kj < handle->desc.R; kj++) {
                for (ki = 0; ki < handle->desc.S; ki++) {
                  A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm2, kj, ki, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block);
                  B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                  ind++;
                }
              }
            }
            n_blocks = ind;
            out_ptr = (handle->avoid_acc_load == 1) ? &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, 0, 0, 0, scratch_ofwp, handle->ofmblock) : &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
            br_gemm_kernel(A_ptrs, B_ptrs, out_ptr, &n_blocks);
            if (handle->avoid_acc_load == 1) {
              for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, ojj, 0, 0, handle->fwd_ofw_rb, handle->ofmblock),
                    &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                    handle->fwd_ofw_rb * handle->ofmblock);
              }
            } else if (ifm2 == handle->blocksifm && kj == handle->desc.R && ki == handle->desc.S) {
              for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS( 5, output_fp32, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                    &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                    handle->fwd_ofw_rb * handle->ofmblock);
              }
            }
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
                    float* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output_fp32, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
                    for (oi = 0; oi < handle->ofw; ++oi) {
                      LIBXSMM_PRAGMA_SIMD
                        for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                          temp_ptr[ofm2] = (float)0;
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
                          last_ki = (handle->shuffle_filter_accesses == 1) ?  (handle->desc.S-1+ltid)%handle->desc.S : handle->desc.S-1;
                          last_kj = (handle->shuffle_filter_accesses == 1) ?  (handle->desc.R-1+ltid)%handle->desc.R : handle->desc.R-1;
                          next_kj = (handle->shuffle_filter_accesses == 1) ?  (kj1+1+ltid)%handle->desc.R : kj1+1;

                          if (kj == 0 && oj == 0) {
                            /* Do no FLOPS  */
                          } else if (kj == handle->desc.R-1 && oj == handle->ofh-1 ) {
                            /* Do no FLOPS  */
                          } else if ( oi == 0 && ki == 0 ) {
                            ind = 0;
                            for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm2, kj, ki, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block);
                              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki + 1, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                              ind++;
                            }
                            n_blocks = ind;
                            out_ptr = (handle->avoid_acc_load == 1) ? &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, 0, 0, 0, scratch_ofwp, handle->ofmblock) : &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, oi_use + 1, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                            br_gemm_kernel2(A_ptrs, B_ptrs, out_ptr, &n_blocks);
                            if (handle->avoid_acc_load == 1) {
                              for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                                LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, ojj, 0, 0, handle->fwd_ofw_rb, handle->ofmblock),
                                    &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use+1, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                                    (handle->fwd_ofw_rb-1) * handle->ofmblock);
                              }
                            } else if (ifm2 == handle->blocksifm &&
                                ((kj == last_kj && ki == last_ki) ||
                                 (next_kj == 0 && next_kj == last_kj && oj == 0) ||
                                 (next_kj == handle->desc.R-1 && next_kj == last_kj && oj == handle->ofh-1))) {
                              for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                                LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16(  &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                                    &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                                    handle->fwd_ofw_rb * handle->ofmblock);
                              }
                            }
                          } else if (oi == handle->ofw-handle->fwd_ofw_rb  && ki == handle->desc.S-1) {
                            ind = 0;
                            for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm2, kj, ki, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block);
                              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                              ind++;
                            }
                            n_blocks = ind;
                            out_ptr = (handle->avoid_acc_load == 1) ? &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, 0, 0, 0, scratch_ofwp, handle->ofmblock) : &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                            br_gemm_kernel2(A_ptrs, B_ptrs, out_ptr, &n_blocks);
                            if (handle->avoid_acc_load == 1) {
                              for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                                LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, ojj, 0, 0, handle->fwd_ofw_rb, handle->ofmblock),
                                    &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                                    (handle->fwd_ofw_rb-1) * handle->ofmblock);
                              }
                            } else if (ifm2 == handle->blocksifm &&
                                ((kj == last_kj && ki == last_ki) ||
                                 (next_kj == 0 && next_kj == last_kj && oj == 0) ||
                                 (next_kj == handle->desc.R-1 && next_kj == last_kj && oj == handle->ofh-1))) {
                              for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                                LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                                    &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                                    handle->fwd_ofw_rb * handle->ofmblock);
                              }
                            }
                          } else {
                            ind = 0;
                            for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm2, kj, ki, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block);
                              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                              ind++;
                            }
                            n_blocks = ind;
                            out_ptr = (handle->avoid_acc_load == 1) ? &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, 0, 0, 0, scratch_ofwp, handle->ofmblock) : &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                            br_gemm_kernel(A_ptrs, B_ptrs, out_ptr, &n_blocks);
                            if (handle->avoid_acc_load == 1) {
                              for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                                LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, ojj, 0, 0, handle->fwd_ofw_rb, handle->ofmblock),
                                    &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                                    handle->fwd_ofw_rb * handle->ofmblock);
                              }
                            } else if (ifm2 == handle->blocksifm &&
                                ((kj == last_kj && ki == last_ki) ||
                                 (next_kj == 0 && next_kj == last_kj && oj == 0) ||
                                 (next_kj == handle->desc.R-1 && next_kj == last_kj && oj == handle->ofh-1))) {
                              for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                                LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                                    &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                                    handle->fwd_ofw_rb * handle->ofmblock);
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
        for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
          for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {
            for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
              for (ofm11 = ofmb; ofm11 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm11++ ) {
                ofm1 = (handle->shuffle_filter_accesses == 1) ? (ofm11+ltid)%handle->blocksofm : ofm11;
                if ( (ifmb == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && handle->avoid_acc_load == 0 && ojb == 0) {
                  /* set output feature map to zero */
                  for (oj = 0; oj < handle->ofh; ++oj) {
                    float* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output_fp32, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
                    for (oi = 0; oi < handle->ofw; ++oi) {
                      LIBXSMM_PRAGMA_SIMD
                        for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                          temp_ptr[ofm2] = (float)0;
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
                      ind = 0;
                      kj1 = 0;
                      ki1 = 0;
                      for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                        for (kj1 = 0; kj1 < handle->desc.R; kj1++) {
                          for (ki1 = 0; ki1 < handle->desc.S; ki1++) {
                            ki = (handle->shuffle_filter_accesses == 1) ?  (ki1+ltid)%handle->desc.S : ki1;
                            kj = (handle->shuffle_filter_accesses == 1) ?  (kj1+ltid)%handle->desc.R : kj1;
                            A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm2, kj, ki, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block);
                            B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                            ind++;
                          }
                        }
                      }
                      n_blocks = ind;
                      out_ptr = (handle->avoid_acc_load == 1) ? &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, 0, 0, 0, scratch_ofwp, handle->ofmblock) : &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                      br_gemm_kernel(A_ptrs, B_ptrs, out_ptr, &n_blocks);
                      if (handle->avoid_acc_load == 1) {
                        for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                          LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, ojj, 0, 0, handle->fwd_ofw_rb, handle->ofmblock),
                              &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                              handle->fwd_ofw_rb * handle->ofmblock);
                        }
                      } else if (kj1 == handle->desc.R && ki1 == handle->desc.S && ifm2 == handle->blocksifm) {
                        for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                          LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                              &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                              handle->fwd_ofw_rb * handle->ofmblock);
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
                    float* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output_fp32, img, ofm1, ojj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
                    for (oii = 0; oii < handle->ofw; ++oii) {
                      LIBXSMM_PRAGMA_SIMD
                        for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                          temp_ptr[ofm2] = (float)0;
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
                    ind = 0;
                    kj = 0;
                    ki = 0;
                    for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                      for (kj = 0; kj < handle->desc.R; kj++) {
                        for (ki = 0; ki < handle->desc.S; ki++) {
                          A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm2, kj, ki, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block);
                          B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock);
                          ind++;
                        }
                      }
                    }
                    n_blocks = ind;
                    out_ptr = (handle->avoid_acc_load == 1) ? &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, 0, 0, 0, scratch_ofwp, handle->ofmblock) : &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                    br_gemm_kernel(A_ptrs, B_ptrs, out_ptr, &n_blocks);

                    if (handle->avoid_acc_load == 1) {
                      for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                        LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS( 3, scratch_fp32, ojj, 0, 0, handle->fwd_ofw_rb, handle->ofmblock),
                            &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                            handle->fwd_ofw_rb * handle->ofmblock);
                      }
                    } else if (kj == handle->desc.R && ki == handle->desc.S && ifm2 == handle->blocksifm) {
                      for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                        LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS(5, output_fp32, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                            &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj_use+ojj, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                            handle->fwd_ofw_rb * handle->ofmblock);
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

#if 0
  /* In case we used intermediate fp32 buffer, now downconvert the result to the actual bf16 output */
  if (handle->avoid_acc_load == 0) {
    for (img = my_img_start; img < my_img_end; img++) {
      for (ofm1 = my_ofm_start; ofm1 < my_ofm_end; ofm1++) {
        for (oj = 0; oj < handle->ofh; oj++) {
          LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS( 5, output_fp32, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
              &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
              handle->ofw * handle->ofmblock);
        }
      }
    }
  }
#endif

}
#endif

handle->tilerelease_kernel(NULL, NULL, NULL);
libxsmm_barrier_wait(handle->barrier, ltid);

