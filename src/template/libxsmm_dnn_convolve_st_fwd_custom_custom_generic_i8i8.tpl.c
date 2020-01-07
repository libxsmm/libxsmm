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

#define LIBXSMM_DNN_CONVOLUTION_FWD_DOWNCONVERT_I32_I8(in, out, length, _vscf) do { \
  int __i = 0; \
  for ( __i = 0; __i < length; __i+= 16) { \
      _mm_store_epi32((char*)out+__i,  _mm512_cvtepi32_epi8(_mm512_cvt_roundps_epi32( _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_load_epi32((int*)in+__i)), _vscf), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))); \
  } \
} while(0)

int img, ofm1, ofm2, ifm1, ifm2, oj, oi, kj, ki, oi_use, oj_use, ii_use, ij_use, ofmb, ifmb, ojb, myOfmId, nOfmBlocks, ind, ofm11, ki1, kj1, ojj, oii, spread_out = 1;
int last_ki, last_kj, next_kj;
/* computing first logical thread */
const int ltid = tid - start_thread;

/* Calclate scaling factor here for output... */
float qscf = (float)(handle->reg_filter->scf + handle->reg_input->scf - handle->reg_output->scf);
__m512  vscf = _mm512_set1_ps(libxsmm_sexp2_i8i(-qscf));

/* number of tasks that could be run in parallel */
const int work = handle->desc.N * handle->blocksofm * handle->ofh;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
int imgofm1ofh;

int imgpt = (handle->desc.N + handle->desc.threads - 1)/handle->desc.threads;
int threads_per_image = handle->desc.threads / handle->desc.N;
int my_img_start = LIBXSMM_MIN( ltid * imgpt, handle->desc.N);
int my_img_end = LIBXSMM_MIN( (ltid+1) * imgpt, handle->desc.N);
int my_ofm_start = 0;
int my_ofm_end = handle->blocksofm;
int ifmblock_lp =  handle->ifmblock/handle->fm_lp_block;
/* Batch reduce related variables */
unsigned long long n_blocks;

/* offset output pointer in case of physical output padding */
int* out_int32 = (int*)handle->scratch6 + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
int* out_scratch = (int*)handle->scratch6 + ((size_t) handle->desc.N * handle->ofwp * handle->ofhp * handle->desc.K + ltid * handle->fwd_ofw_rb * handle->fwd_ofh_rb * handle->ofmblock);
element_output_type *out_ptr;
element_output_type* out = (element_output_type*)handle->reg_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(5, int, output_int32, out_int32, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(3, int, scratch_int32, out_scratch, handle->fwd_ofw_rb, handle->ofmblock);
element_input_type *input_ptr = (handle->pack_input == 1) ?(element_input_type*)handle->scratch1 + handle->blocksifm * handle->ifmblock * handle->blocksofm * handle->ofmblock * handle->desc.R * handle->desc.S : (element_input_type*)handle->reg_input->data;
const int IFW = (handle->pack_input == 1) ? handle->ofwp : handle->ifwp;
const int IFH = (handle->pack_input == 1) ? handle->ofhp : handle->ifhp;
LIBXSMM_VLA_DECL(5, element_input_type, input, input_ptr, handle->blocksifm, IFH, IFW, handle->ifmblock);
LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block);

libxsmm_barrier_init(handle->barrier, ltid);

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

if (handle->avoid_fmas_in_rim == 1) {
  n_blocks = handle->blocksifm_blocking;
  for (imgofm1ofh = thr_begin; imgofm1ofh < thr_end; ++imgofm1ofh) {
    img = imgofm1ofh / (handle->blocksofm*handle->ofh);
    ofm1 = (imgofm1ofh % (handle->blocksofm*handle->ofh))/handle->ofh;
    oj = (imgofm1ofh % (handle->blocksofm*handle->ofh))%handle->ofh;
    for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {
      if ( (ifmb == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
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

              if (kj == 0 && oj == 0) {
                /* Do no FLOPS  */
              } else if (kj == handle->desc.R-1 && oj == handle->ofh-1 ) {
                /* Do no FLOPS  */
              } else if ( oi == 0 && ki == 0 ) {
                br_gemm_kernel_strided2( &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, kj, ki, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block),
                    &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use+kj, ii_use+ki+1, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                    &LIBXSMM_VLA_ACCESS(5, output_int32, img, ofm1, oj, oi+1, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks);
              } else if (oi == handle->ofw-handle->fwd_ofw_rb  && ki == handle->desc.S-1) {
                 br_gemm_kernel_strided2( &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, kj, ki, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block),
                    &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use+kj, ii_use+ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                    &LIBXSMM_VLA_ACCESS(5, output_int32, img, ofm1, oj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks);
              } else {
                 br_gemm_kernel_strided( &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, kj, ki, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block),
                    &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use+kj, ii_use+ki, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                    &LIBXSMM_VLA_ACCESS(5, output_int32, img, ofm1, oj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks);
              }
              /* If last R/S iteration, then downconvert */
              if ((ifm1+handle->blocksifm_blocking == handle->blocksifm) && (kj == handle->desc.R-1) && (ki == handle->desc.S-1)) {
                /* Convert int32 chunk to int8 */
                for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                  LIBXSMM_DNN_CONVOLUTION_FWD_DOWNCONVERT_I32_I8( &LIBXSMM_VLA_ACCESS( 5, output_int32, img, ofm1, oj+ojj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                      &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj+ojj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                      handle->fwd_ofw_rb * handle->ofmblock, vscf);
                }
              }
            }
          }
        }
      }
    }
  }
} else {
  n_blocks = handle->blocksifm_blocking * handle->desc.R * handle->desc.S;
  if (handle->desc.R == 1 && handle->desc.S == 1) {  /* Strided based BRGEMM  */
    if (handle->avoid_acc_load == 0) {
      for (imgofm1ofh = thr_begin; imgofm1ofh < thr_end; ++imgofm1ofh) {
        img = imgofm1ofh / (handle->blocksofm*handle->ofh);
        ofm1 = (imgofm1ofh % (handle->blocksofm*handle->ofh))/handle->ofh;
        oj = (imgofm1ofh % (handle->blocksofm*handle->ofh))%handle->ofh;
        for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {
          if ( (ifmb == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0)) {
            /* set output feature map to zero */
            int* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output_int32, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
            for (oi = 0; oi < handle->ofw; ++oi) {
              LIBXSMM_PRAGMA_SIMD
                for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                  temp_ptr[ofm2] = (int)0;
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
              br_gemm_kernel_strided( &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block),
                  &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                  &LIBXSMM_VLA_ACCESS(5, output_int32, img, ofm1, oj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks);
              if (ifm1+handle->blocksifm_blocking == handle->blocksifm) {
                /* Convert int32 chunk to int8 */
                for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                  LIBXSMM_DNN_CONVOLUTION_FWD_DOWNCONVERT_I32_I8( &LIBXSMM_VLA_ACCESS( 5, output_int32, img, ofm1, oj+ojj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                      &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj+ojj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                      handle->fwd_ofw_rb * handle->ofmblock, vscf);
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
              br_gemm_kernel_strided( &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block),
                  &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                  &LIBXSMM_VLA_ACCESS(3, scratch_int32, 0, 0, 0, handle->fwd_ofw_rb, handle->ofmblock), &n_blocks);
              /* Convert int32 chunk to int8 */
              for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                LIBXSMM_DNN_CONVOLUTION_FWD_DOWNCONVERT_I32_I8( &LIBXSMM_VLA_ACCESS( 3, scratch_int32, ojj, 0, 0, handle->fwd_ofw_rb, handle->ofmblock),
                    &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj+ojj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                    handle->fwd_ofw_rb * handle->ofmblock, vscf);
              }
            }
          }
        }
      }
    }
  } else { /* Offset based BRGEMM */
    if (handle->avoid_acc_load == 0)  {
      for (imgofm1ofh = thr_begin; imgofm1ofh < thr_end; ++imgofm1ofh) {
        img = imgofm1ofh / (handle->blocksofm*handle->ofh);
        ofm1 = (imgofm1ofh % (handle->blocksofm*handle->ofh))/handle->ofh;
        oj = (imgofm1ofh % (handle->blocksofm*handle->ofh))%handle->ofh;
        for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {
          if ( (ifmb == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0)) {
            /* set output feature map to zero */
            int* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output_int32, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
            for (oi = 0; oi < handle->ofw; ++oi) {
              LIBXSMM_PRAGMA_SIMD
                for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                  temp_ptr[ofm2] = (int)0;
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
              br_gemm_kernel_offset( &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block),
                  &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                  &LIBXSMM_VLA_ACCESS(5, output_int32, img, ofm1, oj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks, handle->A_offsets, handle->B_offsets);
              if (ifm1+handle->blocksifm_blocking == handle->blocksifm) {
                /* Convert int32 chunk to int8 */
                for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                  LIBXSMM_DNN_CONVOLUTION_FWD_DOWNCONVERT_I32_I8( &LIBXSMM_VLA_ACCESS( 5, output_int32, img, ofm1, oj+ojj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                      &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj+ojj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                      handle->fwd_ofw_rb * handle->ofmblock, vscf);
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
              br_gemm_kernel_offset( &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block),
                  &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij_use, ii_use, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                  &LIBXSMM_VLA_ACCESS(3, scratch_int32, 0, 0, 0, handle->fwd_ofw_rb, handle->ofmblock), &n_blocks, handle->A_offsets, handle->B_offsets);
              /* Convert int32 chunk to int8 */
              for (ojj = 0; ojj < handle->fwd_ofh_rb; ojj++) {
                LIBXSMM_DNN_CONVOLUTION_FWD_DOWNCONVERT_I32_I8( &LIBXSMM_VLA_ACCESS( 3, scratch_int32, ojj, 0, 0, handle->fwd_ofw_rb, handle->ofmblock),
                    &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, oj+ojj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                    handle->fwd_ofw_rb * handle->ofmblock, vscf);
              }
            }
          }
        }
      }
    }
  }
}
libxsmm_barrier_wait(handle->barrier, ltid);

#undef LIBXSMM_DNN_CONVOLUTION_FWD_DOWNCONVERT_I32_I8
