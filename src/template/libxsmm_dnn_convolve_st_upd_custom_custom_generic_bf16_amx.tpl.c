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
#define TRANS_OUTPUT_TO_VNNI_FORMAT(img, ofm1) do {\
  __m512i zero_reg = _mm512_setzero_si512();\
  src_out = (element_output_type*) &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);\
  tr_out = (element_output_type*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, 0, 0, 0, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2);\
  for (pixel_pair = 0; pixel_pair < n_full_pixel_pairs; pixel_pair++) {\
    for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2+=32) {\
      pixel_0 = _mm512_loadu_si512((element_output_type*)src_out+ofm2);\
      pixel_1 = _mm512_loadu_si512(((element_output_type*)src_out+handle->ofmblock+ofm2));\
      ofms_lo = _mm512_permutex2var_epi16(pixel_0, idx_lo, pixel_1);\
      ofms_hi = _mm512_permutex2var_epi16(pixel_0, idx_hi, pixel_1);\
      _mm512_storeu_si512(tr_out+ofm2*2, ofms_lo);\
      _mm512_storeu_si512((element_output_type*)tr_out+32+ofm2*2, ofms_hi);\
    }\
    src_out += 2* handle->ofmblock;\
    tr_out += 2*handle->ofmblock;\
  }\
  if (half_pixel_pair == 1) {\
    for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2+=32) {\
      pixel_0 = _mm512_loadu_si512((element_output_type*)src_out+ofm2);\
      pixel_1 = _mm512_setzero_si512();\
      ofms_lo = _mm512_permutex2var_epi16(pixel_0, idx_lo, pixel_1);\
      ofms_hi = _mm512_permutex2var_epi16(pixel_0, idx_hi, pixel_1);\
      _mm512_storeu_si512(tr_out+ofm2*2, ofms_lo);\
      _mm512_storeu_si512((element_output_type*)tr_out+32+ofm2*2, ofms_hi);\
    }\
    tr_out += 2*handle->ofmblock;\
  } \
  for (oi = (n_full_pixel_pairs+half_pixel_pair)*2; oi < handle->output_pixels; oi+=2) {\
    for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2+=32) {\
      _mm512_storeu_si512((element_output_type*)tr_out+ofm2*2, zero_reg);\
      _mm512_storeu_si512((element_output_type*)tr_out+32+ofm2*2, zero_reg);\
    } \
    tr_out += 2*handle->ofmblock;\
  }\
}while(0)

#define TRANS_INPUT(img, ifm1) do {\
  transpose_input_pixels_bf16((element_input_type*)&LIBXSMM_VLA_ACCESS(5, input, img, ifm1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),(element_input_type*)&LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, 0, handle->blocksifm, handle->ifmblock, handle->input_pixels), handle->ifmblock, handle->ifhp*handle->ifwp, handle->ifmblock, handle->input_pixels);\
  if (handle->input_pixels - handle->ifhp*handle->ifwp > 0) {\
    for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {\
      zero_ptr_in = (element_input_type*)  &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, ifm2, handle->ifhp * handle->ifwp, handle->blocksifm, handle->ifmblock, handle->input_pixels);\
        memset(zero_ptr_in, 0, (handle->input_pixels - handle->ifhp * handle->ifwp)*sizeof(element_input_type));\
    }\
  }\
} while(0)

int img, my_img_start, my_img_end, ofmb, ifmb, ofm1, ifm1, ifm2, ofm2, oj, oi, ii, ij, kj, ki, /*j_br, img_br,*/ i, j, img_block_size = 1, my_ofm_start, my_ofm_end, my_ifm_start, my_ifm_end, block_ofm, block_ifm, pix;
/* computing first logical thread */
const int ltid = tid - start_thread;

const int IFWP = (handle->upd_padding_copy == 1) ? handle->ifwp + 2*handle->desc.pad_w :  handle->ifwp;
const int IFHP = (handle->upd_padding_copy == 1) ? handle->ifhp + 2*handle->desc.pad_h :  handle->ifhp;
const int OFWP = (handle->upd_padding_copy == 1) ? handle->ofwp + 2*handle->desc.pad_w :  handle->ofwp;
const int OFHP = (handle->upd_padding_copy == 1) ? handle->ofhp + 2*handle->desc.pad_h :  handle->ofhp;

element_output_type *const out = (element_output_type*)handle->grad_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
LIBXSMM_VLA_DECL(5, const element_output_type, output, (const element_output_type*)out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(5, const element_input_type, input, (const element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);

element_filter_type *weight_ptr = (element_filter_type*)((char*)handle->scratch + handle->upd_filter_scratch_offset) + ltid * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S;
element_filter_type *filter_dst_ptr = (handle->weight_copies > 1) ? (element_filter_type*)weight_ptr : (element_filter_type*)handle->grad_filter->data;
LIBXSMM_VLA_DECL(7, element_filter_type, weight_dst, (element_filter_type*)filter_dst_ptr, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock/2, handle->ofmblock, 2);

/* This intermediate tensor is used when pixels are NOT fully accumulated  */
float *weight_ptr_f32 = (float*)((char*)handle->scratch + handle->upd_lp_filter_full_scratch_offset) + ltid * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S;
LIBXSMM_VLA_DECL(6, float, weight_private_f32, (float*)weight_ptr_f32, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
/* Accumulation scratch is used when pixels are ully accumulated  */
element_filter_type *filter_scratch = (element_filter_type*)((char*)handle->scratch + handle->upd_lp_filter_full_scratch_offset) + ltid * handle->ofmblock * handle->ifmblock * 2;
LIBXSMM_VLA_DECL(2, float, filter_tmp, (float*)filter_scratch, handle->ofmblock);

element_input_type *scratch_tr_input = (element_input_type*)((char*)handle->scratch + handle->upd_lp_input_full_scratch_offset);
element_input_type *zero_ptr_in;
LIBXSMM_VLA_DECL(4, element_input_type, tr_input, (element_input_type*) scratch_tr_input, handle->blocksifm, handle->ifmblock, handle->input_pixels);
LIBXSMM_VLA_DECL(5, element_input_type, tr_input_2, (element_input_type*) scratch_tr_input, handle->blocksifm, handle->ifmblock, IFHP, handle->ifwp_extended);
LIBXSMM_VLA_DECL(3, element_input_type, tr_input_3, (element_input_type*) scratch_tr_input, handle->ifmblock, handle->input_pixels);

element_output_type *scratch_tr_output = (element_input_type*)((char*)handle->scratch + handle->upd_lp_output_full_scratch_offset);
LIBXSMM_VLA_DECL(5, element_output_type, tr_output, (element_output_type*) scratch_tr_output, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2);
LIBXSMM_VLA_DECL(6, element_output_type, tr_output_2, (element_output_type*) scratch_tr_output, handle->blocksofm, OFHP, handle->ofwp_extended/2, handle->ofmblock, 2);
LIBXSMM_VLA_DECL(4, element_output_type, tr_output_3, (element_output_type*) scratch_tr_output, handle->output_pixels/2, handle->ofmblock, 2);

element_output_type *out_ptr = (element_output_type*)handle->grad_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
element_output_type *zero_ptr_out;

/* transpose, copy and reduce work-related variables  */
const int reduce_work = (handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S)/16 ;
const int reduce_chunksize = (reduce_work % handle->desc.threads == 0) ? (reduce_work / handle->desc.threads) : (reduce_work / handle->desc.threads) + 1;
const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;

#if 0
const float beta = (handle->use_intermediate_f32_wt_tensor) ? 1.0 : 0.0;
#endif
float *dst_ptr;
#if 0
gemm_br_function br_gemm_kernel = 0;
#endif

/* These are used for the vnni reformatting of the f32 output  */
__m512i c01;
const __m512i perm_index = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);

/* Related to the output transpose */
int n_full_pixel_pairs = handle->compute_pixels/2, half_pixel_pair = handle->compute_pixels%2, pixel_pair;
element_output_type *tr_out, *src_out;
const __m512i selector = LIBXSMM_INTRINSICS_MM512_SET_EPI16(32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0);
const __m512i offsets_lo = LIBXSMM_INTRINSICS_MM512_SET_EPI16(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
const __m512i offsets_hi = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24, 23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16);
const __m512i idx_lo =  _mm512_or_epi32(selector, offsets_lo);
const __m512i idx_hi =  _mm512_or_epi32(selector, offsets_hi);
__m512i pixel_0, pixel_1, ofms_lo, ofms_hi;

/* Batch reduce related variables */
#if 0
const element_output_type *A_ptrs[1024];
const element_input_type  *B_ptrs[1024];
#endif
unsigned long long n_blocks;

#if 0
int LDA = handle->ofmblock;
int LDB = handle->input_pixels;
int LDC = handle->ofmblock;
int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
int l_flags = (LIBXSMM_GEMM_FLAGS('N', 'N')) | LIBXSMM_GEMM_FLAG_EXCLUDE_TILECONFIG;
int l_tc_flags = LIBXSMM_GEMM_FLAG_ONLY_TILECONFIG;
gemm_function tile_config_kernel = 0;
#endif

const int img_work = handle->desc.N;
const int img_chunksize = (img_work % handle->desc.threads == 0) ? (img_work / handle->desc.threads) : (img_work / handle->desc.threads) + 1;

/* select kernel */
if (handle->upd_linearized_pixels == 0) {
  br_gemm_kernel = handle->upd_compute_kernel_brgemm_no_linearized_pixels;
  gemm_kernel = handle->upd_compute_kernel_gemm_linearized_pixels_no_hybrid_par; /* @TODO: ci check */
} else {
  if (handle->use_hybrid_imgofm_parallelization == 0) {
    gemm_kernel = handle->upd_compute_kernel_gemm_linearized_pixels_no_hybrid_par;
    br_gemm_kernel = handle->upd_compute_kernel_brgemm_no_linearized_pixels; /* @TODO: ci check */
  } else {
#if 0 /* if/else branches with same outcome */
    if (handle->pack_to_cnhw == 1)
#endif
    {
      gemm_kernel = handle->upd_compute_kernel_gemm_linearized_pixels_hybrid_par_cnhw;
      br_gemm_kernel = handle->upd_compute_kernel_brgemm_linearized_pixels_hybrid_par_no_cnhw; /* @TODO: ci check */
    }
#if 0 /* if/else branches with same outcome */
    else {
      gemm_kernel = handle->upd_compute_kernel_gemm_linearized_pixels_hybrid_par_cnhw; /* @TODO: ci check */
      br_gemm_kernel = handle->upd_compute_kernel_brgemm_linearized_pixels_hybrid_par_no_cnhw;
    }
#endif
  }
}

my_img_start = (ltid * img_chunksize < img_work) ? (ltid * img_chunksize) : img_work;
my_img_end = ((ltid + 1) * img_chunksize < img_work) ? ((ltid + 1) * img_chunksize) : img_work;

libxsmm_barrier_init(handle->barrier, ltid);

if (handle->upd_linearized_pixels == 1) {
  /* First transpose input and output */
  if (handle->pack_to_cnhw == 0) {
    if (handle->fuse_upd_transposes == 0) {
      if (handle->upd_pack_input_upfront == 0) {
        if (handle->upd_padding_copy == 1) {
          for (img = my_img_start; img < my_img_end; img++) {
            for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1++) {
              zero_ptr_in = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, 0, handle->blocksifm, handle->ifmblock, handle->input_pixels);
              memset(zero_ptr_in, 0, handle->ifmblock * handle->input_pixels * sizeof(element_input_type));
              for (ij = 0; ij < handle->ifhp; ij++) {
                for (ii = 0; ii < handle->ifwp; ii++) {
                  for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
                    LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, ifm2, (ij + handle->desc.pad_h) * IFWP + (ii + handle->desc.pad_w), handle->blocksifm, handle->ifmblock, handle->input_pixels) =
                      LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, ii, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                  }
                }
              }
            }
          }
        } else {
          if (handle->ifmblock % 32 == 0) {
            for (img = my_img_start; img < my_img_end; img++) {
              for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1++) {
                TRANS_INPUT(img, ifm1);
              }
            }
          } else {
            for (img = my_img_start; img < my_img_end; img++) {
              zero_ptr_in = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, tr_input, img, 0, 0, 0, handle->blocksifm, handle->ifmblock, handle->input_pixels);
              memset(zero_ptr_in, 0, handle->desc.C * handle->input_pixels * sizeof(element_input_type));
              for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1++) {
                for (ij = 0; ij < handle->ifhp; ij++) {
                  for (ii = 0; ii < handle->ifwp; ii++) {
                    for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
                      LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, ifm2, ij * handle->ifwp + ii, handle->blocksifm, handle->ifmblock, handle->input_pixels) =
                        LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, ii, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        for (img = my_img_start; img < my_img_end; img++) {
          zero_ptr_in = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, tr_input, img, 0, 0, 0, handle->blocksifm, handle->ifmblock, handle->input_pixels);
          memset(zero_ptr_in, 0, handle->desc.C * handle->input_pixels * sizeof(element_input_type));
          for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1++) {
            for (ij = 0; ij < handle->ifhp/handle->desc.u; ij++) {
              for (ii = 0; ii < handle->ifwp/handle->desc.v; ii++) {
                for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
                  LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, ifm2, ij * (handle->ifwp/handle->desc.v) + ii, handle->blocksifm, handle->ifmblock, handle->input_pixels) =
                    LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij*handle->desc.u, ii*handle->desc.v, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                }
              }
            }
          }
        }
      }

      /* Reformat output */
      if (handle->upd_padding_copy == 1) {
        for (img = my_img_start; img < my_img_end; img++) {
          for (ofm1 = 0; ofm1 < handle->blocksofm; ofm1++) {
            zero_ptr_out = (element_output_type*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, 0, 0, 0, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2);
            memset(zero_ptr_out, 0, handle->ofmblock * handle->output_pixels * sizeof(element_output_type));
            for (oj = 0; oj < handle->ofhp; oj++) {
              for (oi = 0; oi < handle->ofwp; oi++) {
                for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++) {
                  LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, (oj*OFWP+oi)/2, ofm2, (oj*OFWP+oi)%2, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2) =
                    LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                }
              }
            }
          }
        }
      } else {
        if (handle->ofmblock % 32 == 0) {
          for (img = my_img_start; img < my_img_end; img++) {
            for (ofm1 = 0; ofm1 < handle->blocksofm; ofm1++) {
              TRANS_OUTPUT_TO_VNNI_FORMAT(img, ofm1);
            }
          }
        } else {
          for (img = my_img_start; img < my_img_end; img++) {
            zero_ptr_out = (element_output_type*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, 0, 0, 0, 0, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2);
            memset(zero_ptr_out, 0, handle->desc.K * handle->output_pixels * sizeof(element_output_type));
            for (ofm1 = 0; ofm1 < handle->blocksofm; ofm1++) {
              for (oi = 0; oi < handle->compute_pixels; oi++) {
                for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++) {
                  LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, oi/2, ofm2, oi%2, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2) =
                    *((element_output_type*)out_ptr + img * handle->blocksofm * handle->ofwp * handle->ofhp * handle->ofmblock + ofm1 * handle->ofwp * handle->ofhp * handle->ofmblock + oi * handle->ofmblock + ofm2);
                }
              }
            }
          }
        }
      }
    }
  } else {
    int img_tile_id, img_in_tile, init_offset, /*pix_id,*/ images_in_tile = handle->desc.N/handle->weight_copies;
    /* Zero out the input padding pixels  */
    for (img = my_img_start; img < my_img_end; img++) {
      img_tile_id = img/images_in_tile;
      img_in_tile = img%images_in_tile;
      if (img_in_tile == images_in_tile-1) {
        for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1++) {
          for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
            zero_ptr_in = (element_input_type*) &LIBXSMM_VLA_ACCESS(3, tr_input_3, ifm1, ifm2, img_tile_id * handle->pixel_blocking + images_in_tile * (handle->ifhp/handle->desc.u) * (handle->ifwp/handle->desc.v),  handle->ifmblock, handle->input_pixels);
            memset(zero_ptr_in, 0, handle->remainder_pixels * sizeof(element_input_type));
          }
        }
      }
    }

    if ((handle->ifmblock % 32 == 0) && (handle->desc.u == 1) && (handle->desc.v == 1)) {
      for (img = my_img_start; img < my_img_end; img++) {
        img_tile_id = img/images_in_tile;
        img_in_tile = img%images_in_tile;
        for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1++) {
          transpose_input_pixels_bf16((element_input_type*)&LIBXSMM_VLA_ACCESS(5, input, img, ifm1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
              (element_input_type*)&LIBXSMM_VLA_ACCESS(3, tr_input_3, ifm1, 0, img_tile_id * handle->pixel_blocking + img_in_tile * handle->ifhp * handle->ifwp, handle->ifmblock, handle->input_pixels) ,
              handle->ifmblock, handle->ifhp*handle->ifwp, handle->ifmblock, handle->input_pixels);
        }
      }
    } else {
      for (img = my_img_start; img < my_img_end; img++) {
        img_tile_id = img/images_in_tile;
        img_in_tile = img%images_in_tile;
        for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1++) {
          for (ij = 0; ij < handle->ifhp/handle->desc.u; ij++) {
            for (ii = 0; ii < handle->ifwp/handle->desc.v; ii++) {
              for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
                LIBXSMM_VLA_ACCESS(3, tr_input_3, ifm1, ifm2, img_tile_id * handle->pixel_blocking + img_in_tile * (handle->ifhp/handle->desc.u) * (handle->ifwp/handle->desc.v) + ij * (handle->ifwp/handle->desc.v) + ii, handle->ifmblock, handle->input_pixels) =
                  LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij*handle->desc.u, ii*handle->desc.v, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              }
            }
          }
        }
      }
    }

    /* Zero out the output padding pixels  */
    for (img = my_img_start; img < my_img_end; img++) {
      img_tile_id = img/images_in_tile;
      img_in_tile = img%images_in_tile;
      if (img_in_tile == images_in_tile-1) {
        for (ofm1 = 0; ofm1 < handle->blocksofm; ofm1++) {
          init_offset = img_tile_id * handle->pixel_blocking + images_in_tile * handle->ofw * handle->ofh;
          tr_out = (element_output_type*) &LIBXSMM_VLA_ACCESS(4, tr_output_3, ofm1, init_offset/2, 0, init_offset%2, handle->output_pixels/2, handle->ofmblock, 2);
          memset(tr_out, 0, handle->remainder_pixels * handle->ofmblock * sizeof(element_input_type));
#if 0
          for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++) {
            for (oi = 0; oi < handle->remainder_pixels; oi++ ) {
              init_offset = img_tile_id * handle->pixel_blocking + images_in_tile * handle->ofw * handle->ofh;
              pix_id = init_offset + oi;
              LIBXSMM_VLA_ACCESS(4, tr_output_3, ofm1, pix_id/2, ofm2, pix_id%2, handle->output_pixels/2, handle->ofmblock, 2) = (element_output_type)0;
            }
          }
#endif
        }
      }
    }

    if (handle->ofmblock % 32 == 0) {
      int _trans_pixels = handle->ofw*handle->ofh, _n_full_pixel_pairs, _half_pixel_pair, init_pixel_pos;
      for (img = my_img_start; img < my_img_end; img++) {
        int pix_id;
        img_tile_id = img/images_in_tile;
        img_in_tile = img%images_in_tile;
        pix_id = img_tile_id * handle->pixel_blocking + img_in_tile * handle->ofh * handle->ofw;
        /* The first-odd pixel is done with scalar code... */
        if (pix_id % 2 == 1) {
          for (ofm1 = 0; ofm1 < handle->blocksofm; ofm1++) {
            for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++) {
              LIBXSMM_VLA_ACCESS(4, tr_output_3, ofm1, pix_id/2, ofm2, 1, handle->output_pixels/2, handle->ofmblock, 2) =
                LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
            }
          }
          pix_id += 1;
          _trans_pixels--;
          init_pixel_pos = 1;
        } else {
          init_pixel_pos = 0;
        }
        _n_full_pixel_pairs = _trans_pixels/2;
        _half_pixel_pair = _trans_pixels%2;
        for (ofm1 = 0; ofm1 < handle->blocksofm; ofm1++) {
          src_out = (element_output_type*) &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, init_pixel_pos, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          tr_out = (element_output_type*) &LIBXSMM_VLA_ACCESS(4, tr_output_3, ofm1, pix_id/2, 0, 0, handle->output_pixels/2, handle->ofmblock, 2);
          for (pixel_pair = 0; pixel_pair < _n_full_pixel_pairs; pixel_pair++) {
            for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2+=32) {
              pixel_0 = _mm512_loadu_si512((element_output_type*)src_out+ofm2);
              pixel_1 = _mm512_loadu_si512(((element_output_type*)src_out+handle->ofmblock+ofm2));
              ofms_lo = _mm512_permutex2var_epi16(pixel_0, idx_lo, pixel_1);
              ofms_hi = _mm512_permutex2var_epi16(pixel_0, idx_hi, pixel_1);
              _mm512_storeu_si512(tr_out+ofm2*2, ofms_lo);
              _mm512_storeu_si512((element_output_type*)tr_out+32+ofm2*2, ofms_hi);
            }
            src_out += 2* handle->ofmblock;
            tr_out += 2*handle->ofmblock;
          }
        }
        /* The last-odd pixel is done with scalar code... */
        if (_half_pixel_pair == 1) {
          pix_id = pix_id + _n_full_pixel_pairs*2;
          for (ofm1 = 0; ofm1 < handle->blocksofm; ofm1++) {
            for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++) {
              LIBXSMM_VLA_ACCESS(4, tr_output_3, ofm1, pix_id/2, ofm2, pix_id%2, handle->output_pixels/2, handle->ofmblock, 2) =
                LIBXSMM_VLA_ACCESS(5, output, img, ofm1, handle->ofh-1, handle->ofw-1, ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
            }
          }
        }
      }
    } else {
      for (img = my_img_start; img < my_img_end; img++) {
        img_tile_id = img/images_in_tile;
        img_in_tile = img%images_in_tile;
        for (ofm1 = 0; ofm1 < handle->blocksofm; ofm1++) {
          for (oj = 0; oj < handle->ofh; oj++) {
            for (oi = 0; oi < handle->ofw; oi++) {
              for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++) {
                int pix_id = img_tile_id * handle->pixel_blocking + img_in_tile * handle->ofh * handle->ofw + oj * handle->ofw + oi;
                LIBXSMM_VLA_ACCESS(4, tr_output_3, ofm1, pix_id/2, ofm2, pix_id%2, handle->output_pixels/2, handle->ofmblock, 2) =
                  LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
              }
            }
          }
        }
      }
    }
  }
} else {
  if (handle->on_the_fly_input_packing == 0) {
    for (img = my_img_start; img < my_img_end; img++) {
      zero_ptr_in = (element_input_type*) &LIBXSMM_VLA_ACCESS(5, tr_input_2, img, 0, 0, 0, 0, handle->blocksifm, handle->ifmblock, IFHP, handle->ifwp_extended);
      memset(zero_ptr_in, 0, handle->desc.C * handle->ifhp * handle->ifwp_extended * sizeof(element_input_type));
      for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1++) {
        for (ij = 0; ij < handle->ifhp; ij++) {
          for (ii = 0; ii < handle->ifwp; ii++) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
              LIBXSMM_VLA_ACCESS(5, tr_input_2, img, ifm1, ifm2, ij, ii, handle->blocksifm, handle->ifmblock, IFHP, handle->ifwp_extended) =
                LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, ii, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
            }
          }
        }
      }
    }
  } else {
    for (img = my_img_start; img < my_img_end; img++) {
      zero_ptr_in = (element_input_type*) &LIBXSMM_VLA_ACCESS(5, tr_input_2, img, 0, 0, 0, 0, handle->blocksifm, handle->ifmblock, IFHP, handle->ifwp_extended);
      memset(zero_ptr_in, 0, handle->desc.C * IFHP * handle->ifwp_extended * sizeof(element_input_type));
    }
  }
  for (img = my_img_start; img < my_img_end; img++) {
    for (ofm1 = 0; ofm1 < handle->blocksofm; ofm1++) {
      for (oj = 0; oj < handle->ofh; oj++) {
        zero_ptr_out = (element_output_type*) &LIBXSMM_VLA_ACCESS(6, tr_output_2, img, ofm1, oj, 0, 0, 0, handle->blocksofm, OFHP, handle->ofwp_extended/2, handle->ofmblock, 2);
        memset(zero_ptr_out, 0, handle->ofmblock * (handle->ofw+handle->remainder_pixels) * sizeof(element_output_type));
        for (oi = 0; oi < handle->ofw; oi++) {
          for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++) {
            LIBXSMM_VLA_ACCESS(6, tr_output_2, img, ofm1, oj, oi/2, ofm2, oi%2, handle->blocksofm, OFHP, handle->ofwp_extended/2, handle->ofmblock, 2) =
              LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          }
        }
      }
    }
  }
}

/* Make sure we initialize intermediate weights to zero */
if (handle->use_intermediate_f32_wt_tensor == 1 && handle->use_hybrid_imgofm_parallelization == 0) {
  memset(weight_ptr_f32, 0, handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S * sizeof(float));
}

tile_config_kernel(NULL, NULL, NULL);

if (handle->upd_linearized_pixels == 0) {
#if 0
  LDA = handle->ofmblock;
  LDB = handle->ifhp*handle->ifwp_extended;
  LDC = handle->ofmblock;
  prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
  br_gemm_kernel =  libxsmm_bsmmdispatch_reducebatch_addr(handle->ofmblock, handle->ifmblock, handle->ofw+handle->remainder_pixels, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);
  tile_config_kernel = libxsmm_bsmmdispatch(handle->ofmblock, handle->ifmblock, handle->ofw+handle->remainder_pixels, &LDA, &LDB, &LDC, NULL, &beta, &l_tc_flags, NULL);
#endif
  n_blocks = handle->batchreduce_h_pixels;

  for (img = my_img_start; img < my_img_end; img++) {
    for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_upd_ofm) {
      for (oj = 0; oj < handle->ofh; oj += handle->batchreduce_h_pixels){
        for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_upd_ifm) {
          for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_upd_ofm, handle->blocksofm); ofm1++ ) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {
              for (kj = 0; kj < handle->desc.R; ++kj) {
                for (ki = 0; ki < handle->desc.S; ++ki) {

                  /* Determine if destination is the accumulation scratch or the intermediate fp32 weight tensor */
                  if (handle->use_intermediate_f32_wt_tensor == 1) {
                    dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(6, weight_private_f32, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                  } else {
                    dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(2, filter_tmp, 0, 0, handle->ofmblock);
                  }

                  /* Copy the input in such a way that we ignore "w-pixels" based on ki value  */
                  if (handle->on_the_fly_input_packing == 1) {
                    if (handle->upd_padding_copy == 1) {
                      for (ij = kj; ij < IFHP; ij+=handle->desc.u) {
                        for (ii = 0; ii < handle->ofw; ii++) {
                          for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
                            if ( (ij >= handle->desc.pad_h) && (ii*handle->desc.v+ki >= handle->desc.pad_w) && (ij < handle->ifhp+handle->desc.pad_h) && (ii*handle->desc.v+ki < handle->ifwp+handle->desc.pad_w) ) {
                              LIBXSMM_VLA_ACCESS(5, tr_input_2, img, ifm1, ifm2, ij, ii, handle->blocksifm, handle->ifmblock, IFHP, handle->ifwp_extended) =
                              LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij-handle->desc.pad_h, ii*handle->desc.v+ki-handle->desc.pad_w, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                            } else {
                              LIBXSMM_VLA_ACCESS(5, tr_input_2, img, ifm1, ifm2, ij, ii, handle->blocksifm, handle->ifmblock, IFHP, handle->ifwp_extended) = (element_input_type)0;
                            }
                          }
                        }
                      }
                    } else {
                      for (ij = 0; ij < handle->ifhp; ij++) {
                        for (ii = 0; ii < handle->ofw; ii++) {
                          for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
                            LIBXSMM_VLA_ACCESS(5, tr_input_2, img, ifm1, ifm2, ij, ii, handle->blocksifm, handle->ifmblock, IFHP, handle->ifwp_extended) =
                              LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, ii*handle->desc.v+ki, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                          }
                        }
                      }
                    }
                  }

#if 0
                  for (j_br = 0; j_br < handle->batchreduce_h_pixels; j_br++) {
                    A_ptrs[j_br] = (element_output_type*) &LIBXSMM_VLA_ACCESS(6, tr_output_2, img, ofm1, oj+j_br, 0, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp_extended/2, handle->ofmblock, 2);
                    B_ptrs[j_br] = (element_input_type*) &LIBXSMM_VLA_ACCESS(5, tr_input_2, img, ifm1, 0, (oj+j_br)*handle->desc.u + kj, 0, handle->blocksifm, handle->ifmblock, handle->ifhp, handle->ifwp_extended);
                  }
                  br_gemm_kernel(A_ptrs, B_ptrs, dst_ptr, &n_blocks);
#endif
                  br_gemm_kernel( &LIBXSMM_VLA_ACCESS(6, tr_output_2, img, ofm1, oj, 0, 0, 0, handle->blocksofm, OFHP, handle->ofwp_extended/2, handle->ofmblock, 2),
                      &LIBXSMM_VLA_ACCESS(5, tr_input_2, img, ifm1, 0, oj*handle->desc.u + kj, 0, handle->blocksifm, handle->ifmblock, IFHP, handle->ifwp_extended), dst_ptr, &n_blocks);

                  /* Convert fully caccumulated buffer to bf16 weight buffer in case of full accumulation has happened */
                  if (oj + handle->batchreduce_h_pixels >= handle->ofh) {
                    LIBXSMM_VLA_DECL(2, float, filter_acc_buffer, (float*)dst_ptr, handle->ofmblock);
                    for (ij = 0; ij < handle->ifmblock; ij+=2) {
                      for (ii = 0; ii < handle->ofmblock; ii+=16) {
                        c01 = LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)&LIBXSMM_VLA_ACCESS(2, filter_acc_buffer, ij+1, ii, handle->ofmblock)), LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)&LIBXSMM_VLA_ACCESS(2, filter_acc_buffer, ij, ii, handle->ofmblock)));
                        _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(7, weight_dst, ofm1, ifm1, kj, ki, ij/2, ii, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock/2, handle->ofmblock, 2), _mm512_permutexvar_epi16(perm_index,(__m512i)c01));
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
#if 0
  LDA = handle->ofmblock;
  LDB = handle->input_pixels;
  LDC = handle->ofmblock;
  prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
#endif
  if (handle->use_hybrid_imgofm_parallelization == 1) {
    /* Here we are using batch-reduce kernel and hybrid minibatch/FM parallelization */
    /* FIXME: Hardcoed logic for N=27  */
    int group_size = (handle->desc.threads == 27 && handle->desc.N == 27 && handle->ofw == 14 && handle->desc.R == 1 && handle->desc.u == 1 && ltid >= 24) ? 3 : ((handle->desc.threads+handle->weight_copies-1)/handle->weight_copies);
    int tile_id = ltid/( (handle->desc.threads+handle->weight_copies-1)/handle->weight_copies );
    int tiles = handle->weight_copies;
    int img_per_tile = (handle->desc.N+tiles-1)/tiles;
    int my_in_tile_id = ltid % group_size;
    int ifms_per_thread = (handle->blocksifm+group_size-1)/group_size;
    int ofms_per_thread = (handle->blocksofm+group_size-1)/group_size;
    int my_R_start = 0;
    int my_R_end = handle->desc.R;
    element_filter_type *weight_ptr_group = (handle->weight_copies > 1) ? (element_filter_type*)((char*)handle->scratch + handle->upd_filter_scratch_offset) + tile_id * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S : (element_filter_type*)handle->grad_filter->data;
    LIBXSMM_VLA_DECL(7, element_filter_type, weight_private_group, (element_filter_type*)weight_ptr_group, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock/2, handle->ofmblock, 2);
    /* This intermediate tensor is used when pixels are NOT fully accumulated  */
    float *weight_tile_ptr_f32 = (float*)((char*)handle->scratch + handle->upd_lp_filter_full_scratch_offset) + tile_id * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S;
    LIBXSMM_VLA_DECL(6, float, weight_private_tile_f32, (float*)weight_tile_ptr_f32, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);

    my_img_start = LIBXSMM_MIN( tile_id * img_per_tile, handle->desc.N);
    my_img_end = LIBXSMM_MIN( (tile_id+1) * img_per_tile, handle->desc.N);
    my_ifm_start = LIBXSMM_MIN( my_in_tile_id * ifms_per_thread, handle->blocksifm  );
    my_ifm_end = LIBXSMM_MIN( (my_in_tile_id+1) * ifms_per_thread, handle->blocksifm  );
    my_ofm_start = 0;
    my_ofm_end = handle->blocksofm;
    /* FIXME: Hardcoed logic for N=27  */
    if (handle->desc.threads == 27 && handle->desc.N == 27 && handle->desc.C == 256 && handle->desc.K == 1024 && handle->ofh == 14 && handle->desc.u == 1) {
      my_ofm_start = LIBXSMM_MIN( my_in_tile_id * ofms_per_thread, handle->blocksofm  );
      my_ofm_end = LIBXSMM_MIN( (my_in_tile_id+1) * ofms_per_thread, handle->blocksofm  );
      my_ifm_start = 0;
      my_ifm_end = handle->blocksifm;
    }
    if (handle->desc.threads == 27 && handle->desc.N == 27 && handle->desc.R == 3 && handle->desc.S == 3 && handle->ofh == 14) {
      int r_per_tile = (handle->desc.R+group_size-1)/group_size;
      my_ifm_start = 0;
      my_ifm_end = handle->blocksifm;
      my_ofm_start = 0;
      my_ofm_end = handle->blocksofm;
      my_R_start = LIBXSMM_MIN( my_in_tile_id * r_per_tile, handle->desc.R );
      my_R_end = LIBXSMM_MIN( (my_in_tile_id+1) * r_per_tile, handle->desc.R );
    }
    if (handle->pack_to_cnhw == 1) {
      my_ofm_start = LIBXSMM_MIN( my_in_tile_id * ofms_per_thread, handle->blocksofm  );
      my_ofm_end = LIBXSMM_MIN( (my_in_tile_id+1) * ofms_per_thread, handle->blocksofm  );
      my_ifm_start = 0;
      my_ifm_end = handle->blocksifm;
    }

    block_ofm = my_ofm_end-my_ofm_start+1;
    block_ifm = my_ifm_end-my_ifm_start+1;
    img_block_size = my_img_end - my_img_start;

    /* Make sure we initialize intermediate weights to zero */
    if (handle->use_intermediate_f32_wt_tensor == 1) {
      for (ofm1 = my_ofm_start; ofm1 < my_ofm_end; ofm1++ ) {
        for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
          for (kj = my_R_start; kj < my_R_end; ++kj) {
            memset((float*)&LIBXSMM_VLA_ACCESS(6, weight_private_tile_f32, ofm1, ifm1, kj, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), 0, handle->ofmblock * handle->ifmblock * handle->desc.S * sizeof(float));
          }
        }
      }
    }

    libxsmm_barrier_wait(handle->barrier, ltid);

    if (handle->pack_to_cnhw == 0) {
#if 0
      br_gemm_kernel = libxsmm_bsmmdispatch_reducebatch_addr(handle->ofmblock, handle->ifmblock, handle->pixel_blocking, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);
      tile_config_kernel = libxsmm_bsmmdispatch(handle->ofmblock, handle->ifmblock, handle->pixel_blocking, &LDA, &LDB, &LDC, NULL, &beta, &l_tc_flags, NULL);
#endif
      n_blocks = img_block_size;

      for (img = my_img_start; img < my_img_end; img += img_block_size) {
        for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += block_ofm) {
          for (pix = 0; pix < handle->n_used_pixels; pix += handle->pixel_blocking){
            for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += block_ifm) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+block_ofm, my_ofm_end); ofm1++ ) {
                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+block_ifm, my_ifm_end); ifm1++) {
                  for (kj = my_R_start; kj < my_R_end; ++kj) {
                    for (ki = 0; ki < handle->desc.S; ++ki) {

                      /* Determine if destination is the accumulation scratch or the intermediate fp32 weight tensor */
                      if (handle->use_intermediate_f32_wt_tensor == 1) {
                        dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(6, weight_private_tile_f32, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                      } else {
                        dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(2, filter_tmp, 0, 0, handle->ofmblock);
                      }

#if 0
                      for (img_br = 0; img_br < img_block_size; img_br++) {
                        A_ptrs[img_br] = &LIBXSMM_VLA_ACCESS(5, tr_output, img + img_br, ofm1, pix/2, 0, 0, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2);
                        B_ptrs[img_br] = &LIBXSMM_VLA_ACCESS(4, tr_input, img + img_br, ifm1, 0, pix + kj * handle->ifwp + ki, handle->blocksifm, handle->ifmblock, handle->input_pixels);
                      }
                      br_gemm_kernel(A_ptrs, B_ptrs, dst_ptr, &n_blocks);
#endif

                      br_gemm_kernel( &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, pix/2, 0, 0, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2),
                          &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, pix + kj * IFWP + ki, handle->blocksifm, handle->ifmblock, handle->input_pixels),
                          dst_ptr, &n_blocks);

                      /* Convert fully caccumulated buffer to bf16 weight buffer in case of full accumulation has happened */
                      if (pix + handle->pixel_blocking >= handle->n_used_pixels) {
                        LIBXSMM_VLA_DECL(2, float, filter_acc_buffer, (float*)dst_ptr, handle->ofmblock);
                        for (ij = 0; ij < handle->ifmblock; ij+=2) {
                          for (ii = 0; ii < handle->ofmblock; ii+=16) {
                            c01 = LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)&LIBXSMM_VLA_ACCESS(2, filter_acc_buffer, ij+1, ii, handle->ofmblock)), LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)&LIBXSMM_VLA_ACCESS(2, filter_acc_buffer, ij, ii, handle->ofmblock)));
                            _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(7, weight_private_group, ofm1, ifm1, kj, ki, ij/2, ii, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock/2, handle->ofmblock, 2), _mm512_permutexvar_epi16(perm_index, (__m512i)c01));
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
#if 0
      gemm_function gemm_kernel = libxsmm_bsmmdispatch(handle->ofmblock, handle->ifmblock, handle->pixel_blocking, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);
      tile_config_kernel = libxsmm_bsmmdispatch(handle->ofmblock, handle->ifmblock, handle->pixel_blocking, &LDA, &LDB, &LDC, NULL, &beta, &l_tc_flags, NULL);
#endif
      for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += block_ofm) {
        for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += block_ifm) {
          for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+block_ofm, my_ofm_end); ofm1++ ) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+block_ifm, my_ifm_end); ifm1++) {
              for (kj = my_R_start; kj < my_R_end; ++kj) {
                for (ki = 0; ki < handle->desc.S; ++ki) {
                  dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(2, filter_tmp, 0, 0, handle->ofmblock);
                  gemm_kernel( &LIBXSMM_VLA_ACCESS(4, tr_output_3, ofm1, tile_id * handle->pixel_blocking/2, 0, 0, handle->output_pixels/2, handle->ofmblock, 2),
                      &LIBXSMM_VLA_ACCESS(3, tr_input_3, ifm1, 0, tile_id * handle->pixel_blocking, handle->ifmblock, handle->input_pixels),
                      dst_ptr);
                  /* Convert fully caccumulated buffer to bf16 weight buffer in case of full accumulation has happened */
                  {
                    LIBXSMM_VLA_DECL(2, float, filter_acc_buffer, (float*)dst_ptr, handle->ofmblock);
                    for (ij = 0; ij < handle->ifmblock; ij+=2) {
                      for (ii = 0; ii < handle->ofmblock; ii+=16) {
                        c01 = LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)&LIBXSMM_VLA_ACCESS(2, filter_acc_buffer, ij+1, ii, handle->ofmblock)), LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)&LIBXSMM_VLA_ACCESS(2, filter_acc_buffer, ij, ii, handle->ofmblock)));
                        _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(7, weight_private_group, ofm1, ifm1, kj, ki, ij/2, ii, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock/2, handle->ofmblock, 2), _mm512_permutexvar_epi16(perm_index, (__m512i)c01));
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
      for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_upd_ofm) {
        for (pix = 0; pix < handle->n_used_pixels; pix += handle->pixel_blocking){
          for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_upd_ifm) {
            for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_upd_ofm, handle->blocksofm); ofm1++ ) {
              if ((handle->fuse_upd_transposes == 1) && (pix == 0) && (ifmb == 0)) {
                /* (img,ofm1) transpose of output */
                if (handle->upd_padding_copy == 1) {
                  zero_ptr_out = (element_output_type*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, 0, 0, 0, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2);
                  memset(zero_ptr_out, 0, handle->ofmblock * handle->output_pixels * sizeof(element_output_type));
                  for (oj = 0; oj < handle->ofhp; oj++) {
                    for (oi = 0; oi < handle->ofwp; oi++) {
                      for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++) {
                        LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, (oj*OFWP+oi)/2, ofm2, (oj*OFWP+oi)%2, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2) =
                          LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                      }
                    }
                  }
                } else {
                  TRANS_OUTPUT_TO_VNNI_FORMAT(img, ofm1);
                }
              }
              for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {
                if ((handle->fuse_upd_transposes == 1) && (pix == 0) && (ofm1 == 0)) {
                  /* (img,ifm1) transpose of input */
                  if (handle->upd_padding_copy == 1) {
                    zero_ptr_in = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, 0, handle->blocksifm, handle->ifmblock, handle->input_pixels);
                    memset(zero_ptr_in, 0, handle->ifmblock * handle->input_pixels * sizeof(element_input_type));
                    for (ij = 0; ij < handle->ifhp; ij++) {
                      for (ii = 0; ii < handle->ifwp; ii++) {
                        for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
                          LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, ifm2, (ij + handle->desc.pad_h) * IFWP + (ii + handle->desc.pad_w), handle->blocksifm, handle->ifmblock, handle->input_pixels) =
                            LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, ii, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                        }
                      }
                    }
                  } else {
                    TRANS_INPUT(img, ifm1);
                  }
                }
                for (kj = 0; kj < handle->desc.R; ++kj) {
                  for (ki = 0; ki < handle->desc.S; ++ki) {
                    /* Determine if destination is the accumulation scratch or the intermediate fp32 weight tensor */
                    if (handle->use_intermediate_f32_wt_tensor == 1) {
                      dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(6, weight_private_f32, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                    } else {
                      dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(2, filter_tmp, 0, 0, handle->ofmblock);
                    }
                    gemm_kernel( &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, pix/2, 0, 0, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2),
                        &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, pix + kj * IFWP + ki, handle->blocksifm, handle->ifmblock, handle->input_pixels),
                        dst_ptr);
                    /* Convert fully caccumulated buffer to bf16 weight buffer in case of full accumulation has happened */
                    if (pix + handle->pixel_blocking >= handle->n_used_pixels) {
                      LIBXSMM_VLA_DECL(2, float, filter_acc_buffer, (float*)dst_ptr, handle->ofmblock);
                      for (ij = 0; ij < handle->ifmblock; ij+=2) {
                        for (ii = 0; ii < handle->ofmblock; ii+=16) {
                          c01 = LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)&LIBXSMM_VLA_ACCESS(2, filter_acc_buffer, ij+1, ii, handle->ofmblock)), LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)&LIBXSMM_VLA_ACCESS(2, filter_acc_buffer, ij, ii, handle->ofmblock)));
                          _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(7, weight_dst, ofm1, ifm1, kj, ki, ij/2, ii, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock/2, handle->ofmblock, 2), _mm512_permutexvar_epi16(perm_index, (__m512i)c01));
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

libxsmm_barrier_wait(handle->barrier, ltid);

if (handle->weight_copies > 1) {
  const int filter_size = handle->desc.R  * handle->desc.S * handle->desc.C * handle->desc.K;
  LIBXSMM_VLA_DECL(2, element_filter_type, weight_copies_buffer, (element_filter_type*)((char*)handle->scratch + handle->upd_filter_scratch_offset), filter_size);
  element_filter_type *weight_global_ptr = (element_filter_type*) handle->grad_filter->data;
  for ( j = reduce_thr_begin; j < reduce_thr_end; j++) {
    __m512 weight_sum = _mm512_setzero_ps();
    for ( i = 0; i < handle->weight_copies; i++ ) {
      weight_sum = _mm512_add_ps(weight_sum, _mm512_loadcvt_bf16_fp32(&LIBXSMM_VLA_ACCESS(2, weight_copies_buffer, i, j*16, filter_size)));
    }
    _mm512_streamstorecvt_fp32_bf16( ((libxsmm_bfloat16*) weight_global_ptr) + j*16, weight_sum);
  }
  libxsmm_barrier_wait(handle->barrier, ltid);
}
handle->tilerelease_kernel(NULL, NULL, NULL);

#undef TRANS_OUTPUT_TO_VNNI_FORMAT
#undef TRANS_INPUT
