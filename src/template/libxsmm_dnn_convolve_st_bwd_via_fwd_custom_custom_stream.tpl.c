/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#define IMG_LOOP_INIT 0
#define IFM_LOOP_INIT 1
#define IFM_LOOP_CLOSE 2
#define CONVOLUTION_KERNEL 3

const int ltid = tid-start_thread;

int BLOCKSIFM = handle->blocksifm_lp;
int BLOCKSOFM = handle->blocksofm_lp;
int oKB = handle->desc.K/16;
int iCB = handle->desc.C/16;

/* number of tasks for transpose that could be run in parallel */
const int transpose_work = (handle->use_lp_kernel == 0
  ? (BLOCKSOFM * (BLOCKSIFM * handle->fm_lp_block))
#if 0
  : (handle->desc.C * handle->desc.K));
#else
  : (oKB * iCB));
#endif

/* compute chunk size */
const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : ((transpose_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

/* fusion flags */
int fuse_postconv_ops_in_kernel = 0, compute_batch_stats_bwd_externally = 0, overwrite_output_externally = 0, fuse_relu_externally = 0, downconvert_to_bf16_externally = 0;

/* Pointer variables  */
element_output_type *input_base;
element_output_type *input_ptr;
element_filter_type *weight_base;
element_input_type *output_base;
element_input_type *regular_input_base;
element_output_type *copy_ptr;
element_output_type *prefetch_ptr;

/* BN Fusion related pointer variables */
float *bmean_ptr = NULL, *brstd_ptr = NULL, *dbeta_ptr = NULL, *dgamma_ptr = NULL;
element_input_type *input_bn_ptr = NULL, *input_add_ptr = NULL;
int *bn_outstats_stream, *bn_instats_stream, *bn_input_stream;
int stats_in_offset = 0, stats_out_offset = 0, bn_input_offset = 0, fm = 0;
int bn_stream_index = 0;
float placeholder = 0.0;
libxsmm_dnn_fusedbn *pre_bn = handle->pre_bn;
const int bn_ifh = pre_bn->desc.H;
const int bn_ifw = pre_bn->desc.W;
const int bn_sh = pre_bn->desc.u;
const int bn_sw = pre_bn->desc.v;
const int bn_ofh = bn_ifh/bn_sh;
const int bn_ofw = bn_ifw/bn_sw;
const int bn_iph = pre_bn->desc.pad_h_in;
const int bn_ipw = pre_bn->desc.pad_w_in;
const int bn_oph = pre_bn->desc.pad_h_out;
const int bn_opw = pre_bn->desc.pad_w_out;
const int bn_ofhp = bn_ofh + 2*bn_oph;
const int bn_ofwp = bn_ofw + 2*bn_opw;
const int bn_ifhp = bn_ifh + 2*bn_iph;
const int bn_ifwp = bn_ifw + 2*bn_ipw;
const int bn_nBlocksFm = pre_bn->blocksifm;
const int nImg = handle->desc.N;

/* Scratch7 zeroing related logistics  */
int imgpt = (handle->desc.N + handle->desc.threads - 1)/handle->desc.threads;
int threads_per_image = handle->desc.threads / handle->desc.N;
int my_img_start = LIBXSMM_MIN( ltid * imgpt, handle->desc.N);
int my_img_end = LIBXSMM_MIN( (ltid+1) * imgpt, handle->desc.N);
int my_ifm_start = 0;
int my_ifm_end = bn_nBlocksFm;

/* Padding related variables */
const int padded_h = handle->ofhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ofwp + 2 * handle->desc.pad_w;
const size_t output_buffer_size = BLOCKSOFM * padded_h * padded_w * handle->ofmblock;
LIBXSMM_VLA_DECL(5, element_output_type, output_buffer,
  (element_output_type*)(((char*)handle->scratch5) + ltid * LIBXSMM_UP2(output_buffer_size * sizeof(element_output_type), LIBXSMM_CACHELINE)),
  padded_h, padded_w, handle->ofmblock_lp, handle->fm_lp_block);

libxsmm_convfunction kernel_bwd = (libxsmm_convfunction)handle->code_bwd[0].xconv.sconv;
libxsmm_convfunction kernel2_bwd = (libxsmm_convfunction)handle->code_bwd[1].xconv.sconv;
libxsmm_convfunction kernel_pool[2];
char *variant = handle->kernel_bwd_variant_ptrs[ltid];

/* accumulation scratch for fp32->bf16 downconvert */
#if !defined(LIBXSMM_DNN_VLA_TLS2)
float *const accumulators_scratch = (float*)(((char*)handle->scratch6) +
  ltid * LIBXSMM_UP2(handle->ifmblock_hp * handle->desc.W * handle->desc.H * sizeof(float), LIBXSMM_CACHELINE));
#else
float accumulators_scratch_array[handle->ifmblock_hp * handle->desc.W * handle->desc.H];
float *const accumulators_scratch = accumulators_scratch_array;
#endif

/* Input tensor declaration */
/* regular/high precision */
element_input_type* del_in = 0;
kernel_pool[0] = kernel_bwd;
kernel_pool[1] = kernel2_bwd;

del_in = ((element_input_type*)handle->grad_input->data) + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * (handle->ifmblock_hp);

{ /* open new scope for additional variable declarations (C89) */
  LIBXSMM_VLA_DECL(5, element_input_type, del_input, del_in, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
  /* Output tensor declaration */
  element_output_type *const out = ((element_output_type*)handle->grad_output->data) /* + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block*/;
  LIBXSMM_VLA_DECL(6, element_output_type, del_out, out, BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);

  /* Weight and transpose_weight tensor declaration */
  LIBXSMM_VLA_DECL(7, element_filter_type, wt, (element_filter_type*)handle->reg_filter->data, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(7, element_filter_type, tr_wt2, (element_filter_type*)handle->scratch1, BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ofmblock_lp, handle->ifmblock_hp, handle->fm_lp_block);

  /* Auxiliary integer variables   */
  int instr, n_segments, offset_i, offset_o, offset_w, pi, po, pw, pc, i, n_convs, conv_i, ifm1, img = 0, ifm2, ij, ii, n_segs, vi = 0, segment_type;
  /* Stream related variables  */
  segment_t *code_stream;
  int *stream = handle->compute_bwd_indices_ptrs[ltid];
  int pool_index;
  int ifm1ofm1, kj, ki, ofm2, ofm1;
  /* Kernel related variables  */
  libxsmm_convfunction kernel = (libxsmm_convfunction)handle->code_bwd[0].xconv.sconv;
  libxsmm_xmcopyfunction jitted_matcopy = handle->matcopy_bwd[0].xmatcopy;
  libxsmm_xmcopyfunction jitted_zero_overwrite = handle->matcopy_bwd[1].xmatcopy;

  /* Initialize base pointers */
  if ( handle->padding_flag == 1  ) {
    input_base = &LIBXSMM_VLA_ACCESS(5, output_buffer, 0, 0, 0, 0, 0,
        padded_h, padded_w, handle->ofmblock_lp, handle->fm_lp_block);
    /* we need to set the scratch to zero */
    /* @TODO: we need to find a better/faster code here, e.g. just setting the rim */
    memset(input_base, 0, output_buffer_size * sizeof(element_output_type));
  } else {
    input_base = &LIBXSMM_VLA_ACCESS(6, del_out, 0, 0, 0, 0, 0, 0,
        BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);
  }

  output_base = &LIBXSMM_VLA_ACCESS(5, del_input, 0, 0, 0, 0, 0,
      handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
  weight_base = &LIBXSMM_VLA_ACCESS(7, tr_wt2, 0, 0, 0, 0, 0, 0, 0,
      BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ofmblock_lp, handle->ifmblock_hp, handle->fm_lp_block);

  instr = handle->n_entries_bwd[ltid];
  n_segments = handle->n_bwd_code_segments[ltid];
  i = 0;
  code_stream = handle->bwd_code_segments[ltid];

  /* lazy barrier init */
  libxsmm_barrier_init(handle->barrier, ltid);

  /* set accumulation scratch initially to zero */
  if (handle->use_accumulation_scratch) {
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
    float *scratch_ptr = accumulators_scratch;
    __m512 zero_reg = _mm512_setzero_ps();
    for ( ij = 0; ij < handle->desc.H; ij++ ) {
      for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
        _mm512_store_ps(scratch_ptr+ii, zero_reg);
      }
      scratch_ptr += handle->desc.W*handle->ifmblock_hp;
    }
#else
    LIBXSMM_ASSERT(0);
#endif
  }

  if ( (handle->options & LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE) > 0 ) {
    weight_base = (element_filter_type*)handle->reg_filter_tr->data;
  } else {
    if (handle->use_lp_kernel == 0) {
      for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
        ofm1 = ifm1ofm1 / BLOCKSIFM;
        ifm1 = ifm1ofm1 % BLOCKSIFM;
        for (kj=0; kj < handle->desc.R; kj++) {
          for (ki=0; ki < handle->desc.S; ki++) {
            for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
              for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                LIBXSMM_VLA_ACCESS(7, tr_wt2, ifm1, ofm1, handle->desc.R-1-kj, handle->desc.S-1-ki, ofm2, ifm2, 0, BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block) =
                  LIBXSMM_VLA_ACCESS(7, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, 0, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
              }
            }
          }
        }
      }
    } else {
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
        int icb, okb, t1;
        const __m512i permute_index = _mm512_set_epi32(15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0);
        const  __m256i scatter_index = _mm256_set_epi32(7*32, 6*32, 5*32, 4*32,  3*32, 2*32, 1*32, 0*32);
        for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
          icb = ifm1ofm1 / oKB;
          okb = ifm1ofm1 % oKB;
          for (kj=0; kj < handle->desc.R; kj++) {
            for (ki=0; ki < handle->desc.S; ki++) {
              for (t1 = 0; t1 < 8; t1++) {
                __m512i cur_cache_line = _mm512_loadu_si512(&LIBXSMM_VLA_ACCESS(7, wt, okb, icb, kj, ki, t1, 0, 0,
                  BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block));
                __m512i permuted_cache_line = _mm512_permutexvar_epi32(permute_index, cur_cache_line);
                __m256i lo_half = LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(permuted_cache_line, 0);
                __m256i hi_half = LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(permuted_cache_line, 1);
                __m256i lo_zipped = _mm256_unpacklo_epi16(lo_half, hi_half);
                __m256i hi_zipped = _mm256_unpackhi_epi16(lo_half, hi_half);
                __m128i part0 = _mm256_extractf128_si256(lo_zipped,0);
                __m128i part2 = _mm256_extractf128_si256(lo_zipped,1);
                __m128i part1 = _mm256_extractf128_si256(hi_zipped,0);
                __m128i part3 =  _mm256_extractf128_si256(hi_zipped,1);
                __m512i compact = _mm512_inserti32x4(LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), part0, 0);
                compact = _mm512_inserti32x4 (compact, part1, 1);
                compact = _mm512_inserti32x4 (compact, part2, 2);
                compact = _mm512_inserti32x4 (compact, part3, 3);
                _mm512_i32scatter_epi64((long long int*)&LIBXSMM_VLA_ACCESS(7, tr_wt2, icb, okb, handle->desc.R-1-kj, handle->desc.S-1-ki, 0, 2*t1, 0,
                  BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block), scatter_index, compact, 2);
              }
            }
          }
        }
#else /* won't happen as this code only runs on AVX512 platforms */
        LIBXSMM_ASSERT(0);
#endif
    }
    weight_base = &LIBXSMM_VLA_ACCESS(7, tr_wt2, 0, 0, 0, 0, 0, 0, 0,
        BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
    libxsmm_barrier_wait(handle->barrier, ltid);
  }
  pool_index = 0;
  i = 0;
  bn_outstats_stream = handle->bn_stats_indices_ptrs[ltid];
  bn_instats_stream = handle->bn_aux_stats_indices_ptrs[ltid];
  bn_input_stream = handle->bn_aux_input_indices_ptrs[ltid];

  /* Initialize potential bn-fuse related base pointers */
  bmean_ptr = (float*) &placeholder;
  brstd_ptr = (float*) &placeholder;
  input_bn_ptr = (element_input_type*) &placeholder;
  dgamma_ptr = (float*) &placeholder;
  dbeta_ptr = (float*) &placeholder;
  input_add_ptr = (element_input_type*) output_base;
  regular_input_base = (element_input_type*) &placeholder;

  if (handle->compute_eltwise_in_kernel_bwd) {
    LIBXSMM_VLA_DECL(5, element_input_type,  dinput_add, (element_input_type*) pre_bn->grad_add->data, bn_nBlocksFm, bn_ifhp, bn_ifwp, 16);
    input_add_ptr = &LIBXSMM_VLA_ACCESS(5, dinput_add, 0, 0, 0, 0, 0, bn_nBlocksFm, bn_ifhp, bn_ifwp, 16);
  }   

  if (handle->perform_relu_in_kernel) {    
    LIBXSMM_VLA_DECL(5, element_input_type, original_input, ((element_input_type*)handle->reg_input->data) + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in * handle->ifmblock), handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
    regular_input_base = &LIBXSMM_VLA_ACCESS(5, original_input, 0, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
  } 

  if (handle->compute_batch_stats_in_kernel_bwd) {
    LIBXSMM_VLA_DECL(5, element_input_type, dinput_add, (element_input_type*) pre_bn->grad_add->data, bn_nBlocksFm, bn_ifhp, bn_ifwp, 16);
    input_add_ptr = &LIBXSMM_VLA_ACCESS(5, dinput_add, 0, 0, 0, 0, 0, bn_nBlocksFm, bn_ifhp, bn_ifwp, 16);    
    LIBXSMM_VLA_DECL(5, element_input_type, original_input, ((element_input_type*)handle->reg_input->data) + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in * handle->ifmblock), handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
    LIBXSMM_VLA_DECL(5, element_input_type, bn_input,     (element_input_type* )pre_bn->reg_input->data,   bn_nBlocksFm, bn_ifhp, bn_ifwp, 16);
    LIBXSMM_VLA_DECL(2, float,  bmean_lcl,      (float*)pre_bn->expvalue->data,   16);
    LIBXSMM_VLA_DECL(2, float,  brstd_lcl,      (float*)pre_bn->stddev->data,     16);
    LIBXSMM_VLA_DECL(4, float,  kernel_stats_lcl, (float*)handle->scratch7, bn_nBlocksFm, handle->desc.N, 16);

    regular_input_base  = &LIBXSMM_VLA_ACCESS(5, original_input, 0, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
    input_add_ptr       = &LIBXSMM_VLA_ACCESS(5, dinput_add, 0, 0, 0, 0, 0, bn_nBlocksFm, bn_ifhp, bn_ifwp, 16);
    bmean_ptr           = &LIBXSMM_VLA_ACCESS(2, bmean_lcl,     0, 0, 16);
    brstd_ptr           = &LIBXSMM_VLA_ACCESS(2, brstd_lcl,     0, 0, 16);
    input_bn_ptr        = &LIBXSMM_VLA_ACCESS(5, bn_input, 0, 0, 0, 0, 0, bn_nBlocksFm, bn_ifhp, bn_ifwp, 16);
    dbeta_ptr           = &LIBXSMM_VLA_ACCESS(4, kernel_stats_lcl, 0, 0, 0, 0, bn_nBlocksFm, handle->desc.N, 16);
    dgamma_ptr          = &LIBXSMM_VLA_ACCESS(4, kernel_stats_lcl, 1, 0, 0, 0, bn_nBlocksFm, handle->desc.N, 16);
  }

  /* Initialize scratch7 to zero in case of batch stats fusion  */
  if (handle->fuse_batchstats_bwd == 1) {
    LIBXSMM_VLA_DECL(4, float, kernel_stats_lcl, (float*)handle->scratch7, bn_nBlocksFm, handle->desc.N, 16);
    const __m512 zero_reg = _mm512_setzero_ps();
    dbeta_ptr  = &LIBXSMM_VLA_ACCESS(4, kernel_stats_lcl, 0, 0, 0, 0, bn_nBlocksFm, handle->desc.N, 16);
    dgamma_ptr = &LIBXSMM_VLA_ACCESS(4, kernel_stats_lcl, 1, 0, 0, 0, bn_nBlocksFm, handle->desc.N, 16);
    for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
      for (img = my_img_start; img < my_img_end; img++) {
        _mm512_storeu_ps(dbeta_ptr+img*16+ifm1*handle->desc.N*16, zero_reg);
        _mm512_storeu_ps(dgamma_ptr+img*16+ifm1*handle->desc.N*16, zero_reg);
      }
    }
  }

  /* Parse properly the first segment out of the hot loop to deal with the case with 0 segments  */
  n_segs =        (n_segments == 0) ? 1 : n_segments;
  n_convs =       (n_segments == 0) ? instr : code_stream[0].n_convs;
  segment_type =  (n_segments == 0) ? CONVOLUTION_KERNEL : code_stream[0].segment_type;

  /* Set properly the fusion flags  */
  fuse_postconv_ops_in_kernel = (handle->compute_batch_stats_in_kernel_bwd || handle->compute_eltwise_in_kernel_bwd || handle->perform_relu_in_kernel) ? 1 : 0;
  overwrite_output_externally = (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_bwd == 0)) ? 1 : 0;
  fuse_relu_externally = (handle->perform_relu_in_kernel) ? 0 : 1;
  downconvert_to_bf16_externally = (handle->use_accumulation_scratch) ? 1 : 0;
  compute_batch_stats_bwd_externally = (handle->compute_batch_stats_in_kernel_bwd) ? 0 : 1;

  for (pc = 0; pc < n_segs; pc++) {
    switch (segment_type)
    {
      case IMG_LOOP_INIT:
        img = code_stream[pc].aux_index;
        if (handle->padding_flag) {
#include "libxsmm_dnn_bwd_custom_custom_padding.tpl.c"
        }
        break;
      case IFM_LOOP_INIT:
        if (overwrite_output_externally) {
          jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
        }
        break;
      case IFM_LOOP_CLOSE:
#include "libxsmm_dnn_bwd_fuse_postconv_ops_externally.tpl.c"
        break;
      case CONVOLUTION_KERNEL:
        /* TODO: Make CONVOLUTION_STREAK segment and move convolution loop  here */
        break;
      default:
        break;
    }

    /* Run the stream of convolutions  */
    for (conv_i = 0; conv_i < n_convs; conv_i++) {
      vi = (handle->n_variants == 1) ? 0 : variant[pool_index]; 
      offset_i = stream[i]; offset_w = stream[i+1]; offset_o = stream[i+2]; pi = stream[i+3]; pw = stream[i+4]; po = stream[i+5];
      stats_in_offset =  (handle->compute_batch_stats_in_kernel_bwd) ? bn_instats_stream[bn_stream_index] : 0;
      stats_out_offset = (handle->compute_batch_stats_in_kernel_bwd) ? bn_outstats_stream[bn_stream_index] : 0;
      bn_input_offset =  (handle->compute_batch_stats_in_kernel_bwd || handle->compute_eltwise_in_kernel_bwd) ? bn_input_stream[bn_stream_index] : 0;
      kernel_pool[vi](input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po,
          regular_input_base + offset_o, accumulators_scratch + offset_o,
          bmean_ptr + stats_in_offset, brstd_ptr + stats_in_offset, input_bn_ptr + bn_input_offset, dgamma_ptr + stats_out_offset, dbeta_ptr + stats_out_offset, input_add_ptr + bn_input_offset);
      bn_stream_index++;
      pool_index++;
      i += 3;
    }   

    /* Set up for next segment */
    if (pc+1 < n_segs) {
      n_convs       = code_stream[pc+1].n_convs;
      segment_type  = code_stream[pc+1].segment_type;
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);

  if (handle->fuse_batchstats_bwd == 1) {
    /* now we need to reduce the del_gamm and del_beta */
    LIBXSMM_VLA_DECL(4, float,  kernel_stats_red, (float*)handle->scratch7, bn_nBlocksFm, handle->desc.N, 16);
    LIBXSMM_VLA_DECL(2, float,  dgamma_result,     (float*)pre_bn->grad_gamma->data, 16);
    LIBXSMM_VLA_DECL(2, float,  dbeta_result,      (float*)pre_bn->grad_beta->data,  16);
    const int work = bn_nBlocksFm;
    const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
    const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
    for ( fm = thr_begin; fm < thr_end; ++fm ) {
      float *del_gamma_img_ptr = &LIBXSMM_VLA_ACCESS(4, kernel_stats_red, 1, fm, 0, 0, bn_nBlocksFm, handle->desc.N, 16);
      float *del_beta_img_ptr  = &LIBXSMM_VLA_ACCESS(4, kernel_stats_red, 0, fm, 0, 0, bn_nBlocksFm, handle->desc.N, 16);
      __m512 lcl_vdgamma = _mm512_setzero_ps();
      __m512 lcl_vdbeta  = _mm512_setzero_ps();

      for ( img=0; img < nImg; img++ ) {
        lcl_vdgamma = _mm512_add_ps( lcl_vdgamma, _mm512_loadu_ps( del_gamma_img_ptr ) );
        lcl_vdbeta  = _mm512_add_ps( lcl_vdbeta,  _mm512_loadu_ps( del_beta_img_ptr  ) );
        del_gamma_img_ptr += 16;
        del_beta_img_ptr  += 16;
      }

      _mm512_storeu_ps( &LIBXSMM_VLA_ACCESS(2, dgamma_result, fm, 0, 16), lcl_vdgamma );
      _mm512_storeu_ps( &LIBXSMM_VLA_ACCESS(2, dbeta_result,  fm, 0, 16), lcl_vdbeta  );
    }
    libxsmm_barrier_wait(handle->barrier, ltid);
  }
}

