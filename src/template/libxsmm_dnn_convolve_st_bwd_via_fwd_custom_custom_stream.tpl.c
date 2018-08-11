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
/* Pointer variables  */
element_output_type *input_base;
element_output_type *input_ptr;
element_filter_type *weight_base;
element_input_type *output_base;
element_output_type *copy_ptr;
element_output_type *prefetch_ptr;

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

LIBXSMM_ALIGNED(float scale_factor, 64);
LIBXSMM_ALIGNED(float *max_vals, 64) = NULL;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
__m512 max_abs;
#else /* won't happen as this code only runs on AVX512 platforms */
  LIBXSMM_ASSERT(0);
#endif

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

/* select pointer based on precision */
if (handle->datatype_in != handle->datatype_out) {
  del_in = ((element_input_type*)handle->grad_input->data) + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * (handle->ifmblock_hp);
} else {
  del_in = ((element_input_type*)handle->grad_input->data) + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * (handle->ifmblock_hp);
}

if (handle->use_lp_kernel == 1) {
  scale_factor = libxsmm_sexp2(-1.f*((float)(handle->reg_filter->scf + handle->grad_output->scf)));
}

if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
  LIBXSMM_VLA_DECL(2, float, maxstats, (float*)handle->maxstats_bwd->data, handle->ifmblock_hp);
  max_vals = (float*) &LIBXSMM_VLA_ACCESS(2, maxstats, ltid, 0, handle->ifmblock_hp);
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  max_abs = _mm512_setzero_ps();
  _mm512_store_ps(max_vals, max_abs);
#else /* won't happen as this code only runs on AVX512 platforms */
  LIBXSMM_ASSERT(0);
#endif
}

{ /* open new scope for additional variable declarations (C89) */
  LIBXSMM_VLA_DECL(5, element_input_type, del_input, del_in, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
  /* Output tensor declaration */
  element_output_type *const out = ((element_output_type*)handle->grad_output->data) /* + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block*/;
  LIBXSMM_VLA_DECL(6, element_output_type, del_out, out, BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);

  /* Weight and transpose_weight tensor declaration */
  LIBXSMM_VLA_DECL(7, element_filter_type, wt, (element_filter_type*)handle->reg_filter->data, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(7, element_filter_type, tr_wt2, (element_filter_type*)handle->scratch1, BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ofmblock_lp, handle->ifmblock_hp, handle->fm_lp_block);

  /* Auxiliary integer variables   */
  int instr, n_segments, offset_i, offset_o, offset_w, pi, po, pw, pc, i, n_convs, conv_i, ifm1, img = 0, ifm2, ij, ii;
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
    float *scratch_ptr = accumulators_scratch;
    __m512 zero_reg = _mm512_setzero_ps();
    for ( ij = 0; ij < handle->desc.H; ij++ ) {
      for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
        _mm512_store_ps(scratch_ptr+ii, zero_reg);
      }
      scratch_ptr += handle->desc.W*handle->ifmblock_hp;
    }
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
      if  (( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32)) && ((handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0))  {
        int fm_lp_ind;
        for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
          ifm1 = ifm1ofm1 / oKB;
          ofm1 = ifm1ofm1 % oKB;
          for (kj=0; kj < handle->desc.R; kj++) {
            for (ki=0; ki < handle->desc.S; ki++) {
              for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                  for (fm_lp_ind = 0; fm_lp_ind < handle->fm_lp_block; fm_lp_ind++) {
                    LIBXSMM_VLA_ACCESS(7, tr_wt2, ifm1, ofm1, handle->desc.R-1-kj, handle->desc.S-1-ki, ofm2/handle->fm_lp_block, ifm2*handle->fm_lp_block+fm_lp_ind, ofm2%handle->fm_lp_block, BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block) =
                      LIBXSMM_VLA_ACCESS(7, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, fm_lp_ind, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
                  }
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
    }
    weight_base = &LIBXSMM_VLA_ACCESS(7, tr_wt2, 0, 0, 0, 0, 0, 0, 0,
        BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
    libxsmm_barrier_wait(handle->barrier, ltid);
  }
  pool_index = 0;
  i = 0;

  if (n_segments) {
    /* We have segmented the stream of convolutions since we need to inject different functionalities...  */
    code_stream = handle->bwd_code_segments[ltid];
    /* TODO: Second condition guarantees we run the img_par code when we have MB=1 -- and hopefully HUGE images */
    if (handle->desc.N*BLOCKSIFM >= handle->desc.threads && !((handle->desc.N == 1) && (handle->bwd_ofh_rb == 1))) {
      if (handle->perform_relu_in_kernel == 1) {/* do RELU stuff in the kernel  */
        LIBXSMM_VLA_DECL(5, element_input_type, original_input, ((element_input_type*)handle->reg_input->data) + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in * handle->ifmblock), handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
        element_input_type *regular_input_base;
        regular_input_base = &LIBXSMM_VLA_ACCESS(5, original_input, 0, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);

        if (handle->n_variants == 2) {
          for (pc = 0; pc < n_segments; pc++) {
            instr = code_stream[pc].segment_type;
            n_convs = code_stream[pc].n_convs;

            if (instr == IMG_LOOP_INIT) {
              img = code_stream[pc].aux_index;
              /* Apply padding  */
              if (handle->padding_flag == 1) {
#               include "libxsmm_dnn_bwd_custom_custom_padding.tpl.c"
              }
            }

            if ( instr == IFM_LOOP_INIT ) {
              /* Apply bias if requested  */
              if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
                /*#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"*/
              }
              /* Overwrite output with zeros if requested */
              if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_bwd == 0) ) {
                jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
              }
            }

            if ( instr == IFM_LOOP_CLOSE) {
              if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
                element_input_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, del_input, img, /*ifm1*/code_stream[pc].aux_index, 0, 0, 0,
                    handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                for ( ij = 0; ij < handle->desc.H; ij++ ) {
                  for ( ii = 0; ii < handle->desc.W*handle->ifmblock; ii+=16 ) {
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
                    max_abs = _mm512_max_ps(max_abs, LIBXSMM_INTRINSICS_MM512_ABS_PS(LIBXSMM_INTRINSICS_MM512_LOAD_PS(cur_vec+ii)));
#else /* won't happen as this code only runs on AVX512 platforms */
                    LIBXSMM_ASSERT(0);
#endif
                  }
                  cur_vec += handle->ifwp*handle->ifmblock;
                }
              }

              /* @TODO this is a hack as it might conflict with MAX STATS fuse */
              /* down-convert to bf16 from fp32 */
              if (handle->use_accumulation_scratch) {
                element_input_type *input_dst = &LIBXSMM_VLA_ACCESS(5, del_input, img, code_stream[pc].aux_index/*ifm1*/, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
                float *scratch_ptr = accumulators_scratch;
                __m512 zero_reg = _mm512_setzero_ps();
                if ( handle->f32_bf16_cvt_rne ) {
                  __m512i vnaninf = _mm512_set1_epi32( 0x7f800000 );
                  __m512i vrneadd = _mm512_set1_epi32( 0x00007fff );
                  __m512i vfixup = _mm512_set1_epi32( 0x00000001 );
                  __m512i vfixupmask = _mm512_set1_epi32( 0x00010000 );
                  for ( ij = 0; ij < handle->desc.H; ij++ ) {
                    for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
                      __m512i vfp32     = _mm512_castps_si512( _mm512_loadu_ps(scratch_ptr+ii) );
                      __m512i vfp32nan  = _mm512_and_epi32( vfp32, vnaninf );
                      __m512i vfp32fixup  = _mm512_and_epi32( vfp32, vfixupmask );
                      __mmask16 rnemask = _mm512_cmp_epi32_mask( vfp32nan, vnaninf, _MM_CMPINT_NE );
                      __mmask16 fixupmask = _mm512_cmp_epi32_mask( vfp32fixup, vfixupmask, _MM_CMPINT_EQ );
                      __m512i vrnd = _mm512_mask_add_epi32( vrneadd , fixupmask, vrneadd, vfixup );
                      __m512i vfp32rne  = _mm512_mask_add_epi32( vfp32, rnemask, vfp32, vrnd );
                      __m512i vbfp16_32 = _mm512_srai_epi32( vfp32rne, 16 );
                      __m256i vbfp16    = _mm512_cvtepi32_epi16( vbfp16_32 );
                      _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
                      _mm256_storeu_si256( (__m256i*)(input_dst+ii), vbfp16 );
                    }
                    scratch_ptr += handle->desc.W*handle->ifmblock_hp;
                    input_dst += handle->ifwp*handle->ifmblock_hp;
                  }
                } else {
                  for ( ij = 0; ij < handle->desc.H; ij++ ) {
                    for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
                      __m512 tmp = _mm512_loadu_ps(scratch_ptr+ii);
                      __m256i vbfp16 =  _mm512_cvtepi32_epi16(_mm512_srai_epi32( _mm512_castps_si512( tmp ), 16));
                      _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
                      _mm256_storeu_si256( (__m256i*)(input_dst+ii), vbfp16 );
                    }
                    scratch_ptr += handle->desc.W*handle->ifmblock_hp;
                    input_dst += handle->ifwp*handle->ifmblock_hp;
                  }
                }
              }
            }

            /* Run the stream of convolutions for this segment */
            for (conv_i = 0; conv_i < n_convs; conv_i++) {
              const int vi = variant[pool_index]; /* avoid warning about char used as array index */
              offset_i = stream[i];
              offset_w = stream[i+1];
              offset_o = stream[i+2];
              pi = stream[i+3];
              pw = stream[i+4];
              po = stream[i+5];
              kernel_pool[vi](
                input_base + offset_i, weight_base + offset_w, output_base + offset_o,
                input_base + pi, weight_base + pw, output_base + po,
                regular_input_base + offset_o, &scale_factor, max_vals, accumulators_scratch + offset_o);
              ++pool_index;
              i += 3;
            }
          }
        } else {
          for (pc = 0; pc < n_segments; pc++) {
            instr = code_stream[pc].segment_type;
            n_convs = code_stream[pc].n_convs;
            if (instr == IMG_LOOP_INIT) {
              img = code_stream[pc].aux_index;
              /* Apply padding  */
              if (handle->padding_flag == 1) {
#include "libxsmm_dnn_bwd_custom_custom_padding.tpl.c"
              }
            }

            if ( instr == IFM_LOOP_INIT ) {
              /* Apply bias if requested  */
              if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
                /*#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"*/
              }
              /* Overwrite output with zeros if requested */
              if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_bwd == 0) ) {
                jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
              }
            }

            if ( instr == IFM_LOOP_CLOSE) {
              if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
                element_input_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, del_input, img, /*ifm1*/code_stream[pc].aux_index, 0, 0, 0,
                    handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                for ( ij = 0; ij < handle->desc.H; ij++ ) {
                  for ( ii = 0; ii < handle->desc.W*handle->ifmblock; ii+=16 ) {
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
                    max_abs = _mm512_max_ps(max_abs, LIBXSMM_INTRINSICS_MM512_ABS_PS(LIBXSMM_INTRINSICS_MM512_LOAD_PS(cur_vec+ii)));
#else /* won't happen as this code only runs on AVX512 platforms */
                    LIBXSMM_ASSERT(0);
#endif
                  }
                  cur_vec += handle->ifwp*handle->ifmblock;
                }
              }

              /* @TODO this is a hack as it might conflict with MAX STATS fuse */
              /* down-convert to bf16 from fp32 */
              if (handle->use_accumulation_scratch) {
                element_input_type *input_dst = &LIBXSMM_VLA_ACCESS(5, del_input, img, code_stream[pc].aux_index/*ifm1*/, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
                float *scratch_ptr = accumulators_scratch;
                __m512 zero_reg = _mm512_setzero_ps();
                if ( handle->f32_bf16_cvt_rne ) {
                  __m512i vnaninf = _mm512_set1_epi32( 0x7f800000 );
                  __m512i vrneadd = _mm512_set1_epi32( 0x00007fff );
                  __m512i vfixup = _mm512_set1_epi32( 0x00000001 );
                  __m512i vfixupmask = _mm512_set1_epi32( 0x00010000 );
                  for ( ij = 0; ij < handle->desc.H; ij++ ) {
                    for ( ii = 0; ii < handle->desc.W*handle->ifmblock; ii+=16 ) {
                      __m512i vfp32     = _mm512_castps_si512( _mm512_loadu_ps(scratch_ptr+ii) );
                      __m512i vfp32nan  = _mm512_and_epi32( vfp32, vnaninf );
                      __m512i vfp32fixup  = _mm512_and_epi32( vfp32, vfixupmask );
                      __mmask16 rnemask = _mm512_cmp_epi32_mask( vfp32nan, vnaninf, _MM_CMPINT_NE );
                      __mmask16 fixupmask = _mm512_cmp_epi32_mask( vfp32fixup, vfixupmask, _MM_CMPINT_EQ );
                      __m512i vrnd = _mm512_mask_add_epi32( vrneadd , fixupmask, vrneadd, vfixup );
                      __m512i vfp32rne  = _mm512_mask_add_epi32( vfp32, rnemask, vfp32, vrnd );
                      __m512i vbfp16_32 = _mm512_srai_epi32( vfp32rne, 16 );
                      __m256i vbfp16    = _mm512_cvtepi32_epi16( vbfp16_32 );
                      _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
                      _mm256_storeu_si256( (__m256i*)(input_dst+ii), vbfp16 );
                    }
                    scratch_ptr += handle->desc.W*handle->ifmblock;
                    input_dst += handle->ifwp*handle->ifmblock;
                  }
                } else {
                  for ( ij = 0; ij < handle->desc.H; ij++ ) {
                    for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
                      __m512 tmp = _mm512_loadu_ps(scratch_ptr+ii);
                      __m256i vbfp16 =  _mm512_cvtepi32_epi16(_mm512_srai_epi32( _mm512_castps_si512( tmp ), 16));
                      _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
                      _mm256_storeu_si256( (__m256i*)(input_dst+ii), vbfp16 );
                    }
                    scratch_ptr += handle->desc.W*handle->ifmblock_hp;
                    input_dst += handle->ifwp*handle->ifmblock_hp;
                  }
                }
              }
            }

            /* Run the stream of convolutions for this segment */
            for (conv_i = 0; conv_i < n_convs; conv_i++) {
              offset_i = stream[i];
              offset_w = stream[i+1];
              offset_o = stream[i+2];
              pi = stream[i+3];
              pw = stream[i+4];
              po = stream[i+5];
              kernel(
                input_base + offset_i, weight_base + offset_w, output_base + offset_o,
                input_base + pi, weight_base + pw, output_base + po,
                regular_input_base + offset_o, &scale_factor, max_vals, accumulators_scratch + offset_o);
              i += 3;
            }
          }
        }
      } else { /* We don't do RELU stuff in the kernel  */
        if (handle->n_variants == 2) {
          for (pc = 0; pc < n_segments; pc++) {
            instr = code_stream[pc].segment_type;
            n_convs = code_stream[pc].n_convs;

            if (instr == IMG_LOOP_INIT) {
              img = code_stream[pc].aux_index;
              /* Apply padding  */
              if (handle->padding_flag == 1) {
#include "libxsmm_dnn_bwd_custom_custom_padding.tpl.c"
              }
            }

            if ( instr == IFM_LOOP_INIT ) {
              /* Apply bias if requested  */
              if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
                /*#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"*/
              }
              /* Overwrite output with zeros if requested */
              if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_bwd == 0) ) {
                jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
              }
            }

            if ( instr == IFM_LOOP_CLOSE ) {
              if ( ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) && (handle->use_accumulation_scratch == 0) ){
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
                LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*) handle->reg_input->data,  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                LIBXSMM_VLA_DECL(5, element_input_type, del_input_2, (element_input_type*) handle->grad_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                element_input_type *orig_input_ptr;
                element_input_type *del_input_ptr;
                __m512 zero_reg  = _mm512_setzero_ps();
                __m512 orig_reg;
                __mmask16 mask;
                orig_input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, /*ifm1*/code_stream[pc].aux_index, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                del_input_ptr = &LIBXSMM_VLA_ACCESS(5, del_input_2, img, /*ifm1*/code_stream[pc].aux_index, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                for (ij = 0; ij < handle->desc.H; ij++) {
                  for (ii = 0; ii < handle->desc.W * 16; ii += 16) {
                    orig_reg  = LIBXSMM_INTRINSICS_MM512_LOAD_PS(orig_input_ptr + ii);
                    mask = _mm512_cmp_ps_mask(zero_reg, orig_reg, _CMP_EQ_OQ);
                    _mm512_mask_storeu_ps(del_input_ptr + ii, mask, zero_reg);
                  }
                  orig_input_ptr += handle->ifwp * 16;
                  del_input_ptr += handle->ifwp *16;
                }
#else /* won't happen as this code only runs on AVX512 platforms */
                LIBXSMM_ASSERT(0);
#endif
              }

              if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
                element_input_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, del_input, img, /*ifm1*/code_stream[pc].aux_index, 0, 0, 0,
                    handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
                for ( ij = 0; ij < handle->desc.H; ij++ ) {
                  for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
                    max_abs = _mm512_max_ps(max_abs, LIBXSMM_INTRINSICS_MM512_ABS_PS(LIBXSMM_INTRINSICS_MM512_LOAD_PS(cur_vec+ii)));
#else /* won't happen as this code only runs on AVX512 platforms */
                    LIBXSMM_ASSERT(0);
#endif
                  }
                  cur_vec += handle->ifwp*handle->ifmblock_hp;
                }
              }

              /* @TODO this is a hack as it might conflict with MAX STATS/ReLU fuse */
              /* down-convert to bf16 from fp32 */
              if (handle->use_accumulation_scratch) {
                if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) {
                  LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*) handle->reg_input->data,  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
                  element_input_type *orig_input_ptr;
                  orig_input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, /*ifm1*/code_stream[pc].aux_index, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
                  element_input_type *input_dst = &LIBXSMM_VLA_ACCESS(5, del_input, img, code_stream[pc].aux_index/*ifm1*/, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
                  float *scratch_ptr = accumulators_scratch;
                  __m512 zero_reg = _mm512_setzero_ps();
                  __mmask16 mask;
                  __m256i zero_reg_relu = _mm256_setzero_si256();
                  __m256i orig_reg;
                  if ( handle->f32_bf16_cvt_rne ) {
                    __m512i vnaninf = _mm512_set1_epi32( 0x7f800000 );
                    __m512i vrneadd = _mm512_set1_epi32( 0x00007fff );
                    __m512i vfixup = _mm512_set1_epi32( 0x00000001 );
                    __m512i vfixupmask = _mm512_set1_epi32( 0x00010000 );
                    for ( ij = 0; ij < handle->desc.H; ij++ ) {
                      for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
                        __m512i vfp32     = _mm512_castps_si512( _mm512_loadu_ps(scratch_ptr+ii) );
                        __m512i vfp32nan  = _mm512_and_epi32( vfp32, vnaninf );
                        __m512i vfp32fixup  = _mm512_and_epi32( vfp32, vfixupmask );
                        __mmask16 rnemask = _mm512_cmp_epi32_mask( vfp32nan, vnaninf, _MM_CMPINT_NE );
                        __mmask16 fixupmask = _mm512_cmp_epi32_mask( vfp32fixup, vfixupmask, _MM_CMPINT_EQ );
                        __m512i vrnd = _mm512_mask_add_epi32( vrneadd , fixupmask, vrneadd, vfixup );
                        __m512i vfp32rne  = _mm512_mask_add_epi32( vfp32, rnemask, vfp32, vrnd );
                        __m512i vbfp16_32 = _mm512_srai_epi32( vfp32rne, 16 );
                        __m256i vbfp16    = _mm512_cvtepi32_epi16( vbfp16_32 );
                        _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
                        orig_reg  = _mm256_loadu_si256( (__m256i*) (orig_input_ptr + ii));
                        mask = _mm256_cmp_epi16_mask(zero_reg_relu, orig_reg, _MM_CMPINT_NE);
                        _mm256_mask_storeu_epi16( (__m256i*) (input_dst+ii), mask, vbfp16);
                      }
                      scratch_ptr += handle->desc.W*handle->ifmblock_hp;
                      input_dst += handle->ifwp*handle->ifmblock_hp;
                      orig_input_ptr += handle->ifwp * 16;
                    }
                  } else {
                    for ( ij = 0; ij < handle->desc.H; ij++ ) {
                      for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
                        __m512 tmp = _mm512_loadu_ps(scratch_ptr+ii);
                        __m256i vbfp16 =  _mm512_cvtepi32_epi16(_mm512_srai_epi32( _mm512_castps_si512( tmp ), 16));
                        _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
                        orig_reg  = _mm256_loadu_si256( (__m256i*) (orig_input_ptr + ii));
                        mask = _mm256_cmp_epi16_mask(zero_reg_relu, orig_reg, _MM_CMPINT_NE);
                        _mm256_mask_storeu_epi16( (__m256i*) (input_dst+ii), mask, vbfp16);
                      }
                      scratch_ptr += handle->desc.W*handle->ifmblock_hp;
                      input_dst += handle->ifwp*handle->ifmblock_hp;
                      orig_input_ptr += handle->ifwp * 16;
                    }
                  }
                } else {
                  element_input_type *input_dst = &LIBXSMM_VLA_ACCESS(5, del_input, img, code_stream[pc].aux_index/*ifm1*/, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
                  float *scratch_ptr = accumulators_scratch;
                  __m512 zero_reg = _mm512_setzero_ps();
                  if ( handle->f32_bf16_cvt_rne ) {
                    __m512i vnaninf = _mm512_set1_epi32( 0x7f800000 );
                    __m512i vrneadd = _mm512_set1_epi32( 0x00007fff );
                    __m512i vfixup = _mm512_set1_epi32( 0x00000001 );
                    __m512i vfixupmask = _mm512_set1_epi32( 0x00010000 );
                    for ( ij = 0; ij < handle->desc.H; ij++ ) {
                      for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
                        __m512i vfp32     = _mm512_castps_si512( _mm512_loadu_ps(scratch_ptr+ii) );
                        __m512i vfp32nan  = _mm512_and_epi32( vfp32, vnaninf );
                        __m512i vfp32fixup  = _mm512_and_epi32( vfp32, vfixupmask );
                        __mmask16 rnemask = _mm512_cmp_epi32_mask( vfp32nan, vnaninf, _MM_CMPINT_NE );
                        __mmask16 fixupmask = _mm512_cmp_epi32_mask( vfp32fixup, vfixupmask, _MM_CMPINT_EQ );
                        __m512i vrnd = _mm512_mask_add_epi32( vrneadd , fixupmask, vrneadd, vfixup );
                        __m512i vfp32rne  = _mm512_mask_add_epi32( vfp32, rnemask, vfp32, vrnd );
                        __m512i vbfp16_32 = _mm512_srai_epi32( vfp32rne, 16 );
                        __m256i vbfp16    = _mm512_cvtepi32_epi16( vbfp16_32 );
                        _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
                        _mm256_storeu_si256( (__m256i*)(input_dst+ii), vbfp16 );
                      }
                      scratch_ptr += handle->desc.W*handle->ifmblock_hp;
                      input_dst += handle->ifwp*handle->ifmblock_hp;
                    }
                  } else {
                    for ( ij = 0; ij < handle->desc.H; ij++ ) {
                      for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
                        __m512 tmp = _mm512_loadu_ps(scratch_ptr+ii);
                        __m256i vbfp16 =  _mm512_cvtepi32_epi16(_mm512_srai_epi32( _mm512_castps_si512( tmp ), 16));
                        _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
                        _mm256_storeu_si256( (__m256i*)(input_dst+ii), vbfp16 );
                      }
                      scratch_ptr += handle->desc.W*handle->ifmblock_hp;
                      input_dst += handle->ifwp*handle->ifmblock_hp;
                    }
                  }
                }
              }
            }

            /* Run the stream of convolutions for this segment */
            for (conv_i = 0; conv_i < n_convs; conv_i++) {
              const int vi = variant[pool_index]; /* avoid warning about char used as array index */
              offset_i = stream[i];
              offset_w = stream[i+1];
              offset_o = stream[i+2];
              pi = stream[i+3];
              pw = stream[i+4];
              po = stream[i+5];
              kernel_pool[vi](
                input_base + offset_i, weight_base + offset_w, output_base + offset_o,
                input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals, accumulators_scratch + offset_o);
              ++pool_index;
              i += 3;
            }
          }
        } else {
          for (pc = 0; pc < n_segments; pc++) {
            instr = code_stream[pc].segment_type;
            n_convs = code_stream[pc].n_convs;
            if (instr == IMG_LOOP_INIT) {
              img = code_stream[pc].aux_index;
              /* Apply padding  */
              if (handle->padding_flag == 1) {
#include "libxsmm_dnn_bwd_custom_custom_padding.tpl.c"
              }
            }

            if ( instr == IFM_LOOP_INIT ) {
              /* Apply bias if requested  */
              if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
                /*#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"*/
              }
              /* Overwrite output with zeros if requested */
              if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_bwd == 0) ) {
                jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
              }
            }

            if ( instr == IFM_LOOP_CLOSE ) {
              if ( ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) && (handle->use_accumulation_scratch == 0) ){
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
                LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*) handle->reg_input->data,  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                LIBXSMM_VLA_DECL(5, element_input_type, del_input_2, (element_input_type*) handle->grad_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                element_input_type *orig_input_ptr;
                element_input_type *del_input_ptr;
                __m512 zero_reg  = _mm512_setzero_ps();
                __m512 orig_reg;
                __mmask16 mask;
                orig_input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, /*ifm1*/code_stream[pc].aux_index, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                del_input_ptr = &LIBXSMM_VLA_ACCESS(5, del_input_2, img, /*ifm1*/code_stream[pc].aux_index, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                for (ij = 0; ij < handle->desc.H; ij++) {
                  for (ii = 0; ii < handle->desc.W * 16; ii += 16) {
                    orig_reg  = LIBXSMM_INTRINSICS_MM512_LOAD_PS(orig_input_ptr + ii);
                    mask = _mm512_cmp_ps_mask(zero_reg, orig_reg, _CMP_EQ_OQ);
                    _mm512_mask_storeu_ps(del_input_ptr + ii, mask, zero_reg);
                  }
                  orig_input_ptr += handle->ifwp * 16;
                  del_input_ptr += handle->ifwp *16;
                }
#else /* won't happen as this code only runs on AVX512 platforms */
                LIBXSMM_ASSERT(0);
#endif
              }

              if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
                element_input_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, del_input, img, /*ifm1*/code_stream[pc].aux_index, 0, 0, 0,
                    handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
                for ( ij = 0; ij < handle->desc.H; ij++ ) {
                  for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
                    max_abs = _mm512_max_ps(max_abs, LIBXSMM_INTRINSICS_MM512_ABS_PS(LIBXSMM_INTRINSICS_MM512_LOAD_PS(cur_vec+ii)));
#else /* won't happen as this code only runs on AVX512 platforms */
                    LIBXSMM_ASSERT(0);
#endif
                  }
                  cur_vec += handle->ifwp*handle->ifmblock_hp;
                }
              }

              /* @TODO this is a hack as it might conflict with MAX STATS fuse */
              /* down-convert to bf16 from fp32 */
              if (handle->use_accumulation_scratch) {
                if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) {
                  LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*) handle->reg_input->data,  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
                  element_input_type *orig_input_ptr;
                  orig_input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, /*ifm1*/code_stream[pc].aux_index, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
                  element_input_type *input_dst = &LIBXSMM_VLA_ACCESS(5, del_input, img, code_stream[pc].aux_index/*ifm1*/, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
                  float *scratch_ptr = accumulators_scratch;
                  __m512 zero_reg = _mm512_setzero_ps();
                  __mmask16 mask;
                  __m256i zero_reg_relu = _mm256_setzero_si256();
                  __m256i orig_reg;
                  if ( handle->f32_bf16_cvt_rne ) {
                    __m512i vnaninf = _mm512_set1_epi32( 0x7f800000 );
                    __m512i vrneadd = _mm512_set1_epi32( 0x00007fff );
                    __m512i vfixup = _mm512_set1_epi32( 0x00000001 );
                    __m512i vfixupmask = _mm512_set1_epi32( 0x00010000 );
                    for ( ij = 0; ij < handle->desc.H; ij++ ) {
                      for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
                        __m512i vfp32     = _mm512_castps_si512( _mm512_loadu_ps(scratch_ptr+ii) );
                        __m512i vfp32nan  = _mm512_and_epi32( vfp32, vnaninf );
                        __m512i vfp32fixup  = _mm512_and_epi32( vfp32, vfixupmask );
                        __mmask16 rnemask = _mm512_cmp_epi32_mask( vfp32nan, vnaninf, _MM_CMPINT_NE );
                        __mmask16 fixupmask = _mm512_cmp_epi32_mask( vfp32fixup, vfixupmask, _MM_CMPINT_EQ );
                        __m512i vrnd = _mm512_mask_add_epi32( vrneadd , fixupmask, vrneadd, vfixup );
                        __m512i vfp32rne  = _mm512_mask_add_epi32( vfp32, rnemask, vfp32, vrnd );
                        __m512i vbfp16_32 = _mm512_srai_epi32( vfp32rne, 16 );
                        __m256i vbfp16    = _mm512_cvtepi32_epi16( vbfp16_32 );
                        _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
                        orig_reg  = _mm256_loadu_si256( (__m256i*) (orig_input_ptr + ii));
                        mask = _mm256_cmp_epi16_mask(zero_reg_relu, orig_reg, _MM_CMPINT_NE);
                        _mm256_mask_storeu_epi16( (__m256i*) (input_dst+ii), mask, vbfp16);
                      }
                      scratch_ptr += handle->desc.W*handle->ifmblock_hp;
                      input_dst += handle->ifwp*handle->ifmblock_hp;
                      orig_input_ptr += handle->ifwp * 16;
                    }
                  } else {
                    for ( ij = 0; ij < handle->desc.H; ij++ ) {
                      for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
                        __m512 tmp = _mm512_loadu_ps(scratch_ptr+ii);
                        __m256i vbfp16 =  _mm512_cvtepi32_epi16(_mm512_srai_epi32( _mm512_castps_si512( tmp ), 16));
                        _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
                        orig_reg  = _mm256_loadu_si256( (__m256i*) (orig_input_ptr + ii));
                        mask = _mm256_cmp_epi16_mask(zero_reg_relu, orig_reg, _MM_CMPINT_NE);
                        _mm256_mask_storeu_epi16( (__m256i*) (input_dst+ii), mask, vbfp16);
                      }
                      scratch_ptr += handle->desc.W*handle->ifmblock_hp;
                      input_dst += handle->ifwp*handle->ifmblock_hp;
                      orig_input_ptr += handle->ifwp * 16;
                    }
                  }
                } else {
                  element_input_type *input_dst = &LIBXSMM_VLA_ACCESS(5, del_input, img, code_stream[pc].aux_index/*ifm1*/, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
                  float *scratch_ptr = accumulators_scratch;
                  __m512 zero_reg = _mm512_setzero_ps();
                  if ( handle->f32_bf16_cvt_rne ) {
                    __m512i vnaninf = _mm512_set1_epi32( 0x7f800000 );
                    __m512i vrneadd = _mm512_set1_epi32( 0x00007fff );
                    __m512i vfixup = _mm512_set1_epi32( 0x00000001 );
                    __m512i vfixupmask = _mm512_set1_epi32( 0x00010000 );
                    for ( ij = 0; ij < handle->desc.H; ij++ ) {
                      for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
                        __m512i vfp32     = _mm512_castps_si512( _mm512_loadu_ps(scratch_ptr+ii) );
                        __m512i vfp32nan  = _mm512_and_epi32( vfp32, vnaninf );
                        __m512i vfp32fixup  = _mm512_and_epi32( vfp32, vfixupmask );
                        __mmask16 rnemask = _mm512_cmp_epi32_mask( vfp32nan, vnaninf, _MM_CMPINT_NE );
                        __mmask16 fixupmask = _mm512_cmp_epi32_mask( vfp32fixup, vfixupmask, _MM_CMPINT_EQ );
                        __m512i vrnd = _mm512_mask_add_epi32( vrneadd , fixupmask, vrneadd, vfixup );
                        __m512i vfp32rne  = _mm512_mask_add_epi32( vfp32, rnemask, vfp32, vrnd );
                        __m512i vbfp16_32 = _mm512_srai_epi32( vfp32rne, 16 );
                        __m256i vbfp16    = _mm512_cvtepi32_epi16( vbfp16_32 );
                        _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
                        _mm256_storeu_si256( (__m256i*)(input_dst+ii), vbfp16 );
                      }
                      scratch_ptr += handle->desc.W*handle->ifmblock_hp;
                      input_dst += handle->ifwp*handle->ifmblock_hp;
                    }
                  } else {
                    for ( ij = 0; ij < handle->desc.H; ij++ ) {
                      for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
                        __m512 tmp = _mm512_loadu_ps(scratch_ptr+ii);
                        __m256i vbfp16 =  _mm512_cvtepi32_epi16(_mm512_srai_epi32( _mm512_castps_si512( tmp ), 16));
                        _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
                        _mm256_storeu_si256( (__m256i*)(input_dst+ii), vbfp16 );
                      }
                      scratch_ptr += handle->desc.W*handle->ifmblock_hp;
                      input_dst += handle->ifwp*handle->ifmblock_hp;
                    }
                  }
                }
              }
            }

            /* Run the stream of convolutions for this segment */
            for (conv_i = 0; conv_i < n_convs; conv_i++) {
              offset_i = stream[i];
              offset_w = stream[i+1];
              offset_o = stream[i+2];
              pi = stream[i+3];
              pw = stream[i+4];
              po = stream[i+5];
              kernel(
                  input_base + offset_i, weight_base + offset_w, output_base + offset_o,
                  input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals, accumulators_scratch + offset_o);
              i += 3;
            }
          }
        }
      }
    } else { /* This is the the img par branch...  */
      /* Use fine-grained operations since we are in the img_par path, so update relevant kernel pointers... */
      int input_h_start = LIBXSMM_MAX(0,  handle->ofh_bwd_start[ltid] - handle->desc.R + 1);
      int input_h_end = LIBXSMM_MIN(handle->ifhp, (handle->ofh_bwd_end[ltid] + handle->desc.R - 1) * handle->desc.u);
#if 0
      int my_h_out = handle->ofh_bwd_end[ltid] - handle->ofh_bwd_start[ltid];
#endif
      int ih;
      jitted_zero_overwrite = handle->matcopy_bwd[3].xmatcopy;
      jitted_matcopy = handle->matcopy_bwd[2].xmatcopy;
      for (pc = 0; pc < n_segments; pc++) {
        instr = code_stream[pc].segment_type;
        n_convs = code_stream[pc].n_convs;
        if (instr == IMG_LOOP_INIT) {
          /* Padding code via jitted matcopy kernel */
#         include "libxsmm_dnn_bwd_custom_custom_padding_img_par.tpl.c"
        }

        if ( instr == IFM_LOOP_INIT ) {
          /* Overwrite output with zeros if requested */
          if ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) {
#if 0
            int ih;
            for (ih = 0; ih < my_h_out * handle->ifmblock_hp * handle->ifwp; ih += handle->ifmblock_hp * handle->ifwp) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2] + ih, NULL, NULL);
            }
#endif
            int h, w;
            __m512 zero_reg = _mm512_setzero_ps();
            for (h = 0; h<handle->bwd_ofh_rb; h++) {
              for (w = 0; w<handle->bwd_ofw_rb; w++) {
                _mm512_store_ps(output_base+stream[i+2]+w*handle->ifmblock_hp+h*handle->ifwp*handle->ifmblock_hp, zero_reg);
              }
            }
          }
        }

        /* Run the stream of convolutions for this segment */
        for (conv_i = 0; conv_i < n_convs; conv_i++) {
          offset_i = stream[i];
          offset_w = stream[i+1];
          offset_o = stream[i+2];
          pi = stream[i+3];
          pw = stream[i+4];
          po = stream[i+5];
          kernel(
              input_base + offset_i, weight_base + offset_w, output_base + offset_o,
              input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
          i += 3;
        }
      }
    }
  } else {
    /* TODO: Second condition guarantees we run the img_par code when we have MB=1 -- and hopefully HUGE images */
    if (handle->desc.N*BLOCKSIFM >= handle->desc.threads && !((handle->desc.N == 1) && (handle->bwd_ofh_rb == 1))) {
      /* Run the stream of convolutions, no extra operations are required... */
      if (handle->perform_relu_in_kernel == 1) { /* do RELU stuff in the kernel  */
        LIBXSMM_VLA_DECL(5, element_input_type, original_input, ((element_input_type*)handle->reg_input->data) + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in * handle->ifmblock), BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
        element_input_type *regular_input_base;
        regular_input_base = &LIBXSMM_VLA_ACCESS(5, original_input, 0, 0, 0, 0, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);

        if (handle->n_variants == 2) {
          for (pc = 0; pc < instr; pc += 1) {
            const int vi = variant[pc]; /* avoid warning about char used as array index */
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            pi = stream[i+3];
            pw = stream[i+4];
            po = stream[i+5];
            kernel_pool[vi](
                input_base + offset_i, weight_base + offset_w, output_base + offset_o,
                input_base + pi, weight_base + pw, output_base + po,
                regular_input_base + offset_o, &scale_factor, max_vals);
            i += 3;
          }
        } else {
          for (pc = 0; pc < instr; pc++) {
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            pi = stream[i+3];
            pw = stream[i+4];
            po = stream[i+5];
            kernel(
                input_base + offset_i, weight_base + offset_w, output_base + offset_o,
                input_base + pi, weight_base + pw, output_base + po,
                regular_input_base + offset_o, &scale_factor, max_vals);
            i += 3;
          }
        }
      } else {
        if (handle->n_variants == 2) {
          for (pc = 0; pc < instr; pc += 1) {
            const int vi = variant[pc]; /* avoid warning about char used as array index */
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            pi = stream[i+3];
            pw = stream[i+4];
            po = stream[i+5];
            kernel_pool[vi](
                input_base + offset_i, weight_base + offset_w, output_base + offset_o,
                input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
            i += 3;
          }
        } else {
          for (pc = 0; pc < instr; pc++) {
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            pi = stream[i+3];
            pw = stream[i+4];
            po = stream[i+5];
            kernel(
                input_base + offset_i, weight_base + offset_w, output_base + offset_o,
                input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
            i += 3;
          }
        }
      }
    } else {
      /* This is the the img par branch...  */
      for (pc = 0; pc < instr; pc++) {
        offset_i = stream[i];
        offset_w = stream[i+1];
        offset_o = stream[i+2];
        pi = stream[i+3];
        pw = stream[i+4];
        po = stream[i+5];
        kernel(
            input_base + offset_i, weight_base + offset_w, output_base + offset_o,
            input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
        i += 3;
      }
    }
  }

  if ( ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) && (handle->use_lp_kernel == 1) && (handle->compute_max_in_kernel_bwd == 0) ) {
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
    _mm512_store_ps(max_vals, max_abs);
#else /* won't happen as this code only runs on AVX512 platforms */
    LIBXSMM_ASSERT(0);
#endif
  }
  libxsmm_barrier_wait(handle->barrier, ltid);

#if 0
  /* Fuse ReLu here*/
  if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) {
    int ii, ij, ifm1, ifm2, img;
    img = ltid;
    LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*) handle->reg_input->data,  BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
    LIBXSMM_VLA_DECL(5, element_input_type, del_input_2, (element_input_type*) handle->grad_input->data, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
    element_input_type *orig_input_ptr;
    element_input_type *del_input_ptr;
    __m512 zero_reg  = _mm512_setzero_ps();
    __m512 orig_reg;
    __mmask16 mask;
    for (ifm1 = 0; ifm1 < BLOCKSIFM; ifm1++ ) {
      orig_input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
      del_input_ptr = &LIBXSMM_VLA_ACCESS(5, del_input_2, img, ifm1, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
      for (ij = 0; ij < handle->desc.H; ij++) {
        for (ii = 0; ii < handle->desc.W * 16; ii += 16) {
          orig_reg  = LIBXSMM_INTRINSICS_MM512_LOAD_PS(orig_input_ptr + ii);
          mask = _mm512_cmp_ps_mask(zero_reg, orig_reg, _CMP_EQ_OQ);
          _mm512_mask_storeu_ps(del_input_ptr + ii, mask, zero_reg);
        }
        orig_input_ptr += handle->ifwp * 16;
        del_input_ptr += handle->ifwp *16;
      }
    }
    libxsmm_barrier_wait(handle->barrier, ltid);
  }
#endif
}

