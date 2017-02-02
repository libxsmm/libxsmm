/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **d
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
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/

int imgofm1, img, ofm1, ifm1, oj, ij, oi, ii;
/* computing first logical thread */
const int ltid = tid-start_thread;
/* number of tasks that could be run in parallel */
const int work = handle->desc.N*handle->blocksofm;
/* compute chunck size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
const element_input_type *l_input;
const element_filter_type *l_wt;
element_output_type* l_output;

element_output_type *const out = ((element_output_type*)handle->reg_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->blocksofm * handle->ofmblock;
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->ofhp, handle->ofwp,handle->blocksofm,  handle->ofmblock);
LIBXSMM_VLA_DECL(6, const element_input_type, input, (element_input_type*)handle->reg_input->data, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);

#if defined(INPUT_PADDING)
/* Variables and initializations related to padding */
int iii, iij, prev_image = -1;
element_input_type *input_ptr;
element_input_type *copy_ptr;
const int padded_h = handle->ifhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ifwp + 2 * handle->desc.pad_w;
LIBXSMM_VLA_DECL(5, element_input_type, input_buffer, ((element_input_type*)handle->scratch5) + ltid * handle->blocksifm * padded_h * padded_w * handle->ifmblock * handle->fm_lp_block, padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
const int block_size = handle->ifwp * handle->blocksifm * handle->ifmblock * handle->fm_lp_block;
const int big_block_size = padded_w * handle->blocksifm * handle->ifmblock * handle->fm_lp_block;
const size_t small_block_size = handle->ifwp * handle->blocksifm * handle->ifmblock * handle->fm_lp_block * libxsmm_dnn_typesize(handle->datatype) * 8;
/* Based on the input datatype select the right intrinsics */
#ifdef INPUT_F32

#if defined(__AVX512F__)
#define LOAD(x)             _mm512_load_ps(x)
#define LOADU(x)            _mm512_loadu_ps(x)
#define MASK_LOADU(x,y)     _mm512_maskz_loadu_ps(x,y)
#define STORE(x,y)          _mm512_store_ps(x,y)
#define STOREU(x,y)         _mm512_storeu_ps(x,y)
#define MASK_STOREU(x,y,z)  _mm512_mask_storeu_ps(x,y,z)
#define INT_TO_MASK(x)      ( (__mmask16) x)
#endif

#if defined(__AVX__)
#define LOAD_256(x)         _mm256_load_ps(x)
#define STORE_256(x,y)      _mm256_store_ps(x,y)
#endif
#define CHUNK_SIZE          16

#endif

#ifdef INPUT_I16

#if defined(__AVX512F__)
#define LOAD(x)             _mm512_load_si512 (x)
#define LOADU(x)            _mm512_loadu_si512(x)
#define MASK_LOADU(x,y)     _mm512_maskz_loadu_epi16(x,y)
#define STORE(x,y)          _mm512_store_si512(x,y)
#define STOREU(x,y)         _mm512_storeu_si512(x,y)
#define MASK_STOREU(x,y,z)  _mm512_mask_storeu_epi16(x,y,z)
#define INT_TO_MASK(x)      ( (__mmask32) x)
#endif

#if defined(__AVX__)
#define LOAD_256(x)         _mm256_load_si256((__m256i const *)x)
#define STORE_256(x,y)      _mm256_store_si256((__m256i*)x,y)
#endif
#define CHUNK_SIZE          32

#endif

#ifdef INPUT_I8

#if defined(__AVX512F__)
#define LOAD(x)             _mm512_load_si512 (x)
#define LOADU(x)            _mm512_loadu_si512(x)
#define MASK_LOADU(x,y)     _mm512_maskz_loadu_epi8(x,y)
#define STORE(x,y)          _mm512_store_si512(x,y)
#define STOREU(x,y)         _mm512_storeu_si512(x,y)
#define MASK_STOREU(x,y,z)  _mm512_mask_storeu_epi8(x,y,z)
#define INT_TO_MASK(x)      ( (__mmask64) x)
#endif

#if defined(__AVX__)
#define LOAD_256(x)         _mm256_load_si256((__m256i const *)x)
#define STORE_256(x,y)      _mm256_store_si256((__m256i*)x,y)
#endif
#define CHUNK_SIZE          64

#endif

#if defined(__AVX512F__)
element_input_type *prefetch_ptr;
#if !defined(LIBXSMM_INTRINSICS_AVX512_NOMASK)
const int64_t remainder_mask = (block_size % CHUNK_SIZE != 0) ? (1 << (block_size % CHUNK_SIZE)) - 1 : -1;
#endif
#endif

#endif

/* JIT kernel function pointers */
libxsmm_convfunction jitted_conv_fp_one, jitted_conv_fp_two, jitted_conv_fp_zero;

/* select kernels based on architecture */
if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC ||
     libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ) {
  jitted_conv_fp_one = (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;
  jitted_conv_fp_two = (libxsmm_convfunction)handle->code_fwd[2].xconv.sconv;
#if defined(LIBXSMM_CONV_NO_PREFETCH)
  jitted_conv_fp_zero = (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
#endif

  for (imgofm1 = thr_begin; imgofm1 < thr_end; ++imgofm1) {
    img = imgofm1/handle->blocksofm;
    ofm1 = imgofm1%handle->blocksofm;

#if defined(INPUT_PADDING)
    if (prev_image != img) {
      /* The img has changed so we should copy all the ifms */
      input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img, 0, 0, 0, 0, 0, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
      copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_buffer, handle->desc.pad_h, handle->desc.pad_w, 0, 0, 0, padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
#if defined(__AVX512F__)
      prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img+1, 0, 0, 0, 0, 0, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
#endif

      if (small_block_size % 512 == 0) {
        for (oj = handle->ifhp-1; oj >= 0; oj--) {
#if defined(__AVX512F__)
          for (oi = 0; oi < block_size; oi += CHUNK_SIZE) {
            STORE(&copy_ptr[oi+oj*big_block_size], LOAD(&input_ptr[oi+oj*block_size]));
            _mm_prefetch((const char*)&prefetch_ptr[oi+oj*block_size], _MM_HINT_T1);
          }
#else
          for (oi = 0; oi < block_size; oi++) {
            copy_ptr[oi+oj*big_block_size] = input_ptr[oi+oj*block_size];
          }
#endif
        }
      } else {
        for (oj = handle->ifhp-1; oj >= 0; oj--) {
#if defined(__AVX512F__) && !defined(LIBXSMM_INTRINSICS_AVX512_NOMASK)
          for (oi = 0; oi < block_size-CHUNK_SIZE; oi += CHUNK_SIZE) {
            STOREU(&copy_ptr[oi+oj*big_block_size], LOADU(&input_ptr[oi+oj*block_size]));
            _mm_prefetch((const char*)&prefetch_ptr[oi+oj*block_size], _MM_HINT_T1);
          }
          MASK_STOREU(&copy_ptr[oi+oj*big_block_size],
                      INT_TO_MASK(remainder_mask),
                      MASK_LOADU(INT_TO_MASK(remainder_mask),
                                 &input_ptr[oi+oj*block_size]));
          _mm_prefetch((const char*)&prefetch_ptr[oi+oj*block_size], _MM_HINT_T1);
#else
          for (oi = 0; oi < block_size; oi++) {
            copy_ptr[oi+oj*big_block_size] = input_ptr[oi+oj*block_size];
          }
#endif
        }
      }
      prev_image = img;
    }
#endif

    for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
      for (oj = 0; oj < handle->ofh; oj += handle->fwd_ofh_rb) {
        ij = oj * handle->desc.u;
        for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
          ii = oi * handle->desc.v;
#if defined(INPUT_PADDING)
          l_input  = &LIBXSMM_VLA_ACCESS(5, input_buffer, ij, ii, ifm1,  0, 0,
                                         padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
#else
          l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ij, ii, ifm1, 0, 0,
                                         handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
#endif
          l_wt     = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0,
                      handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
          l_output = &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0,
                      handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
#if !defined(LIBXSMM_CONV_NO_PREFETCH)
          /* check we are not at the end */
          if (oj < handle->ofh-handle->fwd_ofh_rb) {
            jitted_conv_fp_one(l_input, l_wt, l_output,
#if defined(INPUT_PADDING)
              &LIBXSMM_VLA_ACCESS(5, input_buffer, (oj + handle->fwd_ofh_rb) * handle->desc.u, ii, ifm1, 0, 0,
                                  padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block),
#else
              &LIBXSMM_VLA_ACCESS(6, input, img, (oj + handle->fwd_ofh_rb) * handle->desc.u, ii, ifm1, 0, 0,
              handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block),
#endif
              NULL,
              &LIBXSMM_VLA_ACCESS(5, output, img, oj + handle->fwd_ofh_rb, oi, ofm1, 0,
                                     handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock));
          }
          else {
            if ((ofm1+1 == handle->blocksofm) &&  (ifm1+1 == handle->blocksifm)) {
              jitted_conv_fp_two(l_input, l_wt, l_output,
#if defined(INPUT_PADDING)
                &LIBXSMM_VLA_ACCESS(5, input_buffer, 0, 0, 0, 0, 0,
                  padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block),
#else
                &LIBXSMM_VLA_ACCESS(6, input, img + 1, 0, 0, 0, 0, 0,
                handle->ifhp, handle->ifwp, handle->ifmblock, handle->blocksifm, handle->fm_lp_block),
#endif
                &LIBXSMM_VLA_ACCESS(7, weight, 0, 0, 0, 0, 0, 0, 0,
                  handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
                &LIBXSMM_VLA_ACCESS(5, output, img + 1, 0, 0, 0, 0,
                  handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock));
            }
            else {
              if ((ifm1+1 == handle->blocksifm)) {
                jitted_conv_fp_two(l_input, l_wt, l_output,
#if defined(INPUT_PADDING)
                  &LIBXSMM_VLA_ACCESS(5, input_buffer, 0, 0, 0, 0, 0,
                    padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block),
#else
                  &LIBXSMM_VLA_ACCESS(6, input, img, 0, 0, 0, 0, 0,
                handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block),
#endif
                  &LIBXSMM_VLA_ACCESS(7, weight, ofm1 + 1, 0, 0, 0, 0, 0, 0,
                    handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
                  &LIBXSMM_VLA_ACCESS(5, output, img, ofm1 + 1, 0, 0, 0,
                    handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock));
              }
              else {
                jitted_conv_fp_two(l_input, l_wt, l_output,
#if defined(INPUT_PADDING)
                  &LIBXSMM_VLA_ACCESS(5, input_buffer, 0, 0, ifm1 + 1, 0, 0,
                    padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block),
#else
                  &LIBXSMM_VLA_ACCESS(6, input, img, 0, 0, ifm1 + 1, 0, 0,
                  handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block),
#endif
                  &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1 + 1, 0, 0, 0, 0, 0,
                    handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
                  &LIBXSMM_VLA_ACCESS(5, output, img, 0, 0, ofm1, 0,
                    handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock));
              }
            }
          }
#else
          jitted_conv_fp_zero(l_input, l_wt, l_output, NULL, NULL, NULL);
#endif
        }
      }
    }
  }
} else if ( libxsmm_target_archid == LIBXSMM_X86_AVX2 ){
  jitted_conv_fp_zero = (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
  jitted_conv_fp_one = (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;

  for (imgofm1 = thr_begin; imgofm1 < thr_end; ++imgofm1) {
    img = imgofm1/handle->blocksofm;
    ofm1 = imgofm1%handle->blocksofm;
#if defined(INPUT_PADDING)
    if (prev_image != img) {
      /* The img has changed so we should copy all the ifms */
      input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img, 0, 0, 0, 0, 0, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
      copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_buffer, handle->desc.pad_h, handle->desc.pad_w, 0, 0, 0, padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);

      if (small_block_size % 256 == 0) {
        for (oj = handle->ifhp-1; oj >= 0; oj--) {
#if defined(__AVX__)
          for (oi = 0; oi < block_size; oi += CHUNK_SIZE/2) {
            STORE_256(&copy_ptr[oi+oj*big_block_size], LOAD_256(&input_ptr[oi+oj*block_size]));
          }
#else
          for (oi = 0; oi < block_size; oi++) {
            copy_ptr[oi+oj*big_block_size] = input_ptr[oi+oj*block_size];
          }
#endif
        }
      } else {
        for (oj = handle->ifhp-1; oj >= 0; oj--) {
          for (oi = 0; oi < handle->ifwp; oi++) {
            for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1++) {
              for (iij = 0; iij < handle->ifmblock; iij++) {
                for (iii = 0; iii < handle->fm_lp_block; iii++) {
                  LIBXSMM_VLA_ACCESS(5, input_buffer, oj + handle->desc.pad_h, oi + handle->desc.pad_w, ifm1, iij, iii, padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block) =
                  LIBXSMM_VLA_ACCESS(6, input, img, oj, oi, ifm1, iij, iii, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
                }
              }
            }
          }
        }
      }
      prev_image = img;
    }
#endif
    for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
      for (oj = 0; oj < handle->ofh; oj += handle->fwd_ofh_rb) {
        ij = oj * handle->desc.u;
        for (oi = 0; oi < (handle->ofw - handle->fwd_ofw_rb_2); oi += handle->fwd_ofw_rb) {
          ii = oi * handle->desc.v;
#if defined(INPUT_PADDING)
          l_input  = &LIBXSMM_VLA_ACCESS(5, input_buffer, ij, ii, ifm1,  0, 0,
                                         padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
#else
          l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ij, ii, ifm1, 0, 0,
                      handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
#endif
          l_wt     = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0,
                      handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
          l_output = &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0,
                      handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
          jitted_conv_fp_zero(l_input, l_wt, l_output, NULL, NULL, NULL);
        }
        if (handle->fwd_ofw_rb_2 != 0) {
          ii = oi * handle->desc.v;
#if defined(INPUT_PADDING)
          l_input  = &LIBXSMM_VLA_ACCESS(5, input_buffer, ij, ii, ifm1,  0, 0,
                                         padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
#else
          l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ij, ii, ifm1, 0, 0,
                      handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
#endif
          l_wt     = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0,
                      handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
          l_output = &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0,
                      handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
          jitted_conv_fp_one(l_input, l_wt, l_output, NULL, NULL, NULL);
        }
      }
    }
  }
/* should never happen, this is just an additional check */
} else {
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
}

#if defined(INPUT_PADDING)
#undef LOAD
#undef LOAD_256
#undef LOADU
#undef MASK_LOADU
#undef STORE
#undef STORE_256
#undef STOREU
#undef MASK_STOREU
#undef INT_TO_MASK
#undef CHUNK_SIZE
#endif
