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
/* Rajkishore Barik (Intel Corp.)
 ******************************************************************************/
int imgifm1, img, ofm1, ifm1, oj, ij, kj, ki, ifm2, ofm2, ifm1ofm1, ifh;
#ifndef INPUT_PADDING
int ii;
#endif
/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = handle->desc.N * handle->blocksifm;
/* compute chunck size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;


/* number of tasks that could be run in parallel */
const int transpose_work = handle->blocksofm * handle->blocksifm;
/* compute chunck size */
const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : ((transpose_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

element_output_type *const out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->blocksofm * handle->ofmblock;
LIBXSMM_VLA_DECL(5, element_output_type, del_out, out, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
LIBXSMM_VLA_DECL(5, element_input_type, del_input, (element_input_type*)handle->grad_input->data, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
LIBXSMM_VLA_DECL(6, element_filter_type, wt, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
LIBXSMM_VLA_DECL(6, element_filter_type, tr_wt, (element_filter_type*)handle->scratch1, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);

/* avoid warning by using the xconv.sconv sequence to get some fn. ptr. to act as source of the type-cast */
libxsmm_convfunction jitted_conv_bp_no_pf = (libxsmm_convfunction)handle->code_bwd[0].xconv.sconv;

element_input_type *l_input;
element_filter_type *l_wt;
element_output_type* l_output;

#if defined(INPUT_PADDING)
/* Based on the input datatype select the right intrinsics */
#ifdef INPUT_F32

#if defined(__AVX512F__)
#define LOAD(x)             _mm512_load_ps(x)
#define STORE(x,y)          _mm512_store_ps(x,y)
#endif

#if defined(__AVX__)
#define STORE_256(x,y)      _mm256_store_ps(x,y)
#define LOAD_256(x)         _mm256_load_ps(x)
#endif
#define CHUNK_SIZE          16

#endif

#ifdef INPUT_I16

#if defined(__AVX512F__)
#define LOAD(x)             _mm512_load_si512 (x)
#define STORE(x,y)          _mm512_store_si512(x,y)
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
#define STORE(x,y)          _mm512_store_si512(x,y)
#endif

#if defined(__AVX__)
#define LOAD_256(x)         _mm256_load_si256((__m256i const *)x)
#define STORE_256(x,y)      _mm256_store_si256((__m256i*)x,y)
#endif
#define CHUNK_SIZE          64

#endif

/* Variables and initializations related to padding */
int iij;
const int padded_h = handle->ifhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ifwp + 2 * handle->desc.pad_w;
#if defined(__AVX512F__) || defined(__AVX__)
element_input_type (* __restrict input_ptr);
element_input_type (* __restrict copy_ptr);
const int big_block_size = padded_w * handle->blocksifm * handle->ifmblock;
const int block_size = handle->ifwp * handle->blocksifm * handle->ifmblock;
element_input_type *prefetch_ptr;
#endif
LIBXSMM_VLA_DECL(4, element_input_type, input_buffer, ((element_input_type*)handle->scratch5) + ltid * padded_h * padded_w * handle->blocksifm * handle->ifmblock, padded_w, handle->blocksifm, handle->ifmblock);
const size_t small_block_size = handle->ifmblock * libxsmm_dnn_typesize(handle->datatype) * 8;
memset(&LIBXSMM_VLA_ACCESS(4, input_buffer, 0, 0, 0, 0, padded_w, handle->blocksifm, handle->ifmblock), 0,
       padded_w * padded_h * handle->blocksifm * handle->ifmblock * sizeof(element_input_type));
ifh = handle->ifhp + 2 * handle->desc.pad_h;
#else
ifh = handle->ifhp;
#endif

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
  ofm1 = ifm1ofm1/handle->blocksifm;
  ifm1 = ifm1ofm1%handle->blocksifm;
  for(kj=0; kj < handle->desc.R; ++kj) {
    for(ki=0; ki < handle->desc.S; ++ki) {
      /* TODO: enable this later */
      /*transpose<VLEN,VLEN>(&wt[ofm1][ifm1][kj][ki][0][0],&tr_wt[ofm1][ifm1][kj][ki][0][0]);*/
      for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
        for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
          LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kj, ki, ofm2, ifm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock) =
            LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
        }
      }
    }
  }
}
libxsmm_barrier_wait(handle->barrier, ltid);

if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC ||
     libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE || /*  ) {
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
} else if (*/ libxsmm_target_archid == LIBXSMM_X86_AVX2 ){
  for (imgifm1 = thr_begin; imgifm1 < thr_end; ++imgifm1) {
    img = imgifm1/handle->blocksifm;
    ifm1 = imgifm1%handle->blocksifm;

#if defined(INPUT_PADDING)

#if defined(__AVX512F__) || defined(__AVX__)
    input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img, 0, 0, ifm1, 0, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
    copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(4, input_buffer, handle->desc.pad_h, handle->desc.pad_w, ifm1, 0, padded_w, handle->blocksifm, handle->ifmblock);

    if (ifm1+1 != handle->blocksifm) {
      /* Prefetch next ifm, same image */
      prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img, 0, 0, ifm1+1, 0, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
    } else {
      /* Prefetch ifm 0 from next image */
      prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img+1, 0, 0, 0, 0, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
    }
#endif

    if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC || libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ) {
      if (small_block_size == 512) {
        for (oj = 0; oj < handle->ifhp; oj++) {
#if defined(__AVX512F__)
          for (ij = 0; ij < handle->ifwp; ij++) {
            STORE(&copy_ptr[ij * handle->blocksifm * handle->ifmblock+oj*big_block_size], LOAD(&input_ptr[ij * handle->blocksifm * handle->ifmblock +oj*block_size]));
            _mm_prefetch((const char*)&prefetch_ptr[ij+oj*block_size], _MM_HINT_T1);
          }
#else
          for (ij = 0; ij < handle->ifwp; ij++) {
            for (iij = 0; iij < handle->ifmblock; iij++) {
              LIBXSMM_VLA_ACCESS(4, input_buffer, oj + handle->desc.pad_h, ij + handle->desc.pad_w, ifm1, iij, padded_w, handle->blocksifm, handle->ifmblock)
              = LIBXSMM_VLA_ACCESS(5, del_input, img, oj, ij, ifm1, iij, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
            }
          }
#endif
        }
      } else {
        for (oj = 0; oj < handle->ifhp; oj++) {
          for (ij = 0; ij < handle->ifwp; ij++) {
            for (iij = 0; iij < handle->ifmblock; iij++) {
              LIBXSMM_VLA_ACCESS(4, input_buffer, oj + handle->desc.pad_h, ij + handle->desc.pad_w, ifm1, iij, padded_w, handle->blocksifm, handle->ifmblock)
              = LIBXSMM_VLA_ACCESS(5, del_input, img, oj, ij, ifm1, iij, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
            }
          }
#if defined(__AVX512F__)
          _mm_prefetch((const char*)&prefetch_ptr[ij*handle->ifmblock*handle->blocksifm+oj*block_size], _MM_HINT_T1);
#endif
        }
      }
    } else if (libxsmm_target_archid == LIBXSMM_X86_AVX2) {
      if (small_block_size == 256) {
        for (oj = 0; oj < handle->ifhp; oj++) {
#if defined(__AVX__)
          for (ij = 0; ij < handle->ifwp; ij++) {
            STORE_256(&copy_ptr[ij * handle->blocksifm * handle->ifmblock+oj*big_block_size], LOAD_256(&input_ptr[ij * handle->blocksifm * handle->ifmblock +oj*block_size]));
            _mm_prefetch((const char*)&prefetch_ptr[ij+oj*block_size], _MM_HINT_T1);
          }
#else
          for (ij = 0; ij < handle->ifwp; ij++) {
            for (iij = 0; iij < handle->ifmblock; iij++) {
              LIBXSMM_VLA_ACCESS(4, input_buffer, oj + handle->desc.pad_h, ij + handle->desc.pad_w, ifm1, iij, padded_w, handle->blocksifm, handle->ifmblock)
              = LIBXSMM_VLA_ACCESS(5, del_input, img, oj, ij, ifm1, iij, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
            }
          }
#endif
        }
      } else {
        for (oj = 0; oj < handle->ifhp; oj++) {
          for (ij = 0; ij < handle->ifwp; ij++) {
            for (iij = 0; iij < handle->ifmblock; iij++) {
              LIBXSMM_VLA_ACCESS(4, input_buffer, oj + handle->desc.pad_h, ij + handle->desc.pad_w, ifm1, iij, padded_w, handle->blocksifm, handle->ifmblock)
              = LIBXSMM_VLA_ACCESS(5, del_input, img, oj, ij, ifm1, iij, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
            }
          }
#if defined(__AVX__)
          _mm_prefetch((const char*)&prefetch_ptr[ij*handle->ifmblock*handle->blocksifm+oj*block_size], _MM_HINT_T1);
#endif
        }
      }
    }
#endif

    for (ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1) {
      for(ij= 0 ; ij < ifh; ++ij) {
        for(kj=0; kj < handle->desc.R; ++kj) {
          oj = ij - handle->desc.R + kj + 1;
          if(oj >= 0 && oj < handle->ofh) {
#if defined(INPUT_PADDING)
            l_input =  &LIBXSMM_VLA_ACCESS(4, input_buffer, ij, 0, ifm1, 0, padded_w, handle->blocksifm, handle->ifmblock);
#else
            l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ij, 0, ifm1, 0, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
#endif
            l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, handle->desc.R-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
            l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, oj, 0, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
            jitted_conv_bp_no_pf(l_input, l_wt, l_output, NULL, NULL, NULL );
          }
        }
      }
    }

#if defined(INPUT_PADDING)
    /* Write back input buffer */
    if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC || libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ) {
      if (small_block_size == 512) {
        for (oj = 0; oj < handle->ifhp; oj++) {
#if defined(__AVX512F__)
          for (ij = 0; ij < handle->ifwp; ij++) {
            STORE(&input_ptr[ij * handle->blocksifm * handle->ifmblock+oj*block_size], LOAD(&copy_ptr[ij * handle->blocksifm * handle->ifmblock+oj*big_block_size]));
          }
#else
          for (ij = 0; ij < handle->ifwp; ij++) {
            for (iij = 0; iij < handle->ifmblock; iij++) {
              LIBXSMM_VLA_ACCESS(5, del_input, img, oj, ij, ifm1, iij, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock) =
              LIBXSMM_VLA_ACCESS(4, input_buffer, oj + handle->desc.pad_h, ij + handle->desc.pad_w, ifm1, iij, padded_w, handle->blocksifm, handle->ifmblock);
            }
          }
#endif
        }
      } else {
        for (oj = 0; oj < handle->ifhp; oj++) {
          for (ij = 0; ij < handle->ifwp; ij++) {
            for (iij = 0; iij < handle->ifmblock; iij++) {
              LIBXSMM_VLA_ACCESS(5, del_input, img, oj, ij, ifm1, iij, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock) =
              LIBXSMM_VLA_ACCESS(4, input_buffer, oj + handle->desc.pad_h, ij + handle->desc.pad_w, ifm1, iij, padded_w, handle->blocksifm, handle->ifmblock);
            }
          }
        }
      }
    } else if (libxsmm_target_archid == LIBXSMM_X86_AVX2) {
      if (small_block_size == 256) {
        for (oj = 0; oj < handle->ifhp; oj++) {
#if defined(__AVX__)
          for (ij = 0; ij < handle->ifwp; ij++) {
            STORE_256(&input_ptr[ij * handle->blocksifm * handle->ifmblock+oj*block_size], LOAD_256(&copy_ptr[ij * handle->blocksifm * handle->ifmblock+oj*big_block_size]));
          }
#else
          for (ij = 0; ij < handle->ifwp; ij++) {
            for (iij = 0; iij < handle->ifmblock; iij++) {
              LIBXSMM_VLA_ACCESS(5, del_input, img, oj, ij, ifm1, iij, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock) =
              LIBXSMM_VLA_ACCESS(4, input_buffer, oj + handle->desc.pad_h, ij + handle->desc.pad_w, ifm1, iij, padded_w, handle->blocksifm, handle->ifmblock);
            }
          }
#endif
        }
      } else {
        for (oj = 0; oj < handle->ifhp; oj++) {
          for (ij = 0; ij < handle->ifwp; ij++) {
            for (iij = 0; iij < handle->ifmblock; iij++) {
              LIBXSMM_VLA_ACCESS(5, del_input, img, oj, ij, ifm1, iij, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock) =
              LIBXSMM_VLA_ACCESS(4, input_buffer, oj + handle->desc.pad_h, ij + handle->desc.pad_w, ifm1, iij, padded_w, handle->blocksifm, handle->ifmblock);
            }
          }
        }
      }
    }
#else
#include "libxsmm_dnn_zero_rim_st_input_nhwc.tpl.c"
#endif
  }
/* should never happen, this is just an additional check */
} else {
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
}

#if defined(INPUT_PADDING)
#undef LOAD
#undef STORE
#undef CHUNK_SIZE
#endif
