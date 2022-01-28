/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Sasikanth Avancha (Intel Corp.)
******************************************************************************/

#if defined(LIBXSMM_DNN_POOLING_FWD_BF16)
# define _mm512_load_act(A)     _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(A))),16))
#if 1
# define _mm512_roundbf16rne(A) LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(A)
# define _mm512_stream_act(A,B) _mm256_stream_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_roundbf16rne((B)),16)))
# define _mm512_store_act(A,B)  _mm256_storeu_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_roundbf16rne((B)),16)))
#else
# define _mm512_stream_act(A,B) _mm256_stream_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_castps_si512((B)),16)))
# define _mm512_store_act(A,B)  _mm256_storeu_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_castps_si512((B)),16)))
#endif
#else
# define _mm512_load_act(A)     _mm512_loadu_ps(A)
# define _mm512_stream_act(A,B) LIBXSMM_INTRINSICS_MM512_STREAM_PS(A,B)
# define _mm512_store_act(A,B)  _mm512_storeu_ps(A,B)
#endif

/* size variables, all const */
const int nImg = handle->desc.N;
const int ifh = handle->desc.H;
const int ifw = handle->desc.W;
const int sh = handle->desc.u;
const int sw = handle->desc.v;
const int ofh = handle->ofh;
const int ofw = handle->ofw;
const int iph = handle->desc.pad_h_in;
const int ipw = handle->desc.pad_w_in;
const int oph = handle->desc.pad_h_out;
const int opw = handle->desc.pad_w_out;
const int ofhp = ofh + 2*oph;
const int ofwp = ofw + 2*opw;
const int ifhp = ifh + 2*iph;
const int ifwp = ifw + 2*ipw;
/* here we assume that input and output blocking is similar */
const int nBlocksFm = handle->blocksifm;

/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = nImg * nBlocksFm;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* loop variables */
int img = 0;
int fm = 0;
int imgfm = 0;
int ho = 0;
int wo = 0;
int hi = 0;
int wi = 0;
int kh = 0;
int kw = 0;
int v = 0;
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
#if defined(LIBXSMM_DNN_POOLING_FWD_BF16)
float recp_pool_size = 1.0f/((float)handle->desc.R*(float)handle->desc.S);
#else
element_output_type recp_pool_size = 1.0f/((element_output_type)handle->desc.R*(element_output_type)handle->desc.S);
#endif
#endif

/* multi-dim arrays declaration */
#if defined(LIBXSMM_DNN_POOLING_FWD_BF16)
float* lcl_buffer_ptr = ((float*)handle->scratch)+((size_t)ofh*(size_t)ofw*(size_t)32*(size_t)ltid);
LIBXSMM_VLA_DECL(3,       float, lcl_output, lcl_buffer_ptr,                                                   ofw, 32);
#else
element_output_type* lcl_buffer_ptr = ((element_output_type*)handle->scratch)+((size_t)ofh*(size_t)ofw*(size_t)32*(size_t)ltid);
LIBXSMM_VLA_DECL(3,       element_output_type, lcl_output, lcl_buffer_ptr,                                                   ofw, 32);
#endif
LIBXSMM_VLA_DECL(5, const element_input_type,  input,      (element_input_type* )handle->reg_input->data,  nBlocksFm, ifhp, ifwp, 32);
LIBXSMM_VLA_DECL(5,       element_output_type, output,     (element_output_type*)handle->reg_output->data, nBlocksFm, ofhp, ofwp, 32);
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
LIBXSMM_VLA_DECL(5,       element_mask_type,   mask,       (element_mask_type*  )handle->mask->data,       nBlocksFm,  ofh,  ofw, 32);
#endif

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

for (imgfm = thr_begin; imgfm < thr_end; ++imgfm) {
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
  __m512i lcl_viadd  = _mm512_set_epi32( 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0 );
#endif
  img = imgfm / nBlocksFm;
  fm = imgfm % nBlocksFm;

  for( v = 0; v < ofh*ofw*32; v+=16 ) {
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
    _mm512_storeu_ps( &(lcl_buffer_ptr[v]), _mm512_set1_ps(-FLT_MAX) );
#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
    _mm512_storeu_ps( &(lcl_buffer_ptr[v]), _mm512_setzero_ps() );
#endif
  }

  for( ho = oph; ho < (ofh+oph); ho++ ) {
    hi = ((ho-oph) * sh) - handle->desc.pad_h;
    for( wo = opw; wo < (ofw+opw); wo++ ) {
      float*               lcl_output_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_output, ho-oph, wo-opw, 0, ofw, 32);
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
      __m512i lcl_vmask  = _mm512_loadu_si512( &LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-oph, wo-opw,  0, nBlocksFm, ofh, ofw, 32) );
      __m512i lcl_vmask2 = _mm512_loadu_si512( &LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-oph, wo-opw, 16, nBlocksFm, ofh, ofw, 32) );
#endif
      __m512 lcl_voutput  = _mm512_loadu_ps( lcl_output_ptr );
      __m512 lcl_voutput2 = _mm512_loadu_ps( lcl_output_ptr+16 );

      wi = ((wo-opw) * sw) - handle->desc.pad_w;
      for( kh = 0; kh < handle->desc.R; kh++ ) {
        if (hi+kh < 0 || hi+kh >= ifh) continue;
        for( kw = 0; kw < handle->desc.S; kw++ ) {
          if (wi+kw < 0 || wi+kw >= ifw) {
            continue;
          } else {
            const element_input_type*      input_ptr  = &LIBXSMM_VLA_ACCESS(5, input,      img, fm, hi+kh+iph, wi+kw+ipw, 0, nBlocksFm, ifhp, ifwp, 32);
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
            __m512i lcl_vnewmask  = _mm512_add_epi32( lcl_viadd, _mm512_set1_epi32((hi+kh)*ifw*32 + (wi+kw)*32) );
            __m512i lcl_vnewmask2 = _mm512_add_epi32( lcl_viadd, _mm512_set1_epi32((hi+kh)*ifw*32 + (wi+kw)*32 + 16) );
            __m512 lcl_vinput  = _mm512_load_act( input_ptr );
            __m512 lcl_vinput2 = _mm512_load_act( input_ptr+16 );
            __mmask16 lcl_mlt  = _mm512_cmp_ps_mask( lcl_voutput,  lcl_vinput,  _CMP_LT_OS );
            __mmask16 lcl_mlt2 = _mm512_cmp_ps_mask( lcl_voutput2, lcl_vinput2, _CMP_LT_OS );
            lcl_voutput   = _mm512_mask_blend_ps( lcl_mlt,  lcl_voutput,  lcl_vinput );
            lcl_voutput2  = _mm512_mask_blend_ps( lcl_mlt2, lcl_voutput2, lcl_vinput2 );
            lcl_vmask  = _mm512_mask_blend_epi32( lcl_mlt,  lcl_vmask,    lcl_vnewmask );
            lcl_vmask2 = _mm512_mask_blend_epi32( lcl_mlt2, lcl_vmask2,   lcl_vnewmask2 );
#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
            lcl_voutput  = _mm512_add_ps( lcl_voutput,  _mm512_load_act( input_ptr ) );
            lcl_voutput2 = _mm512_add_ps( lcl_voutput2, _mm512_load_act( input_ptr+16 ) );
#endif
          }
        }
      }
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
      _mm512_storeu_si512( &LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-oph, wo-opw,  0, nBlocksFm, ofh, ofw, 32), lcl_vmask );
      _mm512_storeu_si512( &LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-oph, wo-opw, 16, nBlocksFm, ofh, ofw, 32), lcl_vmask2 );
#endif
      _mm512_storeu_ps( lcl_output_ptr,    lcl_voutput );
      _mm512_storeu_ps( lcl_output_ptr+16, lcl_voutput2 );
    }
  }

  /* copy the local buffer into output activations */
  for( ho = oph; ho < (ofh+oph); ho++ ) {
    element_output_type*     output_ptr = &LIBXSMM_VLA_ACCESS(5, output,     img, fm,     ho, opw, 0, nBlocksFm, ofhp, ofwp, 32);
    float*               lcl_output_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_output,          ho-oph,   0, 0,                   ofw, 32);
    for( wo = opw; wo < (ofw+opw); wo++ ) {
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
      const __m512 recp_pool_size_ps = _mm512_set1_ps( recp_pool_size );
      _mm512_stream_act( output_ptr,    _mm512_mul_ps( _mm512_loadu_ps( lcl_output_ptr ),    recp_pool_size_ps ) );
      _mm512_stream_act( output_ptr+16, _mm512_mul_ps( _mm512_loadu_ps( lcl_output_ptr+16 ), recp_pool_size_ps ) );
#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
      _mm512_stream_act( output_ptr,    _mm512_loadu_ps( lcl_output_ptr ) );
      _mm512_stream_act( output_ptr+16, _mm512_loadu_ps( lcl_output_ptr+16 ) );
#endif
      output_ptr += 32;
      lcl_output_ptr += 32;
    }
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

# undef _mm512_load_act
# undef _mm512_stream_act
# undef _mm512_store_act

