/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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
/* Alexander Heinecke, Sasikanth Avancha (Intel Corp.)
******************************************************************************/

#if defined(LIBXSMM_DNN_FUSEDBN_BWD_BF16)
# define _mm512_load_act(A)   _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(A))),16))
#if 1
__m512i vnaninf = _mm512_set1_epi32( 0x7f800000 );
__m512i vrneadd = _mm512_set1_epi32( 0x00007fff );
__m512i vfixup = _mm512_set1_epi32( 0x00000001 );
__m512i vfixupmask = _mm512_set1_epi32( 0x00010000 );
# define _mm512_roundbf16rne(A) _mm512_mask_add_epi32( _mm512_castps_si512( A ), _mm512_cmp_epi32_mask( _mm512_and_epi32( _mm512_castps_si512( A ), vnaninf ), vnaninf, _MM_CMPINT_NE ), _mm512_castps_si512( A ), _mm512_mask_add_epi32( vrneadd , _mm512_cmp_epi32_mask( _mm512_and_epi32( _mm512_castps_si512( A ), vfixupmask ), vfixupmask, _MM_CMPINT_EQ ), vrneadd, vfixup ) )
# define _mm512_stream_act(A,B) _mm256_stream_si256((__m256i*)A,_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_roundbf16rne((B)),16)))
# define _mm512_store_act(A,B)  _mm256_storeu_si256((__m256i*)A,_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_roundbf16rne((B)),16)))
#else
# define _mm512_stream_act(A,B) _mm256_stream_si256((__m256i*)A,_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_castps_si512((B)),16)))
# define _mm512_store_act(A,B)  _mm256_storeu_si256((__m256i*)A,_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_castps_si512((B)),16)))
#endif
#else
# define _mm512_load_act(A)   _mm512_loadu_ps(A)
# define _mm512_stream_act(A,B) _mm512_stream_ps(A,B)
# define _mm512_store_act(A,B)  _mm512_storeu_ps(A,B)
#endif

/* size variables, all const */
const int nImg = handle->desc.N;
const int ifh = handle->desc.H;
const int ifw = handle->desc.W;
const int sh = handle->desc.u;
const int sw = handle->desc.v;
const int ofh = ifh/sh;
const int ofw = ifw/sw;
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

const element_stats_type nhw = (element_stats_type)(nImg * ifh * ifw);
const element_stats_type recp_nhw = 1.0f/nhw;

/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = nImg * nBlocksFm;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* number of tasks that could be run in parallel, delta gamma and beta reduction */
const int work2 = nBlocksFm;
/* compute chunk size */
const int chunksize2 = (work2 % handle->desc.threads == 0) ? (work2 / handle->desc.threads) : ((work2 / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin2 = (ltid * chunksize2 < work2) ? (ltid * chunksize2) : work2;
const int thr_end2 = ((ltid + 1) * chunksize2 < work2) ? ((ltid + 1) * chunksize2) : work2;

/* loop variables */
int img = 0;
int fm = 0;
int imgfm = 0;
int hi = 0;
int wi = 0;
int ho = 0;
int wo = 0;

LIBXSMM_VLA_DECL(5,       element_input_type,  dinput,     (element_input_type* )handle->grad_input->data,  nBlocksFm, ifhp, ifwp, 16);
LIBXSMM_VLA_DECL(5,       element_input_type,   input,     (element_input_type* )handle->reg_input->data,   nBlocksFm, ifhp, ifwp, 16);
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_ELTWISE)
LIBXSMM_VLA_DECL(5,       element_input_type,  dinput_add, (element_input_type* )handle->grad_add->data,    nBlocksFm, ifhp, ifwp, 16);
#endif
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_RELU)
LIBXSMM_VLA_DECL(5, const element_output_type, output,     (element_output_type*)handle->reg_output->data,  nBlocksFm, ofhp, ofwp, 16);
#endif
LIBXSMM_VLA_DECL(5,       element_output_type, doutput,    (element_output_type*)handle->grad_output->data, nBlocksFm, ofhp, ofwp, 16);

LIBXSMM_VLA_DECL(2, const element_stats_type,  gamma,      (element_stats_type*)handle->reg_gamma->data,  16);
LIBXSMM_VLA_DECL(2,       element_stats_type,  dgamma,     (element_stats_type*)handle->grad_gamma->data, 16);
LIBXSMM_VLA_DECL(2,       element_stats_type,  dbeta,      (element_stats_type*)handle->grad_beta->data,  16);
LIBXSMM_VLA_DECL(2, const element_stats_type,  bmean,      (element_stats_type*)handle->expvalue->data,   16);
LIBXSMM_VLA_DECL(2, const element_stats_type,  brstd,      (element_stats_type*)handle->rcpstddev->data,  16);
LIBXSMM_VLA_DECL(3,       element_stats_type,  dgamma_img, (element_stats_type*)handle->scratch,                                                    nImg, 16);
LIBXSMM_VLA_DECL(3,       element_stats_type,  dbeta_img, ((element_stats_type*)handle->scratch) + ((size_t)nImg * (size_t)nBlocksFm * (size_t)16), nImg, 16);

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BN) > 0 ) {
  for ( imgfm = thr_begin; imgfm < thr_end; ++imgfm ) {
    __m512 lcl_vdgamma = _mm512_setzero_ps();
    __m512 lcl_vdbeta  = _mm512_setzero_ps();
    __m512 lcl_vbmean, lcl_vbrstd;
    element_stats_type* del_gamma_img_ptr;
    element_stats_type* del_beta_img_ptr;

    img = imgfm / nBlocksFm;
    fm = imgfm % nBlocksFm;
    del_gamma_img_ptr = &LIBXSMM_VLA_ACCESS(3, dgamma_img, fm, img, 0, nImg, 16);
    del_beta_img_ptr  = &LIBXSMM_VLA_ACCESS(3, dbeta_img,  fm, img, 0, nImg, 16);
    lcl_vbmean = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, bmean, fm, 0, 16) );
    lcl_vbrstd = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, brstd, fm, 0, 16) );

    for ( hi=iph, ho=oph; hi < (ifh + iph); hi+=sh, ho++ ) {
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_ELTWISE)
            element_input_type*  del_input_add_ptr = &LIBXSMM_VLA_ACCESS(5, dinput_add, img, fm, hi, ipw, 0, nBlocksFm, ifhp, ifwp, 16);
#endif
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_RELU)
      const element_output_type* output_ptr        = &LIBXSMM_VLA_ACCESS(5,     output, img, fm, ho, opw, 0, nBlocksFm, ofhp, ofwp, 16);
#endif
      const element_input_type*  input_ptr         = &LIBXSMM_VLA_ACCESS(5,      input, img, fm, hi, ipw, 0, nBlocksFm, ifhp, ifwp, 16);
            element_output_type* del_output_ptr    = &LIBXSMM_VLA_ACCESS(5,    doutput, img, fm, ho, opw, 0, nBlocksFm, ofhp, ofwp, 16);
      for ( wi=ipw, wo=opw; wi < (ifw + ipw); wi+=sw, wo++ ) {
        __m512 lcl_vdeloutput = _mm512_load_act( del_output_ptr );
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_RELU)
        const __m512 zero_ps = _mm512_setzero_ps();
        const __mmask16 lcl_mzero = _mm512_cmp_ps_mask( _mm512_load_act( output_ptr ), zero_ps, _CMP_EQ_OQ );
        lcl_vdeloutput = _mm512_mask_blend_ps( lcl_mzero, lcl_vdeloutput, zero_ps );
        _mm512_store_act( del_output_ptr, lcl_vdeloutput );
        output_ptr += 16;
#endif
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_ELTWISE)
        _mm512_stream_act( del_input_add_ptr, lcl_vdeloutput );
        del_input_add_ptr += sw*16;
#endif
        lcl_vdgamma = _mm512_add_ps( lcl_vdgamma, _mm512_mul_ps( _mm512_mul_ps( _mm512_sub_ps( _mm512_load_act( input_ptr ), lcl_vbmean ), lcl_vdeloutput ), lcl_vbrstd ) );
        lcl_vdbeta  = _mm512_add_ps( lcl_vdbeta, lcl_vdeloutput );

        input_ptr += sw*16;
        del_output_ptr += 16;
      }
    }

    _mm512_storeu_ps( del_gamma_img_ptr, lcl_vdgamma );
    _mm512_storeu_ps( del_beta_img_ptr,  lcl_vdbeta );
  }

  libxsmm_barrier_wait(handle->barrier, ltid);

  /* now we need to reduce the del_gamm and del_beta */
  for ( fm = thr_begin2; fm < thr_end2; ++fm ) {
    element_stats_type* del_gamma_img_ptr = &LIBXSMM_VLA_ACCESS(3, dgamma_img, fm, 0, 0, nImg, 16);
    element_stats_type* del_beta_img_ptr  = &LIBXSMM_VLA_ACCESS(3, dbeta_img,  fm, 0, 0, nImg, 16);
    __m512 lcl_vdgamma = _mm512_setzero_ps();
    __m512 lcl_vdbeta  = _mm512_setzero_ps();

    for ( img=0; img < nImg; img++ ) {
      lcl_vdgamma = _mm512_add_ps( lcl_vdgamma, _mm512_loadu_ps( del_gamma_img_ptr ) );
      lcl_vdbeta  = _mm512_add_ps( lcl_vdbeta,  _mm512_loadu_ps( del_beta_img_ptr  ) );
      del_gamma_img_ptr += 16;
      del_beta_img_ptr  += 16;
    }

    _mm512_storeu_ps( &LIBXSMM_VLA_ACCESS(2, dgamma, fm, 0, 16), lcl_vdgamma );
    _mm512_storeu_ps( &LIBXSMM_VLA_ACCESS(2, dbeta,  fm, 0, 16), lcl_vdbeta  );
  }

  libxsmm_barrier_wait(handle->barrier, ltid);
}

/* now we apply the actual backward batch norm */
for ( imgfm = thr_begin; imgfm < thr_end; ++imgfm ) {
  __m512 lcl_vgamma, lcl_vbmean, lcl_vbrstd, lcl_vdgamma, lcl_vdbeta;
  __m512 lcl_vnhw      = _mm512_set1_ps( nhw );
  __m512 lcl_vrec_nhw  = _mm512_set1_ps( recp_nhw );

  img = imgfm / nBlocksFm;
  fm = imgfm % nBlocksFm;
  lcl_vgamma  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, gamma,     fm, 0, 16) );
  lcl_vbmean  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, bmean,     fm, 0, 16) );
  lcl_vbrstd  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, brstd,     fm, 0, 16) );
  lcl_vdgamma = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, dgamma,    fm, 0, 16) );
  lcl_vdbeta  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, dbeta,     fm, 0, 16) );

  for ( hi=iph, ho=oph; hi < (ifh + iph); hi+=sh, ho++ ) {
          element_input_type*  del_input_ptr     = &LIBXSMM_VLA_ACCESS(5,     dinput, img, fm, hi, ipw, 0, nBlocksFm, ifhp, ifwp, 16);
    const element_input_type*  input_ptr         = &LIBXSMM_VLA_ACCESS(5,      input, img, fm, hi, ipw, 0, nBlocksFm, ifhp, ifwp, 16);
    const element_output_type* del_output_ptr    = &LIBXSMM_VLA_ACCESS(5,    doutput, img, fm, ho, opw, 0, nBlocksFm, ofhp, ofwp, 16);
    for ( wi=ipw, wo=opw; wi < (ifw + ipw); wi+=sw, wo++ ) {
      __m512 lcl_vdelinput;

      lcl_vdelinput = _mm512_sub_ps( _mm512_load_act( input_ptr ), lcl_vbmean );
      lcl_vdelinput = _mm512_mul_ps( lcl_vdelinput, lcl_vdgamma );
      lcl_vdelinput = _mm512_mul_ps( lcl_vdelinput, lcl_vbrstd  );
      lcl_vdelinput = _mm512_add_ps( lcl_vdbeta, lcl_vdelinput  );
      lcl_vdelinput = _mm512_sub_ps( _mm512_mul_ps( lcl_vnhw, _mm512_load_act( del_output_ptr ) ), lcl_vdelinput );
      lcl_vdelinput = _mm512_mul_ps( lcl_vrec_nhw, lcl_vdelinput );
      lcl_vdelinput = _mm512_mul_ps( lcl_vbrstd, lcl_vdelinput );
      lcl_vdelinput = _mm512_mul_ps( lcl_vgamma, lcl_vdelinput );
      _mm512_stream_act( del_input_ptr, lcl_vdelinput );

      del_input_ptr += sw*16;
      input_ptr += sw*16;
      del_output_ptr += 16;
    }
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

# undef _mm512_load_act
# undef _mm512_stream_act
# undef _mm512_store_act

