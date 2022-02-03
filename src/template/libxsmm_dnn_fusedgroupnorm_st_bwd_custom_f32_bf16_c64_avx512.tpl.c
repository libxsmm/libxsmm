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

#if defined(LIBXSMM_DNN_FUSEDGN_BWD_BF16)
# define _mm512_load_act(A)   _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(A))),16))
#if 1
# define _mm512_roundbf16rne(A) LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(A)
# define _mm512_stream_act(A,B) _mm256_stream_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_roundbf16rne((B)),16)))
# define _mm512_store_act(A,B)  _mm256_storeu_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_roundbf16rne((B)),16)))
#else
# define _mm512_stream_act(A,B) _mm256_stream_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_castps_si512((B)),16)))
# define _mm512_store_act(A,B)  _mm256_storeu_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_castps_si512((B)),16)))
#endif
#else
# define _mm512_load_act(A)   _mm512_loadu_ps(A)
# define _mm512_stream_act(A,B) LIBXSMM_INTRINSICS_MM512_STREAM_PS(A,B)
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

const element_stats_type nhw = (element_stats_type)(handle->desc.N * ifh * ifw);
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
const int work2 = nBlocksFm * 4;
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

LIBXSMM_VLA_DECL(5,       element_input_type,  dinput,     (element_input_type* )handle->grad_input->data,  nBlocksFm, ifhp, ifwp, 64);
LIBXSMM_VLA_DECL(5,       element_input_type,   input,     (element_input_type* )handle->reg_input->data,   nBlocksFm, ifhp, ifwp, 64);
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE)
LIBXSMM_VLA_DECL(5,       element_input_type,  dinput_add, (element_input_type* )handle->grad_add->data,    nBlocksFm, ifhp, ifwp, 64);
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU)
LIBXSMM_VLA_DECL(5, const element_output_type, output,     (element_output_type*)handle->reg_output->data,  nBlocksFm, ofhp, ofwp, 64);
#endif
LIBXSMM_VLA_DECL(5,       element_output_type, doutput,    (element_output_type*)handle->grad_output->data, nBlocksFm, ofhp, ofwp, 64);

LIBXSMM_VLA_DECL(2, const element_stats_type,  gamma,      (element_stats_type*)handle->reg_gamma->data,  64);
LIBXSMM_VLA_DECL(2,       element_stats_type,  dgamma,     (element_stats_type*)handle->grad_gamma->data, 64);
LIBXSMM_VLA_DECL(2,       element_stats_type,  dbeta,      (element_stats_type*)handle->grad_beta->data,  64);
LIBXSMM_VLA_DECL(2, const element_stats_type,  bmean,      (element_stats_type*)handle->expvalue->data,   64);
LIBXSMM_VLA_DECL(2, const element_stats_type,  brstd,      (element_stats_type*)handle->rcpstddev->data,  64);
LIBXSMM_VLA_DECL(3,       element_stats_type,  dgamma_img, (element_stats_type*)handle->scratch,                                                    nImg, 64);
LIBXSMM_VLA_DECL(3,       element_stats_type,  dbeta_img, ((element_stats_type*)handle->scratch) + ((size_t)nImg * (size_t)nBlocksFm * (size_t)64), nImg, 64);
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK)
LIBXSMM_VLA_DECL(5, const unsigned char,       relumask,   (unsigned char*)handle->relumask->data, nBlocksFm, ofhp, ofwp, 8);
#endif

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

  for ( imgfm = thr_begin; imgfm < thr_end; ++imgfm ) {
    __m512 lcl_vdgamma  = _mm512_setzero_ps();
    __m512 lcl_vdbeta   = _mm512_setzero_ps();
    __m512 lcl_vdgamma2 = _mm512_setzero_ps();
    __m512 lcl_vdbeta2  = _mm512_setzero_ps();
    __m512 lcl_vdgamma3 = _mm512_setzero_ps();
    __m512 lcl_vdbeta3  = _mm512_setzero_ps();
    __m512 lcl_vdgamma4 = _mm512_setzero_ps();
    __m512 lcl_vdbeta4  = _mm512_setzero_ps();
    __m512 lcl_vbmean,  lcl_vbrstd;
    __m512 lcl_vbmean2, lcl_vbrstd2;
    __m512 lcl_vbmean3, lcl_vbrstd3;
    __m512 lcl_vbmean4, lcl_vbrstd4;
    element_stats_type* del_gamma_img_ptr;
    element_stats_type* del_beta_img_ptr;

    img = imgfm / nBlocksFm;
    fm = imgfm % nBlocksFm;
    del_gamma_img_ptr = &LIBXSMM_VLA_ACCESS(3, dgamma_img, fm, img, 0, nImg, 64);
    del_beta_img_ptr  = &LIBXSMM_VLA_ACCESS(3, dbeta_img,  fm, img, 0, nImg, 64);
    lcl_vbmean  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, bmean, fm, 0,  64) );
    lcl_vbrstd  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, brstd, fm, 0,  64) );
    lcl_vbmean2 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, bmean, fm, 16, 64) );
    lcl_vbrstd2 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, brstd, fm, 16, 64) );
    lcl_vbmean3 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, bmean, fm, 32, 64) );
    lcl_vbrstd3 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, brstd, fm, 32, 64) );
    lcl_vbmean4 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, bmean, fm, 48, 64) );
    lcl_vbrstd4 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, brstd, fm, 48, 64) );

    for ( hi=iph, ho=oph; hi < (ifh + iph); hi+=sh, ho++ ) {
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE)
            element_input_type*  del_input_add_ptr = &LIBXSMM_VLA_ACCESS(5, dinput_add, img, fm, hi, ipw, 0, nBlocksFm, ifhp, ifwp, 64);
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU)
      const element_output_type* output_ptr        = &LIBXSMM_VLA_ACCESS(5,     output, img, fm, ho, opw, 0, nBlocksFm, ofhp, ofwp, 64);
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK)
      const unsigned char*       relumask_ptr      = &LIBXSMM_VLA_ACCESS(5,   relumask, img, fm, ho, opw, 0, nBlocksFm, ofhp, ofwp, 8);
#endif
      const element_input_type*  input_ptr         = &LIBXSMM_VLA_ACCESS(5,      input, img, fm, hi, ipw, 0, nBlocksFm, ifhp, ifwp, 64);
            element_output_type* del_output_ptr    = &LIBXSMM_VLA_ACCESS(5,    doutput, img, fm, ho, opw, 0, nBlocksFm, ofhp, ofwp, 64);
      for ( wi=ipw, wo=opw; wi < (ifw + ipw); wi+=sw, wo++ ) {
        __m512 lcl_vdeloutput, lcl_vdeloutput2, lcl_vdeloutput3, lcl_vdeloutput4;
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU)
        __mmask16 lcl_relumask, lcl_relumask2, lcl_relumask3, lcl_relumask4;
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK)
        __mmask16 lcl_relumask, lcl_relumask2, lcl_relumask3, lcl_relumask4;
#endif

        lcl_vdeloutput = _mm512_load_act( del_output_ptr );
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU)
        lcl_relumask = _mm512_cmp_ps_mask( _mm512_load_act( output_ptr ), _mm512_setzero_ps(), _CMP_NEQ_OQ );
        lcl_vdeloutput = _mm512_mask_blend_ps( lcl_relumask, _mm512_setzero_ps(), lcl_vdeloutput );
        _mm512_store_act( del_output_ptr, lcl_vdeloutput );
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK)
        lcl_relumask = LIBXSMM_INTRINSICS_MM512_LOAD_MASK16( relumask_ptr );
        lcl_vdeloutput = _mm512_mask_blend_ps( lcl_relumask, _mm512_setzero_ps(), lcl_vdeloutput );
        _mm512_store_act( del_output_ptr, lcl_vdeloutput );
        relumask_ptr += 2;
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE)
        _mm512_stream_act( del_input_add_ptr, lcl_vdeloutput );
#endif
        lcl_vdgamma = _mm512_add_ps( lcl_vdgamma, _mm512_mul_ps( _mm512_mul_ps( _mm512_sub_ps( _mm512_load_act( input_ptr ), lcl_vbmean ), lcl_vdeloutput ), lcl_vbrstd ) );
        lcl_vdbeta  = _mm512_add_ps( lcl_vdbeta, lcl_vdeloutput );

        lcl_vdeloutput2 = _mm512_load_act( del_output_ptr+16 );
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU)
        lcl_relumask2 = _mm512_cmp_ps_mask( _mm512_load_act( output_ptr+16 ), _mm512_setzero_ps(), _CMP_NEQ_OQ );
        lcl_vdeloutput2 = _mm512_mask_blend_ps( lcl_relumask2, _mm512_setzero_ps(), lcl_vdeloutput2 );
        _mm512_store_act( del_output_ptr+16, lcl_vdeloutput2 );
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK)
        lcl_relumask2 = LIBXSMM_INTRINSICS_MM512_LOAD_MASK16( relumask_ptr );
        lcl_vdeloutput2 = _mm512_mask_blend_ps( lcl_relumask2, _mm512_setzero_ps(), lcl_vdeloutput2 );
        _mm512_store_act( del_output_ptr+16, lcl_vdeloutput2 );
        relumask_ptr += 2;
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE)
        _mm512_stream_act( del_input_add_ptr+16, lcl_vdeloutput2 );
#endif
        lcl_vdgamma2 = _mm512_add_ps( lcl_vdgamma2, _mm512_mul_ps( _mm512_mul_ps( _mm512_sub_ps( _mm512_load_act( input_ptr+16 ), lcl_vbmean2 ), lcl_vdeloutput2 ), lcl_vbrstd2 ) );
        lcl_vdbeta2  = _mm512_add_ps( lcl_vdbeta2, lcl_vdeloutput2 );

        lcl_vdeloutput3 = _mm512_load_act( del_output_ptr+32 );
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU)
        lcl_relumask3 = _mm512_cmp_ps_mask( _mm512_load_act( output_ptr+32 ), _mm512_setzero_ps(), _CMP_NEQ_OQ );
        lcl_vdeloutput3 = _mm512_mask_blend_ps( lcl_relumask3, _mm512_setzero_ps(), lcl_vdeloutput3 );
        _mm512_store_act( del_output_ptr+32, lcl_vdeloutput3 );
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK)
        lcl_relumask3 = LIBXSMM_INTRINSICS_MM512_LOAD_MASK16( relumask_ptr );
        lcl_vdeloutput3 = _mm512_mask_blend_ps( lcl_relumask3, _mm512_setzero_ps(), lcl_vdeloutput3 );
        _mm512_store_act( del_output_ptr+32, lcl_vdeloutput3 );
        relumask_ptr += 2;
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE)
        _mm512_stream_act( del_input_add_ptr+32, lcl_vdeloutput3 );
#endif
        lcl_vdgamma3 = _mm512_add_ps( lcl_vdgamma3, _mm512_mul_ps( _mm512_mul_ps( _mm512_sub_ps( _mm512_load_act( input_ptr+32 ), lcl_vbmean3 ), lcl_vdeloutput3 ), lcl_vbrstd3 ) );
        lcl_vdbeta3  = _mm512_add_ps( lcl_vdbeta3, lcl_vdeloutput3 );

        lcl_vdeloutput4 = _mm512_load_act( del_output_ptr+48 );
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU)
        lcl_relumask4 = _mm512_cmp_ps_mask( _mm512_load_act( output_ptr+48 ), _mm512_setzero_ps(), _CMP_NEQ_OQ );
        lcl_vdeloutput4 = _mm512_mask_blend_ps( lcl_relumask4, _mm512_setzero_ps(), lcl_vdeloutput4 );
        _mm512_store_act( del_output_ptr+48, lcl_vdeloutput4 );
        output_ptr += 64;
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK)
        lcl_relumask4 = LIBXSMM_INTRINSICS_MM512_LOAD_MASK16( relumask_ptr );
        lcl_vdeloutput4 = _mm512_mask_blend_ps( lcl_relumask4, _mm512_setzero_ps(), lcl_vdeloutput4 );
        _mm512_store_act( del_output_ptr+48, lcl_vdeloutput4 );
        relumask_ptr += 2;
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE)
        _mm512_stream_act( del_input_add_ptr+48, lcl_vdeloutput4 );
        del_input_add_ptr += sw*64;
#endif
        lcl_vdgamma4 = _mm512_add_ps( lcl_vdgamma4, _mm512_mul_ps( _mm512_mul_ps( _mm512_sub_ps( _mm512_load_act( input_ptr+48 ), lcl_vbmean4 ), lcl_vdeloutput4 ), lcl_vbrstd4 ) );
        lcl_vdbeta4  = _mm512_add_ps( lcl_vdbeta4, lcl_vdeloutput4 );

        input_ptr += sw*64;
        del_output_ptr += 64;
      }
    }

    _mm512_storeu_ps( del_gamma_img_ptr,    lcl_vdgamma );
    _mm512_storeu_ps( del_beta_img_ptr,     lcl_vdbeta );
    _mm512_storeu_ps( del_gamma_img_ptr+16, lcl_vdgamma2 );
    _mm512_storeu_ps( del_beta_img_ptr+16,  lcl_vdbeta2 );
    _mm512_storeu_ps( del_gamma_img_ptr+32, lcl_vdgamma3 );
    _mm512_storeu_ps( del_beta_img_ptr+32,  lcl_vdbeta3 );
    _mm512_storeu_ps( del_gamma_img_ptr+48, lcl_vdgamma4 );
    _mm512_storeu_ps( del_beta_img_ptr+48,  lcl_vdbeta4 );
  }

  libxsmm_barrier_wait(handle->barrier, ltid);

    /* now we need to reduce the del_gamm and del_beta */
    for ( fm = thr_begin2; fm < thr_end2; ++fm ) {
      element_stats_type* del_gamma_img_ptr = &LIBXSMM_VLA_ACCESS(3, dgamma_img, (fm/4), 0, ((fm%4)*16), nImg, 64);
      element_stats_type* del_beta_img_ptr  = &LIBXSMM_VLA_ACCESS(3, dbeta_img,  (fm/4), 0, ((fm%4)*16), nImg, 64);
      __m512 lcl_vdgamma  = _mm512_setzero_ps();
      __m512 lcl_vdbeta   = _mm512_setzero_ps();

      for ( img=0; img < nImg; img++ ) {
        lcl_vdgamma  = _mm512_add_ps( lcl_vdgamma,  _mm512_loadu_ps( del_gamma_img_ptr ) );
        lcl_vdbeta   = _mm512_add_ps( lcl_vdbeta,   _mm512_loadu_ps( del_beta_img_ptr  ) );
        del_gamma_img_ptr += 64;
        del_beta_img_ptr  += 64;
      }

      _mm512_storeu_ps( &LIBXSMM_VLA_ACCESS(2, dgamma, (fm/4), ((fm%4)*16),  64), lcl_vdgamma );
      _mm512_storeu_ps( &LIBXSMM_VLA_ACCESS(2, dbeta,  (fm/4), ((fm%4)*16),  64), lcl_vdbeta  );
    }

  libxsmm_barrier_wait(handle->barrier, ltid);

  /* now we apply the actual backward batch norm */
  for ( imgfm = thr_begin; imgfm < thr_end; ++imgfm ) {
    __m512 lcl_vgamma,  lcl_vbmean,  lcl_vbrstd,  lcl_vdgamma,  lcl_vdbeta;
    __m512 lcl_vgamma2, lcl_vbmean2, lcl_vbrstd2, lcl_vdgamma2, lcl_vdbeta2;
    __m512 lcl_vgamma3, lcl_vbmean3, lcl_vbrstd3, lcl_vdgamma3, lcl_vdbeta3;
    __m512 lcl_vgamma4, lcl_vbmean4, lcl_vbrstd4, lcl_vdgamma4, lcl_vdbeta4;
    __m512 lcl_vnhw      = _mm512_set1_ps( nhw );
    __m512 lcl_vrec_nhw  = _mm512_set1_ps( recp_nhw );

    img = imgfm / nBlocksFm;
    fm = imgfm % nBlocksFm;
    lcl_vgamma   = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, gamma,     fm, 0, 64) );
    lcl_vbmean   = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, bmean,     fm, 0, 64) );
    lcl_vbrstd   = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, brstd,     fm, 0, 64) );
    lcl_vdgamma  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, dgamma,    fm, 0, 64) );
    lcl_vdbeta   = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, dbeta,     fm, 0, 64) );

    lcl_vgamma2  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, gamma,     fm, 16, 64) );
    lcl_vbmean2  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, bmean,     fm, 16, 64) );
    lcl_vbrstd2  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, brstd,     fm, 16, 64) );
    lcl_vdgamma2 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, dgamma,    fm, 16, 64) );
    lcl_vdbeta2  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, dbeta,     fm, 16, 64) );

    lcl_vgamma3  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, gamma,     fm, 32, 64) );
    lcl_vbmean3  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, bmean,     fm, 32, 64) );
    lcl_vbrstd3  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, brstd,     fm, 32, 64) );
    lcl_vdgamma3 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, dgamma,    fm, 32, 64) );
    lcl_vdbeta3  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, dbeta,     fm, 32, 64) );

    lcl_vgamma4  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, gamma,     fm, 48, 64) );
    lcl_vbmean4  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, bmean,     fm, 48, 64) );
    lcl_vbrstd4  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, brstd,     fm, 48, 64) );
    lcl_vdgamma4 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, dgamma,    fm, 48, 64) );
    lcl_vdbeta4  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, dbeta,     fm, 48, 64) );

    for ( hi=iph, ho=oph; hi < (ifh + iph); hi+=sh, ho++ ) {
      element_input_type*  del_input_ptr     = &LIBXSMM_VLA_ACCESS(5,     dinput, img, fm, hi, ipw, 0, nBlocksFm, ifhp, ifwp, 64);
      const element_input_type*  input_ptr         = &LIBXSMM_VLA_ACCESS(5,      input, img, fm, hi, ipw, 0, nBlocksFm, ifhp, ifwp, 64);
      const element_output_type* del_output_ptr    = &LIBXSMM_VLA_ACCESS(5,    doutput, img, fm, ho, opw, 0, nBlocksFm, ofhp, ofwp, 64);
      for ( wi=ipw, wo=opw; wi < (ifw + ipw); wi+=sw, wo++ ) {
        __m512 lcl_vdelinput;
        __m512 lcl_vdelinput2;
        __m512 lcl_vdelinput3;
        __m512 lcl_vdelinput4;

        lcl_vdelinput = _mm512_sub_ps( _mm512_load_act( input_ptr ), lcl_vbmean );
        lcl_vdelinput = _mm512_mul_ps( lcl_vdelinput, lcl_vdgamma );
        lcl_vdelinput = _mm512_mul_ps( lcl_vdelinput, lcl_vbrstd  );
        lcl_vdelinput = _mm512_add_ps( lcl_vdbeta, lcl_vdelinput  );
        lcl_vdelinput = _mm512_sub_ps( _mm512_mul_ps( lcl_vnhw, _mm512_load_act( del_output_ptr ) ), lcl_vdelinput );
        lcl_vdelinput = _mm512_mul_ps( lcl_vrec_nhw, lcl_vdelinput );
        lcl_vdelinput = _mm512_mul_ps( lcl_vbrstd, lcl_vdelinput );
        lcl_vdelinput = _mm512_mul_ps( lcl_vgamma, lcl_vdelinput );

        lcl_vdelinput2 = _mm512_sub_ps( _mm512_load_act( input_ptr+16 ), lcl_vbmean2 );
        lcl_vdelinput2 = _mm512_mul_ps( lcl_vdelinput2, lcl_vdgamma2 );
        lcl_vdelinput2 = _mm512_mul_ps( lcl_vdelinput2, lcl_vbrstd2  );
        lcl_vdelinput2 = _mm512_add_ps( lcl_vdbeta2, lcl_vdelinput2  );
        lcl_vdelinput2 = _mm512_sub_ps( _mm512_mul_ps( lcl_vnhw, _mm512_load_act( del_output_ptr+16 ) ), lcl_vdelinput2 );
        lcl_vdelinput2 = _mm512_mul_ps( lcl_vrec_nhw, lcl_vdelinput2 );
        lcl_vdelinput2 = _mm512_mul_ps( lcl_vbrstd2, lcl_vdelinput2 );
        lcl_vdelinput2 = _mm512_mul_ps( lcl_vgamma2, lcl_vdelinput2 );

        lcl_vdelinput3 = _mm512_sub_ps( _mm512_load_act( input_ptr+32 ), lcl_vbmean3 );
        lcl_vdelinput3 = _mm512_mul_ps( lcl_vdelinput3, lcl_vdgamma3 );
        lcl_vdelinput3 = _mm512_mul_ps( lcl_vdelinput3, lcl_vbrstd3  );
        lcl_vdelinput3 = _mm512_add_ps( lcl_vdbeta3, lcl_vdelinput3  );
        lcl_vdelinput3 = _mm512_sub_ps( _mm512_mul_ps( lcl_vnhw, _mm512_load_act( del_output_ptr+32 ) ), lcl_vdelinput3 );
        lcl_vdelinput3 = _mm512_mul_ps( lcl_vrec_nhw, lcl_vdelinput3 );
        lcl_vdelinput3 = _mm512_mul_ps( lcl_vbrstd3, lcl_vdelinput3 );
        lcl_vdelinput3 = _mm512_mul_ps( lcl_vgamma3, lcl_vdelinput3 );

        lcl_vdelinput4 = _mm512_sub_ps( _mm512_load_act( input_ptr+48 ), lcl_vbmean4 );
        lcl_vdelinput4 = _mm512_mul_ps( lcl_vdelinput4, lcl_vdgamma4 );
        lcl_vdelinput4 = _mm512_mul_ps( lcl_vdelinput4, lcl_vbrstd4  );
        lcl_vdelinput4 = _mm512_add_ps( lcl_vdbeta4, lcl_vdelinput4  );
        lcl_vdelinput4 = _mm512_sub_ps( _mm512_mul_ps( lcl_vnhw, _mm512_load_act( del_output_ptr+48 ) ), lcl_vdelinput4 );
        lcl_vdelinput4 = _mm512_mul_ps( lcl_vrec_nhw, lcl_vdelinput4 );
        lcl_vdelinput4 = _mm512_mul_ps( lcl_vbrstd4, lcl_vdelinput4 );
        lcl_vdelinput4 = _mm512_mul_ps( lcl_vgamma4, lcl_vdelinput4 );

        _mm512_stream_act( del_input_ptr,    lcl_vdelinput );
        _mm512_stream_act( del_input_ptr+16, lcl_vdelinput2 );
        _mm512_stream_act( del_input_ptr+32, lcl_vdelinput3 );
        _mm512_stream_act( del_input_ptr+48, lcl_vdelinput4 );

        del_input_ptr += sw*64;
        input_ptr += sw*64;
        del_output_ptr += 64;
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);

# undef _mm512_load_act
# undef _mm512_stream_act
# undef _mm512_store_act

