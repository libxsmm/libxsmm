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

#if defined(LIBXSMM_DNN_FUSEDGN_FWD_BF16)
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

/* eps to avoid sqrt of zero */
const element_stats_type sqrt_eps = 1e-7f;
const element_stats_type nhw = (element_stats_type)(handle->desc.N * ifh * ifw);
const element_stats_type recp_nhw = 1.0f/nhw;

/* loop variables */
int img = 0;
int fm = 0;
int imgfm = 0;
int hi = 0;
int wi = 0;
int ho = 0;
int wo = 0;

LIBXSMM_VLA_DECL(5, const element_input_type, input,     (element_input_type* )handle->reg_input->data,  nBlocksFm, ifhp, ifwp, 64);
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE)
LIBXSMM_VLA_DECL(5, const element_input_type, input_add, (element_input_type* )handle->reg_add->data,    nBlocksFm, ifhp, ifwp, 64);
#endif
LIBXSMM_VLA_DECL(5, element_output_type,      output,    (element_output_type*)handle->reg_output->data, nBlocksFm, ofhp, ofwp, 64);
LIBXSMM_VLA_DECL(2, const element_stats_type, gamma,     (element_stats_type*)handle->reg_gamma->data,   64);
LIBXSMM_VLA_DECL(2, const element_stats_type, beta,      (element_stats_type*)handle->reg_beta->data,    64);
LIBXSMM_VLA_DECL(2,       element_stats_type, bmean,     (element_stats_type*)handle->expvalue->data,    64);
LIBXSMM_VLA_DECL(2,       element_stats_type, brstd,     (element_stats_type*)handle->rcpstddev->data,   64);
LIBXSMM_VLA_DECL(2,       element_stats_type, variance,  (element_stats_type*)handle->variance->data,    64);
LIBXSMM_VLA_DECL(3,       element_stats_type, sum_img,   (element_stats_type*)handle->scratch,                                             nImg, 64);
LIBXSMM_VLA_DECL(3,       element_stats_type, sumsq_img, ((element_stats_type*)handle->scratch) + ((size_t)nImg * (size_t)nBlocksFm * 64), nImg, 64);
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK)
LIBXSMM_VLA_DECL(5,       unsigned char,      relumask,  (unsigned char*)handle->relumask->data, nBlocksFm, ofhp, ofwp, 8);
#endif

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

  for ( imgfm = thr_begin; imgfm < thr_end; ++imgfm ) {
    __m512 lcl_vsum    = _mm512_setzero_ps();
    __m512 lcl_vsumsq  = _mm512_setzero_ps();
    __m512 lcl_vsum2   = _mm512_setzero_ps();
    __m512 lcl_vsumsq2 = _mm512_setzero_ps();
    __m512 lcl_vsum3   = _mm512_setzero_ps();
    __m512 lcl_vsumsq3 = _mm512_setzero_ps();
    __m512 lcl_vsum4   = _mm512_setzero_ps();
    __m512 lcl_vsumsq4 = _mm512_setzero_ps();
    element_stats_type* sum_img_ptr;
    element_stats_type* sumsq_img_ptr;

    img = imgfm / nBlocksFm;
    fm = imgfm % nBlocksFm;
    sum_img_ptr = &LIBXSMM_VLA_ACCESS(3, sum_img, fm, img, 0, nImg, 64);
    sumsq_img_ptr = &LIBXSMM_VLA_ACCESS(3, sumsq_img,  fm, img, 0, nImg, 64);

    for ( hi=iph; hi < (ifh + iph); hi++ ) {
      const element_input_type* input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, fm, hi, ipw, 0, nBlocksFm, ifhp, ifwp, 64);
      for ( wi=ipw; wi < (ifw + ipw); wi++ ) {
        __m512 lcl_vinput  = _mm512_load_act( input_ptr );
        __m512 lcl_vinput2 = _mm512_load_act( input_ptr+16 );
        __m512 lcl_vinput3 = _mm512_load_act( input_ptr+32 );
        __m512 lcl_vinput4 = _mm512_load_act( input_ptr+48 );

        lcl_vsum    = _mm512_add_ps( lcl_vsum, lcl_vinput );
        lcl_vsumsq  = _mm512_add_ps( lcl_vsumsq, _mm512_mul_ps( lcl_vinput, lcl_vinput ) );

        lcl_vsum2   = _mm512_add_ps( lcl_vsum2, lcl_vinput2 );
        lcl_vsumsq2 = _mm512_add_ps( lcl_vsumsq2, _mm512_mul_ps( lcl_vinput2, lcl_vinput2 ) );

        lcl_vsum3   = _mm512_add_ps( lcl_vsum3, lcl_vinput3 );
        lcl_vsumsq3 = _mm512_add_ps( lcl_vsumsq3, _mm512_mul_ps( lcl_vinput3, lcl_vinput3 ) );

        lcl_vsum4   = _mm512_add_ps( lcl_vsum4, lcl_vinput4 );
        lcl_vsumsq4 = _mm512_add_ps( lcl_vsumsq4, _mm512_mul_ps( lcl_vinput4, lcl_vinput4 ) );

        input_ptr += 64;
      }
    }

    _mm512_storeu_ps( sum_img_ptr,      lcl_vsum );
    _mm512_storeu_ps( sumsq_img_ptr,    lcl_vsumsq );

    _mm512_storeu_ps( sum_img_ptr+16,   lcl_vsum2 );
    _mm512_storeu_ps( sumsq_img_ptr+16, lcl_vsumsq2 );

    _mm512_storeu_ps( sum_img_ptr+32,   lcl_vsum3 );
    _mm512_storeu_ps( sumsq_img_ptr+32, lcl_vsumsq3 );

    _mm512_storeu_ps( sum_img_ptr+48,   lcl_vsum4 );
    _mm512_storeu_ps( sumsq_img_ptr+48, lcl_vsumsq4 );
  }

  libxsmm_barrier_wait(handle->barrier, ltid);

  /* now we need to reduce the sum and sum^2, we use the final  */
  for ( fm = thr_begin2; fm < thr_end2; ++fm ) {
    __m512 lcl_vsum      = _mm512_setzero_ps();
    __m512 lcl_vsumsq    = _mm512_setzero_ps();
    element_stats_type* sum_img_ptr   = &LIBXSMM_VLA_ACCESS(3, sum_img,   (fm/4), 0, ((fm%4)*16), nImg, 64);
    element_stats_type* sumsq_img_ptr = &LIBXSMM_VLA_ACCESS(3, sumsq_img, (fm/4), 0, ((fm%4)*16), nImg, 64);

    for ( img=0; img < nImg; img++ ) {
      lcl_vsum    = _mm512_add_ps( lcl_vsum,    _mm512_loadu_ps( sum_img_ptr ) );
      lcl_vsumsq  = _mm512_add_ps( lcl_vsumsq,  _mm512_loadu_ps( sumsq_img_ptr ) );

      sum_img_ptr   += 64;
      sumsq_img_ptr += 64;
    }

      __m512 lcl_vsqrt_eps = _mm512_set1_ps(sqrt_eps);
      __m512 lcl_vrec_nhw  = _mm512_set1_ps(recp_nhw);
      __m512 lcl_vone      = _mm512_set1_ps(1.0);
      __m512 lcl_vbmean,  lcl_vbmeansq,  lcl_vsqbmean,  lcl_vbrstd,  lcl_vvar;

      lcl_vbmean    = _mm512_mul_ps( lcl_vrec_nhw, lcl_vsum   );   /* E(X) */
      lcl_vbmeansq  = _mm512_mul_ps( lcl_vbmean,   lcl_vbmean );   /* E(X)^2 */
      lcl_vsqbmean  = _mm512_mul_ps( lcl_vrec_nhw, lcl_vsumsq );   /* E(X^2) */
      lcl_vvar      = _mm512_sub_ps( lcl_vsqbmean, lcl_vbmeansq ); /* variance */
      lcl_vbrstd    = _mm512_div_ps( lcl_vone, _mm512_sqrt_ps( _mm512_add_ps( lcl_vvar, lcl_vsqrt_eps ) ) );

      _mm512_storeu_ps( &LIBXSMM_VLA_ACCESS(2, bmean,    (fm/4), ((fm%4)*16), 64), lcl_vbmean );
      _mm512_storeu_ps( &LIBXSMM_VLA_ACCESS(2, brstd,    (fm/4), ((fm%4)*16), 64), lcl_vbrstd );
      _mm512_storeu_ps( &LIBXSMM_VLA_ACCESS(2, variance, (fm/4), ((fm%4)*16), 64), lcl_vvar );
  }

  libxsmm_barrier_wait(handle->barrier, ltid);

  /* now we apply the actual forward batch norm */
  for ( imgfm = thr_begin; imgfm < thr_end; ++imgfm ) {
    __m512 lcl_vgamma,  lcl_vbeta,  lcl_vbmean,  lcl_vbrstd;
    __m512 lcl_vgamma2, lcl_vbeta2, lcl_vbmean2, lcl_vbrstd2;
    __m512 lcl_vgamma3, lcl_vbeta3, lcl_vbmean3, lcl_vbrstd3;
    __m512 lcl_vgamma4, lcl_vbeta4, lcl_vbmean4, lcl_vbrstd4;

    img = imgfm / nBlocksFm;
    fm = imgfm % nBlocksFm;
    lcl_vgamma  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, gamma,     fm, 0, 64) );
    lcl_vbeta   = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, beta,      fm, 0, 64) );
    lcl_vbmean  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, bmean,     fm, 0, 64) );
    lcl_vbrstd  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, brstd,     fm, 0, 64) );

    lcl_vgamma2 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, gamma,     fm, 16, 64) );
    lcl_vbeta2  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, beta,      fm, 16, 64) );
    lcl_vbmean2 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, bmean,     fm, 16, 64) );
    lcl_vbrstd2 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, brstd,     fm, 16, 64) );

    lcl_vgamma3 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, gamma,     fm, 32, 64) );
    lcl_vbeta3  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, beta,      fm, 32, 64) );
    lcl_vbmean3 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, bmean,     fm, 32, 64) );
    lcl_vbrstd3 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, brstd,     fm, 32, 64) );

    lcl_vgamma4 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, gamma,     fm, 48, 64) );
    lcl_vbeta4  = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, beta,      fm, 48, 64) );
    lcl_vbmean4 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, bmean,     fm, 48, 64) );
    lcl_vbrstd4 = _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, brstd,     fm, 48, 64) );

    for ( hi=iph, ho=oph; hi < (ifh+iph); hi+=sh, ho++ ) {
      const element_input_type*  input_ptr     = &LIBXSMM_VLA_ACCESS(5, input,     img, fm, hi, ipw, 0, nBlocksFm, ifhp, ifwp, 64);
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE)
      const element_input_type*  input_add_ptr = &LIBXSMM_VLA_ACCESS(5, input_add, img, fm, hi, ipw, 0, nBlocksFm, ifhp, ifwp, 64);
#endif
            element_output_type* output_ptr    = &LIBXSMM_VLA_ACCESS(5, output,    img, fm, ho, opw, 0, nBlocksFm, ofhp, ofwp, 64);
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK)
            unsigned char*       relumask_ptr  = &LIBXSMM_VLA_ACCESS(5, relumask,  img, fm, ho, opw, 0, nBlocksFm, ofhp, ofwp, 8);
#endif
      for ( wi=ipw, wo=opw; wi < (ifw+ipw); wi+=sw, wo++ ) {
        __m512 lcl_vo;
        __m512 lcl_vo2;
        __m512 lcl_vo3;
        __m512 lcl_vo4;
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK)
        __mmask16 lcl_relumask;
        __mmask16 lcl_relumask2;
        __mmask16 lcl_relumask3;
        __mmask16 lcl_relumask4;
#endif

        /* BN + scale (gamma, beta) */
        lcl_vo = _mm512_sub_ps( _mm512_load_act( input_ptr ), lcl_vbmean );
        lcl_vo = _mm512_mul_ps( lcl_vgamma, lcl_vo );
        lcl_vo = _mm512_fmadd_ps( lcl_vo, lcl_vbrstd, lcl_vbeta );
        /* eltwise add */
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE)
        lcl_vo = _mm512_add_ps( lcl_vo, _mm512_load_act( input_add_ptr ) );
#endif
        /* ReLU */
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU)
        lcl_vo = _mm512_max_ps( lcl_vo, _mm512_setzero_ps() );
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK)
        lcl_relumask = _mm512_cmp_ps_mask( lcl_vo, _mm512_setzero_ps(), _CMP_GT_OQ );
        lcl_vo = _mm512_mask_blend_ps( lcl_relumask, _mm512_setzero_ps(), lcl_vo );
        LIBXSMM_INTRINSICS_MM512_STORE_MASK16( relumask_ptr, lcl_relumask );
        relumask_ptr += 2;
#endif

        /* BN + scale (gamma, beta) */
        lcl_vo2 = _mm512_sub_ps( _mm512_load_act( input_ptr+16 ), lcl_vbmean2 );
        lcl_vo2 = _mm512_mul_ps( lcl_vgamma2, lcl_vo2 );
        lcl_vo2 = _mm512_fmadd_ps( lcl_vo2, lcl_vbrstd2, lcl_vbeta2 );
        /* eltwise add */
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE)
        lcl_vo2 = _mm512_add_ps( lcl_vo2, _mm512_load_act( input_add_ptr+16 ) );
#endif
        /* ReLU */
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU)
        lcl_vo2 = _mm512_max_ps( lcl_vo2, _mm512_setzero_ps() );
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK)
        lcl_relumask2 = _mm512_cmp_ps_mask( lcl_vo2, _mm512_setzero_ps(), _CMP_GT_OQ );
        lcl_vo2 = _mm512_mask_blend_ps( lcl_relumask2, _mm512_setzero_ps(), lcl_vo2 );
        LIBXSMM_INTRINSICS_MM512_STORE_MASK16( relumask_ptr, lcl_relumask2 );
        relumask_ptr += 2;
#endif

        /* BN + scale (gamma, beta) */
        lcl_vo3 = _mm512_sub_ps( _mm512_load_act( input_ptr+32 ), lcl_vbmean3 );
        lcl_vo3 = _mm512_mul_ps( lcl_vgamma3, lcl_vo3 );
        lcl_vo3 = _mm512_fmadd_ps( lcl_vo3, lcl_vbrstd3, lcl_vbeta3 );
        /* eltwise add */
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE)
        lcl_vo3 = _mm512_add_ps( lcl_vo3, _mm512_load_act( input_add_ptr+32 ) );
#endif
        /* ReLU */
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU)
        lcl_vo3 = _mm512_max_ps( lcl_vo3, _mm512_setzero_ps() );
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK)
        lcl_relumask3 = _mm512_cmp_ps_mask( lcl_vo3, _mm512_setzero_ps(), _CMP_GT_OQ );
        lcl_vo3 = _mm512_mask_blend_ps( lcl_relumask3, _mm512_setzero_ps(), lcl_vo3 );
        LIBXSMM_INTRINSICS_MM512_STORE_MASK16( relumask_ptr, lcl_relumask3 );
        relumask_ptr += 2;
#endif

        /* BN + scale (gamma, beta) */
        lcl_vo4 = _mm512_sub_ps( _mm512_load_act( input_ptr+48 ), lcl_vbmean4 );
        lcl_vo4 = _mm512_mul_ps( lcl_vgamma4, lcl_vo4 );
        lcl_vo4 = _mm512_fmadd_ps( lcl_vo4, lcl_vbrstd4, lcl_vbeta4 );
        /* eltwise add */
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE)
        lcl_vo4 = _mm512_add_ps( lcl_vo4, _mm512_load_act( input_add_ptr+48 ) );
#endif
        /* ReLU */
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU)
        lcl_vo4 = _mm512_max_ps( lcl_vo4, _mm512_setzero_ps() );
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK)
        lcl_relumask4 = _mm512_cmp_ps_mask( lcl_vo4, _mm512_setzero_ps(), _CMP_GT_OQ );
        lcl_vo4 = _mm512_mask_blend_ps( lcl_relumask4, _mm512_setzero_ps(), lcl_vo4 );
        LIBXSMM_INTRINSICS_MM512_STORE_MASK16( relumask_ptr, lcl_relumask4 );
        relumask_ptr += 2;
#endif

        _mm512_stream_act( output_ptr, lcl_vo );
        _mm512_stream_act( output_ptr+16, lcl_vo2 );
        _mm512_stream_act( output_ptr+32, lcl_vo3 );
        _mm512_stream_act( output_ptr+48, lcl_vo4 );

        input_ptr += sw*64;
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE)
        input_add_ptr += sw*64;
#endif
        output_ptr += 64;
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);

# undef _mm512_load_act
# undef _mm512_stream_act
# undef _mm512_store_act

