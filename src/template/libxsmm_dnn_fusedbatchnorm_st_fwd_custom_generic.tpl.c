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

/* size variables, all const */
const int nImg = handle->desc.partN;
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
const int nFmBlock = handle->ifmblock;

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

/* eps to avoid sqrt of zero */
const element_stats_type sqrt_eps = 1e-7f;
const element_stats_type nhw = (element_stats_type)(handle->desc.fullN * ifh * ifw);
const element_stats_type recp_nhw = 1.0f/nhw;

/* loop variables */
int img = 0;
int fm = 0;
int imgfm = 0;
int hi = 0;
int wi = 0;
int v = 0;
int ho = 0;
int wo = 0;

LIBXSMM_VLA_DECL(5, const element_input_type, input,     (element_input_type* )handle->reg_input->data,  nBlocksFm, ifhp, ifwp, nFmBlock);
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_ENABLE_ELTWISE)
LIBXSMM_VLA_DECL(5, const element_input_type, input_add, (element_input_type* )handle->reg_add->data,    nBlocksFm, ifhp, ifwp, nFmBlock);
#endif
LIBXSMM_VLA_DECL(5, element_output_type,      output,    (element_output_type*)handle->reg_output->data, nBlocksFm, ofhp, ofwp, nFmBlock);
LIBXSMM_VLA_DECL(2, const element_stats_type, gamma,     (element_stats_type*)handle->reg_gamma->data,   nFmBlock);
LIBXSMM_VLA_DECL(2, const element_stats_type, beta,      (element_stats_type*)handle->reg_beta->data,    nFmBlock);
LIBXSMM_VLA_DECL(2,       element_stats_type, bmean,     (element_stats_type*)handle->expvalue->data,    nFmBlock);
LIBXSMM_VLA_DECL(2,       element_stats_type, brstd,     (element_stats_type*)handle->rcpstddev->data,   nFmBlock);
LIBXSMM_VLA_DECL(2,       element_stats_type, variance,  (element_stats_type*)handle->variance->data,    nFmBlock);
LIBXSMM_VLA_DECL(3,       element_stats_type, sum_img,   (element_stats_type*)handle->scratch,                                                           nImg, nFmBlock);
LIBXSMM_VLA_DECL(3,       element_stats_type, sumsq_img, ((element_stats_type*)handle->scratch) + ((size_t)nImg * (size_t)nBlocksFm * (size_t)nFmBlock), nImg, nFmBlock);
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_ENABLE_RELU_WITH_MASK)
LIBXSMM_VLA_DECL(5,       unsigned char,      relumask,  (unsigned char*)handle->relumask->data, nBlocksFm, ofhp, ofwp, nFmBlock);
#endif

#if defined(LIBXSMM_DNN_FUSEDBN_FWD_BF16)
union libxsmm_bfloat16_hp input_f32;
union libxsmm_bfloat16_hp output_f32;
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_ENABLE_ELTWISE)
union libxsmm_bfloat16_hp input_add_f32;
input_add_f32.i[1]  = 0;
input_add_f32.i[0]  = 0;
#endif
input_f32.i[1]  = 0;
input_f32.i[0]  = 0;
output_f32.i[1] = 0;
output_f32.i[0] = 0;
#endif

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

if ( ((handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BN) > 0)            ||
     ((handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS) > 0)       ||
     ((handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED) > 0)    ) {
  for ( imgfm = thr_begin; imgfm < thr_end; ++imgfm ) {
    /* @TODO check if we can bake this in into scratch */
    element_stats_type lcl_sum_ptr[64];
    element_stats_type lcl_sumsq_ptr[64];
    element_stats_type* sum_img_ptr;
    element_stats_type* sumsq_img_ptr;

    img = imgfm / nBlocksFm;
    fm = imgfm % nBlocksFm;
    sum_img_ptr = &LIBXSMM_VLA_ACCESS(3, sum_img, fm, img, 0, nImg, nFmBlock);
    sumsq_img_ptr = &LIBXSMM_VLA_ACCESS(3, sumsq_img,  fm, img, 0, nImg, nFmBlock);

    LIBXSMM_PRAGMA_SIMD
    for ( v=0; v < nFmBlock; v++ ) {
      lcl_sum_ptr[v] = (element_stats_type)0;
      lcl_sumsq_ptr[v] = (element_stats_type)0;
    }

    for ( hi=iph; hi < (ifh + iph); hi++ ) {
      for ( wi=ipw; wi < (ifw + ipw); wi++ ) {
        const element_input_type* input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, fm, hi, wi, 0, nBlocksFm, ifhp, ifwp, nFmBlock);

#if !defined(LIBXSMM_DNN_FUSEDBN_FWD_BF16)
        LIBXSMM_PRAGMA_SIMD
#endif
        for (v=0; v < nFmBlock; v++) {
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_BF16)
          input_f32.i[1] = input_ptr[v];
          lcl_sum_ptr[v]   += input_f32.f;
          lcl_sumsq_ptr[v] += (input_f32.f * input_f32.f);
#else
          lcl_sum_ptr[v]   += input_ptr[v];
          lcl_sumsq_ptr[v] += (input_ptr[v] * input_ptr[v]);
#endif
        }
      }
    }

    LIBXSMM_PRAGMA_SIMD
    for (v=0; v < nFmBlock; v++) {
      sum_img_ptr[v] = lcl_sum_ptr[v];
      sumsq_img_ptr[v] = lcl_sumsq_ptr[v];
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);

 /* now we need to reduce the sum and sum^2, we use the final  */
  for ( fm = thr_begin2; fm < thr_end2; ++fm ) {
    /* @TODO check if we can bake this in into scratch */
    element_stats_type lcl_sum_ptr[64];
    element_stats_type lcl_sumsq_ptr[64];
    element_stats_type* bmean_ptr = &LIBXSMM_VLA_ACCESS(2, bmean,    fm, 0, nFmBlock);
    element_stats_type* brstd_ptr = &LIBXSMM_VLA_ACCESS(2, brstd,    fm, 0, nFmBlock);
    element_stats_type* tvar_ptr  = &LIBXSMM_VLA_ACCESS(2, variance, fm, 0, nFmBlock);

    LIBXSMM_PRAGMA_SIMD
    for ( v=0; v < nFmBlock; v++ ) {
      lcl_sum_ptr[v]   = (element_stats_type)0;
      lcl_sumsq_ptr[v] = (element_stats_type)0;
    }

    for ( img=0; img < nImg; img++ ) {
      element_stats_type* sum_img_ptr   = &LIBXSMM_VLA_ACCESS(3, sum_img,   fm, img, 0, nImg, nFmBlock);
      element_stats_type* sumsq_img_ptr = &LIBXSMM_VLA_ACCESS(3, sumsq_img, fm, img, 0, nImg, nFmBlock);

      LIBXSMM_PRAGMA_SIMD
      for ( v=0; v < nFmBlock; v++ ) {
        lcl_sum_ptr[v] += sum_img_ptr[v];
        lcl_sumsq_ptr[v] += sumsq_img_ptr[v];
      }
    }

    if ( ((handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BN) > 0)      ||
         ((handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS) > 0)    ) {
      LIBXSMM_PRAGMA_SIMD
      for ( v=0; v < nFmBlock; v++ ) {
        const element_stats_type tbmean = (recp_nhw * lcl_sum_ptr[v]);
        const element_stats_type tbmeansq = tbmean * tbmean;
        const element_stats_type tsqbmean = recp_nhw * lcl_sumsq_ptr[v];
        const element_stats_type tvar     = tsqbmean - tbmeansq;
        const element_stats_type tbrstd = (element_stats_type)(1.0/sqrt((double)tvar + sqrt_eps));
        bmean_ptr[v] = tbmean;
        brstd_ptr[v] = tbrstd;
        tvar_ptr[v] = tvar;
      }
    } else {
      element_stats_type* sum_ptr   = &LIBXSMM_VLA_ACCESS(3, sum_img,   fm, 0, 0, nImg, nFmBlock);
      element_stats_type* sumsq_ptr = &LIBXSMM_VLA_ACCESS(3, sumsq_img, fm, 0, 0, nImg, nFmBlock);

      LIBXSMM_PRAGMA_SIMD
      for ( v=0; v < nFmBlock; v++ ) {
        sum_ptr[v]   = lcl_sum_ptr[v];
        sumsq_ptr[v] = lcl_sumsq_ptr[v];
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);
}

if ( ((handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BN) > 0)      ||
     ((handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE) > 0)    ) {
  /* now we apply the actual forward batch norm */
  for ( imgfm = thr_begin; imgfm < thr_end; ++imgfm ) {
    img = imgfm / nBlocksFm;
    fm = imgfm % nBlocksFm;
    for ( hi=iph, ho=oph; hi < (ifh+iph); hi+=sh, ho++ ) {
      for ( wi=ipw, wo=opw; wi < (ifw+ipw); wi+=sw, wo++ ) {
        const element_input_type*  input_ptr     = &LIBXSMM_VLA_ACCESS(5, input,     img, fm, hi, wi, 0, nBlocksFm, ifhp, ifwp, nFmBlock);
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_ENABLE_ELTWISE)
        const element_input_type*  input_add_ptr = &LIBXSMM_VLA_ACCESS(5, input_add, img, fm, hi, wi, 0, nBlocksFm, ifhp, ifwp, nFmBlock);
#endif
        const element_stats_type*  gamma_ptr     = &LIBXSMM_VLA_ACCESS(2, gamma,     fm, 0, nFmBlock);
        const element_stats_type*  beta_ptr      = &LIBXSMM_VLA_ACCESS(2, beta,      fm, 0, nFmBlock);
        const element_stats_type*  bmean_ptr     = &LIBXSMM_VLA_ACCESS(2, bmean,     fm, 0, nFmBlock);
        const element_stats_type*  brstd_ptr     = &LIBXSMM_VLA_ACCESS(2, brstd,     fm, 0, nFmBlock);
              element_output_type* output_ptr    = &LIBXSMM_VLA_ACCESS(5, output,    img, fm, ho, wo, 0, nBlocksFm, ofhp, ofwp, nFmBlock);
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_ENABLE_RELU_WITH_MASK)
              unsigned char*       relumask_ptr  = &LIBXSMM_VLA_ACCESS(5, relumask,  img, fm, ho, wo, 0, nBlocksFm, ofhp, ofwp, nFmBlock);
#endif
        float o;

#if !defined(LIBXSMM_DNN_FUSEDBN_FWD_BF16)
        LIBXSMM_PRAGMA_SIMD
#endif
        for (v = 0; v < nFmBlock; v++ ) {
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_BF16)
          input_f32.i[1] = input_ptr[v];
          o = gamma_ptr[v]*(input_f32.f - bmean_ptr[v])*brstd_ptr[v] + beta_ptr[v];
#else
          /* BN + scale (gamma, beta) */
          o = gamma_ptr[v]*(input_ptr[v] - bmean_ptr[v])*brstd_ptr[v] + beta_ptr[v];
#endif
          /* Eltwise */
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_ENABLE_ELTWISE)
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_BF16)
          input_add_f32.i[1] = input_add_ptr[v];
          o += input_add_f32.f;
#else
          o += input_add_ptr[v];
#endif
#endif
          /* ReLU */
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_ENABLE_RELU)
          o = ( o > 0.0f ) ? o : 0.0f;
#endif
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_ENABLE_RELU_WITH_MASK)
          o = ( o > 0.0f ) ? o : 0.0f;
          relumask_ptr[v] = (unsigned char)(o > 0.0f ? 1 : 0);
#endif
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_BF16)
          output_f32.f = o;
          output_ptr[v] = output_f32.i[1];
#else
          output_ptr[v] = o;
#endif
        }
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);
}

