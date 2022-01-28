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
const int nImg = handle->desc.N;
const int nG = handle->desc.G;
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
/* derive channels per group */
const int nFmG = (nBlocksFm * nFmBlock) / nG;
/* size of sample */
const element_stats_type ghw = (element_stats_type)(nFmG * ifh * ifw);
const element_stats_type recp_ghw = 1.0f/ghw;

/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
/* @TODO let's fix parallelization to include channel groups while avoiding conflict misses */
const int work = nImg;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* eps to avoid sqrt of zero */
const element_stats_type sqrt_eps = 1e-7f;

/* loop variables */
int img = 0;
int fm = 0;
/*int imgfm = 0;*/
int hi = 0;
int wi = 0;
int v = 0;
int ho = 0;
int wo = 0;
int g = 0;

LIBXSMM_VLA_DECL(5, const element_input_type, input,     (element_input_type* )handle->reg_input->data,  nBlocksFm, ifhp, ifwp, nFmBlock);
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE)
LIBXSMM_VLA_DECL(5, const element_input_type, input_add, (element_input_type* )handle->reg_add->data,    nBlocksFm, ifhp, ifwp, nFmBlock);
#endif
LIBXSMM_VLA_DECL(5, element_output_type,      output,    (element_output_type*)handle->reg_output->data, nBlocksFm, ofhp, ofwp, nFmBlock);
LIBXSMM_VLA_DECL(2, const element_stats_type, gamma,     (element_stats_type*)handle->reg_gamma->data,   nFmBlock);
LIBXSMM_VLA_DECL(2, const element_stats_type, beta,      (element_stats_type*)handle->reg_beta->data,    nFmBlock);
LIBXSMM_VLA_DECL(2,       element_stats_type, bmean,     (element_stats_type*)handle->expvalue->data,    nG);
LIBXSMM_VLA_DECL(2,       element_stats_type, brstd,     (element_stats_type*)handle->rcpstddev->data,   nG);
LIBXSMM_VLA_DECL(2,       element_stats_type, variance,  (element_stats_type*)handle->variance->data,    nG);
LIBXSMM_VLA_DECL(3,       element_stats_type, sum_img,   (element_stats_type*)handle->scratch,                                                           nBlocksFm, nFmBlock);
LIBXSMM_VLA_DECL(3,       element_stats_type, sumsq_img, ((element_stats_type*)handle->scratch) + ((size_t)nImg * (size_t)nBlocksFm * (size_t)nFmBlock), nBlocksFm, nFmBlock);
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK)
LIBXSMM_VLA_DECL(5,       unsigned char,      relumask,  (unsigned char*)handle->relumask->data, nBlocksFm, ofhp, ofwp, nFmBlock);
#endif

#if defined(LIBXSMM_DNN_FUSEDGN_FWD_BF16)
union libxsmm_bfloat16_hp input_f32;
union libxsmm_bfloat16_hp output_f32;
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE)
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

for ( img = thr_begin; img < thr_end; ++img ) {
  element_stats_type* bmean_ptr = &LIBXSMM_VLA_ACCESS(2, bmean,    img, 0, nG);
  element_stats_type* brstd_ptr = &LIBXSMM_VLA_ACCESS(2, brstd,    img, 0, nG);
  element_stats_type* tvar_ptr  = &LIBXSMM_VLA_ACCESS(2, variance, img, 0, nG);
  element_stats_type* sum_img_ptr = NULL;
  element_stats_type* sumsq_img_ptr = NULL;

  /* create reduction over all pixels per channel */
  for ( fm = 0; fm < nBlocksFm; ++fm ) {
    /* @TODO check if we can bake this in into scratch */
    element_stats_type lcl_sum_ptr[64];
    element_stats_type lcl_sumsq_ptr[64];

    sum_img_ptr = &LIBXSMM_VLA_ACCESS(3, sum_img, img, fm, 0, nBlocksFm, nFmBlock);
    sumsq_img_ptr = &LIBXSMM_VLA_ACCESS(3, sumsq_img, img, fm, 0, nBlocksFm, nFmBlock);

    LIBXSMM_PRAGMA_SIMD
    for ( v=0; v < nFmBlock; v++ ) {
      lcl_sum_ptr[v] = (element_stats_type)0;
      lcl_sumsq_ptr[v] = (element_stats_type)0;
    }

    for ( hi=iph; hi < (ifh + iph); hi++ ) {
      for ( wi=ipw; wi < (ifw + ipw); wi++ ) {
        const element_input_type* input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, fm, hi, wi, 0, nBlocksFm, ifhp, ifwp, nFmBlock);

#if !defined(LIBXSMM_DNN_FUSEDGN_FWD_BF16)
        LIBXSMM_PRAGMA_SIMD
#endif
        for (v=0; v < nFmBlock; v++) {
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_BF16)
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

  /* new we compute mean, variance and rstd per channel group */
  sum_img_ptr = &LIBXSMM_VLA_ACCESS(3, sum_img, img, 0, 0, nImg, nFmBlock);
  sumsq_img_ptr = &LIBXSMM_VLA_ACCESS(3, sumsq_img, img, 0, 0, nImg, nFmBlock);
  for ( g = 0; g < nG; ++g ) {
    element_stats_type lcl_fm_sum = 0.0f;
    element_stats_type lcl_fm_sumsq = 0.0f;

    for ( fm = g*nFmG; fm < (g+1)*nFmG; ++fm ) {
      lcl_fm_sum   += sum_img_ptr[fm];
      lcl_fm_sumsq += sumsq_img_ptr[fm];
    }

    {
      const element_stats_type tbmean = (recp_ghw * lcl_fm_sum);
      const element_stats_type tbmeansq = tbmean * tbmean;
      const element_stats_type tsqbmean = recp_ghw * lcl_fm_sumsq;
      const element_stats_type tvar     = tsqbmean - tbmeansq;
      const element_stats_type tbrstd = (element_stats_type)(1.0/sqrt((double)tvar + sqrt_eps));
      bmean_ptr[g] = tbmean;
      brstd_ptr[g] = tbrstd;
      tvar_ptr[g] = tvar;
    }
  }

  /* let's scale the data */
  for ( fm = 0; fm < nBlocksFm; ++fm ) {
    for ( hi=iph, ho=oph; hi < (ifh+iph); hi+=sh, ho++ ) {
      for ( wi=ipw, wo=opw; wi < (ifw+ipw); wi+=sw, wo++ ) {
        const element_input_type*  input_ptr     = &LIBXSMM_VLA_ACCESS(5, input,     img, fm, hi, wi, 0, nBlocksFm, ifhp, ifwp, nFmBlock);
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE)
        const element_input_type*  input_add_ptr = &LIBXSMM_VLA_ACCESS(5, input_add, img, fm, hi, wi, 0, nBlocksFm, ifhp, ifwp, nFmBlock);
#endif
        const element_stats_type*  gamma_ptr     = &LIBXSMM_VLA_ACCESS(2, gamma,     fm, 0, nFmBlock);
        const element_stats_type*  beta_ptr      = &LIBXSMM_VLA_ACCESS(2, beta,      fm, 0, nFmBlock);
              element_output_type* output_ptr    = &LIBXSMM_VLA_ACCESS(5, output,    img, fm, ho, wo, 0, nBlocksFm, ofhp, ofwp, nFmBlock);
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK)
              unsigned char*       relumask_ptr  = &LIBXSMM_VLA_ACCESS(5, relumask,  img, fm, ho, wo, 0, nBlocksFm, ofhp, ofwp, nFmBlock);
#endif
        float o;

#if 0
#if !defined(LIBXSMM_DNN_FUSEDGN_FWD_BF16)
        LIBXSMM_PRAGMA_SIMD
#endif
#endif
        for (v = 0; v < nFmBlock; v++ ) {
          g = ((fm*nFmBlock)+v)/nFmG;
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_BF16)
          input_f32.i[1] = input_ptr[v];
          o = gamma_ptr[v]*(input_f32.f - bmean_ptr[g])*brstd_ptr[g] + beta_ptr[v];
#else
          /* BN + scale (gamma, beta) */
          o = gamma_ptr[v]*(input_ptr[v] - bmean_ptr[g])*brstd_ptr[g] + beta_ptr[v];
#endif
          /* Eltwise */
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE)
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_BF16)
          input_add_f32.i[1] = input_add_ptr[v];
          o += input_add_f32.f;
#else
          o += input_add_ptr[v];
#endif
#endif
          /* ReLU */
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU)
          o = ( o > 0.0f ) ? o : 0.0f;
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK)
          o = ( o > 0.0f ) ? o : 0.0f;
          relumask_ptr[v] = (unsigned char)(o > 0.0f ? 1 : 0);
#endif
#if defined(LIBXSMM_DNN_FUSEDGN_FWD_BF16)
          output_f32.f = o;
          output_ptr[v] = output_f32.i[1];
#else
          output_ptr[v] = o;
#endif
        }
      }
    }
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

