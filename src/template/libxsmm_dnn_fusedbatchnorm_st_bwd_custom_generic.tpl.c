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

const element_stats_type nhw = (element_stats_type)(handle->desc.fullN * ifh * ifw);
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
int v = 0;
int ho = 0;
int wo = 0;

LIBXSMM_VLA_DECL(5,       element_input_type,  dinput,     (element_input_type* )handle->grad_input->data,  nBlocksFm, ifhp, ifwp, nFmBlock);
LIBXSMM_VLA_DECL(5,       element_input_type,   input,     (element_input_type* )handle->reg_input->data,   nBlocksFm, ifhp, ifwp, nFmBlock);
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_ELTWISE)
LIBXSMM_VLA_DECL(5,       element_input_type,  dinput_add, (element_input_type* )handle->grad_add->data,    nBlocksFm, ifhp, ifwp, nFmBlock);
#endif
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_RELU)
LIBXSMM_VLA_DECL(5, const element_output_type, output,     (element_output_type*)handle->reg_output->data,  nBlocksFm, ofhp, ofwp, nFmBlock);
#endif
LIBXSMM_VLA_DECL(5,       element_output_type, doutput,    (element_output_type*)handle->grad_output->data, nBlocksFm, ofhp, ofwp, nFmBlock);

LIBXSMM_VLA_DECL(2, const element_stats_type,  gamma,      (element_stats_type*)handle->reg_gamma->data,  nFmBlock);
LIBXSMM_VLA_DECL(2,       element_stats_type,  dgamma,     (element_stats_type*)handle->grad_gamma->data, nFmBlock);
LIBXSMM_VLA_DECL(2,       element_stats_type,  dbeta,      (element_stats_type*)handle->grad_beta->data,  nFmBlock);
LIBXSMM_VLA_DECL(2, const element_stats_type,  bmean,      (element_stats_type*)handle->expvalue->data,   nFmBlock);
LIBXSMM_VLA_DECL(2, const element_stats_type,  brstd,      (element_stats_type*)handle->rcpstddev->data,  nFmBlock);
LIBXSMM_VLA_DECL(3,       element_stats_type,  dgamma_img, (element_stats_type*)handle->scratch,                                                          nImg, nFmBlock);
LIBXSMM_VLA_DECL(3,       element_stats_type,  dbeta_img, ((element_stats_type*)handle->scratch) + ((size_t)nImg * (size_t)nBlocksFm * (size_t)nFmBlock), nImg, nFmBlock);
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK)
LIBXSMM_VLA_DECL(5,       unsigned char,       relumask,   (unsigned char*)handle->relumask->data, nBlocksFm, ofhp, ofwp, nFmBlock);
#endif

#if defined(LIBXSMM_DNN_FUSEDBN_BWD_BF16)
union libxsmm_bfloat16_hp input_f32;
union libxsmm_bfloat16_hp del_input_f32;
union libxsmm_bfloat16_hp del_output_f32;
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_RELU)
union libxsmm_bfloat16_hp output_f32;
output_f32.i[1] = 0;
output_f32.i[0] = 0;
#endif
input_f32.i[1]  = 0;
input_f32.i[0]  = 0;
del_output_f32.i[1] = 0;
del_output_f32.i[0] = 0;
del_input_f32.i[1] = 0;
del_input_f32.i[0] = 0;
#endif

assert( nFmBlock <= 64 );

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

if ( ((handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BN) > 0)            ||
     ((handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS) > 0)       ||
     ((handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED) > 0)    ) {
  for ( imgfm = thr_begin; imgfm < thr_end; ++imgfm ) {
    /* @TODO check if we can bake this in into scratch */
    element_stats_type lcl_gamma_ptr[64];
    element_stats_type lcl_beta_ptr[64];
    element_stats_type* del_gamma_img_ptr;
    element_stats_type* del_beta_img_ptr;

    img = imgfm / nBlocksFm;
    fm = imgfm % nBlocksFm;
    del_gamma_img_ptr = &LIBXSMM_VLA_ACCESS(3, dgamma_img, fm, img, 0, nImg, nFmBlock);
    del_beta_img_ptr  = &LIBXSMM_VLA_ACCESS(3, dbeta_img,  fm, img, 0, nImg, nFmBlock);

    LIBXSMM_PRAGMA_SIMD
    for ( v=0; v < nFmBlock; v++ ) {
      lcl_gamma_ptr[v] = 0.0f;
      lcl_beta_ptr[v] = 0.0f;
    }

    for ( hi=iph, ho=oph; hi < (ifh + iph); hi+=sh, ho++ ) {
      for ( wi=ipw, wo=opw; wi < (ifw + ipw); wi+=sw, wo++ ) {
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_ELTWISE)
              element_input_type*  del_input_add_ptr = &LIBXSMM_VLA_ACCESS(5, dinput_add, img, fm, hi, wi, 0, nBlocksFm, ifhp, ifwp, nFmBlock);
#endif
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_RELU)
        const element_output_type* output_ptr        = &LIBXSMM_VLA_ACCESS(5,     output, img, fm, ho, wo, 0, nBlocksFm, ofhp, ofwp, nFmBlock);
#endif
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK)
        const unsigned char*       relumask_ptr      = &LIBXSMM_VLA_ACCESS(5,   relumask, img, fm, ho, wo, 0, nBlocksFm, ofhp, ofwp, nFmBlock);
#endif
        const element_input_type*  input_ptr         = &LIBXSMM_VLA_ACCESS(5,      input, img, fm, hi, wi, 0, nBlocksFm, ifhp, ifwp, nFmBlock);
              element_output_type* del_output_ptr    = &LIBXSMM_VLA_ACCESS(5,    doutput, img, fm, ho, wo, 0, nBlocksFm, ofhp, ofwp, nFmBlock);
        const element_stats_type*  bmean_ptr         = &LIBXSMM_VLA_ACCESS(2, bmean,     fm, 0, nFmBlock);
        const element_stats_type*  brstd_ptr         = &LIBXSMM_VLA_ACCESS(2, brstd,     fm, 0, nFmBlock);

#if !defined(LIBXSMM_DNN_FUSEDBN_BWD_BF16)
        LIBXSMM_PRAGMA_SIMD
#endif
        for ( v=0; v < nFmBlock; v++ ) {
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_BF16)
          del_output_f32.i[1] = del_output_ptr[v];
          del_output_f32.i[0] = 0;
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_RELU)
          output_f32.i[1] = output_ptr[v];
          del_output_f32.f = LIBXSMM_FEQ(output_f32.f, 0) ? 0 : del_output_f32.f;
          del_output_ptr[v] = del_output_f32.i[1];
#endif
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK)
          del_output_ptr[v] = (element_output_type)(relumask_ptr[v] == 1 ? del_output_ptr[v] : 0);
#endif
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_ELTWISE)
          del_input_add_ptr[v] = del_output_ptr[v];
#endif
          input_f32.i[1] = input_ptr[v];
          lcl_gamma_ptr[v] += (input_f32.f - bmean_ptr[v]) * del_output_f32.f * brstd_ptr[v];
          lcl_beta_ptr[v]  += del_output_f32.f;
#else
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_RELU)
          del_output_ptr[v] = LIBXSMM_FEQ(output_ptr[v], 0) ? 0 : del_output_ptr[v];
#endif
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK)
          del_output_ptr[v] = (element_output_type)(relumask_ptr[v] == 1 ? del_output_ptr[v] : 0);
#endif
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_ENABLE_ELTWISE)
          del_input_add_ptr[v] = del_output_ptr[v];
#endif
          lcl_gamma_ptr[v] += (input_ptr[v] - bmean_ptr[v]) * del_output_ptr[v] * brstd_ptr[v];
          lcl_beta_ptr[v]  += del_output_ptr[v];
#endif
        }
      }
    }

    LIBXSMM_PRAGMA_SIMD
    for ( v=0; v < nFmBlock; v++ ) {
      del_gamma_img_ptr[v] = lcl_gamma_ptr[v];
      del_beta_img_ptr[v]  = lcl_beta_ptr[v];
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);

  if ( ((handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BN) > 0)      ||
       ((handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS) > 0)    ) {
    /* now we need to reduce the del_gamm and del_beta */
    for ( fm = thr_begin2; fm < thr_end2; ++fm ) {
      element_stats_type* del_gamma_ptr = &LIBXSMM_VLA_ACCESS(2, dgamma, fm, 0, nFmBlock);
      element_stats_type* del_beta_ptr  = &LIBXSMM_VLA_ACCESS(2, dbeta,  fm, 0, nFmBlock);

      LIBXSMM_PRAGMA_SIMD
      for ( v=0; v < nFmBlock; v++ ) {
        del_gamma_ptr[v] = (element_stats_type)0;
        del_beta_ptr[v]  = (element_stats_type)0;
      }

      for ( img=0; img < nImg; img++ ) {
        element_stats_type* del_gamma_img_ptr = &LIBXSMM_VLA_ACCESS(3, dgamma_img, fm, img, 0, nImg, nFmBlock);
        element_stats_type* del_beta_img_ptr  = &LIBXSMM_VLA_ACCESS(3, dbeta_img,  fm, img, 0, nImg, nFmBlock);

        LIBXSMM_PRAGMA_SIMD
        for ( v=0; v < nFmBlock; v++ ) {
          del_gamma_ptr[v] += del_gamma_img_ptr[v];
          del_beta_ptr[v]  += del_beta_img_ptr[v];
        }
      }
    }
  } else {
    /* now we need to reduce the del_gamm and del_beta */
    for ( fm = thr_begin2; fm < thr_end2; ++fm ) {
      element_stats_type* del_gamma_ptr = &LIBXSMM_VLA_ACCESS(3, dgamma_img, fm, 0, 0, nImg, nFmBlock);
      element_stats_type* del_beta_ptr  = &LIBXSMM_VLA_ACCESS(3, dbeta_img,  fm, 0, 0, nImg, nFmBlock);

      for ( img=1; img < nImg; img++ ) {
        element_stats_type* del_gamma_img_ptr = &LIBXSMM_VLA_ACCESS(3, dgamma_img, fm, img, 0, nImg, nFmBlock);
        element_stats_type* del_beta_img_ptr  = &LIBXSMM_VLA_ACCESS(3, dbeta_img,  fm, img, 0, nImg, nFmBlock);

        LIBXSMM_PRAGMA_SIMD
        for ( v=0; v < nFmBlock; v++ ) {
          del_gamma_ptr[v] += del_gamma_img_ptr[v];
          del_beta_ptr[v]  += del_beta_img_ptr[v];
        }
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);
}

if ( ((handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BN) > 0)      ||
     ((handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE) > 0)    ) {
  /* now we apply the actual backward batch norm */
  for ( imgfm = thr_begin; imgfm < thr_end; ++imgfm ) {
    img = imgfm / nBlocksFm;
    fm = imgfm % nBlocksFm;
    for ( hi=iph, ho=oph; hi < (ifh + iph); hi+=sh, ho++ ) {
      for ( wi=ipw, wo=opw; wi < (ifw + ipw); wi+=sw, wo++ ) {
        element_input_type*  del_input_ptr     = &LIBXSMM_VLA_ACCESS(5,     dinput, img, fm, hi, wi, 0, nBlocksFm, ifhp, ifwp, nFmBlock);
        const element_input_type*  input_ptr         = &LIBXSMM_VLA_ACCESS(5,      input, img, fm, hi, wi, 0, nBlocksFm, ifhp, ifwp, nFmBlock);
        const element_output_type* del_output_ptr    = &LIBXSMM_VLA_ACCESS(5,    doutput, img, fm, ho, wo, 0, nBlocksFm, ofhp, ofwp, nFmBlock);
        const element_stats_type*  bmean_ptr         = &LIBXSMM_VLA_ACCESS(2, bmean,     fm, 0, nFmBlock);
        const element_stats_type*  brstd_ptr         = &LIBXSMM_VLA_ACCESS(2, brstd,     fm, 0, nFmBlock);
        const element_stats_type*  gamma_ptr         = &LIBXSMM_VLA_ACCESS(2, gamma,     fm, 0, nFmBlock);
        const element_stats_type*  del_gamma_ptr     = &LIBXSMM_VLA_ACCESS(2, dgamma,    fm, 0, nFmBlock);
        const element_stats_type*  del_beta_ptr      = &LIBXSMM_VLA_ACCESS(2, dbeta,     fm, 0, nFmBlock);

#if !defined(LIBXSMM_DNN_FUSEDBN_BWD_BF16)
        LIBXSMM_PRAGMA_SIMD
#endif
        for ( v=0; v < nFmBlock; v++ ) {
#if defined(LIBXSMM_DNN_FUSEDBN_BWD_BF16)
          del_output_f32.i[1] = del_output_ptr[v];
          input_f32.i[1] = input_ptr[v];
          del_input_f32.f = gamma_ptr[v] * brstd_ptr[v] * recp_nhw * (nhw*del_output_f32.f -
                   (del_beta_ptr[v] + (input_f32.f - bmean_ptr[v]) * del_gamma_ptr[v] * brstd_ptr[v]));
          del_input_ptr[v] = del_input_f32.i[1];
#else
          del_input_ptr[v] = gamma_ptr[v] * brstd_ptr[v] * recp_nhw * (nhw*del_output_ptr[v] -
                   (del_beta_ptr[v] + (input_ptr[v] - bmean_ptr[v]) * del_gamma_ptr[v] * brstd_ptr[v]));
#endif
        }
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);
}

