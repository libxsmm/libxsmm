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

/* size variables, all const */
const int nImg = handle->desc.N;
const int fhi = handle->desc.H;
const int fwi = handle->desc.W;
const int sh = handle->desc.u;
const int sw = handle->desc.v;
const int fho = fhi/sh;
const int fwo = fwi/sw;
const int iph = handle->desc.pad_h_in;
const int ipw = handle->desc.pad_w_in;
const int oph = handle->desc.pad_h_out;
const int opw = handle->desc.pad_w_out;
const int fhpo = fho + 2*oph;
const int fwpo = fwo + 2*opw;
const int fhpi = fhi + 2*iph;
const int fwpi = fwi + 2*ipw;
/* here we assume that input and output blocking is similar */
const int nBlocksFm = handle->blocksifm;
const int nFmBlock = handle->fm_lp_block*handle->ifmblock;

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
const element_stats_type sqrt_eps = 1e-7;
const element_stats_type nhw = (element_stats_type)(nImg * fhi * fwi);
const element_stats_type recp_nhw = 1.0f/nhw;

/* loop variables */
int img = 0;
int fm = 0;
int imgfm = 0;
int h = 0;
int w = 0;
int v = 0;
int hp = 0;
int wp = 0;

LIBXSMM_VLA_DECL(5, const element_input_type, input,     (element_input_type* )handle->reg_input->data,  nBlocksFm, fhpi, fwpi, nFmBlock);
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_ENABLE_ELTWISE)
LIBXSMM_VLA_DECL(5, const element_input_type, input_add, (element_input_type* )handle->reg_add->data,    nBlocksFm, fhpi, fwpi, nFmBlock);
#endif
LIBXSMM_VLA_DECL(5, element_output_type,      output,    (element_output_type*)handle->reg_output->data, nBlocksFm, fhpo, fwpo, nFmBlock);
LIBXSMM_VLA_DECL(2, const element_stats_type, gamma, (element_stats_type*)handle->reg_gamma->data, nFmBlock);
LIBXSMM_VLA_DECL(2, const element_stats_type, beta,  (element_stats_type*)handle->reg_beta->data,  nFmBlock);
LIBXSMM_VLA_DECL(2,       element_stats_type, bmean, (element_stats_type*)handle->expvalue->data,  nFmBlock);
LIBXSMM_VLA_DECL(2,       element_stats_type, brstd, (element_stats_type*)handle->stddev->data,    nFmBlock);
LIBXSMM_VLA_DECL(3,       element_stats_type, sum_img,   (element_stats_type*)handle->scratch,                                            nImg, nFmBlock);
LIBXSMM_VLA_DECL(3,       element_stats_type, sumsq_img, ((element_stats_type*)handle->scratch) + ((size_t)nImg * nBlocksFm * nFmBlock),  nImg, nFmBlock);

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BN) > 0 ) {
  for (imgfm = thr_begin; imgfm < thr_end; ++imgfm) {
    /* @TODO check if we can bake this in into scratch */
    element_stats_type lcl_sum_ptr[64];
    element_stats_type lcl_sumsq_ptr[64];
    element_stats_type* sum_img_ptr;
    element_stats_type* sumsq_img_ptr;

    img = imgfm / nBlocksFm;
    fm = imgfm % nBlocksFm;
    sum_img_ptr = &LIBXSMM_VLA_ACCESS(3, sum_img, fm, img, 0, nImg, nFmBlock);
    sumsq_img_ptr  = &LIBXSMM_VLA_ACCESS(3, sumsq_img,  fm, img, 0, nImg, nFmBlock);

    LIBXSMM_PRAGMA_SIMD
    LIBXSMM_PRAGMA_VALIGNED
    for (v=0; v < nFmBlock; v++) {
      lcl_sum_ptr[v] = (element_stats_type)0;
      lcl_sumsq_ptr[v] = (element_stats_type)0;
    }

    for (h=iph; h < (fhi + iph); h++) {
      for (w=ipw; w < (fwi + ipw); w++) {
        const element_input_type* input_ptr  = &LIBXSMM_VLA_ACCESS(5, input, img, fm, h, w, 0, nBlocksFm, fhpi, fwpi, nFmBlock);

        LIBXSMM_PRAGMA_SIMD
        LIBXSMM_PRAGMA_VALIGNED
        for (v=0; v < nFmBlock; v++) {
          lcl_sum_ptr[v]   += input_ptr[v];
          lcl_sumsq_ptr[v] += (input_ptr[v] * input_ptr[v]);
        }
      }
    }

    LIBXSMM_PRAGMA_SIMD
    LIBXSMM_PRAGMA_VALIGNED
    for (v=0; v < nFmBlock; v++) {
      sum_img_ptr[v] = lcl_sum_ptr[v];
      sumsq_img_ptr[v]  = lcl_sumsq_ptr[v];
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);

  /* now we need to reduce the sum and sum^2, we use the final  */
  for ( fm = thr_begin2; fm < thr_end2; ++fm ) {
    /* @TODO check if we can bake this in into scratch */
    element_stats_type lcl_sum_ptr[64];
    element_stats_type lcl_sumsq_ptr[64];
    element_stats_type* bmean_ptr = &LIBXSMM_VLA_ACCESS(2, bmean, fm, 0, nFmBlock);
    element_stats_type* brstd_ptr = &LIBXSMM_VLA_ACCESS(2, brstd, fm, 0, nFmBlock);

    LIBXSMM_PRAGMA_SIMD
    LIBXSMM_PRAGMA_VALIGNED
    for (v=0; v < nFmBlock; v++) {
      lcl_sum_ptr[v]   = (element_stats_type)0;
      lcl_sumsq_ptr[v] = (element_stats_type)0;
    }

    for (img=0; img < nImg; img++ ) {
      element_stats_type* sum_img_ptr   = &LIBXSMM_VLA_ACCESS(3, sum_img,   fm, img, 0, nImg, nFmBlock);
      element_stats_type* sumsq_img_ptr = &LIBXSMM_VLA_ACCESS(3, sumsq_img, fm, img, 0, nImg, nFmBlock);

      LIBXSMM_PRAGMA_SIMD
      LIBXSMM_PRAGMA_VALIGNED
      for (v=0; v < nFmBlock; v++) {
        lcl_sum_ptr[v] = sum_img_ptr[v];
        lcl_sumsq_ptr[v] = sumsq_img_ptr[v];
      }
    }

    LIBXSMM_PRAGMA_SIMD
    LIBXSMM_PRAGMA_VALIGNED
    for (v=0; v < nFmBlock; v++) {
      const element_stats_type tbmean = (recp_nhw * lcl_sum_ptr[v]) ;
      const element_stats_type tbmeansq  = tbmean * tbmean;
      const element_stats_type tsqbmean = recp_nhw * lcl_sumsq_ptr[v];
      const element_stats_type tbrstd = 1.0/sqrt(tsqbmean - tbmeansq + sqrt_eps);
      bmean_ptr[v] += tbmean;
      brstd_ptr[v] += tbrstd;
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);
}

/* now we apply the actual forward batch norm */
for (imgfm = thr_begin; imgfm < thr_end; ++imgfm) {
  img = imgfm / nBlocksFm;
  fm = imgfm % nBlocksFm;
  for (h=iph, hp=oph; h < (fhi+iph); h+=sh, hp++) {
    for (w=ipw, wp=opw; w < (fwi+ipw); w+=sw, wp++) {
      const element_input_type*  input_ptr     = &LIBXSMM_VLA_ACCESS(5, input,     img, fm, h,  w,  0, nBlocksFm, fhpi, fwpi, nFmBlock);
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_ENABLE_ELTWISE)
      const element_input_type*  input_add_ptr = &LIBXSMM_VLA_ACCESS(5, input_add, img, fm, h,  w,  0, nBlocksFm, fhpi, fwpi, nFmBlock);
#endif
            element_output_type* output_ptr    = &LIBXSMM_VLA_ACCESS(5, output,    img, fm, hp, wp, 0, nBlocksFm, fhpo, fwpo, nFmBlock);
      const element_stats_type*  gamma_ptr     = &LIBXSMM_VLA_ACCESS(2, gamma,     fm, 0, nFmBlock);
      const element_stats_type*  beta_ptr      = &LIBXSMM_VLA_ACCESS(2, beta,      fm, 0, nFmBlock);
      const element_stats_type*  bmean_ptr     = &LIBXSMM_VLA_ACCESS(2, bmean,     fm, 0, nFmBlock);
      const element_stats_type*  brstd_ptr     = &LIBXSMM_VLA_ACCESS(2, brstd,     fm, 0, nFmBlock);

      LIBXSMM_PRAGMA_SIMD
      LIBXSMM_PRAGMA_VALIGNED
      LIBXSMM_PRAGMA_NONTEMPORAL
      for (v = 0; v < nFmBlock; v++ ) {
        /* BN + scale (gamma, beta) */
        float o = gamma_ptr[v]*(input_ptr[v] - bmean_ptr[v])*brstd_ptr[v] + beta_ptr[v];
        /* Eltwise */
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_ENABLE_ELTWISE)
        o += input_add_ptr[v];
#endif
        /* ReLU */
#if defined(LIBXSMM_DNN_FUSEDBN_FWD_ENABLE_RELU)
        o = ( o < 0.0f ) ? 0.0f : o;
#endif
        output_ptr[v] = o;
      }
    }
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

