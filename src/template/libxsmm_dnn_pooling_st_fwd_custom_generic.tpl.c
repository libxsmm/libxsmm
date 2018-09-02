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
element_output_type recp_pool_size = 1.0f/((element_output_type)handle->desc.R*(element_output_type)handle->desc.S);
#endif

/* multi-dim arrays declaration */
element_output_type* lcl_buffer_ptr = ((element_output_type*)handle->scratch)+(fho*fwo*nFmBlock*ltid);
LIBXSMM_VLA_DECL(5, const element_input_type,  input,      (element_input_type* )handle->reg_input->data,  nBlocksFm, fhpi, fwpi, nFmBlock);
LIBXSMM_VLA_DECL(5,       element_output_type, output,     (element_output_type*)handle->reg_output->data, nBlocksFm, fhpo, fwpo, nFmBlock);
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
LIBXSMM_VLA_DECL(5,       element_mask_type,   mask,       (element_mask_type*  )handle->mask->data,       nBlocksFm,  fho,  fwo, nFmBlock);
#endif
LIBXSMM_VLA_DECL(3,       element_output_type, lcl_output, lcl_buffer_ptr,                                                   fwo, nFmBlock);

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

for (imgfm = thr_begin; imgfm < thr_end; ++imgfm) {
  img = imgfm / nBlocksFm;
  fm = imgfm % nBlocksFm;

  LIBXSMM_PRAGMA_SIMD
  LIBXSMM_PRAGMA_VALIGNED
  for( v = 0; v < fho*fwo*nFmBlock; v++ ) {
    lcl_buffer_ptr[v] = -FLT_MAX;
  }

  for( ho = oph; ho < (fho+oph); ho++ ) {
    hi = ((ho-oph) * sh) - handle->desc.pad_h;
    for( wo = opw; wo < (fwo+opw); wo++ ) {
      wi = ((wo-opw) * sw) - handle->desc.pad_w;
      for( kh = 0; kh < handle->desc.R; kh++ ) {
        if(hi+kh < 0 || hi+kh >= fhi) continue;
        for( kw = 0; kw < handle->desc.S; kw++ ) {
          if(wi+kw < 0 || wi+kw >= fwi) continue;
            const int                           index = (hi+kh)*fwi*nFmBlock + (wi+kw)*nFmBlock;
            const element_input_type*      input_ptr  = &LIBXSMM_VLA_ACCESS(5, input,      img, fm, hi+kh+iph, wi+kw+ipw, 0, nBlocksFm, fhpi, fwpi, nFmBlock);
                  element_output_type* lcl_output_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_output,             ho-oph,    wo-opw, 0,                   fwo, nFmBlock);
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
                  element_mask_type*       mask_ptr   = &LIBXSMM_VLA_ACCESS(5, mask,       img, fm,    ho-oph,    wo-opw, 0, nBlocksFm,  fho,  fwo, nFmBlock);
#endif

          LIBXSMM_PRAGMA_SIMD
          LIBXSMM_PRAGMA_VALIGNED
          for( v = 0; v < nFmBlock; v++ ) {
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
            lcl_output_ptr[v] = (input_ptr[v] > lcl_output_ptr[v]) ? input_ptr[v] : lcl_output_ptr[v];
            mask_ptr[v] = (input_ptr[v] > lcl_output_ptr[v]) ? index + v : mask_ptr[v];
#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
            lcl_output_ptr[v] += input_ptr[v];
#endif
          }
        }
      }
    }
  }

  /* copy the local buffer into output activations */
  for( ho = oph; ho < (fho+oph); ho++ ) {
    for( wo = opw; wo < (fwo+opw); wo++ ) {
      element_output_type*     output_ptr = &LIBXSMM_VLA_ACCESS(5, output,     img, fm,        ho,        wo, 0, nBlocksFm, fhpo, fwpo, nFmBlock);
      element_output_type* lcl_output_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_output,             ho-oph,    wo-opw, 0,                   fwo, nFmBlock);

      LIBXSMM_PRAGMA_SIMD
      LIBXSMM_PRAGMA_VALIGNED
      LIBXSMM_PRAGMA_NONTEMPORAL
      for( v = 0; v < nFmBlock; v++ ) {
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
        output_ptr[v] = lcl_output_ptr[v];
#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
        output_ptr[v] = lcl_output_ptr[v] * recp_pool_size;
#endif
      }
    }
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

