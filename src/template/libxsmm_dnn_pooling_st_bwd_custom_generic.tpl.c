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
const int ifh = handle->desc.H;
const int ifw = handle->desc.W;
#if defined(LIBXSMM_DNN_POOLING_BWD_AVG)
const int sh = handle->desc.u;
const int sw = handle->desc.v;
#endif
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

/* loop variables */
int img = 0;
int fm = 0;
int imgfm = 0;
int ho = 0;
int wo = 0;
int hi = 0;
int wi = 0;
int v = 0;
#if defined(LIBXSMM_DNN_POOLING_BWD_AVG)
int kh = 0;
int kw = 0;
#if defined(LIBXSMM_DNN_POOLING_BWD_BF16)
float recp_pool_size = 1.0f/((float)handle->desc.R*(float)handle->desc.S);
#else
element_input_type recp_pool_size = 1.0f/((element_input_type)handle->desc.R*(element_input_type)handle->desc.S);
#endif
#endif

/* multi-dim arrays declaration */
#if defined(LIBXSMM_DNN_POOLING_BWD_BF16)
float *const lcl_buffer_ptr = (float*)handle->scratch + (size_t)ifh*ifw*nFmBlock*ltid;
LIBXSMM_VLA_DECL(3,                    float, lcl_dinput, lcl_buffer_ptr,                                                    ifw, nFmBlock);
#else
element_output_type *const lcl_buffer_ptr = (element_input_type*)handle->scratch + (size_t)ifh*ifw*nFmBlock*ltid;
LIBXSMM_VLA_DECL(3,       element_input_type, lcl_dinput, lcl_buffer_ptr,                                                    ifw, nFmBlock);
#endif
LIBXSMM_VLA_DECL(5,       element_input_type,     dinput, (element_input_type* )handle->grad_input->data,  nBlocksFm, ifhp, ifwp, nFmBlock);
LIBXSMM_VLA_DECL(5, const element_output_type,   doutput, (element_output_type*)handle->grad_output->data, nBlocksFm, ofhp, ofwp, nFmBlock);
#if defined(LIBXSMM_DNN_POOLING_BWD_MAX)
LIBXSMM_VLA_DECL(5, const element_mask_type,        mask, (element_mask_type*  )handle->mask->data,        nBlocksFm,  ofh,  ofw, nFmBlock);
#endif

#if defined(LIBXSMM_DNN_POOLING_BWD_BF16)
union libxsmm_bfloat16_hp del_input_f32;
union libxsmm_bfloat16_hp del_output_f32;
del_input_f32.i[1]  = 0;
del_input_f32.i[0]  = 0;
del_output_f32.i[1] = 0;
del_output_f32.i[0] = 0;
#endif

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

for (imgfm = thr_begin; imgfm < thr_end; ++imgfm) {
  img = imgfm / nBlocksFm;
  fm = imgfm % nBlocksFm;

  LIBXSMM_PRAGMA_SIMD
  for ( v = 0; v < ifh*ifw*nFmBlock; v++ ) {
#if defined(LIBXSMM_DNN_POOLING_BWD_BF16)
    lcl_buffer_ptr[v] = (float)0;
#else
    lcl_buffer_ptr[v] = (element_input_type)0;
#endif
  }

#if defined(LIBXSMM_DNN_POOLING_BWD_MAX)
  for ( ho = oph; ho < (ofh+oph); ho++ ) {
    for ( wo = opw; wo < (ofw+opw); wo++ ) {
      const element_output_type* doutput_ptr = &LIBXSMM_VLA_ACCESS(5, doutput, img, fm,     ho,     wo, 0, nBlocksFm, ofhp, ofwp, nFmBlock);
      const element_mask_type*      mask_ptr = &LIBXSMM_VLA_ACCESS(5, mask,    img, fm, ho-oph, wo-opw, 0, nBlocksFm,  ofh,  ofw, nFmBlock);

#if !defined(LIBXSMM_DNN_POOLING_BWD_BF16)
      LIBXSMM_PRAGMA_SIMD
#endif
      for ( v = 0; v < nFmBlock; v++ ) {
#if defined(LIBXSMM_DNN_POOLING_BWD_BF16)
        del_output_f32.i[1] = doutput_ptr[v];
        lcl_buffer_ptr[mask_ptr[v]] += del_output_f32.f;
#else
        lcl_buffer_ptr[mask_ptr[v]] += doutput_ptr[v];
#endif
      }
    }
  }
#endif
#if defined(LIBXSMM_DNN_POOLING_BWD_AVG)
  for ( ho = oph; ho < (ofh+oph); ho++ ) {
    hi = ((ho-oph) * sh) - handle->desc.pad_h;
    for ( wo = opw; wo < (ofw+opw); wo++ ) {
      wi = ((wo-opw) * sw) - handle->desc.pad_w;
      for ( kh = 0; kh < handle->desc.R; kh++ ) {
        if (hi+kh < 0 || hi+kh >= ifh) continue;
        for ( kw = 0; kw < handle->desc.S; kw++ ) {
          if (wi+kw < 0 || wi+kw >= ifw) {
            continue;
          } else {
            const element_output_type*   doutput_ptr = &LIBXSMM_VLA_ACCESS(5, doutput,    img, fm,    ho,    wo, 0, nBlocksFm, ofhp, ofwp, nFmBlock);
#if defined(LIBXSMM_DNN_POOLING_BWD_BF16)
                  float*              lcl_dinput_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_dinput,          hi+kh, wi+kw, 0,                   ifw, nFmBlock);
#else
                  element_input_type* lcl_dinput_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_dinput,          hi+kh, wi+kw, 0,                   ifw, nFmBlock);
#endif

#if !defined(LIBXSMM_DNN_POOLING_BWD_BF16)
            LIBXSMM_PRAGMA_SIMD
#endif
            for ( v = 0; v < nFmBlock; v++ ) {
#if defined(LIBXSMM_DNN_POOLING_BWD_BF16)
              del_output_f32.i[1] = doutput_ptr[v];
              lcl_dinput_ptr[v] += (del_output_f32.f * recp_pool_size);
#else
              lcl_dinput_ptr[v] += (doutput_ptr[v] * recp_pool_size);
#endif
            }
          }
        }
      }
    }
  }
#endif

  /* copy the local buffer into dinput activations */
  for ( hi = iph; hi < (ifh+iph); hi++ ) {
    for ( wi = ipw; wi < (ifw+ipw); wi++ ) {
      element_input_type*     dinput_ptr = &LIBXSMM_VLA_ACCESS(5, dinput,     img, fm,        hi,        wi, 0, nBlocksFm, ifhp, ifwp, nFmBlock);
#if defined(LIBXSMM_DNN_POOLING_BWD_BF16)
      float*              lcl_dinput_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_dinput,             hi-iph,    wi-ipw, 0,                   ifw, nFmBlock);
#else
      element_input_type* lcl_dinput_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_dinput,             hi-iph,    wi-ipw, 0,                   ifw, nFmBlock);
#endif

#if !defined(LIBXSMM_DNN_POOLING_BWD_BF16)
      LIBXSMM_PRAGMA_SIMD
#endif
      for ( v = 0; v < nFmBlock; v++ ) {
#if defined(LIBXSMM_DNN_POOLING_BWD_BF16)
        del_input_f32.f = lcl_dinput_ptr[v];
        dinput_ptr[v] = del_input_f32.i[1];
#else
        dinput_ptr[v] = lcl_dinput_ptr[v];
#endif
      }
    }
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

