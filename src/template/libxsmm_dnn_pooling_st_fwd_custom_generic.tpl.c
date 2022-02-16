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
float *const lcl_buffer_ptr = (float*)handle->scratch + (size_t)ofh*ofw*nFmBlock*ltid;
LIBXSMM_VLA_DECL(3,                     float, lcl_output, lcl_buffer_ptr,                                                   ofw, nFmBlock);
#else
element_output_type *const lcl_buffer_ptr = (element_output_type*)handle->scratch + (size_t)ofh*ofw*nFmBlock*ltid;
LIBXSMM_VLA_DECL(3,       element_output_type, lcl_output, lcl_buffer_ptr,                                                   ofw, nFmBlock);
#endif
LIBXSMM_VLA_DECL(5, const element_input_type,       input, (element_input_type* )handle->reg_input->data,  nBlocksFm, ifhp, ifwp, nFmBlock);
LIBXSMM_VLA_DECL(5,       element_output_type,     output, (element_output_type*)handle->reg_output->data, nBlocksFm, ofhp, ofwp, nFmBlock);
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
LIBXSMM_VLA_DECL(5,       element_mask_type,         mask, (element_mask_type*  )handle->mask->data,       nBlocksFm,  ofh,  ofw, nFmBlock);
#endif

#if defined(LIBXSMM_DNN_POOLING_FWD_BF16)
union libxsmm_bfloat16_hp input_f32;
union libxsmm_bfloat16_hp output_f32;
input_f32.i[1]  = 0;
input_f32.i[0]  = 0;
output_f32.i[1] = 0;
output_f32.i[0] = 0;
#endif

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

for (imgfm = thr_begin; imgfm < thr_end; ++imgfm) {
  img = imgfm / nBlocksFm;
  fm = imgfm % nBlocksFm;

  LIBXSMM_PRAGMA_SIMD
  for ( v = 0; v < ofh*ofw*nFmBlock; v++ ) {
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
    lcl_buffer_ptr[v] = -FLT_MAX;
#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
#if defined(LIBXSMM_DNN_POOLING_FWD_BF16)
    lcl_buffer_ptr[v] = (float)0.0;
#else
    lcl_buffer_ptr[v] = (element_output_type)0.0;
#endif
#endif
  }

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
            const element_input_type*      input_ptr  = &LIBXSMM_VLA_ACCESS(5, input,      img, fm, hi+kh+iph, wi+kw+ipw, 0, nBlocksFm, ifhp, ifwp, nFmBlock);
#if defined(LIBXSMM_DNN_POOLING_FWD_BF16)
                  float*               lcl_output_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_output,             ho-oph,    wo-opw, 0,                   ofw, nFmBlock);
#else
                  element_output_type* lcl_output_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_output,             ho-oph,    wo-opw, 0,                   ofw, nFmBlock);
#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
            const int                             idx = (hi+kh)*ifw*nFmBlock + (wi+kw)*nFmBlock;
                  element_mask_type*         mask_ptr = &LIBXSMM_VLA_ACCESS(5, mask,       img, fm,    ho-oph,    wo-opw, 0, nBlocksFm,  ofh,  ofw, nFmBlock);
#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_BF16)
            for ( v = 0; v < nFmBlock; v++ ) {
              input_f32.i[1] = input_ptr[v];
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
              if ( input_f32.f > lcl_output_ptr[v] ) {
                lcl_output_ptr[v] =  input_f32.f;
                mask_ptr[v] = idx + v;
              }
#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
              lcl_output_ptr[v] += input_f32.f;
#endif
            }
#else
            LIBXSMM_PRAGMA_SIMD
            for ( v = 0; v < nFmBlock; v++ ) {
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
              if ( input_ptr[v] > lcl_output_ptr[v] ) {
                lcl_output_ptr[v] =  input_ptr[v];
                mask_ptr[v] = idx + v;
              }
#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
              lcl_output_ptr[v] += input_ptr[v];
#endif
            }
#endif
          }
        }
      }
    }
  }

  /* copy the local buffer into output activations */
  for ( ho = oph; ho < (ofh+oph); ho++ ) {
    for ( wo = opw; wo < (ofw+opw); wo++ ) {
      element_output_type*     output_ptr = &LIBXSMM_VLA_ACCESS(5, output,     img, fm,        ho,        wo, 0, nBlocksFm, ofhp, ofwp, nFmBlock);
#if defined(LIBXSMM_DNN_POOLING_FWD_BF16)
      float*               lcl_output_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_output,             ho-oph,    wo-opw, 0,                   ofw, nFmBlock);
#else
      element_output_type* lcl_output_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_output,             ho-oph,    wo-opw, 0,                   ofw, nFmBlock);
#endif

#if defined(LIBXSMM_DNN_POOLING_FWD_BF16)
      for ( v = 0; v < nFmBlock; v++ ) {
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
        output_f32.f = lcl_output_ptr[v];
#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
        output_f32.f = lcl_output_ptr[v] * recp_pool_size;
#endif
        output_ptr[v] = output_f32.i[1];
      }
#else
      LIBXSMM_PRAGMA_SIMD
      for ( v = 0; v < nFmBlock; v++ ) {
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
        output_ptr[v] = lcl_output_ptr[v];
#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
        output_ptr[v] = lcl_output_ptr[v] * recp_pool_size;
#endif
      }
#endif
    }
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

