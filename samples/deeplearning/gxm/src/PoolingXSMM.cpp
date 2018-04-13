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
/* Sasikanth Avancha, Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/


#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "PoolingXSMM.hpp"

#define VLEN 16

void PoolXSMM::forwardPropagate(TensorBuf *inpb, TensorBuf *outpb, int *maskp, int tid)
{
  float *inp = (float*)inpb->getBuffer();
  float *outp = (float*)outpb->getBuffer();
  float *outpp = (float*)outpb->getPrivBuffer();

  int nImg = gp->batch_size;
  int nIfm = gp->nInput;
  int nBIfm = gp->nInput/VLEN;
  int nOfm = gp->nOutput;
  int nBOfm = gp->nOutput/VLEN;
  int ifh  = gp->iHeight;
  int ifw  = gp->iWidth;
  int ofh  = gp->oHeight;
  int ofw  = gp->oWidth;
  int l_pad = gp->pad_w;
  int t_pad = gp->pad_h;
  int kh    = gp->kh;
  int kw    = gp->kw;
  int stride_h  = gp->stride_h;
  int stride_w  = gp->stride_w;
  int pool = gp->pool_mode;
  int threads = gp->num_threads;

  int lp_ipad = gp->ipad_w;
  int tp_ipad = gp->ipad_h;
  int tp_opad = gp->opad_h;
  int lp_opad = gp->opad_w;

  bool needs_conversion = false;

  if(t_pad || l_pad)
  {
    if ((ofh - 1) * stride_h >= ifh + t_pad) ofh--;
    if ((ofw - 1) * stride_w >= ifw + l_pad) ofw--;
  }

  __assume_aligned(inp,64);
  __assume_aligned(outp,64);
  __assume_aligned(maskp,64);

  float (* __restrict input )[nBOfm][ifh][ifw][VLEN] = (float (*)[*][*][*][VLEN])(inp + (tp_ipad * ifw + lp_ipad) * VLEN);
  float (* __restrict output)[nBOfm][ofh][ofw][VLEN] = (float (*)[*][*][*][VLEN])(outp + (tp_opad *ofw + lp_opad) * VLEN);

  switch(pool)
  {
    case MAX:
      {
        int (* __restrict mask)[nBOfm][ofh][ofw][VLEN] = (int (*)[*][*][*][VLEN])maskp;

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for(int img = 0; img < nImg; img++) {
          for(int ofm1=0; ofm1 < nBOfm; ofm1++) {
            float lcl_buffer[ofh*ofw*VLEN] __attribute__((aligned(64)));
            float (* __restrict lcl_output)[ofw][VLEN] = (float (*)[*][VLEN])lcl_buffer;
#pragma simd
#pragma vector aligned
            for(int i=0; i<ofh*ofw*VLEN; i++) {
              lcl_buffer[i] = -FLT_MAX;
            }
            for(int oj=0; oj < ofh; oj++) {
              int ij = oj * stride_h - t_pad;
              for(int oi=0; oi < ofw; oi++) {
                int ii = oi * stride_w - l_pad;

                for(int kj = 0; kj < kh; kj++) {
                  if(ij+kj < 0 || ij+kj >= ifh) continue;
                  for(int ki = 0; ki < kw; ki++) {
                    if(ii+ki < 0 || ii+ki >= ifw) continue;

                    int index = img*nBOfm*ifh*ifw*VLEN + ofm1*ifh*ifw*VLEN + (ij+kj)*ifw*VLEN + (ii + ki)*VLEN;
#pragma simd
                    for(int ofm2=0; ofm2 < VLEN; ofm2++) {
                      if(input[img][ofm1][ij+kj][ii+ki][ofm2] > lcl_output[oj][oi][ofm2])
                      {
                        lcl_output[oj][oi][ofm2] = input[img][ofm1][ij+kj][ii+ki][ofm2];
                        mask[img][ofm1][oj][oi][ofm2] = index + ofm2;
                      }
                    }
                  }
                }
              }
            }
            for(int oj=0; oj < ofh; oj++) {
              for(int oi=0; oi < ofw; oi++) {
#pragma simd
#pragma vector aligned
#pragma vector nontemporal
                for(int ofm2=0; ofm2 < VLEN; ofm2++) {
                  output[img][ofm1][oj][oi][ofm2] = lcl_output[oj][oi][ofm2];
                }
              }
            }
          }
        }
      }
      break;


    case AVE:
      {
        float recp_pool_size = 1.0/(kh*kw);
        float val;

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for(int img = 0; img < nImg; img++) {
          for(int ofm1 = 0; ofm1 < nBOfm; ofm1++) {
            float lcl_buffer[ofh*ofw*VLEN] __attribute__((aligned(64)));
            float (* __restrict lcl_output)[ofw][VLEN] = (float (*)[*][VLEN])lcl_buffer;
#pragma simd
#pragma vector aligned
            for(int i=0; i<ofh*ofw*VLEN; i++) {
              lcl_buffer[i] = 0.0f;
            }
            for(int oj=0; oj < ofh; oj++) {
              int ij = oj * stride_h - t_pad;
              for(int oi=0; oi < ofw; oi++) {
                int ii = oi * stride_w - l_pad;

                for(int kj = 0; kj < kh; kj++) {
                  if(ij+kj < 0 || ij+kj >= ifh) continue;
                  for(int ki = 0; ki < kw; ki++) {
                    if(ii+ki < 0 || ii+ki >= ifw) continue;
#pragma simd
                    for(int ofm2=0; ofm2 < VLEN; ofm2++)
                      lcl_output[oj][oi][ofm2] += input[img][ofm1][ij+kj][ii+ki][ofm2];
                  }
                }
              }
            }

            for(int oj=0; oj < ofh; oj++) {
              for(int oi=0; oi < ofw; oi++) {
#pragma simd
#pragma vector aligned
#pragma vector nontemporal
                for(int ofm2=0; ofm2 < VLEN; ofm2++) {
                  output[img][ofm1][oj][oi][ofm2] = lcl_output[oj][oi][ofm2] * recp_pool_size;
                }
              }
            }
          }
        }
      }
      break;
  }

  if(gp->data_type == DT_DFP16)
  {
    short* i16_outp = (short*)outpb->getLPBuffer();
    unsigned char scf_output;
    libxsmm_dnn_quantize_act((float*)outpb->getBuffer(), i16_outp, nImg, nOfm, ofh, ofw, 16, 8, 2, 2, &scf_output, LIBXSMM_DNN_QUANT_FPHW_ROUND);
    outpb->setLPSF(scf_output);
  }
}

void PoolXSMM::backPropagate(TensorBuf *deloutpb, int *maskp, TensorBuf *delinpb, int tid)
{
  float *deloutp = (float*)deloutpb->getBuffer();
  float *delinp = (float*)delinpb->getBuffer();
  float *deloutpp = (float*)deloutpb->getPrivBuffer();

  int nImg  = gp->batch_size;
  int nIfm = gp->nInput;
  int nOfm = gp->nOutput;
  int nBIfm = nIfm/VLEN;
  int nBOfm = nOfm/VLEN;
  int ifh  = gp->iHeight;
  int ifw  = gp->iWidth;
  int ofh  = gp->oHeight;
  int ofw  = gp->oWidth;
  int l_pad = gp->pad_w;
  int t_pad = gp->pad_h;
  int kh    = gp->kh;
  int kw    = gp->kw;
  int stride_h  = gp->stride_h;
  int stride_w  = gp->stride_w;
  int pool = gp->pool_mode;
  int threads = gp->num_threads;

  int lp_ipad = gp->ipad_w;
  int tp_ipad = gp->ipad_h;
  int tp_opad = gp->opad_h;
  int lp_opad = gp->opad_w;

  bool needs_conversion = false;

  if(t_pad || l_pad)
  {
    if ((ofh - 1) * stride_h >= ifh + t_pad) ofh--;
    if ((ofw - 1) * stride_w >= ifw + l_pad) ofw--;
  }

  __assume_aligned(delinp,64);
  __assume_aligned(deloutp,64);
  __assume_aligned(maskp,64);

  switch(pool)
  {
    case MAX:
      {
        float (* __restrict del_output)[nBOfm][ofh][ofw][VLEN] = (float (*)[*][*][*][VLEN])(deloutp + (tp_opad *ofw + lp_opad) * VLEN);
        int (* __restrict mask)[nBOfm][ofh][ofw][VLEN] = (int (*)[*][*][*][VLEN])maskp;

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for(int img = 0; img < nImg; img++) {
          for(int ifm1=0; ifm1 < nBIfm; ifm1++) {
            int idx_mask_offset = img*nBOfm*ifh*ifw*VLEN + ifm1*ifh*ifw*VLEN;
            int idx_input_offset = img*nBIfm*ifh*ifw*VLEN + ifm1*ifh*ifw*VLEN;
            float lcl_buffer[ifh*ifw*VLEN] __attribute__((aligned(64)));
#pragma simd
#pragma vector aligned
            for(int i=0; i<ifh*ifw*VLEN; i++) {
              lcl_buffer[i] = 0.0f;
            }
            for(int oj=0; oj < ofh; oj++) {
              for(int oi=0; oi < ofw; oi++) {
#pragma simd
                for(int ifm2 = 0; ifm2 <VLEN; ifm2++) {
                  /*delinp[mask[img][ifm1][oj][oi][ifm2]] += del_output[img][ifm1][oj][oi][ifm2];*/
                  lcl_buffer[(mask[img][ifm1][oj][oi][ifm2])-idx_mask_offset] += del_output[img][ifm1][oj][oi][ifm2];
                }
              }
            }
#pragma simd
#pragma vector aligned
#pragma vector nontemporal
            for(int i=0; i<ifh*ifw*VLEN; i++) {
              delinp[idx_input_offset + i] = lcl_buffer[i];
            }
          }
        }
      }
      break;

    case AVE:
      {
        float (* __restrict del_input )[nBIfm][ifh][ifw][VLEN] = (float (*)[*][*][*][VLEN])(delinp + (tp_ipad * ifw + lp_ipad) * VLEN);
        float (* __restrict del_output)[nBOfm][ofh][ofw][VLEN] = (float (*)[*][*][*][VLEN])(deloutp + (tp_opad *ofw + lp_opad) * VLEN);

        float recp_pool_size = 1.0/(kh*kw);

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for(int img = 0; img < nImg; img++) {
          for(int ifm1 = 0; ifm1 < nBIfm; ifm1++) {
            float lcl_buffer[ifh*ifw*VLEN] __attribute__((aligned(64)));
            float (* __restrict lcl_input)[ifw][VLEN] = (float (*)[*][VLEN])lcl_buffer;
            int idx_input_offset = img*nBIfm*ifh*ifw*VLEN + ifm1*ifh*ifw*VLEN;
#pragma simd
#pragma vector aligned
            for(int i=0; i<ifh*ifw*VLEN; i++) {
              lcl_buffer[i] = 0.0f;
            }
            for(int oj=0; oj < ofh; oj++) {
              int ij = oj*stride_h - t_pad;
              for(int oi=0; oi < ofw; oi++) {
                int ii = oi*stride_w - l_pad;
                for(int kj=0; kj < kh; kj++) {
                  if(ij+kj < 0 || ij+kj >= ifh) continue;
                  for(int ki=0; ki < kw; ki++) {
                    if(ii+ki < 0 || ii+ki >= ifw) continue;
#pragma simd
                    for(int ifm2 = 0; ifm2 <VLEN; ifm2++) {
                      lcl_input[ij+kj][ii+ki][ifm2] += del_output[img][ifm1][oj][oi][ifm2] * recp_pool_size;
                    }
                  }
                }
              }
            }
            for(int ij=0; ij < ifh; ij++) {
              for(int ii=0; ii < ifw; ii++) {
#pragma simd
#pragma vector aligned
#pragma vector nontemporal
                for(int ifm2 = 0; ifm2 <VLEN; ifm2++) {
                  del_input[img][ifm1][ij][ii][ifm2] = lcl_input[ij][ii][ifm2];
                }
              }
            }
          }
        }
      }
      break;
  }

  delinpb->setLayoutType(LIBXSMM_CUSTOM_LAYOUT);
}
