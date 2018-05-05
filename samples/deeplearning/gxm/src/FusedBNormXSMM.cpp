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
/* Sasikanth Avancha, Dhiraj Kalamkar, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>
#include <immintrin.h>
#include "FusedBNormXSMM.hpp"

#define VLEN 16

#define NO_APPX_RCP

void FusedBNormXSMM::forwardPropagate(vector<TensorBuf *> inpb, TensorBuf *gammapb, TensorBuf *betapb, float *gmeanp, float *grstdp, TensorBuf *outpb, int tid)
{
  int nImg = gp->batch_size;
  int nFM = gp->nInput[0];
  int nBfm = nFM/VLEN;
  int fh = gp->iHeight;
  int fw = gp->iWidth;
  int ph = gp->pad_h;
  int pw = gp->pad_w;
  int iph = gp->ipad_h;
  int ipw = gp->ipad_w;
  int sh = gp->stride_h;
  int sw = gp->stride_w;
  int fhs = fh/sh;
  int fws = fw/sw;
  int fhp = fhs + 2*ph;
  int fwp = fws + 2*pw;
  int fhi = fh + 2*iph;
  int fwi = fw + 2*ipw;

  float *inp_r = (float*)inpb[0]->getBuffer();
  float *inp_l = gp->eltwise ? (float*)inpb[1]->getBuffer() : NULL;
  float *outp = (float*)outpb->getBuffer();
  float *gammap = (float*)gammapb->getBuffer();
  float *betap = (float*)betapb->getBuffer();

  int offset = nImg * nFM * fhi * fwi;
  float* bstats_ip = inp_r + offset;

  if(bmeanp == NULL)
  {
    bmeanp = (float*)_mm_malloc(nFM*sizeof(float), 64);
#ifndef NDEBUG
    printf("%s allocated %lu bytes for mean\n",nname.c_str(), nFM*sizeof(float));
#endif
  }
  if(brstdp == NULL)
  {
    brstdp = (float*)_mm_malloc(nFM*sizeof(float), 64);
#ifndef NDEBUG
    printf("%s allocated %lu bytes for stdev\n",nname.c_str(), nFM*sizeof(float));
#endif
  }

  __assume_aligned(inp_r,64);
  if(inp_l)
    __assume_aligned(inp_l,64);

  __assume_aligned(gammap, 64);
  __assume_aligned(betap, 64);
  __assume_aligned(bmeanp, 64);
  __assume_aligned(brstdp, 64);
  __assume_aligned(gmeanp, 64);
  __assume_aligned(grstdp, 64);
  __assume_aligned(outp,64);

  float (* __restrict input_r)[nBfm][fhi][fwi][VLEN] = (float (*)[*][*][*][VLEN])inp_r;
  float (* __restrict input_l)[nBfm][fhi][fwi][VLEN] = gp->eltwise ? (float (*)[*][*][*][VLEN])inp_l : NULL;
  float (* __restrict output)[nBfm][fhp][fwp][VLEN]  = (float (*)[*][*][*][VLEN])outp;
  float (* __restrict bmean)[VLEN]                   = (float (*)[VLEN])bmeanp;
  float (* __restrict brstd)[VLEN]                   = (float (*)[VLEN])brstdp;
  float (* __restrict gmean)[VLEN]                   = (float (*)[VLEN])gmeanp;
  float (* __restrict grstd)[VLEN]                   = (float (*)[VLEN])grstdp;
  float (* __restrict gamma)[VLEN]                   = (float (*)[VLEN])gammap;
  float (* __restrict beta)[VLEN]                    = (float (*)[VLEN])betap;
  float (* __restrict ibstats)[nImg][VLEN]          = (float (*)[*][VLEN])bstats_ip;
  float (* __restrict ibstats2)[nImg][VLEN]         = (float (*)[*][VLEN])((float*)(bstats_ip + nFM*nImg ));

  float recp_nhw = 1./(float)(nImg * fh * fw);

  /* Perform physical padding tests */
#ifndef NDEBUG
  if ( (ph > 0 || pw > 0) && (iph > 0 || ipw > 0) ) {
    printf("node %s: batchnorm forward input and output is padded which cannot be :-(\n", nname.c_str());
  }
  /* check rims */
  check_physical_pad(nname.c_str(), outp, nImg, nBfm, fhs, fws, VLEN, ph,  pw );
  check_physical_pad(nname.c_str(), inp_r, nImg, nBfm, fh,  fw,  VLEN, iph, ipw );
  if (gp->eltwise) check_physical_pad(nname.c_str(), inp_l, nImg, nBfm, fh,  fw,  VLEN, iph, ipw );
#endif

  if(!gp->use_global_stats)
  {
    float recp_nhw2 = recp_nhw * recp_nhw;
    float nhw_ratio = float(nImg*fh*fw)/float(nImg*fh*fw - 1);

    /* reducing the batch stats computed during convolution to compute E(X) and Sigma(X).
       We need double precision for this as this computation is numerically slightly
       unstable */
#if 1
#ifdef __AVX512F__
    __m512  vrecp_nhw  = _mm512_set1_ps(recp_nhw);
    __m512  veps       = _mm512_set1_ps(gp->eps);
    __m512  vmmf       = _mm512_set1_ps(gp->mmf);
    __m512  vnhw_ratio = _mm512_set1_ps(nhw_ratio);
    float one          = 1.0;
    __m512  vone       = _mm512_set1_ps(one);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int b = 0; b < nBfm; ++b) {
      __m512 tmp1  = _mm512_setzero_ps();
      __m512 tmpd1 = _mm512_setzero_ps();

      /* reduce over images */
      for (int n = 0; n < nImg; ++n) {
        tmp1 = _mm512_add_ps( tmp1, _mm512_load_ps(&(ibstats[b][n][0]) ) );
        tmpd1 = _mm512_add_ps( tmpd1, _mm512_load_ps(&(ibstats2[b][n][0]) ) );
      }

      __m512 vtbmeanA   = _mm512_mul_ps( vrecp_nhw, tmp1 );
      __m512 vtbmean2A  = _mm512_mul_ps( vtbmeanA, vtbmeanA );
      __m512 vtbmean_2A = _mm512_mul_ps( vrecp_nhw, tmpd1 );
#ifdef __AVX512ER__
      __m512 vtbrstd_A  = _mm512_rsqrt28_ps( _mm512_add_ps( _mm512_sub_ps( vtbmean_2A, vtbmean2A ), veps) );
#else
#ifdef NO_APPX_RCP
      __m512 vtbrstd_A  = _mm512_div_ps(vone, _mm512_sqrt_ps(_mm512_add_ps( _mm512_sub_ps( vtbmean_2A, vtbmean2A ), veps)));
#else
      __m512 vtbrstd_A  = _mm512_rsqrt14_ps( _mm512_add_ps( _mm512_sub_ps( vtbmean_2A, vtbmean2A ), veps) );
#endif
#endif
      _mm512_store_ps( &(bmean[b][0]), vtbmeanA );
      _mm512_store_ps( &(brstd[b][0]), vtbrstd_A );

      _mm512_store_ps( &(gmeanp[b*16]), _mm512_add_ps( _mm512_mul_ps( _mm512_load_ps( &(gmeanp[b*16]) ), vmmf), vtbmeanA));
      _mm512_store_ps( &(grstdp[b*16]), _mm512_add_ps( _mm512_mul_ps( _mm512_load_ps( &(grstdp[b*16]) ), vmmf), _mm512_mul_ps(vnhw_ratio, vtbrstd_A)));
    }
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int b = 0; b < nBfm; ++b) {
      float tmp[16];
      float tmpd[16];
#pragma omp simd
      for (int v = 0; v < 16; ++v) {
        tmp[v] = 0.0;
        tmpd[v] = 0.0;
      }
      /* reduce over images */
      for (int n = 0; n < nImg; ++n) {
#pragma omp simd
        for(int v = 0; v < 16; ++v) {
          tmp[v] += ibstats[b][n][v];
          tmpd[v] += ibstats2[b][n][v];
        }
      }
      /* calculate expectation and standard derivation */
#pragma omp simd
      for (int v = 0; v < 16; ++v) {
        const float tbmean = (recp_nhw*tmp[v]) ;
        const float tbmean2  = tbmean * tbmean;
        const float tbmean_2 = recp_nhw * tmpd[v];
        const float tbrstd = 1.0/sqrt(tbmean_2 - tbmean2 + gp->eps);
        bmean[b][v] = tbmean;
        brstd[b][v] = tbrstd;
        gmeanp[(b*16)+v] = gmeanp[(b*16)+v] * gp->mmf + tbmean;
        grstdp[(b*16)+v] = grstdp[(b*16)+v] * gp->mmf + nhw_ratio*tbrstd;
      }
    }
#endif
#endif

    /* This commened code is performing a stable computation of E(X) and Sigma(X),
       however it requires several passes over the data and is potentially slower */
#if 0
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i < nFM; i++) {
      bmeanp[i] = 0.0;
      brstdp[i] = 0.0;
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int fm=0; fm < nBfm; fm++) {
      for(int v=0; v<VLEN; v++) {
        for(int img=0; img < nImg; img++) {
          for(int h=0; h < fh; h++) {
            for(int w=0; w < fw; w++) {
              bmean[fm][v] += input_r[img][fm][h][w][v];
            }
          }
        }
      }
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i < nFM; i++)
      bmeanp[i] *= recp_nhw;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int fm=0; fm < nBfm; fm++) {
      for(int v=0; v<VLEN; v++) {
        for(int img=0; img < nImg; img++) {
          for(int h=0; h < fh; h++) {
            for(int w=0; w < fw; w++) {
              brstd[fm][v] += (input_r[img][fm][h][w][v] - bmean[fm][v]) * (input_r[img][fm][h][w][v] - bmean[fm][v]) * recp_nhw;
            }
          }
        }
      }
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int fm=0; fm < nFM; fm++)
      brstdp[fm] = 1/sqrt(brstdp[fm] + gp->eps);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int fm=0; fm < nFM; fm++)
    {
      gmeanp[fm] = gmeanp[fm] * gp->mmf + bmeanp[fm] ;
      grstdp[fm] = grstdp[fm] * gp->mmf + nhw_ratio*brstdp[fm];
    }
#endif
    /* multiple pass computatoin of E(X) and Sigma(X) ends here */

#ifdef USE_XSMM_TIMING
    struct timeval tvsc, tvec;
    double gb, gib;
    gettimeofday(&tvsc, NULL);

    /* @TODO: fix when physical paddding is working */
    gb = (2.0*(double)nImg*(double)nFM*(double)fhs*(double)fws*(double)sizeof(float)) / (1000*1000*1000);
    gib = (2.0*(double)nImg*(double)nFM*(double)fhs*(double)fws*(double)sizeof(float)) / (1024*1024*1024);
#endif

    if(gp->eltwise)
    {
      // ReLU
      if(gp->relu) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int img=0; img < nImg; img++) {
          for(int fm=(nBfm-1); fm >= 0; fm--) {
            for(int h=(fh+iph)-sh, hp=(fhs+ph)-1; h >= iph; h-=sh, hp--) {
              for(int w=(fw+ipw)-sw, wp=(fws+pw)-1; w >= ipw; w-=sw, wp--) {
#if 0
          for(int fm=0; fm < nBfm; fm++) {
            for(int h=iph, hp=ph; h < (fh+iph); h+=sh, hp++) {
              for(int w=ipw, wp=pw; w < (fw+ipw); w+=sw, wp++) {
#endif
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
                for(int v=0; v<VLEN; v++) {
                  // BN + scale (gamma, beta)
                  float o = gamma[fm][v]*(input_r[img][fm][h][w][v] - bmean[fm][v]) * brstd[fm][v] + beta[fm][v] + input_l[img][fm][h][w][v];
                  output[img][fm][hp][wp][v] = ( o < 0.0f ) ? 0.0f : o;
                }
              }
            }
          }
        }
      } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int img=0; img < nImg; img++) {
          for(int fm=(nBfm-1); fm >= 0; fm--) {
            for(int h=(fh+iph)-sh, hp=(fhs+ph)-1; h >= iph; h-=sh, hp--) {
              for(int w=(fw+ipw)-sw, wp=(fws+pw)-1; w >= ipw; w-=sw, wp--) {
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
                for(int v=0; v<VLEN; v++) {
                  // BN + scale (gamma, beta)
                  output[img][fm][hp][wp][v] = gamma[fm][v]*(input_r[img][fm][h][w][v] - bmean[fm][v]) * brstd[fm][v] + beta[fm][v] + input_l[img][fm][h][w][v];
                }
              }
            }
          }
        }
      }
#ifdef USE_XSMM_TIMING
      gb += ((double)nImg*(double)nFM*(double)fhs*(double)fws*(double)sizeof(float)) / (1000*1000*1000);
      gib += ((double)nImg*(double)nFM*(double)fhs*(double)fws*(double)sizeof(float)) / (1024*1024*1024);
#endif
    }
    else
    {
      if(gp->relu) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int img=0; img < nImg; img++) {
          for(int fm=(nBfm-1); fm >= 0; fm--) {
            for(int h=(fh+iph)-sh, hp=(fhs+ph)-1; h >= iph; h-=sh, hp--) {
              for(int w=(fw+ipw)-sw, wp=(fws+pw)-1; w >= ipw; w-=sw, wp--) {
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
                for(int v=0; v<VLEN; v++) {
                  // BN + scale (gamma, beta)
                  float o = gamma[fm][v]*(input_r[img][fm][h][w][v] - bmean[fm][v]) * brstd[fm][v] + beta[fm][v];
                  output[img][fm][hp][wp][v] = ( o < 0.0f ) ? 0.0f : o;
                }
              }
            }
          }
        }
      } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int img=0; img < nImg; img++) {
          for(int fm=(nBfm-1); fm >= 0; fm--) {
            for(int h=(fh+iph)-sh, hp=(fhs+ph)-1; h >= iph; h-=sh, hp--) {
              for(int w=(fw+ipw)-sw, wp=(fws+pw)-1; w >= ipw; w-=sw, wp--) {
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
                for(int v=0; v<VLEN; v++) {
                  // BN + scale (gamma, beta)
                  output[img][fm][hp][wp][v] = gamma[fm][v]*(input_r[img][fm][h][w][v] - bmean[fm][v]) * brstd[fm][v] + beta[fm][v];
                }
              }
            }
          }
        }
      }
    }
    scaling_factor_ = scaling_factor_ * gp->mmf + 1.;

    // Quantization
    if(gp->out_data_type == DT_DFP16)
    {
      unsigned char scf_output;
      i16_outp = (short*)outpb->getLPBuffer();
      libxsmm_dnn_quantize_act(outp, i16_outp, nImg, nFM, fhp, fwp, 16, 8, 2, 2, &scf_output, LIBXSMM_DNN_QUANT_FPHW_ROUND);

      outpb->setLPSF(scf_output);
    }

#ifdef USE_XSMM_TIMING
    gettimeofday(&tvec, NULL);
    double fp_time = (tvec.tv_sec + tvec.tv_usec*1e-6) - (tvsc.tv_sec + tvsc.tv_usec*1e-6);

#ifdef USE_MLSL
    if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
    {
      if(gp->stride_h == 1 && gp->pad_h == 0)
        printf("XSMM-BN-FP mb%dic%dih%d time = %g ms, GB/s = %.1f GiB/s = %.1f\n",gp->batch_size,gp->nOutput,gp->oHeight,fp_time*1000.0, gb/fp_time, gib/fp_time);
      else if(gp->stride_h == 2)
        printf("XSMM-BN-FP mb%dic%dih%dsh%dn time = %g ms, GB/s = %.1f GiB/s = %.1f\n",gp->batch_size,gp->nOutput,gp->oHeight,gp->stride_h,fp_time*1000.0, gb/fp_time, gib/fp_time);
      else if(gp->pad_h == 1)
        printf("XSMM-BN-FP mb%dic%dih%dph%dn time = %g ms, GB/s= %.1f GiB/s = %.1f\n",gp->batch_size,gp->nOutput,gp->oHeight,gp->pad_h,fp_time*1000.0, gb/fp_time, gib/fp_time);
    }
#endif
  }
  else
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i < nFM; i++) {
      grstdp[i] /= scaling_factor_;
      gmeanp[i] /= scaling_factor_;
    }

    if(gp->eltwise)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int img=0; img < nImg; img++) {
        for(int fm=0; fm < nBfm; fm++) {
          for(int h=iph, hp=ph; h < (fh+iph); h+=sh, hp++) {
            for(int w=ipw, wp=pw; w < (fw+ipw); w+=sw, wp++) {
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
              for(int v=0; v<VLEN; v++) {
                // BN + scale (gamma, beta)
                output[img][fm][hp][wp][v] = gamma[fm][v]*(input_r[img][fm][h][w][v] - gmean[fm][v]) * grstd[fm][v] + beta[fm][v] + input_l[img][fm][h][w][v];
                // ReLU
                if(gp->relu) {
                  if(output[img][fm][hp][wp][v] < 0)
                    output[img][fm][hp][wp][v] = 0;
                }
              }
            }
          }
        }
      }
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int img=0; img < nImg; img++) {
        for(int fm=0; fm < nBfm; fm++) {
          for(int h=iph, hp=ph; h < (fh+iph); h+=sh, hp++) {
            for(int w=ipw, wp=pw; w < (fw+ipw); w+=sw, wp++) {
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
              for(int v=0; v<VLEN; v++) {
                // BN + scale (gamma, beta)
                output[img][fm][hp][wp][v] = gamma[fm][v]*(input_r[img][fm][h][w][v] - gmean[fm][v]) * grstd[fm][v] + beta[fm][v];
                // ReLU
                if(gp->relu) {
                  if(output[img][fm][hp][wp][v] < 0)
                    output[img][fm][hp][wp][v] = 0;
                }
              }
            }
          }
        }
      }
    }
  }

  /* Perform physical padding tests */
#ifndef NDEBUG
  check_physical_pad(nname.c_str(), outp, nImg, nBfm, fhs, fws, VLEN, ph, pw );
  check_physical_pad(nname.c_str(), inp_r, nImg, nBfm, fh,  fw,  VLEN, iph, ipw );
  if (gp->eltwise) check_physical_pad(nname.c_str(), inp_l, nImg, nBfm, fh,  fw,  VLEN, iph, ipw );
#endif

  outpb->setLayoutType(inpb[0]->getLayoutType());
  outpb->setLayout(inpb[0]->getLayout());
}

void FusedBNormXSMM::backPropagate(vector<TensorBuf*> inpb, TensorBuf* outpb, TensorBuf *gammapb, TensorBuf *deloutpb, TensorBuf *delgammapb, TensorBuf *delbetapb, vector<TensorBuf*> delinpb, int tid)
{
  int nImg  = gp->batch_size;
  int nFM = gp->nOutput;
  int nBfm = nFM/VLEN;
  int fh = gp->oHeight;
  int fw = gp->oWidth;
  int ph = gp->pad_h;
  int pw = gp->pad_w;
  int sh = gp->stride_h;
  int sw = gp->stride_w;
  int iph = gp->ipad_h;
  int ipw = gp->ipad_w;
  int fhs = fh/sh;
  int fws = fw/sw;
  int fhp = fhs + 2*ph;
  int fwp = fws + 2*pw;
  int fhi = fh + 2*iph;
  int fwi = fw + 2*ipw;

  int threads = gp->num_threads;

  const float nhw = nImg * fh * fw;
  const float recp_nhw = 1.0f/nhw;

  float *inp_r = (float*)inpb[0]->getBuffer();
  float *outp = (float*)outpb->getBuffer();
  float *gammap = (float*)gammapb->getBuffer();
  float *deloutp = (float*)deloutpb->getBuffer();
  float *delinp_r = (float*)delinpb[0]->getBuffer();
  float *delgammap = (float*)delgammapb->getBuffer();
  float *delbetap = (float*)delbetapb->getBuffer();

  __assume_aligned(inp_r,64);
  __assume_aligned(outp,64);
  __assume_aligned(delinp_r,64);
  __assume_aligned(deloutp,64);
  __assume_aligned(bmeanp,64);
  __assume_aligned(brstdp,64);

  float (* __restrict input_r)[nBfm][fhi][fwi][VLEN]     = (float (*)[*][*][*][VLEN])inp_r;
  float (* __restrict del_input_r)[nBfm][fhi][fwi][VLEN] = (float (*)[*][*][*][VLEN])delinp_r;
  float (* __restrict output)[nBfm][fhp][fwp][VLEN]      = (float (*)[*][*][*][VLEN])outp;
  float (* __restrict del_output)[nBfm][fhp][fwp][VLEN]  = (float (*)[*][*][*][VLEN])deloutp;
  float (* __restrict gamma)[VLEN]                       = (float (*)[VLEN])gammap;
  float (* __restrict del_gamma)[VLEN]                   = (float (*)[VLEN])delgammap;
  float (* __restrict del_beta)[VLEN]                    = (float (*)[VLEN])delbetap;
  float (* __restrict bmean)[VLEN]                       = (float (*)[VLEN])bmeanp;
  float (* __restrict brstd)[VLEN]                       = (float (*)[VLEN])brstdp;

  if ( del_gamma_imgp == NULL) {
    del_gamma_imgp = (float*)_mm_malloc( nImg * nBfm * VLEN * sizeof(float), 64);
#ifndef NDEBUG
    printf("%s allocated %lu bytes for del_gamma reduce\n", nname.c_str(), nImg * nBfm * VLEN * sizeof(float));
#endif
  }
  if ( del_beta_imgp == NULL) {
    del_beta_imgp = (float*)_mm_malloc( nImg * nBfm * VLEN * sizeof(float), 64);
#ifndef NDEBUG
    printf("%s allocated %lu bytes for del_beta reduce\n", nname.c_str(), nImg * nBfm * VLEN * sizeof(float));
#endif
  }

  float (* __restrict del_gamma_img)[nImg][VLEN] = (float (*)[nImg][VLEN])del_gamma_imgp;
  float (* __restrict del_beta_img)[nImg][VLEN] = (float (*)[nImg][VLEN])del_beta_imgp;

  /* zero the rims in case of physical padding */
  /* @TODO, we need to do the same thing with del_input_l?! */
  if (iph > 0 || iph > 0) {
#pragma omp parallel for
    for (int img = 0; img < nImg; img++) {
      for (int fm = 0; fm < nBfm; fm++) {
        for (int w = 0; w < fwi; w++) {
          for (int ph = 0; ph < iph; ph++) {
#ifdef __AVX512F__
            _mm512_stream_ps( &(del_input_r[img][fm][ph      ][w][0]), _mm512_setzero_ps() );
            _mm512_stream_ps( &(del_input_r[img][fm][fhi-1-ph][w][0]), _mm512_setzero_ps() );
#else
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
            for(int v=0; v < VLEN; v++) {
              del_input_r[img][fm][ph][w][v] = 0.0f;
              del_input_r[img][fm][fhi-1-ph][w][v] = 0.0f;
            }
#endif
          }
        }
        for (int h = iph; h < fh+iph; h++) {
          for (int pw = 0; pw < ipw; pw++) {
#ifdef __AVX512F__
            _mm512_stream_ps( &(del_input_r[img][fm][h][pw      ][0]), _mm512_setzero_ps() );
            _mm512_stream_ps( &(del_input_r[img][fm][h][fwi-1-pw][0]), _mm512_setzero_ps() );
#else
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
            for(int v=0; v < VLEN; v++) {
              del_input_r[img][fm][h][pw][v] = 0.0f;
              del_input_r[img][fm][h][fwi-1-pw][v] = 0.0f;
            }
#endif
          }
        }
      }
    }
  }

  /* Perform physical padding tests */
#ifndef NDEBUG
  if ( (ph > 0 || pw > 0) && (iph > 0 || ipw > 0) ) {
    printf("node %s: batchnorm backward input and output is padded which cannot be :-(\n", nname.c_str());
  }

  /* check rims */
  check_physical_pad( nname.c_str(), delinp_r, nImg, nBfm, fh,  fw,  VLEN, iph, ipw );
  check_physical_pad( nname.c_str(),    inp_r, nImg, nBfm, fh,  fw,  VLEN, iph, ipw );
  check_physical_pad( nname.c_str(),  deloutp, nImg, nBfm, fhs, fws, VLEN, ph,  pw );
  check_physical_pad( nname.c_str(),     outp, nImg, nBfm, fhs, fws, VLEN, ph,  pw );
#endif

  int start_fm = 0;
  int inc_fm = nBfm;
#ifdef USE_BLOCKING_BN
  double blocks = 2000.0 / ((5.0*(double)16*(double)fhs*(double)fws*(double)sizeof(float)) / (1024.0)) ;
  if (blocks >= 1.0) {
    inc_fm = std::min<int>((1 << (int)log2(blocks)), nBfm);
  }
#endif

#ifdef USE_XSMM_TIMING
  struct timeval tvsc, tvec;
  double gb, gib;
  gettimeofday(&tvsc, NULL);

  /* @TODO: fix when physical paddding is working */
  gb = (5.0*(double)nImg*(double)nFM*(double)fhs*(double)fws*(double)sizeof(float)) / (1000*1000*1000);
  gib = (5.0*(double)nImg*(double)nFM*(double)fhs*(double)fws*(double)sizeof(float)) / (1024*1024*1024);
#endif


  if(gp->eltwise)
    delinpb[1]->setBuffer(deloutp);

  if(gp->bwd_relu)
  {
    for ( int fmb = 0; fmb < nBfm; fmb += inc_fm ) {
    int stop_fm = std::min<int>(fmb + inc_fm, nBfm);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#pragma omp for
#pragma vector aligned
      for(int img=0; img < nImg; img++) {
        for(int fm=fmb; fm < stop_fm; fm++) {
        /*for(int fm=0; fm < nBfm; fm++) {*/
          float lcl_gamma[VLEN];
          float lcl_beta[VLEN];
#pragma omp simd
#pragma vector aligned
          for(int v=0; v < VLEN; v++) {
            lcl_gamma[v] = 0.0f;
            lcl_beta[v] = 0.0f;
          }
          for(int h=iph, hp=ph; h < (fh + iph); h+=sh, hp++) {
            for(int w=ipw, wp=pw; w < (fw + ipw); w+=sw, wp++) {
#pragma omp simd
#pragma vector aligned
              for(int v=0; v < VLEN; v++) {
                del_output[img][fm][hp][wp][v] = (output[img][fm][hp][wp][v] == 0.0) ? 0.0 : del_output[img][fm][hp][wp][v];
                lcl_gamma[v] += (input_r[img][fm][h][w][v] - bmean[fm][v]) * del_output[img][fm][hp][wp][v] * brstd[fm][v];
                lcl_beta[v] += del_output[img][fm][hp][wp][v];
              }
            }
          }
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
          for(int v=0; v < VLEN; v++) {
            del_gamma_img[fm][img][v] = lcl_gamma[v];
            del_beta_img[fm][img][v]  = lcl_beta[v];
          }
        }
      }
#pragma omp for
#pragma vector aligned
      for(int fm=fmb; fm < stop_fm; fm++) {
      /*for(int fm=0; fm < nBfm; fm++) {*/
        for(int img=0; img < nImg; img++) {
#pragma omp simd
#pragma vector aligned
          for(int v=0; v < VLEN; v++) {
            del_gamma[fm][v] += del_gamma_img[fm][img][v];
            del_beta[fm][v] += del_beta_img[fm][img][v];
          }
        }
      }
#pragma omp for nowait
      for(int img=0; img < nImg; img++) {
#ifdef __AVX512F__
        __m512 vrecp_nhw = _mm512_set1_ps(recp_nhw);
        __m512 vnhw      = _mm512_set1_ps(nhw);
#endif
        for(int fm=(stop_fm-1); fm >= fmb; fm--) {
          /*for(int fm=(nBfm-1); fm >= 0; fm--) {*/
#ifdef __AVX512F__
          __m512 vgamma     = _mm512_load_ps( &(gamma[fm][0]) );
          __m512 vbrstd     = _mm512_load_ps( &(brstd[fm][0]) );
          __m512 vdel_beta  = _mm512_load_ps( &(del_beta[fm][0]) );
          __m512 vbmean     = _mm512_load_ps( &(bmean[fm][0]) );
          __m512 vdel_gamma = _mm512_load_ps( &(del_gamma[fm][0]) );
#endif
          for(int h=(fh+iph)-sh, hp=(fhs+ph)-1; h >= iph; h-=sh, hp--) {
            for(int w=(fw+ipw)-sw, wp=(fws+pw)-1; w >= ipw; w-=sw, wp--) {
#ifdef __AVX512F__
              __m512 vdel_output = _mm512_load_ps( &(del_output[img][fm][hp][wp][0]) );
              __m512 vinput_r    = _mm512_load_ps( &(input_r[img][fm][h][w][0]) );
              __m512 vtmp0 = _mm512_mul_ps( _mm512_mul_ps( _mm512_sub_ps( vinput_r, vbmean ), vdel_gamma ), vbrstd );
              __m512 vtmp1 = _mm512_sub_ps( _mm512_mul_ps( vnhw, vdel_output ), _mm512_add_ps( vdel_beta, vtmp0 ) );
#ifdef USE_NTS_BN
              _mm512_stream_ps( &(del_input_r[img][fm][h][w][0]),
                  _mm512_mul_ps( vgamma, _mm512_mul_ps( vbrstd, _mm512_mul_ps( vrecp_nhw, vtmp1 ))) );
#else
              _mm512_store_ps( &(del_input_r[img][fm][h][w][0]),
                  _mm512_mul_ps( vgamma, _mm512_mul_ps( vbrstd, _mm512_mul_ps( vrecp_nhw, vtmp1 ))) );
#endif
#else
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
              for(int v=0; v < VLEN; v++) {
                del_input_r[img][fm][h][w][v] = gamma[fm][v] * brstd[fm][v] * recp_nhw * (nhw*del_output[img][fm][hp][wp][v] -
                    (del_beta[fm][v] + (input_r[img][fm][h][w][v] - bmean[fm][v]) * del_gamma[fm][v] * brstd[fm][v]));
              }
#endif
            }
          }
        }
        }
      }
    }

#ifdef USE_XSMM_TIMING
    gb += ((double)nImg*(double)nFM*(double)fhs*(double)fws*(double)sizeof(float)) / (1000*1000*1000);
    gib += ((double)nImg*(double)nFM*(double)fhs*(double)fws*(double)sizeof(float)) / (1024*1024*1024);
#endif
  }
  else
  {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#pragma omp for
#pragma vector aligned
      for(int img=0; img < nImg; img++) {
        for(int fm=0; fm < nBfm; fm++) {
          float lcl_gamma[VLEN];
          float lcl_beta[VLEN];
#pragma omp simd
#pragma vector aligned
          for(int v=0; v < VLEN; v++) {
            lcl_gamma[v] = 0.0f;
            lcl_beta[v] = 0.0f;
          }
          for(int h=iph, hp=ph; h < (fh + iph); h+=sh, hp++) {
            for(int w=ipw, wp=pw; w < (fw + ipw); w+=sw, wp++) {
#pragma omp simd
#pragma vector aligned
              for(int v=0; v < VLEN; v++) {
                lcl_gamma[v] += (input_r[img][fm][h][w][v] - bmean[fm][v]) * del_output[img][fm][hp][wp][v] * brstd[fm][v];
                lcl_beta[v] += del_output[img][fm][hp][wp][v];
              }
            }
          }
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
          for(int v=0; v < VLEN; v++) {
            del_gamma_img[fm][img][v] = lcl_gamma[v];
            del_beta_img[fm][img][v]  = lcl_beta[v];
          }
        }
      }
#pragma omp for
#pragma vector aligned
      for(int fm=0; fm < nBfm; fm++) {
        for(int img=0; img < nImg; img++) {
#pragma omp simd
#pragma vector aligned
          for(int v=0; v < VLEN; v++) {
            del_gamma[fm][v] += del_gamma_img[fm][img][v];
            del_beta[fm][v] += del_beta_img[fm][img][v];
          }
        }
      }
#pragma omp for nowait
      for(int img=0; img < nImg; img++) {
#ifdef __AVX512F__
        __m512 vrecp_nhw = _mm512_set1_ps(recp_nhw);
        __m512 vnhw      = _mm512_set1_ps(nhw);
#endif
        for(int fm=(nBfm-1); fm >= 0; fm--) {
#ifdef __AVX512F__
          __m512 vgamma     = _mm512_load_ps( &(gamma[fm][0]) );
          __m512 vbrstd     = _mm512_load_ps( &(brstd[fm][0]) );
          __m512 vdel_beta  = _mm512_load_ps( &(del_beta[fm][0]) );
          __m512 vbmean     = _mm512_load_ps( &(bmean[fm][0]) );
          __m512 vdel_gamma = _mm512_load_ps( &(del_gamma[fm][0]) );
#endif
          for(int h=(fh+iph)-sh, hp=(fhs+ph)-1; h >= iph; h-=sh, hp--) {
            for(int w=(fw+ipw)-sw, wp=(fws+pw)-1; w >= ipw; w-=sw, wp--) {
#ifdef __AVX512F__
               __m512 vdel_output = _mm512_load_ps( &(del_output[img][fm][hp][wp][0]) );
               __m512 vinput_r    = _mm512_load_ps( &(input_r[img][fm][h][w][0]) );
               __m512 vtmp0 = _mm512_mul_ps( _mm512_mul_ps( _mm512_sub_ps( vinput_r, vbmean ), vdel_gamma ), vbrstd );
               __m512 vtmp1 = _mm512_sub_ps( _mm512_mul_ps( vnhw, vdel_output ), _mm512_add_ps( vdel_beta, vtmp0 ) );
#ifdef USE_NTS_BN
               _mm512_stream_ps( &(del_input_r[img][fm][h][w][0]),
                  _mm512_mul_ps( vgamma, _mm512_mul_ps( vbrstd, _mm512_mul_ps( vrecp_nhw, vtmp1 ))) );
#else
               _mm512_store_ps( &(del_input_r[img][fm][h][w][0]),
                  _mm512_mul_ps( vgamma, _mm512_mul_ps( vbrstd, _mm512_mul_ps( vrecp_nhw, vtmp1 ))) );
#endif
#else
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
              for(int v=0; v < VLEN; v++) {
                del_input_r[img][fm][h][w][v] = gamma[fm][v] * brstd[fm][v] * recp_nhw * (nhw*del_output[img][fm][hp][wp][v] -
                                    (del_beta[fm][v] + (input_r[img][fm][h][w][v] - bmean[fm][v]) * del_gamma[fm][v] * brstd[fm][v]));
              }
#endif
            }
          }
        }
      }
    }
  }

  // Quantization
  if(gp->in_data_type == DT_DFP16 || gp->out_data_type == DT_DFP16)
  {
    unsigned char scf_delinput_r;
    i16_delinp_r = (short*)delinpb[0]->getLPBuffer();

    libxsmm_dnn_quantize_act(delinp_r, i16_delinp_r, nImg, nFM, fhi, fwi, 16, 8, 2, 2, &scf_delinput_r, LIBXSMM_DNN_QUANT_FPHW_ROUND);
    delinpb[0]->setLPSF(scf_delinput_r);
  }

#ifdef USE_XSMM_TIMING
  gettimeofday(&tvec, NULL);
  double bp_time = (tvec.tv_sec + tvec.tv_usec*1e-6) - (tvsc.tv_sec + tvsc.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    if(gp->stride_h == 1 && gp->pad_h == 0)
      printf("XSMM-BN-BP mb%dic%dih%d time = %g ms, GB/s = %.1f GiB/s = %.1f\n",gp->batch_size,gp->nOutput,gp->oHeight,bp_time*1000.0, gb/bp_time, gib/bp_time);
    else if(gp->stride_h == 2)
      printf("XSMM-BN-BP mb%dic%dih%dsh%dn time = %g ms, GB/s = %.1f GiB/s = %.1f\n",gp->batch_size,gp->nOutput,gp->oHeight,gp->stride_h,bp_time*1000.0, gb/bp_time, gib/bp_time);
    else if(gp->pad_h == 1)
      printf("XSMM-BN-BP mb%dic%dih%dph%dn time = %g ms, GB/s= %.1f GiB/s = %.1f\n",gp->batch_size,gp->nOutput,gp->oHeight,gp->pad_h,bp_time*1000.0, gb/bp_time, gib/bp_time);
  }
#endif

#ifndef NDEBUG
  /* check rims */
  check_physical_pad( nname.c_str(), delinp_r, nImg, nBfm, fh,  fw,  VLEN, iph, ipw );
  check_physical_pad( nname.c_str(),    inp_r, nImg, nBfm, fh,  fw,  VLEN, iph, ipw );
  check_physical_pad( nname.c_str(),  deloutp, nImg, nBfm, fhs, fws, VLEN, ph,  pw );
  check_physical_pad( nname.c_str(),     outp, nImg, nBfm, fhs, fws, VLEN, ph,  pw );
#endif

  bpdone = true;
  delinpb[0]->setLayoutType(deloutpb->getLayoutType());
  delinpb[0]->setLayout(deloutpb->getLayout());
}

