/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
/* Alexander Heinecke, Hans Pabst, Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#include <libxsmm_timer.h>

#if defined(_WIN32)
/* note: later on, this leads to (correct but) different than expected norm-values */
# define drand48() ((double)rand() / RAND_MAX)
# define srand48 srand
#endif

#define CHKERR_LIBXSMM_CONV(A) if ( A != LIBXSMM_CONV_SUCCESS ) fprintf(stderr, "%s\n", libxsmm_conv_get_error(A) );

typedef struct {
  int nImg;
  int nIfm;
  int nOfm;
  int ifhp;
  int ifwp;
  int ofhp;
  int ofwp;
  int ofh;
  int ofw;
  int pad_h;
  int pad_w;
  int kh;
  int kw;
  int stride_h;
  int stride_w;
  int nSplits;
} naive_conv_t;

typedef struct {
  double max_rel_err;
  double max_abs_err;
  double l2_rel_err;
  double one_norm_ref;
  double one_norm_test;
} correctness_t;

LIBXSMM_INLINE void zero_buf(float* buf, long size) {
  int i;
  for (i = 0; i < size; ++i) {
    buf[i] = 0.0f;
  }
}

LIBXSMM_INLINE void init_buf(float* buf, long size, int initPos, int initOne)
{
  int i;
  zero_buf(buf, size);
  for (i = 0; i < size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? drand48() : (0.5 - drand48())));
  }
}

LIBXSMM_INLINE void compare_buf(float* ref, float* test, long size, correctness_t* norms)
{
  int i;
  double diff, rel_err;

  norms->max_rel_err = 0.;
  norms->max_abs_err = 0.;
  norms->l2_rel_err = 0.;
  norms->one_norm_ref = 0.;
  norms->one_norm_test = 0.;

  for (i = 0; i < size; ++i) {
    norms->one_norm_ref += (double)ref[i];
    norms->one_norm_test += (double)test[i];
    diff = fabs((double)ref[i] - (double)test[i]);
    norms->l2_rel_err += (diff*diff);
    rel_err = 0.0;
    if (diff > 0.0 ) {
      rel_err = diff/fabs((double)ref[i]);
    }
    if (rel_err > norms->max_rel_err) {
      norms->max_rel_err = rel_err;
    }
    if (diff > norms->max_abs_err) {
      norms->max_abs_err = diff;
    }
#if 0
    printf("MISMATCH@ %3d: A=%12.8g  B=%12.8g (E:%12.4e)\n", i, ref[i], test[i], rel_err);
#endif

  }
  norms->l2_rel_err = sqrt(norms->l2_rel_err);
}

LIBXSMM_INLINE void naive_conv_fp(naive_conv_t* param, const float* input, float* output, const float* filter)
{
  int nImg  = param->nImg;
  int nIfm = param->nIfm;
  int nOfm = param->nOfm;
  int ifhp  = param->ifhp;
  int ifwp  = param->ifwp;
  int ofhp  = param->ofhp;
  int ofwp  = param->ofwp;
  int ofh   = param->ofh;
  int ofw   = param->ofw;
  int pad_h = param->pad_h;
  int pad_w = param->pad_w;
  int kh    = param->kh;
  int kw    = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  int nSplits   = param->nSplits;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;
#if defined(__INTEL_COMPILER)
  float (*LIBXSMM_RESTRICT  input_t)[nIfm][ifhp][ifwp] = (float(*)[*][*][*])input;
  float (*LIBXSMM_RESTRICT filter_t)[nIfm][kh][kw]     = (float(*)[*][*][*])filter;
  float (*LIBXSMM_RESTRICT output_t)[nOfm][ofhp][ofwp] = (float(*)[*][*][*])(output + (pad_w * ofwp + pad_h));
#elif defined(LIBXSMM_VLA)
  typedef float (*LIBXSMM_RESTRICT  input_type)[nIfm][ifhp][ifwp];
  typedef float (*LIBXSMM_RESTRICT filter_type)[nIfm][kh][kw];
  typedef float (*LIBXSMM_RESTRICT output_type)[nOfm][ofhp][ofwp];
  const input_type   input_t =  (input_type)input;
  const filter_type filter_t = (filter_type)filter;
  const output_type output_t = (output_type)(output + (pad_w * ofwp + pad_h));
#else
  unsigned int ishape[4], fshape[4], oshape[4], indexi[4], indexf[4], indexo[4];
  const float *LIBXSMM_RESTRICT  input_t = (const float*)input;
  const float *LIBXSMM_RESTRICT filter_t = (const float*)filter;
  float *LIBXSMM_RESTRICT output_t = (float*)(output + (pad_w * ofwp + pad_h));
  ishape[0] = ifwp; ishape[1] = ifhp; ishape[2] = nIfm; ishape[3] = nImg;
  fshape[0] =   kw; fshape[1] =   kh; fshape[2] = nIfm; fshape[3] = nOfm;
  oshape[0] = ofwp; oshape[1] = ofhp; oshape[2] = nOfm; oshape[3] = nImg;
#endif

  if (nSplits != 1) {
    printf("nSplits != 1 not supported yet for naive code!\n");
    exit(1);
  }

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) /*private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)*/
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w;
            for (kj = 0; kj < kh; ++kj) {
              for (ki = 0; ki < kw; ++ki) {
#if defined(__INTEL_COMPILER) || defined(LIBXSMM_VLA)
                output_t[img][ofm][oj][oi] += input_t[img][ifm][ij+kj][ii+ki] * filter_t[ofm][ifm][kj][ki];
#else
                size_t i, f, o;
                indexi[0] = ii + ki; indexi[1] = ij + kj; indexi[2] = ifm; indexi[3] = img;
                indexf[0] = ki; indexf[1] = kj; indexf[2] = ifm; indexf[3] = ofm;
                indexo[0] = oi; indexo[1] = oj; indexo[2] = ofm; indexo[3] = img;
                LIBXSMM_CALC_INDEX1(size_t, i, 4, indexi, ishape);
                LIBXSMM_CALC_INDEX1(size_t, f, 4, indexf, fshape);
                LIBXSMM_CALC_INDEX1(size_t, o, 4, indexo, oshape);
                output_t[o] += input_t[i] * filter_t[f];
#endif
              }
            }
          }
        }
      }
    }
  }
}

int main(int argc, char* argv[])
{
  float *naive_input, *naive_output, *naive_filter, *naive_libxsmm_output;
  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w, pad_h, pad_w;
  naive_conv_t naive_param;
  correctness_t norms;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int ifw = 14;           /* input width, "W" */
  int ifh = 14;           /* input height, "H" */
  int nImg = 1;           /* mini-batch size, "N" */
  int nIfm = 256;         /* number of input feature maps, "C" */
  int nOfm = 512;         /* number of output feature maps, "K" */
  int kh = 3;             /* filter height, "R" */
  int kw = 3;             /* filter width, "S" */
  int pad = 1;            /* padding in output */
  int stride = 1;         /* stride when accessing inputs */
  int nSplits = 1;        /* splits */

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double flops = 0.0;
  int i;

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads();
#else
  int nThreads = 1;
#endif

  libxsmm_conv_desc conv_desc;
  libxsmm_conv_handle* libxsmm_handle;
  libxsmm_conv_layer* libxsmm_input;
  libxsmm_conv_layer* libxsmm_output;
  libxsmm_conv_filter* libxsmm_filter;
  libxsmm_conv_err_t status;

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters inpWidth inpHeight nImg nIfm nOfm kw kh pad stride splits\n", argv[0]);
    return 0;
  }
  srand48(1);

  /* reading new values from cli */
  i=1;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) ifw        = atoi(argv[i++]);
  if (argc > i) ifh        = atoi(argv[i++]);
  if (argc > i) nImg       = atoi(argv[i++]);
  if (argc > i) nIfm       = atoi(argv[i++]);
  if (argc > i) nOfm       = atoi(argv[i++]);
  if (argc > i) kw         = atoi(argv[i++]);
  if (argc > i) kh         = atoi(argv[i++]);
  if (argc > i) pad        = atoi(argv[i++]);
  if (argc > i) stride     = atoi(argv[i++]);
  if (argc > i) nSplits    = atoi(argv[i++]);

  stride_w = stride;
  stride_h = stride;
  pad_h = pad;
  pad_w = pad;

  /* deriving some values for naive code */
  ofh = (ifh - kh) / stride_h + 1;
  ofw = (ifw - kw) / stride_w + 1;
  ifhp = ifh;
  ifwp = ifw;
  ofhp = ofh + 2*pad_h;
  ofwp = ofw + 2*pad_w;

  /* set struct for naive convolution */
  naive_param.nImg = nImg;
  naive_param.nIfm = nIfm;
  naive_param.nOfm = nOfm;
  naive_param.ifhp = ifhp;
  naive_param.ifwp = ifwp;
  naive_param.ofhp = ofhp;
  naive_param.ofwp = ofwp;
  naive_param.ofh = ofh;
  naive_param.ofw = ofw;
  naive_param.pad_h = pad_h;
  naive_param.pad_w = pad_w;
  naive_param.kh = kh;
  naive_param.kw = kw;
  naive_param.stride_h = stride_h;
  naive_param.stride_w = stride_w;
  naive_param.nSplits = nSplits;

  /* print some summary */
  printf("##########################################\n");
  printf("#        Setting Up forward-prop         #\n");
  printf("##########################################\n");
  printf("PARAMS: W:%d  H:%d  N:%d  C:%d  K:%d  R:%d  S:%d  STRIDE:%d\n", ifw, ifh, nImg, nIfm, nOfm, kw, kh, stride);
  printf("PARAMS: ITERS:%d  Threads:%d\n", iters, nThreads);
  printf(" InImg %dx%d Padded (%dx%d)\n", ifh, ifw, ifhp, ifwp);
  printf("OutImg %dx%d Padded (%dx%d)\n", ofh, ofw, ofhp, ofwp);
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIfm*ifhp*ifwp*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOfm*ofhp*ofwp*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nIfm*ifhp*ifwp*   sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nOfm*ofhp*ofwp*   sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Weight     : %10.2f MiB\n", (double)(nIfm*nOfm*kw*kh*    sizeof(float))/(1024.0*1024.0) );

  /* allocate data */
  naive_input =          (float*)malloc(nImg*nIfm*ifhp*ifwp*sizeof(float));
  naive_output =         (float*)malloc(nImg*nOfm*ofhp*ofwp*sizeof(float));
  naive_libxsmm_output = (float*)malloc(nImg*nOfm*ofhp*ofwp*sizeof(float));
  naive_filter =         (float*)malloc(nOfm*nIfm*kh*kw*    sizeof(float));

  /* init data */
  init_buf(naive_input,          nImg*nIfm*ifhp*ifwp, 0, 0);
  zero_buf(naive_output,         nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_libxsmm_output, nImg*nOfm*ofhp*ofwp);
  init_buf(naive_filter,         nOfm*nIfm*kh*kw, 0, 0);

  /* setup LIBXSMM handle */
  conv_desc.N = nImg;
  conv_desc.C = nIfm;
  conv_desc.H = ifh;
  conv_desc.W = ifw;
  conv_desc.K = nOfm;
  conv_desc.R = kh;
  conv_desc.S = kw;
  conv_desc.u = stride_h;
  conv_desc.v = stride_w;
  conv_desc.pad_h = pad_h;
  conv_desc.pad_w = pad_h;
  conv_desc.splits = nSplits;
  libxsmm_handle = libxsmm_conv_create_handle_check( conv_desc, LIBXSMM_CONV_DATATYPE_FP32, LIBXSMM_CONV_ALGO_DIRECT, &status );
  CHKERR_LIBXSMM_CONV( status );

  /* setup LIBXSMM layers */
  libxsmm_input = libxsmm_conv_create_input_layer_check( libxsmm_handle, &status );
  CHKERR_LIBXSMM_CONV( status );
  libxsmm_output = libxsmm_conv_create_output_layer_check( libxsmm_handle, &status );
  CHKERR_LIBXSMM_CONV( status );
  libxsmm_filter = libxsmm_conv_create_filter_check( libxsmm_handle, &status );
  CHKERR_LIBXSMM_CONV( status );

  /* copy in data to LIBXSMM format */
  CHKERR_LIBXSMM_CONV( libxsmm_conv_copyin_layer( libxsmm_input, (void*)naive_input ) );
  CHKERR_LIBXSMM_CONV( libxsmm_conv_zero_layer( libxsmm_output ) );
  CHKERR_LIBXSMM_CONV( libxsmm_conv_copyin_filter( libxsmm_filter, (void*)naive_filter ) );

  /* bind layer to handle */
  CHKERR_LIBXSMM_CONV( libxsmm_conv_bind_input_layer( libxsmm_handle, libxsmm_input ) );
  CHKERR_LIBXSMM_CONV( libxsmm_conv_bind_output_layer( libxsmm_handle, libxsmm_output ) );
  CHKERR_LIBXSMM_CONV( libxsmm_conv_bind_filter( libxsmm_handle, libxsmm_filter ) );

  printf("##########################################\n");
  printf("#           Check Correctness            #\n");
  printf("##########################################\n");
  /* run naive convolution */
  naive_conv_fp(&naive_param, naive_input, naive_output, naive_filter);
  /* run LIBXSMM convolutions */
# pragma omp parallel
  {
    const int tid = omp_get_thread_num(), nthreads = omp_get_num_threads();
    CHKERR_LIBXSMM_CONV( libxsmm_convolve_st( libxsmm_handle, LIBXSMM_CONV_KIND_FWD, 0, tid, nthreads ) );
  }
  /* copy out data */
  CHKERR_LIBXSMM_CONV( libxsmm_conv_copyout_layer( libxsmm_output, (void*)naive_libxsmm_output ) );

  /* compare */
  compare_buf(naive_output, naive_libxsmm_output, nImg*nOfm*ofhp*ofwp, &norms);
  printf("             1-norm of reference: %f\n", norms.one_norm_ref);
  printf("              1-norm of JIT-code: %f\n", norms.one_norm_test);
  printf("       L2-error-norm of JIT-code: %f\n", norms.l2_rel_err);
  printf("    inf-norm of comp. rel. error: %f\n", norms.max_rel_err);
  printf("    inf-norm of comp. abs. error: %f\n", norms.max_abs_err);
  printf("##########################################\n");
  printf("#            Performance Run             #\n");
  printf("##########################################\n");
  /* run LIBXSMM convolution for performance */
  l_start = libxsmm_timer_tick();
  for (i = 0; i < iters; ++i) {
#   pragma omp parallel
    {
      const int tid = omp_get_thread_num(), nthreads = omp_get_num_threads();
      libxsmm_convolve_st( libxsmm_handle, LIBXSMM_CONV_KIND_FWD, 0, tid, nthreads );
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

  printf("GFLOP  = %.5g\n", flops*1e-9);
  printf("fp time = %.5g\n", ((double)(l_total/iters)));
  printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

  printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%.5g,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
     ifw, ifh, kw, kh, stride, pad, nSplits, ((double)(l_total/iters)), (flops*1e-9)/l_total,
     (flops*1e-9)/l_total, norms.max_rel_err, norms.max_abs_err, norms.l2_rel_err, norms.one_norm_ref, norms.one_norm_test );

  /* clean-up */
  CHKERR_LIBXSMM_CONV( libxsmm_conv_destroy_layer( libxsmm_input ) );
  CHKERR_LIBXSMM_CONV( libxsmm_conv_destroy_layer( libxsmm_output ) );
  CHKERR_LIBXSMM_CONV( libxsmm_conv_destroy_filter( libxsmm_filter ) );
  CHKERR_LIBXSMM_CONV( libxsmm_conv_destroy_handle( libxsmm_handle ) );

  return 0;
}

