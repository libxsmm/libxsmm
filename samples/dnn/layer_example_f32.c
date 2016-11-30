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
/* Alexander Heinecke, Hans Pabst, Dhiraj Kalamkar,
   Rajkishore Barik (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
/* note: later on, this leads to (correct but) different than expected norm-values */
# define drand48() ((double)rand() / RAND_MAX)
# define srand48 srand
#endif

#define CHKERR_LIBXSMM_DNN(A) if ( A != LIBXSMM_DNN_SUCCESS ) fprintf(stderr, "%s\n", libxsmm_dnn_get_error(A) );

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
  int pad_h_in;
  int pad_w_in;
  int pad_h_out;
  int pad_w_out;
  int kh;
  int kw;
  int stride_h;
  int stride_w;
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

LIBXSMM_INLINE void copy_buf(float* src, float* dst, long size) {
  int i;
  for (i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}

LIBXSMM_INLINE void init_buf(float* buf, long size, int initPos, int initOne)
{
  int i;
  zero_buf(buf, size);
  for (i = 0; i < size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? drand48() : (0.05 - drand48()/10.0)));
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
    if (diff > 1.0) {
      printf("MISMATCH@ %3d: A=%12.8g  B=%12.8g (E:%12.4e)\n", i, ref[i], test[i], diff);
    }
#endif

  }
  norms->l2_rel_err = sqrt(norms->l2_rel_err);
}


LIBXSMM_INLINE void naive_copy_NCHW_to_NHWC(const float* nchw, float* nhwc, int N, int H, int W, int C)
{
  LIBXSMM_VLA_DECL(4,       float, output, nhwc, H, W, C);
  LIBXSMM_VLA_DECL(4, const float,  input, nchw, C, H, W);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( h = 0; h < H; h++ ) {
      for ( w = 0; w < W; w++ ) {
        for ( c = 0; c < C; c++ ) {
          LIBXSMM_VLA_ACCESS(4, output, n, h, w, c, H, W, C) =
          LIBXSMM_VLA_ACCESS(4,  input, n, c, h, w, C, H, W);
        }
      }
    }
  }
}


LIBXSMM_INLINE void naive_copy_NHWC_to_NCHW(const float* nhwc, float* nchw, int N, int H, int W, int C)
{
  LIBXSMM_VLA_DECL(4,       float, output, nchw, C, H, W);
  LIBXSMM_VLA_DECL(4, const float,  input, nhwc, H, W, C);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( h = 0; h < H; h++ ) {
      for ( w = 0; w < W; w++ ) {
        for ( c = 0; c < C; c++ ) {
          LIBXSMM_VLA_ACCESS(4, output, n, c, h, w, C, H, W) =
          LIBXSMM_VLA_ACCESS(4,  input, n, h, w, c, H, W, C);
        }
      }
    }
  }
}


LIBXSMM_INLINE void naive_copy_KCRS_to_RSCK(const float* kcrs, float* rsck, int R, int S, int C, int K)
{
  LIBXSMM_VLA_DECL(4,       float, output, rsck, S, C, K);
  LIBXSMM_VLA_DECL(4, const float,  input, kcrs, C, R, S);
  int r, s, c, k;

  for ( r = 0; r < R; r++ ) {
    for ( s = 0; s < S; s++ ) {
      for ( c = 0; c < C; c++ ) {
        for ( k = 0; k < K; k++ ) {
          LIBXSMM_VLA_ACCESS(4, output, r, s, c, k, S, C, K) =
          LIBXSMM_VLA_ACCESS(4,  input, k, c, r, s, C, R, S);
        }
      }
    }
  }
}


LIBXSMM_INLINE void naive_copy_RSCK_to_KCRS(const float* rsck, float* kcrs, int R, int S, int C, int K)
{
  LIBXSMM_VLA_DECL(4, const float,  input, rsck, S, C, K);
  LIBXSMM_VLA_DECL(4,       float, output, kcrs, C, R, S);
  int r, s, c, k;

  for ( r = 0; r < R; r++ ) {
    for ( s = 0; s < S; s++ ) {
      for ( c = 0; c < C; c++ ) {
        for ( k = 0; k < K; k++ ) {
          LIBXSMM_VLA_ACCESS(4, output, k, c, r, s, C, R, S) =
            LIBXSMM_VLA_ACCESS(4,  input, r, s, c, k, S, C, K);
        }
      }
    }
  }
}


LIBXSMM_INLINE void naive_conv_fp(naive_conv_t* param, const float* input, float* output, const float* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4,       float, output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const float,  input_t,  input, nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const float, filter_t, filter, nIfm, kh, kw);

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
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
                LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) +=
                  LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_conv_bp(naive_conv_t* param, float* input, const float* output, const float* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4, const float, output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4,       float,  input_t,  input, nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const float, filter_t, filter, nIfm, kh, kw);

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for(img = 0; img < nImg; ++img) {
    for(ifm = 0; ifm < nIfm; ++ifm) {
      for(ofm = 0; ofm < nOfm; ++ofm) {
        for(oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h;
          for(oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w;
            for(kj = 0; kj < kh; ++kj) {
              for(ki = 0; ki < kw; ++ki) {
                LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp) +=
                  LIBXSMM_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp)
                * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_conv_wu(naive_conv_t* param, const float* input, const float* output, float* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4, const float, output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const float,  input_t,  input, nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4,       float, filter_t, filter, nIfm, kh, kw);

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for(ofm = 0; ofm < nOfm; ++ofm) {
    for(ifm = 0; ifm < nIfm; ++ifm) {
      for(img = 0; img < nImg; ++img) {
        for(oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h;
          for(oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w;
            for(kj = 0; kj < kh; ++kj) {
              for(ki = 0; ki < kw; ++ki) {
                LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw) +=
                  LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                * LIBXSMM_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp);
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
  float *naive_input, *naive_output, *naive_filter, *naive_filter_wu, *naive_libxsmm_output;
  float *naive_libxsmm_input, *naive_libxsmm_filter, *naive_input_save, *naive_filter_save, *naive_filter_kcrs;
  float *input_nhwc, *output_nhwc, *filter_rsck, *naive_output_nhwc, *naive_input_nhwc;
  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w, pad_h_out, pad_w_out;
  naive_conv_t naive_param;
  correctness_t norms_fwd, norms_bwd, norms_upd;

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
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
#if defined(_OPENMP)
  int nThreads = omp_get_max_threads();       /* number of threads */
#else
  int nThreads = 1;       /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double flops = 0.0;
  int i;

  libxsmm_dnn_conv_desc conv_desc;
  libxsmm_dnn_conv_handle* libxsmm_handle;
  libxsmm_dnn_buffer* libxsmm_input;
  libxsmm_dnn_buffer* libxsmm_output;
  libxsmm_dnn_filter* libxsmm_filter;
  libxsmm_dnn_err_t status;

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters inpWidth inpHeight nImg nIfm nOfm kw kh pad stride type\n", argv[0]);
    return 0;
  }
  srand48(1);

  /* reading new values from cli */
  i = 1;
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
  if (argc > i) type       = *(argv[i++]);

  if (type != 'A' && type != 'F' && type != 'B' && type != 'U') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only), 'U' (WU only)\n");
    return 0;
  }

  stride_w = stride;
  stride_h = stride;
  pad_h_out = pad;
  pad_w_out = pad;

  /* deriving some values for naive code */
  ofh = (ifh - kh) / stride_h + 1;
  ofw = (ifw - kw) / stride_w + 1;
  ifhp = ifh;
  ifwp = ifw;
  ofhp = ofh + 2 * pad_h_out;
  ofwp = ofw + 2 * pad_w_out;

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
  naive_param.pad_h_in = 0;
  naive_param.pad_w_in = 0;
  naive_param.pad_h_out = pad_h_out;
  naive_param.pad_w_out = pad_w_out;
  naive_param.kh = kh;
  naive_param.kw = kw;
  naive_param.stride_h = stride_h;
  naive_param.stride_w = stride_w;

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
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
  naive_input           = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_input_save      = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_output          = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_libxsmm_output  = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_libxsmm_input   = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_filter          = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_filter_save     = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_filter_wu       = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_filter_kcrs     = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_libxsmm_filter  = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  input_nhwc            = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  output_nhwc           = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_output_nhwc     = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_input_nhwc      = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  filter_rsck           = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);

  /* initialize data */
  init_buf(naive_input,          nImg*nIfm*ifhp*ifwp, 0, 0);
  copy_buf(naive_input, naive_input_save, nImg*nIfm*ifhp*ifwp);
  zero_buf(naive_output,         nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_libxsmm_output, nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_libxsmm_input,  nImg*nIfm*ifhp*ifwp);
  init_buf(naive_filter,         nOfm*nIfm*kh*kw, 0, 0);
  copy_buf(naive_filter, naive_filter_wu, nOfm*nIfm*kh*kw);
  zero_buf(naive_libxsmm_filter, nOfm*nIfm*kh*kw);
  naive_copy_NCHW_to_NHWC(naive_input, input_nhwc, nImg, ifhp, ifwp, nIfm);
  zero_buf(output_nhwc,          nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_output_nhwc,    nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_input_nhwc,     nImg*nIfm*ifhp*ifwp);
  naive_copy_KCRS_to_RSCK(naive_filter, filter_rsck, kh, kw, nIfm, nOfm);

  printf("\n");
  printf("##########################################\n");
  printf("#      Setting Up  (custom-Storage)      #\n");
  printf("##########################################\n");

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
  /* @TODO we need to change the interface to provide CAFFE compatible padding! */
  conv_desc.pad_h_in = 0;
  conv_desc.pad_w_in = 0;
  conv_desc.pad_h_out = pad_h_out;
  conv_desc.pad_w_out = pad_w_out;
  conv_desc.threads = nThreads;
  conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_AUTO;
  conv_desc.buffer_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
  conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
  conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
  conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
  conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
  conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;

  libxsmm_handle = libxsmm_dnn_create_conv_handle_check( conv_desc, &status );
  CHKERR_LIBXSMM_DNN( status );

  /* setup LIBXSMM buffers and filter */
  libxsmm_input = libxsmm_dnn_create_input_buffer_check( libxsmm_handle, &status );
  CHKERR_LIBXSMM_DNN( status );
  libxsmm_output = libxsmm_dnn_create_output_buffer_check( libxsmm_handle, &status );
  CHKERR_LIBXSMM_DNN( status );
  libxsmm_filter = libxsmm_dnn_create_filter_check( libxsmm_handle, &status );
  CHKERR_LIBXSMM_DNN( status );

  /* copy in data to LIBXSMM format */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_buffer( libxsmm_input, (void*)naive_input, LIBXSMM_DNN_CONV_FORMAT_NCHW ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_zero_buffer( libxsmm_output ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_filter( libxsmm_filter, (void*)naive_filter, LIBXSMM_DNN_CONV_FORMAT_KCRS ) );

  /* bind buffers and filter to handle */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_input_buffer( libxsmm_handle, libxsmm_input ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_output_buffer( libxsmm_handle, libxsmm_output ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_filter( libxsmm_handle, libxsmm_filter ) );

  if (type == 'A' || type == 'F') {
    printf("##########################################\n");
    printf("#   Correctness - FWD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run naive convolution */
    naive_conv_fp(&naive_param, naive_input, naive_output, naive_filter);
    /* run LIBXSMM convolutions */
#if defined(_OPENMP)
# pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_FWD, 0, tid ) );
    }
    /* copy out data */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_buffer( libxsmm_output, (void*)naive_libxsmm_output, LIBXSMM_DNN_CONV_FORMAT_NCHW ) );

    /* compare */
    compare_buf(naive_output, naive_libxsmm_output, nImg*nOfm*ofhp*ofwp, &norms_fwd);
    printf("             1-norm of reference: %f\n", norms_fwd.one_norm_ref);
    printf("              1-norm of JIT-code: %f\n", norms_fwd.one_norm_test);
    printf("       L2-error-norm of JIT-code: %f\n", norms_fwd.l2_rel_err);
    printf("    inf-norm of comp. rel. error: %f\n", norms_fwd.max_rel_err);
    printf("    inf-norm of comp. abs. error: %f\n", norms_fwd.max_abs_err);
  }

  if ( (type == 'A' || type == 'B') && (stride == 1 && nIfm > 3) ) {
    printf("##########################################\n");
    printf("#   Correctness - BWD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run naive convolution */
    naive_conv_bp(&naive_param, naive_input, naive_output, naive_filter);
    /* run LIBXSMM convolutions */
#if defined(_OPENMP)
# pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_BWD, 0, tid ) );
    }
    /* copy out data */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_buffer( libxsmm_input, (void*)naive_libxsmm_input, LIBXSMM_DNN_CONV_FORMAT_NCHW ) );

    /* compare */
    compare_buf(naive_input, naive_libxsmm_input, nImg*nIfm*ifhp*ifwp, &norms_bwd);
    printf("             1-norm of reference: %f\n", norms_bwd.one_norm_ref);
    printf("              1-norm of JIT-code: %f\n", norms_bwd.one_norm_test);
    printf("       L2-error-norm of JIT-code: %f\n", norms_bwd.l2_rel_err);
    printf("    inf-norm of comp. rel. error: %f\n", norms_bwd.max_rel_err);
    printf("    inf-norm of comp. abs. error: %f\n", norms_bwd.max_abs_err);
  }

  if (type == 'A' || type == 'U') {
    printf("##########################################\n");
    printf("#   Correctness - UPD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run naive convolution */
    naive_conv_wu(&naive_param, naive_input, naive_output, naive_filter_wu);
    /* run LIBXSMM convolutions */
#if defined(_OPENMP)
# pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_UPD, 0, tid ) );
    }
    /* copy out data */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_filter( libxsmm_filter, (void*)naive_libxsmm_filter, LIBXSMM_DNN_CONV_FORMAT_KCRS ) );

    /* compare */
    compare_buf(naive_filter_wu, naive_libxsmm_filter, nOfm*nIfm*kh*kw, &norms_upd);
    printf("             1-norm of reference: %f\n", norms_upd.one_norm_ref);
    printf("              1-norm of JIT-code: %f\n", norms_upd.one_norm_test);
    printf("       L2-error-norm of JIT-code: %f\n", norms_upd.l2_rel_err);
    printf("    inf-norm of comp. rel. error: %f\n", norms_upd.max_rel_err);
    printf("    inf-norm of comp. abs. error: %f\n", norms_upd.max_abs_err);
  }

  if (type == 'A' || type == 'F') {
    printf("##########################################\n");
    printf("#   Performance - FWD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_FWD, 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

    printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
       ifw, ifh, kw, kh, stride, pad, ((double)(l_total/iters)), (flops*1e-9)/l_total,
       norms_fwd.max_rel_err, norms_fwd.max_abs_err, norms_fwd.l2_rel_err, norms_fwd.one_norm_ref, norms_fwd.one_norm_test );
  }

  if ( (type == 'A' || type == 'B') && (stride == 1 && nIfm > 3) ) {
    printf("##########################################\n");
    printf("#   Performance - BWD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_BWD, 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
    printf("bp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

    printf("PERFDUMP,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
       ifw, ifh, kw, kh, stride, pad, ((double)(l_total/iters)), (flops*1e-9)/l_total,
       norms_bwd.max_rel_err, norms_bwd.max_abs_err, norms_bwd.l2_rel_err, norms_bwd.one_norm_ref, norms_bwd.one_norm_test );
  }

  if (type == 'A' || type == 'U') {
    printf("##########################################\n");
    printf("#   Performance - UPD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_UPD, 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
    printf("wu time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

    printf("PERFDUMP,WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
       ifw, ifh, kw, kh, stride, pad, ((double)(l_total/iters)), (flops*1e-9)/l_total,
       norms_upd.max_rel_err, norms_upd.max_abs_err, norms_upd.l2_rel_err, norms_upd.one_norm_ref, norms_upd.one_norm_test );
  }

  /* clean-up */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_buffer( libxsmm_input ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_buffer( libxsmm_output ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_filter( libxsmm_filter ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_conv_handle( libxsmm_handle ) );

  printf("\n");
  printf("##########################################\n");
  printf("#  Setting Up - FWD (NHWC/RSCK-Storage)  #\n");
  printf("##########################################\n");

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
  /* @TODO we need to change the interface to provide CAFFE compatible padding! */
  conv_desc.pad_h_in = 0;
  conv_desc.pad_w_in = 0;
  conv_desc.pad_h_out = pad_h_out;
  conv_desc.pad_w_out = pad_w_out;
  conv_desc.threads = nThreads;
  conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_AUTO;
  conv_desc.buffer_format = LIBXSMM_DNN_CONV_FORMAT_NHWC;
  conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_RSCK;
  conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
  conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
  conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
  conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;

  libxsmm_handle = libxsmm_dnn_create_conv_handle_check( conv_desc, &status );
  CHKERR_LIBXSMM_DNN( status );

  /* setup LIBXSMM buffers and filter */
  libxsmm_input = libxsmm_dnn_link_input_buffer_check( libxsmm_handle, input_nhwc, LIBXSMM_DNN_CONV_FORMAT_NHWC_PTR, &status );
  CHKERR_LIBXSMM_DNN( status );
  libxsmm_output = libxsmm_dnn_link_output_buffer_check( libxsmm_handle, output_nhwc, LIBXSMM_DNN_CONV_FORMAT_NHWC_PTR, &status );
  CHKERR_LIBXSMM_DNN( status );
  libxsmm_filter = libxsmm_dnn_link_filter_check( libxsmm_handle, filter_rsck, LIBXSMM_DNN_CONV_FORMAT_RSCK_PTR, &status );
  CHKERR_LIBXSMM_DNN( status );

  /* bind buffers and filter to handle */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_input_buffer( libxsmm_handle, libxsmm_input ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_output_buffer( libxsmm_handle, libxsmm_output ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_filter( libxsmm_handle, libxsmm_filter ) );

  if (type == 'A' || type == 'F') {
    printf("##########################################\n");
    printf("#  Correctness - FWD (NHWC/RSCK-Storage) #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolutions */
#if defined(_OPENMP)
# pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_FWD, 0, tid ) );
    }
    /* copy output data into NCHW storage in user code */
    naive_copy_NHWC_to_NCHW(output_nhwc, naive_output_nhwc, nImg, ofhp, ofwp, nOfm);

    /* compare */
    compare_buf(naive_output, naive_output_nhwc, nImg*nOfm*ofhp*ofwp, &norms_fwd);
    printf("             1-norm of reference: %f\n", norms_fwd.one_norm_ref);
    printf("              1-norm of JIT-code: %f\n", norms_fwd.one_norm_test);
    printf("       L2-error-norm of JIT-code: %f\n", norms_fwd.l2_rel_err);
    printf("    inf-norm of comp. rel. error: %f\n", norms_fwd.max_rel_err);
    printf("    inf-norm of comp. abs. error: %f\n", norms_fwd.max_abs_err);
  }

  if ( (type == 'A' || type == 'B') && (stride == 1 && nIfm > 3) ) {
    printf("##########################################\n");
    printf("# Correctness - BWD (NHWC/RSCK-Storage)  #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolutions */
#if defined(_OPENMP)
# pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_BWD, 0, tid ) );
    }
    /* copy input data into NCHW storage in user code */
    naive_copy_NHWC_to_NCHW(input_nhwc, naive_input_nhwc, nImg, ifhp, ifwp, nIfm);

    /* compare */
    compare_buf(naive_input, naive_input_nhwc, nImg*nIfm*ifhp*ifwp, &norms_bwd);
    printf("             1-norm of reference: %f\n", norms_bwd.one_norm_ref);
    printf("              1-norm of JIT-code: %f\n", norms_bwd.one_norm_test);
    printf("       L2-error-norm of JIT-code: %f\n", norms_bwd.l2_rel_err);
    printf("    inf-norm of comp. rel. error: %f\n", norms_bwd.max_rel_err);
    printf("    inf-norm of comp. abs. error: %f\n", norms_bwd.max_abs_err);
  }

  if (type == 'A' || type == 'U') {
    printf("##########################################\n");
    printf("# Correctness - UPD (NHWC/RSCK-Storage)  #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolutions */
#if defined(_OPENMP)
# pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_UPD, 0, tid ) );
    }
    /* copy input data into KCRS storage in user code */
    naive_copy_RSCK_to_KCRS(filter_rsck, naive_filter_kcrs, kh, kw, nIfm, nOfm);

    /* compare */
    compare_buf(naive_filter_wu, naive_filter_kcrs, nOfm*nIfm*kh*kw, &norms_upd);
    printf("             1-norm of reference: %f\n", norms_upd.one_norm_ref);
    printf("              1-norm of JIT-code: %f\n", norms_upd.one_norm_test);
    printf("       L2-error-norm of JIT-code: %f\n", norms_upd.l2_rel_err);
    printf("    inf-norm of comp. rel. error: %f\n", norms_upd.max_rel_err);
    printf("    inf-norm of comp. abs. error: %f\n", norms_upd.max_abs_err);
  }

  if (type == 'A' || type == 'F') {
    printf("##########################################\n");
    printf("#  Performance - FWD (NHWC/RSCK-Storage) #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
     {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_FWD, 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GFLOP (NHWC,RSCK)  = %.5g\n", flops*1e-9/(double)iters);
    printf("fp time (NHWC,RSCK) = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS (NHWC,RSCK) = %.5g\n", (flops*1e-9)/l_total);

    printf("PERFDUMP-NHWC-RSCK,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
       ifw, ifh, kw, kh, stride, pad, ((double)(l_total/iters)), (flops*1e-9)/l_total,
       norms_fwd.max_rel_err, norms_fwd.max_abs_err, norms_fwd.l2_rel_err, norms_fwd.one_norm_ref, norms_fwd.one_norm_test );
  }

  if ( (type == 'A' || type == 'B') && (stride == 1 && nIfm > 3) ) {
    printf("##########################################\n");
    printf("#  Performance - BWD (NHWC/RSCK-Storage) #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_BWD, 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GFLOP (NHWC,RSCK)  = %.5g\n", flops*1e-9/(double)iters);
    printf("fp time (NHWC,RSCK) = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS (NHWC,RSCK) = %.5g\n", (flops*1e-9)/l_total);

    printf("PERFDUMP-NHWC-RSCK,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
       ifw, ifh, kw, kh, stride, pad, ((double)(l_total/iters)), (flops*1e-9)/l_total,
       norms_bwd.max_rel_err, norms_bwd.max_abs_err, norms_bwd.l2_rel_err, norms_bwd.one_norm_ref, norms_bwd.one_norm_test );
  }

  if (type == 'A' || type == 'U') {
    printf("##########################################\n");
    printf("#  Performance - UPD (NHWC/RSCK-Storage) #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_UPD, 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GFLOP (NHWC,RSCK)  = %.5g\n", flops*1e-9/(double)iters);
    printf("fp time (NHWC,RSCK) = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS (NHWC,RSCK) = %.5g\n", (flops*1e-9)/l_total);

    printf("PERFDUMP-NHWC-RSCK,WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
       ifw, ifh, kw, kh, stride, pad, ((double)(l_total/iters)), (flops*1e-9)/l_total,
       norms_upd.max_rel_err, norms_upd.max_abs_err, norms_upd.l2_rel_err, norms_upd.one_norm_ref, norms_upd.one_norm_test );
  }

  /* clean-up */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_buffer( libxsmm_input ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_buffer( libxsmm_output ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_filter( libxsmm_filter ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_conv_handle( libxsmm_handle ) );

  printf("\n");
  printf("##########################################\n");
  printf("# Setting Up - FWD (NHWC/custom-Storage) #\n");
  printf("##########################################\n");

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
  /* @TODO we need to change the interface to provide CAFFE compatible padding! */
  conv_desc.pad_h_in = 0;
  conv_desc.pad_w_in = 0;
  conv_desc.pad_h_out = pad_h_out;
  conv_desc.pad_w_out = pad_w_out;
  conv_desc.threads = nThreads;
  conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_AUTO;
  conv_desc.buffer_format = LIBXSMM_DNN_CONV_FORMAT_NHWC;
  conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
  conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
  conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
  conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
  conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;

  libxsmm_handle = libxsmm_dnn_create_conv_handle_check( conv_desc, &status );
  CHKERR_LIBXSMM_DNN( status );

  /* zero output buffer again */
  zero_buf(output_nhwc,          nImg*nOfm*ofhp*ofwp);
  /* init input agian */
  naive_copy_NCHW_to_NHWC(naive_input_save, input_nhwc, nImg, ifhp, ifwp, nIfm);

  /* setup LIBXSMM buffers and filter */
  libxsmm_input = libxsmm_dnn_link_input_buffer_check( libxsmm_handle, input_nhwc, LIBXSMM_DNN_CONV_FORMAT_NHWC_PTR, &status );
  CHKERR_LIBXSMM_DNN( status );
  libxsmm_output = libxsmm_dnn_link_output_buffer_check( libxsmm_handle, output_nhwc, LIBXSMM_DNN_CONV_FORMAT_NHWC_PTR, &status );
  CHKERR_LIBXSMM_DNN( status );
  libxsmm_filter = libxsmm_dnn_create_filter_check( libxsmm_handle, &status );
  CHKERR_LIBXSMM_DNN( status );

  /* copy in data to LIBXSMM format */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_filter( libxsmm_filter, (void*)naive_filter, LIBXSMM_DNN_CONV_FORMAT_KCRS ) );

  /* bind buffers and filter to handle */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_input_buffer( libxsmm_handle, libxsmm_input ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_output_buffer( libxsmm_handle, libxsmm_output ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_filter( libxsmm_handle, libxsmm_filter ) );

  if (type == 'A' || type == 'F') {
    printf("##########################################\n");
    printf("# Correctness - FWD(NHWC/custom-Storage) #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolutions */
#if defined(_OPENMP)
# pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_FWD, 0, tid ) );
    }
    /* copy output data into NCHW storage in user code */
    naive_copy_NHWC_to_NCHW(output_nhwc, naive_output_nhwc, nImg, ofhp, ofwp, nOfm);

    /* compare */
    compare_buf(naive_output, naive_output_nhwc, nImg*nOfm*ofhp*ofwp, &norms_fwd);
    printf("             1-norm of reference: %f\n", norms_fwd.one_norm_ref);
    printf("              1-norm of JIT-code: %f\n", norms_fwd.one_norm_test);
    printf("       L2-error-norm of JIT-code: %f\n", norms_fwd.l2_rel_err);
    printf("    inf-norm of comp. rel. error: %f\n", norms_fwd.max_rel_err);
    printf("    inf-norm of comp. abs. error: %f\n", norms_fwd.max_abs_err);
  }

  if ( (type == 'A' || type == 'B') && (stride == 1 && nIfm > 3) ) {
    printf("##########################################\n");
    printf("# Correctness - BWD (NHWC/RSCK-Storage)  #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolutions */
#if defined(_OPENMP)
# pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_BWD, 0, tid ) );
    }
    /* copy input data into NCHW storage in user code */
    naive_copy_NHWC_to_NCHW(input_nhwc, naive_input_nhwc, nImg, ifhp, ifwp, nIfm);

    /* compare */
    compare_buf(naive_input, naive_input_nhwc, nImg*nIfm*ifhp*ifwp, &norms_bwd);
    printf("             1-norm of reference: %f\n", norms_bwd.one_norm_ref);
    printf("              1-norm of JIT-code: %f\n", norms_bwd.one_norm_test);
    printf("       L2-error-norm of JIT-code: %f\n", norms_bwd.l2_rel_err);
    printf("    inf-norm of comp. rel. error: %f\n", norms_bwd.max_rel_err);
    printf("    inf-norm of comp. abs. error: %f\n", norms_bwd.max_abs_err);
  }

  if (type == 'A' || type == 'U') {
    printf("##########################################\n");
    printf("# Correctness - UPD(NHWC/custom-Storage) #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolutions */
#if defined(_OPENMP)
# pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_UPD, 0, tid ) );
    }
    /* copy out data */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_filter( libxsmm_filter, (void*)naive_libxsmm_filter, LIBXSMM_DNN_CONV_FORMAT_KCRS ) );

    /* compare */
    compare_buf(naive_filter_wu, naive_libxsmm_filter, nOfm*nIfm*kh*kw, &norms_upd);
    printf("             1-norm of reference: %f\n", norms_upd.one_norm_ref);
    printf("              1-norm of JIT-code: %f\n", norms_upd.one_norm_test);
    printf("       L2-error-norm of JIT-code: %f\n", norms_upd.l2_rel_err);
    printf("    inf-norm of comp. rel. error: %f\n", norms_upd.max_rel_err);
    printf("    inf-norm of comp. abs. error: %f\n", norms_upd.max_abs_err);
  }

  if (type == 'A' || type == 'F') {
    printf("##########################################\n");
    printf("# Performance - FWD(NHWC/custom-Storage) #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_FWD, 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GFLOP (NHWC,custom)  = %.5g\n", flops*1e-9/(double)iters);
    printf("fp time (NHWC,custom) = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS (NHWC,custom) = %.5g\n", (flops*1e-9)/l_total);

    printf("PERFDUMP-NHWC-custom,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
       ifw, ifh, kw, kh, stride, pad, ((double)(l_total/iters)), (flops*1e-9)/l_total,
       norms_fwd.max_rel_err, norms_fwd.max_abs_err, norms_fwd.l2_rel_err, norms_fwd.one_norm_ref, norms_fwd.one_norm_test );
  }

  if ( (type == 'A' || type == 'B') && (stride == 1 && nIfm > 3) ) {
    printf("##########################################\n");
    printf("# Performance - BWD(NHWC/custom-Storage) #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_BWD, 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GFLOP (NHWC,custom)  = %.5g\n", flops*1e-9/(double)iters);
    printf("fp time (NHWC,custom) = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS (NHWC,custom) = %.5g\n", (flops*1e-9)/l_total);

    printf("PERFDUMP-NHWC-custom,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
       ifw, ifh, kw, kh, stride, pad, ((double)(l_total/iters)), (flops*1e-9)/l_total,
       norms_bwd.max_rel_err, norms_bwd.max_abs_err, norms_bwd.l2_rel_err, norms_bwd.one_norm_ref, norms_bwd.one_norm_test );
  }

  if (type == 'A' || type == 'U') {
    printf("##########################################\n");
    printf("# Performance - UPD(NHWC/custom-Storage) #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxsmm_dnn_convolve_st( libxsmm_handle, LIBXSMM_DNN_CONV_KIND_UPD, 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GFLOP (NHWC,custom)  = %.5g\n", flops*1e-9/(double)iters);
    printf("fp time (NHWC,custom) = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS (NHWC,custom) = %.5g\n", (flops*1e-9)/l_total);

    printf("PERFDUMP-NHWC-custom,WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
       ifw, ifh, kw, kh, stride, pad, ((double)(l_total/iters)), (flops*1e-9)/l_total,
       norms_upd.max_rel_err, norms_upd.max_abs_err, norms_upd.l2_rel_err, norms_upd.one_norm_ref, norms_upd.one_norm_test );
  }

  /* clean-up */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_buffer( libxsmm_input ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_buffer( libxsmm_output ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_filter( libxsmm_filter ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_conv_handle( libxsmm_handle ) );

  /* deallocate data */
  libxsmm_free(naive_input);
  libxsmm_free(naive_input_save);
  libxsmm_free(naive_output);
  libxsmm_free(naive_libxsmm_output);
  libxsmm_free(naive_libxsmm_input);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_filter_save);
  libxsmm_free(naive_filter_wu);
  libxsmm_free(naive_filter_kcrs);
  libxsmm_free(naive_libxsmm_filter);
  libxsmm_free(input_nhwc);
  libxsmm_free(output_nhwc);
  libxsmm_free(naive_output_nhwc);
  libxsmm_free(naive_input_nhwc);
  libxsmm_free(filter_rsck);

  /* some empty lines at the end */
  printf("\n\n\n");

  return 0;
}

