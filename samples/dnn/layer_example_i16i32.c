/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Alexander Heinecke, Hans Pabst, Dhiraj Kalamkar, Ankush Mandal (Intel Corp.)
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
  int ifh;
  int ifw;
  int ofhp;
  int ofwp;
  int ofh;
  int ofw;
  int pad_h;
  int pad_w;
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

LIBXSMM_INLINE void zero_buf_int16(short* buf, long size) {
  int i;
  for (i = 0; i < size; ++i) {
    buf[i] = 0;
  }
}

LIBXSMM_INLINE void copy_buf_int16(short* src, short* dst, long size) {
  int i;
  for (i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}

LIBXSMM_INLINE void init_buf_int16(short* buf, long size, int initPos, int initOne)
{
  int i;
  zero_buf_int16(buf, size);
  for (i = 0; i < size; ++i) {
    buf[i] = (short)((initOne != 0) ? 1 : ((initPos != 0) ? (rand()%7) : (rand()%7)-3));
  }
}

LIBXSMM_INLINE void set_zeropad_nchw_int16(short* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, short, input, nchw, C, H, W);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          if (h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w)
            LIBXSMM_VLA_ACCESS(4,  input, n, c, h, w, C, H, W) = 0;
        }
      }
    }
  }
}

LIBXSMM_INLINE void compare_buf_int16(short* ref, short* test, long size, correctness_t* norms)
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

LIBXSMM_INLINE void naive_conv_fp_int16(naive_conv_t* param, const short* input, short* output, const short* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  int* output_itm = (int*)libxsmm_aligned_malloc(nImg * nOfm * ofhp * ofwp * sizeof(int), 64);

  LIBXSMM_VLA_DECL(4,         int, output_itm_t, output_itm + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4,       short,     output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const short,      input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const short,     filter_t, filter, nIfm, kh, kw);

  /* up convert */
#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, oj, oi)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (oj = 0; oj < ofh; ++oj) {
        for (oi = 0; oi < ofw; ++oi) {
          LIBXSMM_VLA_ACCESS(  4, output_itm_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) =
           (int)LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp);
        }
      }
    }
  }

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(  4, output_itm_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) += (int)
                  LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }
    }
  }

  /* down convert */
#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, oj, oi)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (oj = 0; oj < ofh; ++oj) {
        for (oi = 0; oi < ofw; ++oi) {
          LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) =
           (short)LIBXSMM_VLA_ACCESS(  4, output_itm_t, img, ofm, oj, oi, nOfm, ofhp, ofwp);
        }
      }
    }
  }

  libxsmm_free(output_itm);
}

LIBXSMM_INLINE void naive_conv_bp_int16(naive_conv_t* param, short* input, const short* output, const short* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  int* input_itm = (int*)libxsmm_aligned_malloc(nImg * nIfm * ifhp * ifwp * sizeof(int), 64);

  LIBXSMM_VLA_DECL(4,         int,  input_itm_t, input_itm + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const short,     output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4,       short,      input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const short,     filter_t, filter, nIfm, kh, kw);

  /* up convert */
#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ifm, ij, ii)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ifm = 0; ifm < nIfm; ++ifm) {
      for (ij = 0; ij < ifh; ++ij) {
        for (ii = 0; ii < ifw; ++ii) {
          LIBXSMM_VLA_ACCESS(  4, input_itm_t, img, ifm, ij, ii, nIfm, ifhp, ifwp) =
           (int)LIBXSMM_VLA_ACCESS(  4, input_t, img, ifm, ij, ii, nIfm, ifhp, ifwp);
        }
      }
    }
  }

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ifm = 0; ifm < nIfm; ++ifm) {
      for (ofm = 0; ofm < nOfm; ++ofm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                  LIBXSMM_VLA_ACCESS(4,  input_itm_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp) += (int)
                  (LIBXSMM_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp)
                    * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw));
              }
            }
          }
        }
      }
    }
  }

  /* down convert */
#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ifm, ij, ii)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ifm = 0; ifm < nIfm; ++ifm) {
      for (ij = 0; ij < ifh; ++ij) {
        for (ii = 0; ii < ifw; ++ii) {
          LIBXSMM_VLA_ACCESS(  4, input_t, img, ifm, ij, ii, nIfm, ifhp, ifwp) =
           (short)LIBXSMM_VLA_ACCESS(  4, input_itm_t, img, ifm, ij, ii, nIfm, ifhp, ifwp);
        }
      }
    }
  }

  libxsmm_free(input_itm);
}

int main(int argc, char* argv[])
{
  short *naive_input, *naive_output, *naive_filter;
  short *naive_output_bp, *naive_input_save;
  short *naive_libxsmm_input, *naive_libxsmm_output;
  short *input_libxsmm, *filter_libxsmm;
  short *output_libxsmm;
  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out;
  naive_conv_t naive_param;
  correctness_t norms_fwd, norms_bwd;
  void* scratch;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int ifw = 14;           /* input width, "W" */
  int ifh = 18;           /* input height, "H" */
  int nImg = 32;          /* mini-batch size, "N" */
  int nIfm = 256;         /* number of input feature maps, "C" */
  int nOfm = 512;         /* number of output feature maps, "K" */
  int kh = 3;             /* filter height, "R" */
  int kw = 3;             /* filter width, "S" */
  int pad = 2;            /* padding in output */
  int stride = 1;         /* stride when accessing inputs */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
#if defined(_OPENMP)
  int nThreads = omp_get_max_threads();       /* number of threads */
#else
  int nThreads = 1;       /* number of threads */
#endif
  int padding_mode = 0;   /* padding mode */

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double lpOps = 0.0; /* number of low precision operations */
  int i;

  libxsmm_dnn_conv_desc conv_desc;
  libxsmm_dnn_layer* libxsmm_handle;
  libxsmm_dnn_buffer* libxsmm_input;
  libxsmm_dnn_buffer* libxsmm_output;
  libxsmm_dnn_filter* libxsmm_filter;
  libxsmm_dnn_err_t status;

  memset(&norms_fwd, 0, sizeof(norms_fwd));
  memset(&norms_bwd, 0, sizeof(norms_bwd));

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters inpWidth inpHeight nImg nIfm nOfm kw kh pad stride type padding_mode\n", argv[0]);
    return 0;
  }
  srand(1);

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
  if (argc > i) padding_mode = atoi(argv[i++]);

  if (type != 'A' && type != 'F' && type != 'B'/* && type != 'U'*/) {
    /*printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only), 'U' (WU only)\n");*/
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only)\n");
    return 0;
  }

  stride_w = stride;
  stride_h = stride;
  pad_h = pad;
  pad_w = pad;

  if (padding_mode == 1) {
    pad_h_in = pad_h;
    pad_w_in = pad_w;
  } else {
    pad_h_in = 0;
    pad_w_in = 0;
  }

  pad_h_out = 0;
  pad_w_out = 0;

  /* deriving some values for naive code */
  ofh = (ifh + 2 * pad_h - kh) / stride_h + 1;
  ofw = (ifw + 2 * pad_w - kw) / stride_w + 1;
  ifhp = ifh + 2 * pad_h_in;
  ifwp = ifw + 2 * pad_w_in;
  ofhp = ofh + 2 * pad_h_out;
  ofwp = ofw + 2 * pad_w_out;

  /* set struct for naive convolution */
  naive_param.nImg = nImg;
  naive_param.nIfm = nIfm;
  naive_param.nOfm = nOfm;
  naive_param.ifhp = ifhp;
  naive_param.ifwp = ifwp;
  naive_param.ifh = ifh;
  naive_param.ifw = ifw;
  naive_param.ofhp = ofhp;
  naive_param.ofwp = ofwp;
  naive_param.ofh = ofh;
  naive_param.ofw = ofw;
  naive_param.pad_h = pad_h;
  naive_param.pad_w = pad_w;
  naive_param.pad_h_in = pad_h_in;
  naive_param.pad_w_in = pad_w_in;
  naive_param.pad_h_out = pad_h_out;
  naive_param.pad_w_out = pad_w_out;
  naive_param.kh = kh;
  naive_param.kw = kw;
  naive_param.stride_h = stride_h;
  naive_param.stride_w = stride_w;

  /* print some summary */
  printf("##########################################\n");
  printf("#    Setting Up Common    #\n");
  printf("##########################################\n");
  printf("PARAMS: W:%d  H:%d  N:%d  C:%d  K:%d  R:%d  S:%d  P:%d  Q:%d  STRIDE:%d\n", ifw, ifh, nImg, nIfm, nOfm, kw, kh, ofh, ofw, stride);
  printf("PARAMS: ITERS:%d  Threads:%d\n", iters, nThreads);
  printf(" InImg %dx%d Padded (%dx%d)\n", ifh, ifw, ifhp, ifwp);
  printf("OutImg %dx%d Padded (%dx%d)\n", ofh, ofw, ofhp, ofwp);
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIfm*ifhp*ifwp*sizeof(short))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOfm*ofhp*ofwp*sizeof(short))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nIfm*ifhp*ifwp*   sizeof(short))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nOfm*ofhp*ofwp*   sizeof(short))/(1024.0*1024.0) );
  printf("SIZE Weight     : %10.2f MiB\n", (double)(nIfm*nOfm*kw*kh*    sizeof(short))/(1024.0*1024.0) );

  /* allocate data */
  naive_input           = (short*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(short), 2097152);
  naive_input_save      = (short*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(short), 2097152);
  naive_output          = (short*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(short), 2097152);
  naive_output_bp       = (short*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(short), 2097152);
  naive_libxsmm_input   = (short*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(short), 2097152);
  naive_libxsmm_output  = (short*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(short), 2097152);
  naive_filter          = (short*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(short), 2097152);
  input_libxsmm         = (short*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(short), 2097152);
  filter_libxsmm        = (short*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(short), 2097152);
  output_libxsmm        = (short*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(short), 2097152);

  /* initialize data */
  init_buf_int16(naive_input,          nImg*nIfm*ifhp*ifwp, 0, 0);
  init_buf_int16(naive_output_bp,      nImg*nOfm*ofhp*ofwp, 0, 0);
  set_zeropad_nchw_int16(naive_input, nImg, nIfm, ifhp, ifwp, pad_h_in, pad_w_in);
  copy_buf_int16(naive_input, naive_input_save, nImg*nIfm*ifhp*ifwp);
  zero_buf_int16(naive_output,         nImg*nOfm*ofhp*ofwp);
  zero_buf_int16(naive_libxsmm_output, nImg*nOfm*ofhp*ofwp);
  zero_buf_int16(naive_libxsmm_input,  nImg*nIfm*ifhp*ifwp);
  init_buf_int16(naive_filter,         nOfm*nIfm*kh*kw, 0, 0);

  printf("##########################################\n");
  printf("#         Computing Reference ...        #\n");
  printf("##########################################\n");
  /* run naive convolutions */
  if (type == 'A' || type == 'F') {
    naive_conv_fp_int16(&naive_param, naive_input, naive_output, naive_filter);
  }
  if (type == 'A' || type == 'B') {
    zero_buf_int16(naive_input, nImg*nIfm*ifhp*ifwp);
    naive_conv_bp_int16(&naive_param, naive_input, naive_output_bp, naive_filter);
  }
  printf("##########################################\n");
  printf("#      Computing Reference ... done      #\n");
  printf("##########################################\n");

  printf("\n");
  printf("##########################################\n");
  printf("#     Setting Up    (custom-Storage)     #\n");
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
  conv_desc.pad_h = pad_h;
  conv_desc.pad_w = pad_w;
  conv_desc.pad_h_in = pad_h_in;
  conv_desc.pad_w_in = pad_w_in;
  conv_desc.pad_h_out = pad_h_out;
  conv_desc.pad_w_out = pad_w_out;
  conv_desc.threads = nThreads;
  conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
  conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
  conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
  conv_desc.datatype = LIBXSMM_DNN_DATATYPE_I16;

  libxsmm_handle = libxsmm_dnn_create_conv_layer( conv_desc, &status );
  CHKERR_LIBXSMM_DNN( status );

  /* setup LIBXSMM buffers and filter */
  libxsmm_input = libxsmm_dnn_link_buffer( libxsmm_handle, LIBXSMM_DNN_INPUT, input_libxsmm, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status );
  CHKERR_LIBXSMM_DNN( status );
  libxsmm_output = libxsmm_dnn_link_buffer( libxsmm_handle, LIBXSMM_DNN_OUTPUT, output_libxsmm, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status );
  CHKERR_LIBXSMM_DNN( status );
  libxsmm_filter = libxsmm_dnn_link_filter( libxsmm_handle, LIBXSMM_DNN_FILTER, filter_libxsmm, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status );
  CHKERR_LIBXSMM_DNN( status );

  /* copy in data to LIBXSMM format */
  /* we can also use the layout functions and set the data on our
     own external to the library, @TODO, we plan to add an example here */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_buffer( libxsmm_input, (void*)naive_input_save, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_zero_buffer( libxsmm_output ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_filter( libxsmm_filter, (void*)naive_filter, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );

  /* bind buffers and filter to handle */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_buffer( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_REGULAR_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_buffer( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_GRADIENT_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_buffer( libxsmm_handle, libxsmm_output, LIBXSMM_DNN_REGULAR_OUTPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_buffer( libxsmm_handle, libxsmm_output, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_filter( libxsmm_handle, libxsmm_filter, LIBXSMM_DNN_REGULAR_FILTER ) );

  /* let's allocate and bind scratch */
  scratch = (void*)libxsmm_aligned_malloc( libxsmm_dnn_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status ), 2097152);
  CHKERR_LIBXSMM_DNN( status );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );

  if (type == 'A' || type == 'F') {
    printf("##############################################\n");
    printf("#  Check Correctness - FWD (custom-Storage)  #\n");
    printf("##############################################\n");
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
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
    }
    /* copy out data */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_buffer( libxsmm_output, (void*)naive_libxsmm_output, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );

    /* compare */
    compare_buf_int16(naive_output, naive_libxsmm_output, nImg*nOfm*ofhp*ofwp, &norms_fwd);
    printf("             1-norm of reference: %f\n", norms_fwd.one_norm_ref);
    printf("              1-norm of JIT-code: %f\n", norms_fwd.one_norm_test);
    printf("       L2-error-norm of JIT-code: %f\n", norms_fwd.l2_rel_err);
    printf("    inf-norm of comp. rel. error: %f\n", norms_fwd.max_rel_err);
    printf("    inf-norm of comp. abs. error: %f\n", norms_fwd.max_abs_err);
  }

  if (type == 'A' || type == 'B') {
    printf("##############################################\n");
    printf("#  Check Correctness - BWD (custom-Storage)  #\n");
    printf("##############################################\n");
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_buffer( libxsmm_output, (void*)naive_output_bp, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_zero_buffer( libxsmm_input ) );
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
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
    }
    /* copy out data */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_buffer( libxsmm_input, (void*)naive_libxsmm_input, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );

    /* compare */
    compare_buf_int16(naive_input, naive_libxsmm_input, nImg*nIfm*ifhp*ifwp, &norms_bwd);
    printf("             1-norm of reference: %f\n", norms_bwd.one_norm_ref);
    printf("              1-norm of JIT-code: %f\n", norms_bwd.one_norm_test);
    printf("       L2-error-norm of JIT-code: %f\n", norms_bwd.l2_rel_err);
    printf("    inf-norm of comp. rel. error: %f\n", norms_bwd.max_rel_err);
    printf("    inf-norm of comp. abs. error: %f\n", norms_bwd.max_abs_err);
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
        libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    lpOps = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GOP  = %.5g\n", lpOps*1e-9/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GOPS  = %.5g\n", (lpOps*1e-9)/l_total);

    printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
       ifw, ifh, kw, kh, stride, pad, ((double)(l_total/iters)), (lpOps*1e-9)/l_total,
       norms_fwd.max_rel_err, norms_fwd.max_abs_err, norms_fwd.l2_rel_err, norms_fwd.one_norm_ref, norms_fwd.one_norm_test );
  }

  if (type == 'A' || type == 'B') {
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
        libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    lpOps = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GOP  = %.5g\n", lpOps*1e-9/(double)iters);
    printf("bp time = %.5g\n", ((double)(l_total/iters)));
    printf("GOPS  = %.5g\n", (lpOps*1e-9)/l_total);

    printf("PERFDUMP,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, pad, ((double)(l_total/iters)), (lpOps*1e-9)/l_total,
        norms_bwd.max_rel_err, norms_bwd.max_abs_err, norms_bwd.l2_rel_err, norms_bwd.one_norm_ref, norms_bwd.one_norm_test );
  }

  /* clean-up */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
  libxsmm_free(scratch);
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_buffer( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_buffer( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_buffer( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_buffer( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_filter( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER ) );
  /*CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_filter( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ) );*/
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_buffer( libxsmm_input ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_buffer( libxsmm_output ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_filter( libxsmm_filter ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_conv_layer( libxsmm_handle ) );

  /* deallocate data */
  libxsmm_free(naive_input);
  libxsmm_free(naive_output);
  libxsmm_free(naive_output_bp);
  libxsmm_free(naive_libxsmm_output);
  libxsmm_free(naive_libxsmm_input);
  libxsmm_free(naive_filter);
  libxsmm_free(input_libxsmm);
  libxsmm_free(output_libxsmm);
  libxsmm_free(filter_libxsmm);

  /* some empty lines at the end */
  printf("\n\n\n");

  return 0;
}

