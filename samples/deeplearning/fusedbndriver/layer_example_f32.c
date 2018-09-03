/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

#define CHKERR_LIBXSMM_DNN(A) { const int chkerr_libxsmm_dnn_ = A; if (LIBXSMM_DNN_SUCCESS != chkerr_libxsmm_dnn_) { \
  fprintf(stderr, "%s\n", libxsmm_dnn_get_error(chkerr_libxsmm_dnn_)); global_status = chkerr_libxsmm_dnn_; } \
}

typedef struct {
  int N;
  int C;
  int H;
  int W;
  int pad_h_in;
  int pad_w_in;
  int pad_h_out;
  int pad_w_out;
  int stride_h;
  int stride_w;
} naive_fusedbn_t;

LIBXSMM_INLINE void zero_buf(float* buf, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0.0f;
  }
}

LIBXSMM_INLINE void copy_buf(float* src, float* dst, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    dst[i] = src[i];
  }
}

LIBXSMM_INLINE void init_buf(float* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf(buf, size);
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? libxsmm_rand_f64() : (0.05 - libxsmm_rand_f64()/10.0)));
  }
}

LIBXSMM_INLINE void set_zeropad_nchw(float* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, float, input, nchw, C, H, W);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          if (h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w)
            LIBXSMM_VLA_ACCESS(4,  input, n, c, h, w, C, H, W) = 0.0;
        }
      }
    }
  }
}

LIBXSMM_INLINE void copy_internal_nchw(float* dst , float* src, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, float, input, src, C, H, W);
  LIBXSMM_VLA_DECL(4, float, new_input, dst, C, H+2*pad_h, W+2*pad_w);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          LIBXSMM_VLA_ACCESS(4, new_input, n, c, h+pad_h, w+pad_w, C, H+2*pad_h, W+2*pad_w) =  LIBXSMM_VLA_ACCESS(4,  input, n, c, h, w, C, H, W);
        }
      }
    }
  }
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


LIBXSMM_INLINE void naive_fusedbn_fp(naive_fusedbn_t* param, const float* input_ptr, float* output_ptr, const float* input_add_ptr,
                                     const float* beta_ptr, const float* gamma_ptr, float* expectval_ptr, float* stddev_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int fhi = param->H;
  const int fwi = param->W;
  const int iph = param->pad_h_in;
  const int ipw = param->pad_w_in;
  const int oph = param->pad_h_out;
  const int opw = param->pad_w_out;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int fho = fhi/sh;
  const int fwo = fwi/sw;
  const int fhpo = fho + 2*oph;
  const int fwpo = fwo + 2*opw;
  const int fhpi = fhi + 2*iph;
  const int fwpi = fwi + 2*ipw;

  int img, fm, h, w, hp, wp;

  LIBXSMM_VLA_DECL(4, const float, input,     input_ptr,     nFm, fhpi, fwpi);
  LIBXSMM_VLA_DECL(4, const float, input_add, input_add_ptr, nFm, fhpi, fwpi);
  LIBXSMM_VLA_DECL(4,       float, output,    output_ptr,    nFm, fhpo, fwpo);

#if defined(_OPENMP)
#pragma omp parallel for private(img, fm, h, w, hp, wp)
#endif
  for (img = 0; img < nImg; img++) {
    for (fm = 0; fm < nFm; fm++) {
      for( h=iph, hp=oph; h < (fhi+iph); h+=sh, hp++) {
        for( w=ipw, wp=opw; w < (fwi+ipw); w+=sw, wp++) {
          const float  input_val     =  LIBXSMM_VLA_ACCESS(4, input,     img, fm, h,  w, nFm, fhpi, fwpi);
          const float  input_add_val =  LIBXSMM_VLA_ACCESS(4, input_add, img, fm, h,  w, nFm, fhpi, fwpi);
                float* output_ptr2   = &LIBXSMM_VLA_ACCESS(4, output,    img, fm, hp, wp, nFm, fhpo, fwpo);

          /* BN + scale (gamma, beta) */
          float o = gamma_ptr[fm]*(input_val - expectval_ptr[fm])*stddev_ptr[fm] + beta_ptr[fm];
          /* Eltwise */
          o += input_add_val;
          /* ReLU */
          o = ( o < 0.0f ) ? 0.0f : o;
          *output_ptr2 = o;
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_fusedbn_bp(naive_fusedbn_t* param, const float* input_ptr, float* dinput_ptr, const float* output_ptr, float* doutput_ptr, float* dinput_add_ptr,
                                     const float* beta_ptr, float* del_beta_ptr, const float* gamma_ptr, float* del_gamma_ptr,
                                     const float* expectval_ptr, const float* stddev_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int fhi = param->H;
  const int fwi = param->W;
  const int iph = param->pad_h_in;
  const int ipw = param->pad_w_in;
  const int oph = param->pad_h_out;
  const int opw = param->pad_w_out;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int fho = fhi/sh;
  const int fwo = fwi/sw;
  const int fhpo = fho + 2*oph;
  const int fwpo = fwo + 2*opw;
  const int fhpi = fhi + 2*iph;
  const int fwpi = fwi + 2*ipw;
  const float nhw = (float)(nImg * fhi * fwi);
  const float recp_nhw = 1.0f/nhw;

  int img, fm, h, w, hp, wp;

  LIBXSMM_VLA_DECL(4, const float, input,      input_ptr,      nFm, fhpi, fwpi);
  LIBXSMM_VLA_DECL(4,       float, dinput,     dinput_ptr,     nFm, fhpi, fwpi);
  LIBXSMM_VLA_DECL(4,       float, dinput_add, dinput_add_ptr, nFm, fhpi, fwpi);
  LIBXSMM_VLA_DECL(4, const float, output,     output_ptr,     nFm, fhpo, fwpo);
  LIBXSMM_VLA_DECL(4,       float, doutput,    doutput_ptr,    nFm, fhpo, fwpo);

#if defined(_OPENMP)
#pragma omp parallel for private(img, fm, h, w, hp, wp)
#endif
  for (fm = 0; fm < nFm; fm++) {
    del_gamma_ptr[fm] = 0.0f;
    del_beta_ptr[fm] = 0.0f;

    for (img = 0; img < nImg; img++) {
      for( h=iph, hp=oph; h < (fhi+iph); h+=sh, hp++) {
        for( w=ipw, wp=opw; w < (fwi+ipw); w+=sw, wp++) {
                float* del_input_add_ptr = &LIBXSMM_VLA_ACCESS(4, dinput_add, img, fm, h,  w,  fm, fhpi, fwpi);
          const float  output_val        =  LIBXSMM_VLA_ACCESS(4,     output, img, fm, hp, wp, fm, fhpo, fwpo);
          const float  input_val         =  LIBXSMM_VLA_ACCESS(4,      input, img, fm, h,  w,  fm, fhpi, fwpi);
                float* del_output_ptr    = &LIBXSMM_VLA_ACCESS(4,    doutput, img, fm, hp, wp, fm, fhpo, fwpo);

          /* ReLU */
          *del_output_ptr    = LIBXSMM_FEQ(output_val, 0) ? 0 : *del_output_ptr;
          /* elementwise */
          *del_input_add_ptr = *del_output_ptr;
          del_gamma_ptr[fm] += (input_val - expectval_ptr[fm]) * (*del_output_ptr) * stddev_ptr[fm];
          del_beta_ptr[fm]  += *del_output_ptr;
        }
      }
    }
  }

#if defined(_OPENMP)
#pragma omp parallel for private(img, fm, h, w, hp, wp)
#endif
  for (img = 0; img < nImg; img++) {
    for (fm = 0; fm < nFm; fm++) {
      for( h=iph, hp=oph; h < (fhi+iph); h+=sh, hp++) {
        for( w=ipw, wp=opw; w < (fwi+ipw); w+=sw, wp++) {
                float* del_input_ptr  = &LIBXSMM_VLA_ACCESS(4,     dinput, img, fm, h,  w,  fm, fhpi, fwpi);
          const float  input_val      =  LIBXSMM_VLA_ACCESS(4,      input, img, fm, h,  w,  fm, fhpi, fwpi);
          const float  del_output_val =  LIBXSMM_VLA_ACCESS(4,    doutput, img, fm, hp, wp, fm, fhpo, fwpo);

          *del_input_ptr = gamma_ptr[fm] * stddev_ptr[fm] * recp_nhw * (nhw * del_output_val -
                    (del_beta_ptr[fm] + (input_val - expectval_ptr[fm]) * del_gamma_ptr[fm] * stddev_ptr[fm]));
        }
      }
    }
  }
}


int main(int argc, char* argv[])
{
  float *naive_input, *naive_output, *naive_input_add, *naive_delinput, *naive_deloutput;
  float *naive_delinput_add, *naive_libxsmm_output, *naive_libxsmm_delinput, *naive_libxsmm_delinput_add;
  float *naive_beta, *naive_gamma, *naive_delbeta, *naive_delgamma, *naive_expectval, *naive_stddev;
  float *input_libxsmm, *output_libxsmm, *input_add_libxsmm, *delinput_libxsmm, *deloutput_libxsmm, *delinput_add_libxsmm;
  float *beta_libxsmm, *gamma_libxsmm, *delbeta_libxsmm, *delgamma_libxsmm, *expectval_libxsmm, *stddev_libxsmm;

  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w;
  naive_fusedbn_t naive_param;
  void* scratch;
  size_t scratch_size = 0;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int ifw = 14;           /* input width, "W" */
  int ifh = 20;           /* input height, "H" */
  int nImg = 32;          /* mini-batch size, "N" */
  int nFm = 256;          /* number of input feature maps, "C" */
  int stride = 1;         /* stride when accessing inputs */
  int pad_h_in = 0;       /* padding mode */
  int pad_w_in = 0;       /* padding mode */
  int pad_h_out = 0;      /* padding mode */
  int pad_w_out = 0;      /* padding mode */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
  char format = 'L';

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double gb = 0.0;
  double gib = 0.0;
  int i;

  libxsmm_dnn_fusedbn_desc fusedbn_desc;
  libxsmm_dnn_fusedbn* libxsmm_handle;
  libxsmm_dnn_tensor*  libxsmm_input;
  libxsmm_dnn_tensor*  libxsmm_delinput;
  libxsmm_dnn_tensor*  libxsmm_output;
  libxsmm_dnn_tensor*  libxsmm_deloutput;
  libxsmm_dnn_tensor*  libxsmm_input_add;
  libxsmm_dnn_tensor*  libxsmm_delinput_add;
  libxsmm_dnn_tensor*  libxsmm_beta;
  libxsmm_dnn_tensor*  libxsmm_gamma;
  libxsmm_dnn_tensor*  libxsmm_delbeta;
  libxsmm_dnn_tensor*  libxsmm_delgamma;
  libxsmm_dnn_tensor*  libxsmm_expectval;
  libxsmm_dnn_tensor*  libxsmm_stddev;
  libxsmm_dnn_tensor_datalayout* libxsmm_layout;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;

  libxsmm_matdiff_info norms_fwd, norms_bwd, diff;
  memset(&norms_fwd, 0, sizeof(norms_fwd));
  memset(&norms_bwd, 0, sizeof(norms_bwd));
  memset(&diff, 0, sizeof(diff));

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters inpWidth inpHeight nImg nFm pad_w_in pad_h_in pad_w_out pad_h_out stride type format\n", argv[0]);
    return 0;
  }
  libxsmm_srand(1);

  /* reading new values from cli */
  i = 1;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) ifw        = atoi(argv[i++]);
  if (argc > i) ifh        = atoi(argv[i++]);
  if (argc > i) nImg       = atoi(argv[i++]);
  if (argc > i) nFm        = atoi(argv[i++]);
  if (argc > i) pad_w_in   = atoi(argv[i++]);
  if (argc > i) pad_h_in   = atoi(argv[i++]);
  if (argc > i) pad_w_out  = atoi(argv[i++]);
  if (argc > i) pad_h_out  = atoi(argv[i++]);
  if (argc > i) stride     = atoi(argv[i++]);
  if (argc > i) type       = *(argv[i++]);

  if (type != 'A' && type != 'F' && type != 'B') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only)\n");
    return 0;
  }

  stride_w = stride;
  stride_h = stride;

  /* deriving some values for naive code */
  ofh  = ifh/stride_h;
  ofw  = ifw/stride_w;
  ifhp = ifh + 2 * pad_h_in;
  ifwp = ifw + 2 * pad_w_in;
  ofhp = ofh + 2 * pad_h_out;
  ofwp = ofw + 2 * pad_w_out;

  /* set struct for naive convolution */
  naive_param.N = nImg;
  naive_param.C = nFm;
  naive_param.H = ifh;
  naive_param.W = ifw;
  naive_param.pad_h_in = pad_h_in;
  naive_param.pad_w_in = pad_w_in;
  naive_param.pad_h_out = pad_h_out;
  naive_param.pad_w_out = pad_w_out;
  naive_param.stride_h = stride_h;
  naive_param.stride_w = stride_w;

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: W:%d  H:%d  N:%d  C:%d  P:%d  Q:%d  STRIDE:%d\n", ifw, ifh, nImg, nFm, ofh, ofw, stride);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf(" InImg %dx%d Padded (%dx%d)\n", ifh, ifw, ifhp, ifwp);
  printf("OutImg %dx%d Padded (%dx%d)\n", ofh, ofw, ofhp, ofwp);
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nFm*ifhp*ifwp*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nFm*ofhp*ofwp*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nFm*ifhp*ifwp*  sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nFm*ofhp*ofwp*  sizeof(float))/(1024.0*1024.0) );
#if defined(USE_OVERWRITE)
  printf("Using Overwrite Option\n");
#endif

  /* allocate data */
  naive_input                = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  naive_input_add            = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  naive_delinput             = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  naive_delinput_add         = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  naive_output               = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);
  naive_deloutput            = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);
  naive_libxsmm_output       = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);
  naive_libxsmm_delinput     = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  naive_libxsmm_delinput_add = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  input_libxsmm              = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  delinput_libxsmm           = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  input_add_libxsmm          = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  delinput_add_libxsmm       = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  output_libxsmm             = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);
  deloutput_libxsmm          = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);
  naive_beta                 = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  naive_gamma                = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  naive_delbeta              = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  naive_delgamma             = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  naive_expectval            = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  naive_stddev               = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  beta_libxsmm               = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  gamma_libxsmm              = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  delbeta_libxsmm            = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  delgamma_libxsmm           = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  expectval_libxsmm          = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  stddev_libxsmm             = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);

  /* initialize data */
  if (pad_h_in == 0 && pad_w_in == 0) {
    init_buf(naive_input,          nImg*nFm*ifhp*ifwp, 0, 0);
    init_buf(naive_delinput,       nImg*nFm*ifhp*ifwp, 0, 0);
    init_buf(naive_input_add,      nImg*nFm*ifhp*ifwp, 0, 0);
    init_buf(naive_delinput_add,   nImg*nFm*ifhp*ifwp, 0, 0);
    init_buf(naive_output,         nImg*nFm*ofhp*ofwp, 0, 0);
    init_buf(naive_deloutput,      nImg*nFm*ofhp*ofwp, 0, 0);
  } else {
    float *naive_tmp = (float*)libxsmm_aligned_scratch( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
    init_buf(naive_tmp,          nImg*nFm*ifh*ifw, 0, 0);
    copy_internal_nchw( naive_input , naive_tmp, nImg, nFm, ifh, ifw, pad_h_in, pad_w_in);
    init_buf(naive_tmp,          nImg*nFm*ifh*ifw, 0, 0);
    copy_internal_nchw( naive_delinput , naive_tmp, nImg, nFm, ifh, ifw, pad_h_in, pad_w_in);
    init_buf(naive_tmp,          nImg*nFm*ifh*ifw, 0, 0);
    copy_internal_nchw( naive_input_add , naive_tmp, nImg, nFm, ifh, ifw, pad_h_in, pad_w_in);
    init_buf(naive_tmp,          nImg*nFm*ifh*ifw, 0, 0);
    copy_internal_nchw( naive_delinput_add , naive_tmp, nImg, nFm, ifh, ifw, pad_h_in, pad_w_in);
    libxsmm_free(naive_tmp);
    naive_tmp = (float*)libxsmm_aligned_scratch( nImg*nFm*ofh*ofw*sizeof(float), 2097152);
    init_buf(naive_tmp,          nImg*nFm*ofh*ofw, 0, 0);
    copy_internal_nchw( naive_output , naive_tmp, nImg, nFm, ofh, ofw, pad_h_out, pad_w_out);
    init_buf(naive_tmp,          nImg*nFm*ofh*ofw, 0, 0);
    copy_internal_nchw( naive_deloutput , naive_tmp, nImg, nFm, ofh, ofw, pad_h_out, pad_w_out);
    libxsmm_free(naive_tmp);
  }
  set_zeropad_nchw(naive_input,        nImg, nFm, ifhp, ifwp, pad_h_in,  pad_w_in);
  set_zeropad_nchw(naive_delinput,     nImg, nFm, ifhp, ifwp, pad_h_in,  pad_w_in);
  set_zeropad_nchw(naive_input_add,    nImg, nFm, ifhp, ifwp, pad_h_in,  pad_w_in);
  set_zeropad_nchw(naive_delinput_add, nImg, nFm, ifhp, ifwp, pad_h_in,  pad_w_in);
  set_zeropad_nchw(naive_output,       nImg, nFm, ifhp, ifwp, pad_h_out, pad_w_out);
  set_zeropad_nchw(naive_deloutput,    nImg, nFm, ifhp, ifwp, pad_h_out, pad_w_out);
  init_buf(naive_beta,      nFm, 0, 0);
  init_buf(naive_gamma,     nFm, 0, 0);
  init_buf(naive_delbeta,   nFm, 0, 0);
  init_buf(naive_delgamma,  nFm, 0, 0);
  init_buf(naive_expectval, nFm, 0, 0);
  init_buf(naive_stddev,    nFm, 0, 0);
  copy_buf(naive_beta,      beta_libxsmm,      nFm);
  copy_buf(naive_gamma,     gamma_libxsmm,     nFm);
  copy_buf(naive_delbeta,   delbeta_libxsmm,   nFm);
  copy_buf(naive_delgamma,  delgamma_libxsmm,  nFm);
  copy_buf(naive_expectval, expectval_libxsmm, nFm);
  copy_buf(naive_stddev,    stddev_libxsmm,    nFm);

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    if (type == 'A' || type == 'F') {
      naive_fusedbn_fp(&naive_param, naive_input, naive_output, naive_input_add, naive_beta, naive_gamma, naive_expectval, naive_stddev);
    }
    if (type == 'A' || type == 'B') {
      naive_fusedbn_bp(&naive_param, naive_input, naive_delinput, naive_output, naive_deloutput, naive_delinput_add,
                       naive_beta, naive_delbeta, naive_gamma, naive_delgamma, naive_expectval, naive_stddev);
    }
    printf("##########################################\n");
    printf("#      Computing Reference ... done      #\n");
    printf("##########################################\n");
  }

  if (format == 'A' || format == 'L') {
    printf("\n");
    printf("##########################################\n");
    printf("#      Setting Up  (custom-Storage)      #\n");
    printf("##########################################\n");

    /* setup LIBXSMM handle */
    fusedbn_desc.N = nImg;
    fusedbn_desc.C = nFm;
    fusedbn_desc.H = ifh;
    fusedbn_desc.W = ifw;
    fusedbn_desc.u = stride_h;
    fusedbn_desc.v = stride_w;
    fusedbn_desc.pad_h_in = pad_h_in;
    fusedbn_desc.pad_w_in = pad_w_in;
    fusedbn_desc.pad_h_out = pad_h_out;
    fusedbn_desc.pad_w_out = pad_w_out;
    fusedbn_desc.threads = nThreads;
    fusedbn_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    fusedbn_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    fusedbn_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    fusedbn_desc.fuse_order = LIBXSMM_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU;
    fusedbn_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE_RELU;

    libxsmm_handle = libxsmm_dnn_create_fusedbn( fusedbn_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    /* setup LIBXSMM buffers */
    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input  = libxsmm_dnn_link_tensor( libxsmm_layout, input_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_delinput  = libxsmm_dnn_link_tensor( libxsmm_layout, delinput_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ADD, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input_add  = libxsmm_dnn_link_tensor( libxsmm_layout, input_add_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_ADD, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_delinput_add  = libxsmm_dnn_link_tensor( libxsmm_layout, delinput_add_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_output  = libxsmm_dnn_link_tensor( libxsmm_layout, output_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_deloutput  = libxsmm_dnn_link_tensor( libxsmm_layout, deloutput_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_BETA, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_beta  = libxsmm_dnn_link_tensor( libxsmm_layout, beta_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_delbeta  = libxsmm_dnn_link_tensor( libxsmm_layout, delbeta_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_gamma  = libxsmm_dnn_link_tensor( libxsmm_layout, gamma_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_delgamma  = libxsmm_dnn_link_tensor( libxsmm_layout, delgamma_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_CHANNEL_EXPECTVAL, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_expectval  = libxsmm_dnn_link_tensor( libxsmm_layout, expectval_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_CHANNEL_STDDEV, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_stddev  = libxsmm_dnn_link_tensor( libxsmm_layout, stddev_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* copy in data to LIBXSMM format */
    /* we can also use the layout functions and set the data on our
       own external to the library */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input,        (void*)naive_input,     LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_output,       (void*)naive_output,     LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input_add,    (void*)naive_input_add, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_delinput,     (void*)naive_delinput,     LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_deloutput,    (void*)naive_deloutput,     LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_delinput_add, (void*)naive_delinput_add, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_input,        LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_delinput,     LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_output,       LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_deloutput,    LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_input_add,    LIBXSMM_DNN_REGULAR_INPUT_ADD ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_delinput_add, LIBXSMM_DNN_GRADIENT_INPUT_ADD ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_beta,         LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_gamma,        LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_delbeta,      LIBXSMM_DNN_GRADIENT_CHANNEL_BETA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_delgamma,     LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_expectval,    LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_stddev,       LIBXSMM_DNN_CHANNEL_STDDEV ) );

    /* let's allocate and bind scratch */
    scratch_size = libxsmm_dnn_fusedbn_get_scratch_size( libxsmm_handle, &status );
    CHKERR_LIBXSMM_DNN( status );
    scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_scratch( libxsmm_handle, scratch ) );
    /* set scratch to bogus to make sure that libxsmm takes care of zeroing internally */
    init_buf( (float*)scratch, scratch_size/4, 0, 0 );

    if ((type == 'A' || type == 'F') && LIBXSMM_NEQ(0, check)) {
      printf("##########################################\n");
      printf("#   Correctness - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
      }
      /* copy out data */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_output, (void*)naive_libxsmm_output, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nImg*nFm*ofhp*ofwp, 1, naive_output, naive_libxsmm_output, 0, 0, &norms_fwd);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
    }

    if ( (type == 'A' || type == 'B') && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("#   Correctness - BWD (custom-Storage)   #\n");
      printf("##########################################\n");

      /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
      }

      /* copy out data */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_delinput,     (void*)naive_libxsmm_delinput,     LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_delinput_add, (void*)naive_libxsmm_delinput_add, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nImg*nFm*ifhp*ifwp, 1, naive_delinput_add, naive_libxsmm_delinput_add, 0, 0, &norms_bwd);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nImg*nFm*ifhp*ifwp, 1, naive_delinput, naive_libxsmm_delinput, 0, 0, &norms_bwd);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
    }

    if (type == 'A' || type == 'F') {
      printf("##########################################\n");
      printf("#   Performance - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolution for performance */
      l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel private(i)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_fusedbn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);

      gb = ((double)nImg*(double)nFm*(((double)ifh*(double)ifw) + ((double)ofh*(double)ofw))*(double)sizeof(float)*(double)iters) / (1000*1000*1000);
      gib = ((double)nImg*(double)nFm*(((double)ifh*(double)ifw) + ((double)ofh*(double)ofw))*(double)sizeof(float)*(double)iters) / (1024*1024*1024);

      printf("GB  = %.5g\n", gb/(double)iters);
      printf("GiB  = %.5g\n", gib/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GB/s  = %.5g\n", gb/l_total);
      printf("GiB/s  = %.5g\n", gib/l_total);

      printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nFm,
        ifw, ifh, stride, pad_w_in, pad_h_in, pad_w_out, pad_h_out, ((double)(l_total/iters)), gb/l_total, gib/l_total, norms_fwd.l1_ref, norms_fwd.l1_tst,
        norms_fwd.l2_abs, norms_fwd.l2_rel, norms_fwd.linf_abs, norms_fwd.linf_rel, norms_fwd.normf_rel);
    }

    if ( (type == 'A' || type == 'B') ) {
      printf("##########################################\n");
      printf("#   Performance - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolution for performance */
      l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#     pragma omp parallel  private(i)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_fusedbn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);

      gb = (2.0*(double)nImg*(double)nFm*(((double)ifh*(double)ifw) + (2.0*(double)ofh*(double)ofw))*(double)sizeof(float)*(double)iters) / (1000*1000*1000);
      gib = (2.0*(double)nImg*(double)nFm*(((double)ifh*(double)ifw) + (2.0*(double)ofh*(double)ofw))*(double)sizeof(float)*(double)iters) / (1024*1024*1024);

      printf("GB  = %.5g\n", gb/(double)iters);
      printf("GiB  = %.5g\n", gib/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GB/s  = %.5g\n", gb/l_total);
      printf("GiB/s  = %.5g\n", gib/l_total);

      printf("PERFDUMP,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nFm,
        ifw, ifh, stride, pad_w_in, pad_h_in, pad_w_out, pad_h_out, ((double)(l_total/iters)), gb/l_total, gib/l_total, norms_bwd.l1_ref, norms_bwd.l1_tst,
        norms_bwd.l2_abs, norms_bwd.l2_rel, norms_bwd.linf_abs, norms_bwd.linf_rel, norms_bwd.normf_rel);
    }

    /* clean-up */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_release_scratch( libxsmm_handle ) );
    libxsmm_free(scratch);
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ADD ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_ADD ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_release_tensor( libxsmm_handle, LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_release_tensor( libxsmm_handle, LIBXSMM_DNN_CHANNEL_STDDEV ) );

    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delinput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_output ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_deloutput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input_add ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delinput_add ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_beta ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delbeta ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_gamma ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delgamma ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_expectval ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_stddev ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_fusedbn( libxsmm_handle ) );
  }

  /* deallocate data */
  libxsmm_free(naive_input);
  libxsmm_free(naive_input_add);
  libxsmm_free(naive_output);
  libxsmm_free(naive_delinput);
  libxsmm_free(naive_delinput_add);
  libxsmm_free(naive_deloutput);
  libxsmm_free(naive_beta);
  libxsmm_free(naive_gamma);
  libxsmm_free(naive_delbeta);
  libxsmm_free(naive_delgamma);
  libxsmm_free(naive_expectval);
  libxsmm_free(naive_stddev);
  libxsmm_free(naive_libxsmm_output);
  libxsmm_free(naive_libxsmm_delinput);
  libxsmm_free(naive_libxsmm_delinput_add);
  libxsmm_free(input_libxsmm);
  libxsmm_free(input_add_libxsmm);
  libxsmm_free(output_libxsmm);
  libxsmm_free(delinput_libxsmm);
  libxsmm_free(delinput_add_libxsmm);
  libxsmm_free(deloutput_libxsmm);
  libxsmm_free(beta_libxsmm);
  libxsmm_free(gamma_libxsmm);
  libxsmm_free(delbeta_libxsmm);
  libxsmm_free(delgamma_libxsmm);
  libxsmm_free(expectval_libxsmm);
  libxsmm_free(stddev_libxsmm);

  { const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXSMM_ABS(0 == env_check_scale ? 1.0 : atof(env_check_scale));
    if (LIBXSMM_NEQ(0, check) && (check < 100.0 * check_scale * diff.normf_rel) && (global_status == LIBXSMM_DNN_SUCCESS)) {
      fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  /* some empty lines at the end */
  printf("\n\n\n");

  return EXIT_SUCCESS;
}

