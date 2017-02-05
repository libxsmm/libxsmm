/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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
/* Kunal Banerjee (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "libxsmm_dnn_convolution_winograd_forward.c"

/* Code for initializing arrays */
void zero_buf(float *buf, long size)
{
  int i;
  LIBXSMM_ASSUME_ALIGNED(buf, 64);
  for (i = 0; i < size; i++)
    buf[i] = 0.0f;
}

void init_buf(float *buf, long size, int initOne)
{
  int i;
  LIBXSMM_ASSUME_ALIGNED(buf, 64);
  zero_buf(buf, size);
  srand(0);
  for (i = 0; i < size; i++)
    buf[i] = initOne ? 1.0 : (0.5 - drand48());
}

/* Code for checking correctness */
#define checksum_buf(x,y) checksum_buf_name(x,y,#x)
void checksum_buf_name(float *buf, long size, char *name)
{
  int i;
  double sum = 0.0;
  LIBXSMM_ASSUME_ALIGNED(buf, 64);
  for (i = 0; i < size; i++)
    sum += (double)buf[i];

  printf("%-10s Checksum = %.10g\n", name, sum);
}

void compare_buf_4d(int d1, int d2, int d3, int d4, float *naive, float*blocked)
{
  int i1;
  int i2;
  int i3;
  int i4;
  int match = 1;
  double max_err = 0.;
  double err;
  float (*A)[d2][d3][d4] = (float (*)[*][*][*])naive;
  float (*B)[d2][d3][d4] = (float (*)[*][*][*])blocked;
  for (i1 = 0; i1 < d1; i1++) {
    for (i2 = 0; i2 < d2; i2++) {
      for (i3 = 0; i3 < d3; i3++) {
        for (i4 = 0; i4 < d4; i4++) {
          err = fabsf((A[i1][i2][i3][i4] - B[i1][i2][i3][i4])/A[i1][i2][i3][i4]);
          if ( err > 1.0e-4 ) {
            if ( err > max_err ) {
              printf("MISMATCH@ %3d,%3d,%3d,%3d: A=%12.8g  B=%12.8g (E:%12.4e)\n", i1, i2, i3, i4, A[i1][i2][i3][i4], B[i1][i2][i3][i4], err);
              max_err = err;
            }
            match = 0;
            /*break;*/
          }
        }
      }
    }
  }
  if ( match == 1 )
    printf("A and B are same\n");
  else
    printf("A and B are NOT same\n");
}

void cvt_4d_5d(int d1, int d2, int d3, int d4, int d5, float *in, float*out)
{
  int i1;
  int i2;
  int i3;
  int i4;
  int v;
  LIBXSMM_VLA_DECL(4, float, src, in, d2, d3, d4);
  LIBXSMM_VLA_DECL(5, float, dst, out, d2/d5, d3, d4, d5);

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(i1, i2, i3, i4, v)
#endif
  for (i1 = 0; i1 < d1; i1++) {
    for (i2 = 0; i2 < d2/d5; i2++) {
      for (i3 = 0; i3 < d3; i3++) {
        for (i4 = 0; i4 < d4; i4++) {
          for (v = 0; v < d5; v++) {
            dst[i1][i2][i3][i4][v] = src[i1][i2*d5+v][i3][i4];
          }
        }
      }
    }
  }
}

void cvt_5d_4d(int d1, int d2, int d3, int d4, int d5, float *in, float*out)
{
  int i1;
  int i2;
  int i3;
  int i4;
  int v;
  LIBXSMM_VLA_DECL(5, float, src, in, d2/d5, d3, d4, d5);
  LIBXSMM_VLA_DECL(4, float, dst, out, d2, d3, d4);

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(i1, i2, i3, i4, v)
#endif
  for (i1 = 0; i1 < d1; i1++) {
    for (i2 = 0; i2 < d2/d5; i2++) {
      for (i3 = 0; i3 < d3; i3++) {
        for (i4 = 0; i4 < d4; i4++) {
          for (v = 0; v < d5; v++) {
            dst[i1][i2*d5+v][i3][i4] = src[i1][i2][i3][i4][v];
          }
        }
      }
    }
  }
}

void cvt_4d_6d(int d1, int d2, int d3, int d4, int d5, int d6, float *in, float*out)
{
  int i1;
  int i2;
  int i3;
  int i4;
  int v1;
  int v2;
  LIBXSMM_VLA_DECL(4, float, src, in, d2, d3, d4);
  LIBXSMM_VLA_DECL(6, float, dst, out, d2/d5, d3, d4, d5, d6);

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(i1, i2, i3, i4, v1, v2)
#endif
  for (i1 = 0; i1 < d1/d6; i1++) {
    for (i2 = 0; i2 < d2/d5; i2++) {
      for (v1 = 0; v1 < d5; v1++) {
        for (i3 = 0; i3 < d3; i3++) {
          for (i4 = 0; i4 < d4; i4++) {
            for (v2 = 0; v2 < d6; v2++) {
              dst[i1][i2][i3][i4][v1][v2] = src[i1*d6+v2][i2*d5+v1][i3][i4];
            }
          }
        }
      }
    }
  }
}

void cvt_6d_4d(int d1, int d2, int d3, int d4, int d5, int d6, float *in, float*out)
{
  int i1;
  int i2;
  int i3;
  int i4;
  int v1;
  int v2;
  LIBXSMM_VLA_DECL(6, float, src, in, d2/d5, d3, d4, d5, d6);
  LIBXSMM_VLA_DECL(4, float, dst, out, d2, d3, d4);

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(i1, i2, i3, i4, v1, v2)
#endif
  for (i1 = 0; i1 < d1/d6; i1++) {
    for (i2 = 0; i2 < d2/d5; i2++) {
      for (v1 = 0; v1 < d5; v1++) {
        for (i3 = 0; i3 < d3; i3++) {
          for (i4 = 0; i4 < d4; i4++) {
            for (v2 = 0; v2 < d6; v2++) {
              dst[i1*d6+v2][i2*d5+v1][i3][i4] = src[i1][i2][i3][i4][v1][v2];
            }
          }
        }
      }
    }
  }
}

void naive_conv_fp(libxsmm_dnn_layer *handle, float *inp, float *outp, float *wp, float *biasp)
{
  int img;
  int ofm;
  int ifm;
  int oj;
  int oi;
  int ij;
  int ii;
  int kj;
  int ki;
  LIBXSMM_VLA_DECL(4, float, input, inp, handle->desc.C, handle->ifhp, handle->ifwp);
  LIBXSMM_VLA_DECL(4, float, output, outp, handle->desc.K, handle->ofhp, handle->ofwp);
  LIBXSMM_VLA_DECL(4, float, wt, wp, handle->desc.C, handle->desc.R, handle->desc.S);
  LIBXSMM_VLA_DECL(1, float, bias, biasp);

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, oj, oi, ifm, ij, ii, kj, ki)
#endif
  for (img = 0; img < handle->desc.N; img++) {
    for (ofm = 0; ofm < handle->desc.K; ofm++) {
      for (oj = 0; oj < handle->ofhp; oj++) {
        for (oi = 0; oi < handle->ofwp; oi++) {
          LIBXSMM_VLA_ACCESS(4, output, img, ofm, oj, oi, handle->desc.K, handle->ofhp, handle->ofwp) =
          LIBXSMM_VLA_ACCESS(1, bias, ofm);
        }
      }
      for (ifm = 0; ifm < handle->desc.C; ifm++) {
        for (oj = 0; oj < handle->ofhp; oj++) {
          ij = oj * handle->desc.u;
          for (oi = 0; oi < handle->ofwp; oi++) {
            ii = oi * handle->desc.v;
            for (kj = 0; kj < handle->desc.R; kj++) {
              for (ki = 0; ki < handle->desc.S; ki++) {
                LIBXSMM_VLA_ACCESS(4, output, img, ofm, oj, oi, handle->desc.K, handle->ofhp, handle->ofwp) +=
                  (LIBXSMM_VLA_ACCESS(4, input, img, ifm, ij+kj, ii+ki, handle->desc.C, handle->ifhp, handle->ifwp) *
                   LIBXSMM_VLA_ACCESS(4, wt, ofm, ifm, kj, ki, handle->desc.C, handle->desc.R, handle->desc.S));
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void print_help(void) {
  printf("\nUsage:\n");
  printf("    inlineasm/plainasm\n");
  printf("    filename to append\n");
  printf("    routine name\n");
  printf("    images\n");
  printf("    ifm\n");
  printf("    ifh\n");
  printf("    ifw\n");
  printf("    ofm\n");
  printf("    kh: 3\n");
  printf("    kw: 3\n");
  printf("    stride_h: 1\n");
  printf("    stride_w: 1\n");
  printf("    pad_h_in: 0\n");
  printf("    pad_w_in: 0\n");
  printf("    pad_h_out: 0\n");
  printf("    pad_w_out: 0\n");
  printf("    alpha: 4/6\n");
  printf("    bimg\n");
  printf("    vratio\n");
  printf("    precision: SP\n");
  printf("    ARCH: knl/skx\n");
  printf("\n\n");
}

int main(int argc, char* argv []) {
  libxsmm_dnn_layer l_handle;
  char* l_type;
  char* l_file_out;
  char* l_routine_name;
  char* l_arch;
  char* l_precision;
  int l_img = 0;        /* number of images */
  int l_ifm = 0;        /* number of input feature maps  */
  int l_ifh = 0;        /* input feature map height */
  int l_ifw = 0;        /* input feature map width  */
  int l_ofm = 0;        /* number of output feature maps */
  int l_kh = 0;         /* kernel height */
  int l_kw = 0;         /* kernel width  */
  int l_stride_h = 0;   /* this we use for offsets in the input */
  int l_stride_w = 0;   /* this we use for offsets in the input */
  int l_pad_h_in = 0;   /* pad for input  feature map height */
  int l_pad_w_in = 0;   /* pad for input  feature map width  */
  int l_pad_h_out = 0;  /* pad for output feature map height */
  int l_pad_w_out = 0;  /* pad for output feature map width  */
  int l_alpha  = 0;     /* tile size = (alpha - 2) */
  int l_bimg   = 0;     /* block on images */
  int l_vratio = 0;     /* ratio between vector lengths in frequency and time domains */
  int l_threads = 0;
  int l_single_precision = 0;

  /* variables for checking correctness */
  float* n_inp;
  float* n_outp;
  float* n_wp;
  float* test_outp;

  /* check argument count for a valid range */
  if ( argc != 22 ) {
    printf("Num of arguments: %d\n", argc);
    print_help();
    return -1;
  }

  /* names of files and routines */
  l_type = argv[1];
  l_file_out = argv[2];
  l_routine_name = argv[3];

  /* convolution sizes */
  l_img = atoi(argv[4]);
  l_ifm = atoi(argv[5]);
  l_ifh = atoi(argv[6]);
  l_ifw = atoi(argv[7]);
  l_ofm = atoi(argv[8]);
  l_kh  = atoi(argv[9]);
  l_kw  = atoi(argv[10]);
  l_stride_h = atoi(argv[11]);
  l_stride_w = atoi(argv[12]);
  l_pad_h_in = atoi(argv[13]);
  l_pad_w_in = atoi(argv[14]);
  l_pad_h_out = atoi(argv[15]);
  l_pad_h_out = atoi(argv[16]);
  l_alpha  = atoi(argv[17]);
  l_bimg   = atoi(argv[18]);
  l_vratio = atoi(argv[19]);

  /* precision */
  l_precision = argv[20];

  /* arch specific stuff */
  l_arch = argv[21];

  /* some intial parameters checks */
  /* check for sparse / dense only */
  if ( (strcmp(l_type, "inlineasm") != 0) &&
       (strcmp(l_type, "plainasm")  != 0) ) {
    print_help();
    return -1;
  }

  /* check value of arch flag */
  if ( strcmp(l_arch, "knl") == 0 ) {
    l_threads = 68;
  } else if ( strcmp(l_arch, "skx") == 0 ) {
    l_threads = 4;
  } else {
    print_help();
    return -1;
  }

  /* check and evaluate precison flag */
  if ( strcmp(l_precision, "SP") == 0 ) {
    l_single_precision = 1;
  } else {
    print_help();
    return -1;
  }

  if ( (l_kh != 3) || (l_kw != 3) ) {
    printf("For winograd convolution, kh and kw must be 3\n");
    print_help();
    return -1;
  }

  if ( (l_stride_h != 1) || (l_stride_w != 1) ) {
    printf("For winograd convolution, stride_h and stride_w must be 1\n");
    print_help();
    return -1;
  }

  if ( (l_pad_h_in != 0)  || (l_pad_w_in != 0) ||
       (l_pad_h_out != 0) || (l_pad_w_out != 0) ) {
    printf("Currently, for winograd convolution, only 0 pads are allowed\n");
    print_help();
    return -1;
  }

  if ( (l_alpha != 4) && (l_alpha != 6) ) {
    printf("Supported values for alpha are 4 and 6\n");
    print_help();
    return -1;
  }

  if ( ((l_ifh - 2)%(l_alpha - 2) != 0) ||
       ((l_ifw - 2)%(l_alpha - 2) != 0) ) {
    printf("Output feature map height and width must be perfectly divisible by (alpha - 2)\n");
    print_help();
    return -1;
  }

  if ( (l_img % l_bimg) != 0 ) {
    printf("Number of images must be perfectly divisible by bimg\n");
    print_help();
    return -1;
  }

  /* set up convolution descriptor */
  l_handle.desc.N = l_img;
  l_handle.desc.C = l_ifm;
  l_handle.desc.H = l_ifh;
  l_handle.desc.W = l_ifw;
  l_handle.desc.K = l_ofm;
  l_handle.desc.R = l_kh;
  l_handle.desc.S = l_kw;
  l_handle.desc.u = l_stride_h;
  l_handle.desc.v = l_stride_w;
  l_handle.desc.pad_h_in = l_pad_h_in;
  l_handle.desc.pad_w_in = l_pad_w_in;
  l_handle.desc.pad_h_out = l_pad_h_out;
  l_handle.desc.pad_w_out = l_pad_w_out;
  l_handle.desc.threads = l_threads;
  l_handle.desc.algo = LIBXSMM_DNN_CONV_ALGO_WINOGRAD;
  l_handle.desc.datatype_in  = LIBXSMM_DNN_DATATYPE_F32;
  l_handle.desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;

  /* set up winograd descriptor */
  l_handle.cwino_fwd.alpha = l_alpha;
  l_handle.cwino_fwd.itiles = (l_ifw - 2) / (l_alpha - 2);
  l_handle.cwino_fwd.jtiles = (l_ifh - 2) / (l_alpha - 2);
  l_handle.cwino_fwd.bimg  = l_bimg;
  l_handle.cwino_fwd.ur_i  = 1;
  l_handle.cwino_fwd.ur_j  = 1;
  l_handle.cwino_fwd.ur_m  = 1;
  l_handle.cwino_fwd.vratio = l_vratio;
  l_handle.cwino_fwd.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;

  /* set up handle */
  l_handle.datatype_in  = LIBXSMM_DNN_DATATYPE_F32;
  l_handle.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  l_handle.algo = LIBXSMM_DNN_CONV_ALGO_WINOGRAD;
  l_handle.ifhp = l_ifh + 2*l_pad_h_in;
  l_handle.ifwp = l_ifw + 2*l_pad_w_in;
  l_handle.ofh = l_handle.ifhp - 2;
  l_handle.ofw = l_handle.ifwp - 2;
  l_handle.ofhp = l_handle.ofh + 2*l_pad_h_out;
  l_handle.ofwp = l_handle.ofw + 2*l_pad_w_out;
  l_handle.ifmblock = 16;
  l_handle.ofmblock = 16;
  l_handle.blocksifm = l_ifm / 16; /* since tdvlen = 16 */
  l_handle.blocksofm = l_ofm / 16; /* since tdvlen = 16 */

  l_handle.input  = (libxsmm_dnn_buffer*)libxsmm_malloc(sizeof(libxsmm_dnn_buffer));
  l_handle.output = (libxsmm_dnn_buffer*)libxsmm_malloc(sizeof(libxsmm_dnn_buffer));
  l_handle.filter = (libxsmm_dnn_filter*)libxsmm_malloc(sizeof(libxsmm_dnn_filter));
  l_handle.bias   = (libxsmm_dnn_bias*)libxsmm_malloc(sizeof(libxsmm_dnn_bias));

  l_handle.input->data  = (float*)libxsmm_aligned_malloc(l_img*l_ifm*l_handle.ifhp*l_handle.ifwp*sizeof(float), 64);
  l_handle.output->data = (float*)libxsmm_aligned_malloc(l_img*l_ofm*l_handle.ofhp*l_handle.ofwp*sizeof(float), 64);
  l_handle.filter->data = (float*)libxsmm_aligned_malloc(l_ofm*l_ifm*l_kh*l_kw*sizeof(float), 64);
  l_handle.bias->data   = (float*)libxsmm_aligned_malloc(l_ofm*sizeof(float), 64);

  n_inp  = (float*)libxsmm_aligned_malloc(l_img*l_ifm*l_handle.ifhp*l_handle.ifwp*sizeof(float), 64);
  n_outp = (float*)libxsmm_aligned_malloc(l_img*l_ofm*l_handle.ofhp*l_handle.ofwp*sizeof(float), 64);
  n_wp   = (float*)libxsmm_aligned_malloc(l_ofm*l_ifm*l_kh*l_kw*sizeof(float), 64);

  test_outp = (float*)libxsmm_aligned_malloc(l_img*l_ofm*l_handle.ofhp*l_handle.ofwp*sizeof(float), 64);

  init_buf(n_inp, l_img*l_ifm*l_handle.ifhp*l_handle.ifwp, 0);
  zero_buf(n_outp, l_img*l_ofm*l_handle.ofhp*l_handle.ofwp);
  init_buf(l_handle.bias->data, l_ofm, 0);
  init_buf(n_wp, l_ofm*l_ifm*l_kh*l_kw, 0);
  zero_buf(l_handle.bias->data, l_ofm);

  zero_buf(l_handle.input->data,  l_img*l_ifm*l_handle.ifhp*l_handle.ifwp);
  zero_buf(l_handle.output->data, l_img*l_ofm*l_handle.ofhp*l_handle.ofwp);
  zero_buf(l_handle.filter->data, l_ofm*l_ifm*l_kh*l_kw);

  cvt_4d_5d(l_img, l_ifm, l_handle.ifhp, l_handle.ifwp, l_handle.ifmblock, n_inp, l_handle.input->data);
  cvt_4d_6d(l_ofm, l_ifm, l_kh, l_kw, l_handle.ifmblock, l_handle.ofmblock, n_wp, l_handle.filter->data);

  checksum_buf(n_inp, l_img*l_ifm*l_handle.ifhp*l_handle.ifwp);
  checksum_buf(l_handle.input->data, l_img*l_ifm*l_handle.ifhp*l_handle.ifwp);
  checksum_buf(n_wp, l_ofm*l_ifm*l_kh*l_kw);
  checksum_buf(l_handle.filter->data, l_ofm*l_ifm*l_kh*l_kw);
  checksum_buf(l_handle.bias->data, l_ofm);

  /* Perform winograd convolution */
  libxsmm_dnn_convolve_winograd_st_fwd_custom_custom( &l_handle, 0, 0 );
  checksum_buf(l_handle.output->data, l_img*l_ofm*l_handle.ofhp*l_handle.ofwp);

  naive_conv_fp(&l_handle, n_inp, n_outp, n_wp, l_handle.bias->data);

  cvt_5d_4d(l_img, l_ofm, l_handle.ofhp, l_handle.ofwp, l_handle.ofmblock, l_handle.output->data, test_outp);
  checksum_buf(test_outp, l_img*l_ofm*l_handle.ofhp*l_handle.ofwp);
  checksum_buf(n_outp, l_img*l_ofm*l_handle.ofhp*l_handle.ofwp);
  compare_buf_4d(l_img, l_ofm, l_handle.ofhp, l_handle.ofwp, n_outp, test_outp);

  /* generate code */
  /*
  if ( strcmp(l_type, "inlineasm")  == 0 ) {
    libxsmm_generator_dnn_convolution_winograd_forward_inlineasm( l_file_out, l_routine_name, &l_conv_desc, l_arch );
  } else {
    libxsmm_generator_dnn_convolution_winograd_forward_directasm( l_file_out, l_routine_name, &l_conv_desc, l_arch );
  }
  */

  libxsmm_free(n_inp);
  libxsmm_free(n_outp);
  libxsmm_free(n_wp);
  libxsmm_free(test_outp);

  return 0;
}

