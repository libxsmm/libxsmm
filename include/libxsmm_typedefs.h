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
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_TYPEDEFS_H
#define LIBXSMM_TYPEDEFS_H

#include "libxsmm_macros.h"


/** Flag enumeration which can be binary ORed. */
typedef enum libxsmm_gemm_flags {
  /** Transpose matrix A. */
  LIBXSMM_GEMM_FLAG_TRANS_A = 1,
  /** Transpose matrix B. */
  LIBXSMM_GEMM_FLAG_TRANS_B = 2,
  /** Generate aligned load instructions. */
  LIBXSMM_GEMM_FLAG_ALIGN_A = 4,
  /** Aligned load/store instructions. */
  LIBXSMM_GEMM_FLAG_ALIGN_C = 8
} libxsmm_gemm_flags;

/** Enumeration of the available prefetch strategies. */
typedef enum libxsmm_gemm_prefetch_type {
  /** Automatically select strategy (frontend). */
  LIBXSMM_PREFETCH_AUTO               = -1,
  /** No prefetching and no prefetch fn. signature. */
  LIBXSMM_PREFETCH_NONE               = 0,
  /** Only function prefetch signature. */
  LIBXSMM_PREFETCH_SIGONLY            = 1,
  /** Prefetch PA using accesses to A. */
  LIBXSMM_PREFETCH_AL2                = 2,
  /** Prefetch PA (aggressive). */
  LIBXSMM_PREFETCH_AL2_JPST           = 4,
  /** Prefetch PB using accesses to C. */
  LIBXSMM_PREFETCH_BL2_VIA_C          = 8,
  /** Prefetch A ahead. */
  LIBXSMM_PREFETCH_AL2_AHEAD          = 16,
  /** Prefetch PC using accesses to C. */
  LIBXSMM_PREFETCH_CL2                = 32,
  LIBXSMM_PREFETCH_AL2BL2_VIA_C       = LIBXSMM_PREFETCH_BL2_VIA_C | LIBXSMM_PREFETCH_AL2,
  LIBXSMM_PREFETCH_AL2CL2BL2_VIA_C    = LIBXSMM_PREFETCH_AL2BL2_VIA_C | LIBXSMM_PREFETCH_CL2,
  LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST  = LIBXSMM_PREFETCH_BL2_VIA_C | LIBXSMM_PREFETCH_AL2_JPST,
  LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD = LIBXSMM_PREFETCH_BL2_VIA_C | LIBXSMM_PREFETCH_AL2_AHEAD
} libxsmm_gemm_prefetch_type;

/** Provided for compatibility with older codes. */
typedef libxsmm_gemm_prefetch_type libxsmm_prefetch_type;

/** Flag enumeration which can be binary ORed. */
typedef enum libxsmm_convolution_prefetch_type {
  /** no prefetch */
  LIBXSMM_CONVOLUTION_PREFETCH_NONE = 0,
  /** prefetch input into L1 */
  LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 = 1,
  /** prefetch weight into L2 */
  LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2 = 2,
  /** prefetch output into L1 */
  LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 = 4,

  /** prefetch weight into L1 */
  LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1 = 8,
  /** prefetch output into L2 */
  LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2 = 16,
  /** prefetch input into L2 */
  LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2 = 32,

  /** combination 1: all */
  LIBXSMM_CONVOLUTION_PREFETCH_ALL = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2 | LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2,

  /** combination 2: no weight */
  LIBXSMM_CONVOLUTION_PREFETCH_NO_WEIGHT = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1,
  /** combination 3: no output */
  LIBXSMM_CONVOLUTION_PREFETCH_NO_OUTPUT = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2,
  /** combination 4: no output L2 */
  LIBXSMM_CONVOLUTION_PREFETCH_NO_OUTPUT_L2 = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1  | LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2,
  /** combination 5: no input L2 */
  LIBXSMM_CONVOLUTION_PREFETCH_NO_INPUT_L2 = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2 ,
  /** combination 7: no output L2  and no input L2*/
  LIBXSMM_CONVOLUTION_PREFETCH_NO_OUTPUT_NO_INPUT_L2 = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1 ,
  /** combination 8: no output L2  and no input L2 and no weight L2*/
  LIBXSMM_CONVOLUTION_PREFETCH_NO_OUTPUT_NO_INPUT_NO_WEIGHT_L2 = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1,
  /** combination 9: no output L2 no weight L2 */
  LIBXSMM_CONVOLUTION_PREFETCH_NO_OUTPUT_NO_WEIGHT_L2 = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1  | LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2,

  /** combination 10: no input and no output L1 */
  LIBXSMM_CONVOLUTION_PREFETCH_NO_OUTPUT_NO_INPUT_L1 = LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2 | LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2,

  /** combination 11: no weight L2 */
  LIBXSMM_CONVOLUTION_PREFETCH_NO_WEIGHT_L2 = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2 | LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2,
  /** combination 12: no input L1 */
  LIBXSMM_CONVOLUTION_PREFETCH_NO_INPUT_L1 = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2,

  /** combination 12: no input L1 no weight L2*/
  LIBXSMM_CONVOLUTION_PREFETCH_NO_INPUT_L1_NO_WEIGHT_L2 = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2
} libxsmm_convolution_prefetch_type;


typedef enum libxsmm_dnn_conv_format{
  /* use LIBXSMM internal format, we need to copy data into that */
  LIBXSMM_DNN_CONV_FORMAT_LIBXSMM = 1,
  /* use NHWC format internally, this allows no-copy operations */
  LIBXSMM_DNN_CONV_FORMAT_NHWC = 2,
  /* use NCHW format internally, this will include shadow copies, not preferred */
  LIBXSMM_DNN_CONV_FORMAT_NCHW = 4,
  /* use RSCK format internally, this allows no-copy operations  */
  LIBXSMM_DNN_CONV_FORMAT_RSCK = 8,
  /* use KCRS format internally, this will include shadow copies, not preferred */
  LIBXSMM_DNN_CONV_FORMAT_KCRS = 16,
  /* use ptr copy when copying in -> no copy takes place, this is just an additional option */
  LIBXSMM_DNN_CONV_FORMAT_PTR = 32,
  /* now some combinded types */
  LIBXSMM_DNN_CONV_FORMAT_NHWC_PTR = LIBXSMM_DNN_CONV_FORMAT_NHWC | LIBXSMM_DNN_CONV_FORMAT_PTR,
  LIBXSMM_DNN_CONV_FORMAT_RSCK_PTR = LIBXSMM_DNN_CONV_FORMAT_RSCK | LIBXSMM_DNN_CONV_FORMAT_PTR,
  LIBXSMM_DNN_CONV_FORMAT_NHWC_RSCK = LIBXSMM_DNN_CONV_FORMAT_NHWC | LIBXSMM_DNN_CONV_FORMAT_RSCK,
  LIBXSMM_DNN_CONV_FORMAT_LIBXSMM_PTR = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM | LIBXSMM_DNN_CONV_FORMAT_PTR
} libxsmm_dnn_conv_format;

/** Denotes the element/pixel type of an image/channel. */
typedef enum libxsmm_dnn_datatype {
  LIBXSMM_DNN_DATATYPE_F32,
  LIBXSMM_DNN_DATATYPE_I32,
  LIBXSMM_DNN_DATATYPE_I16,
  LIBXSMM_DNN_DATATYPE_I8
} libxsmm_dnn_datatype;

typedef enum libxsmm_dnn_conv_option {
  /* we get default settings */
  LIBXSMM_DNN_CONV_OPTION_NONE = 0,
  /* activations are stored unsigned */
  LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED = 1
} libxsmm_dnn_conv_option;

/** Structure storing the convolution argument description. */
typedef struct LIBXSMM_MAY_ALIAS libxsmm_convolution_forward_descriptor {
  unsigned int kh;                              /* kernel height */
  unsigned int kw;                              /* kernel width */
  unsigned int unroll_kh;                       /* kernel height, unrolled */
  unsigned int unroll_kw;                       /* kernel width, unrolled */
  unsigned int blocks_ofm;
  unsigned int blocks_ifm;
  unsigned int ofm_block;                       /* should be VLEN */
  unsigned int ifm_block;                       /* should be VLEN */
  unsigned int ofh_padded;                      /* this we need for 2D register block */
  unsigned int ofw_padded;                      /* this we use for 1D and 2D register block */
  unsigned int ofh_rb;                          /* UR, register block of ofh */
  unsigned int ofw_rb;                          /* UR, register block of ofw */
  unsigned int ifh_padded;                      /* this we need for 2D register block */
  unsigned int ifw_padded;                      /* this we use for 1D and 2D register block */
  unsigned int stride_h;                        /* this we use for offsets in the input */
  unsigned int stride_w;                        /* this we use for offsets in the input */
  unsigned int fm_lp_block;                    /* additional blocking for low precision datatypes of ifm */
  libxsmm_dnn_conv_format format;
  libxsmm_dnn_conv_option option;
  libxsmm_dnn_datatype datatype_in;
  libxsmm_dnn_datatype datatype_out;
  libxsmm_convolution_prefetch_type prefetch;   /* prefetch type, can be ORed vales of libxsmm_convolution_prefetch_type */
} libxsmm_convolution_forward_descriptor;

/** Structure storing the convolution backward argument description. */
typedef struct LIBXSMM_MAY_ALIAS libxsmm_convolution_backward_descriptor {
  unsigned int kw;                              /* kernel width */
  unsigned int unroll_kw;                       /* kernel width, unrolled */
  unsigned int blocks_ofm;
  unsigned int blocks_ifm;
  unsigned int ofm_block;                       /* should be VLEN */
  unsigned int ifm_block;                       /* should be VLEN */
  unsigned int ofh_padded;                      /* this we need for 2D register block */
  unsigned int ofw_padded;                      /* this we use for 1D and 2D register block */
  unsigned int ofh_rb;                          /* UR, register block of ofh */
  unsigned int ofw_rb;                          /* UR, register block of ofw */
  unsigned int ifh_padded;                      /* this we need for 2D register block */
  unsigned int ifw_padded;                      /* this we use for 1D and 2D register block */
  unsigned int stride_h;                        /* this we use for offsets in the input */
  unsigned int stride_w;                        /* this we use for offsets in the input */

  unsigned int ofw;                             /* upper bound for oi loop */
  unsigned int ofw_unroll;                      /* this we use for ofw unroll factor */
  unsigned int kh;                              /* kernel height */
  unsigned int unroll_kh;                       /* kernel height, unrolled */
  unsigned int peeled;                          /* generate multi version code for peeled and non-peeled loop -- that avoids conditional in back propagation */

  unsigned int prefetch_output_ahead;           /* prefetch all outputs of kj when you jump from non-peeled to peeled */

  libxsmm_dnn_conv_format format;
  libxsmm_dnn_conv_option option;
  libxsmm_dnn_datatype datatype_in;
  libxsmm_dnn_datatype datatype_out;
  libxsmm_convolution_prefetch_type prefetch;   /* prefetch type, can be ORed vales of libxsmm_convolution_prefetch_type */
} libxsmm_convolution_backward_descriptor;

/** Structure storing the convolution weight update argument description. */
typedef struct LIBXSMM_MAY_ALIAS libxsmm_convolution_weight_update_descriptor {
  unsigned int kw;                              /* kernel width */
  unsigned int unroll_kw;                       /* kernel width, unrolled */
  unsigned int kh;                              /* kernel height */
  unsigned int blocks_ofm;
  unsigned int blocks_ifm;
  unsigned int ofm_block;                       /* should be VLEN */
  unsigned int ifm_block;                       /* should be VLEN */
  unsigned int ofh_padded;                      /* this we need for 2D register block */
  unsigned int ofw_padded;                      /* this we use for 1D and 2D register block */
  unsigned int ofh_rb;                          /* UR, register block of ofh */
  unsigned int ofw_rb;                          /* UR, register block of ofw */
  unsigned int ifh_padded;                      /* this we need for 2D register block */
  unsigned int ifw_padded;                      /* this we use for 1D and 2D register block */
  unsigned int stride_h;                        /* this we use for offsets in the input */
  unsigned int stride_w;                        /* this we use for offsets in the input */

  unsigned int ifm_unroll;                      /* this we use to unroll ifm loop */
  unsigned int ofh;                             /* upper bound of oj loop */
  unsigned int ofh_unroll;                      /* this we use to unroll ofh loop */
  unsigned int ofw;                             /* upper bound of oi loop */
  unsigned int ofw_unroll;                      /* this we use to unroll ofw loop */

  unsigned int transpose_ofw_ifm;               /* transpose ofw and ifm */
  libxsmm_dnn_conv_format format;
  libxsmm_dnn_conv_option option;
  libxsmm_dnn_datatype datatype_in;
  libxsmm_dnn_datatype datatype_out;
  libxsmm_convolution_prefetch_type prefetch;   /* prefetch type, can be ORed vales of libxsmm_convolution_prefetch_type */
} libxsmm_convolution_weight_update_descriptor;

/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (single-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_smmfunction)(const float* a, const float* b, float* c, ...);
/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (double-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_dmmfunction)(const double* a, const double* b, double* c, ...);
/** Function type which is either libxsmm_smmfunction or libxsmm_dmmfunction (weak-typed). */
typedef union LIBXSMM_RETARGETABLE libxsmm_xmmfunction { libxsmm_smmfunction smm; libxsmm_dmmfunction dmm; } libxsmm_xmmfunction;

#endif /*LIBXSMM_TYPEDEFS_H*/

