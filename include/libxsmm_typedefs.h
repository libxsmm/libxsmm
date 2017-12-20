/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
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

#define LIBXSMM_TYPESIZE(ENUM) ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_F64 ? 8 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_F32 ? 4 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_I32 ? 4 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_I16 ? 2 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_I8  ? 1 : ( \
  0/*invalid*/))))))


/** Enumerates element/data types. */
typedef enum libxsmm_datatype {
  LIBXSMM_DATATYPE_F64,
  LIBXSMM_DATATYPE_F32,
  LIBXSMM_DATATYPE_I32,
  LIBXSMM_DATATYPE_I16,
  LIBXSMM_DATATYPE_I8
} libxsmm_datatype;

/** Denotes the precision/data type of GEMM. */
typedef enum libxsmm_gemm_precision {
  LIBXSMM_GEMM_PRECISION_F64  = LIBXSMM_DATATYPE_F64,
  LIBXSMM_GEMM_PRECISION_F32  = LIBXSMM_DATATYPE_F32,
  LIBXSMM_GEMM_PRECISION_I16  = LIBXSMM_DATATYPE_I16
} libxsmm_gemm_precision;

/** Flag enumeration which can be binary ORed. */
typedef enum libxsmm_gemm_flags {
  LIBXSMM_GEMM_FLAG_NONE    = 0,
  /** Transpose matrix A. */
  LIBXSMM_GEMM_FLAG_TRANS_A = 1,
  /** Transpose matrix B. */
  LIBXSMM_GEMM_FLAG_TRANS_B = 2,
#if 0
  /** Alpha=0|1 */
  LIBXSMM_GEMM_FLAG_ALPHA_0 = 4,
  /** Alpha=neg|pos */
  LIBXSMM_GEMM_FLAG_ALPHA_S = 8,
  /** Beta=0|1 */
  LIBXSMM_GEMM_FLAG_BETA_0  = 16,
  /** Beta=neg|pos */
  LIBXSMM_GEMM_FLAG_BETA_S  = 32,
#endif
  /** Generate aligned load instructions. */
  LIBXSMM_GEMM_FLAG_ALIGN_A = 64,
  /** Aligned load/store instructions. */
  LIBXSMM_GEMM_FLAG_ALIGN_C = 128,
  /** Marker flag; do not use. */
  LIBXSMM_GEMM_FLAG_INVALID = 256
} libxsmm_gemm_flags;

/** Auto-batch flags (can be ORed) applicable to mmbatch_begin/mmbatch_end. */
typedef enum libxsmm_mmbatch_flags {
  /** Handle recorded batch unsynchronized-parallel. */
  LIBXSMM_MMBATCH_FLAG_DEFAULT      = LIBXSMM_GEMM_FLAG_INVALID * 0,
  /** Synchronize among C matrices. */
  LIBXSMM_MMBATCH_FLAG_SYNCHRONIZED = LIBXSMM_GEMM_FLAG_INVALID * 1,
  /** Handle recorded batch sequentially. */
  LIBXSMM_MMBATCH_FLAG_SEQUENTIAL   = LIBXSMM_GEMM_FLAG_INVALID * 2,
  /** Only record a statistic of potential SMMs. */
  LIBXSMM_MMBATCH_FLAG_STATISTIC    = LIBXSMM_GEMM_FLAG_INVALID * 4
} libxsmm_mmbatch_flags;

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
  LIBXSMM_PREFETCH_AL2BL2_VIA_C       = LIBXSMM_PREFETCH_BL2_VIA_C | LIBXSMM_PREFETCH_AL2,
  LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST  = LIBXSMM_PREFETCH_BL2_VIA_C | LIBXSMM_PREFETCH_AL2_JPST,
  LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD = LIBXSMM_PREFETCH_BL2_VIA_C | LIBXSMM_PREFETCH_AL2_AHEAD,
  /** Prefetch PA/PB/PC in L1 (using accesses to A, B, C) */
  LIBXSMM_PREFETCH_AL1                = 32,
  LIBXSMM_PREFETCH_BL1                = 64,
  LIBXSMM_PREFETCH_CL1                = 128,
  LIBXSMM_PREFETCH_AL1_BL1            = LIBXSMM_PREFETCH_AL1 | LIBXSMM_PREFETCH_BL1,
  LIBXSMM_PREFETCH_BL1_CL1            = LIBXSMM_PREFETCH_BL1 | LIBXSMM_PREFETCH_CL1,
  LIBXSMM_PREFETCH_AL1_CL1            = LIBXSMM_PREFETCH_AL1 | LIBXSMM_PREFETCH_CL1,
  LIBXSMM_PREFETCH_AL1_BL1_CL1        = LIBXSMM_PREFETCH_AL1_BL1 | LIBXSMM_PREFETCH_CL1,
  /** Backward compatibility: AL2CL2BL2_VIA_C is an alias for AL2BL2_VIA_C. */
  LIBXSMM_PREFETCH_AL2CL2BL2_VIA_C    = LIBXSMM_PREFETCH_AL2BL2_VIA_C
} libxsmm_gemm_prefetch_type;

/** DEPRECATED: provided for compatibility with older codes. */
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
  LIBXSMM_CONVOLUTION_PREFETCH_NO_INPUT_L2 = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2,
  /** combination 7: no output L2  and no input L2*/
  LIBXSMM_CONVOLUTION_PREFETCH_NO_OUTPUT_NO_INPUT_L2 = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2 | LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1,
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


typedef enum libxsmm_dnn_tensor_format{
  /* use LIBXSMM internal format, we need to copy data into that */
  LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM = 1,
  /* use NHWC format internally, this allows no-copy operations */
  LIBXSMM_DNN_TENSOR_FORMAT_NHWC = 2,
  /* use NCHW format internally, this will include shadow copies, not preferred */
  LIBXSMM_DNN_TENSOR_FORMAT_NCHW = 4,
  /* use RSCK format internally, this allows no-copy operations  */
  LIBXSMM_DNN_TENSOR_FORMAT_RSCK = 8,
  /* use KCRS format internally, this will include shadow copies, not preferred */
  LIBXSMM_DNN_TENSOR_FORMAT_KCRS = 16
} libxsmm_dnn_tensor_format;

typedef enum libxsmm_dnn_internal_format {
  /* use LIBXSMM internal format NC_bHWc */
  LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1 = 1,
  /* use LIBXSMM internal format C_bN_bHWnc */
  LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2 = 2,
  /* use LIBXSMM internal format HWN_bC_bnc */
  LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_3 = 3
} libxsmm_dnn_internal_format;

/** Denotes the element/pixel type of an image/channel. */
typedef enum libxsmm_dnn_datatype {
  LIBXSMM_DNN_DATATYPE_F64  = LIBXSMM_DATATYPE_F64,
  LIBXSMM_DNN_DATATYPE_F32  = LIBXSMM_DATATYPE_F32,
  LIBXSMM_DNN_DATATYPE_I32  = LIBXSMM_DATATYPE_I32,
  LIBXSMM_DNN_DATATYPE_I16  = LIBXSMM_DATATYPE_I16,
  LIBXSMM_DNN_DATATYPE_I8   = LIBXSMM_DATATYPE_I8
} libxsmm_dnn_datatype;

typedef enum libxsmm_dnn_conv_option {
  /* we get default settings */
  LIBXSMM_DNN_CONV_OPTION_NONE = 0,
  /* activations are stored unsigned */
  LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED = 1,
  /* reduce filters externally to op */
  LIBXSMM_DNN_CONV_OPTION_WU_EXT_FILTER_REDUCE = 2,
  /* use 16 bit accumulate instead of 32 bit accumulate for fix point */
  LIBXSMM_DNN_CONV_OPTION_16BIT_ACC = 4,
  /* overwrite results buffer (set it to zero before running the operations) */
  LIBXSMM_DNN_CONV_OPTION_OVERWRITE = 8,
  /* compound types */
  LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED_16BIT_ACC = LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED | LIBXSMM_DNN_CONV_OPTION_16BIT_ACC,
  LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED_16BIT_ACC_OVERWRITE = LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED | LIBXSMM_DNN_CONV_OPTION_16BIT_ACC | LIBXSMM_DNN_CONV_OPTION_OVERWRITE,
  LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED_OVERWRITE = LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED | LIBXSMM_DNN_CONV_OPTION_OVERWRITE,
  LIBXSMM_DNN_CONV_OPTION_WU_EXT_FILTER_REDUCE_OVERWRITE = LIBXSMM_DNN_CONV_OPTION_WU_EXT_FILTER_REDUCE | LIBXSMM_DNN_CONV_OPTION_OVERWRITE
} libxsmm_dnn_conv_option;

/** Structure storing the convolution argument description. */
typedef struct LIBXSMM_MAY_ALIAS libxsmm_convolution_forward_descriptor {
  unsigned int kh;                              /* kernel height */
  unsigned int kw;                              /* kernel width */
  unsigned int unroll_kh;                       /* kernel height, unrolled */
  unsigned int unroll_kw;                       /* kernel width, unrolled */
  unsigned int blocks_ofm;
  unsigned int blocks_ifm;
  unsigned int blocks_ifm_blocking;
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
  unsigned int fm_lp_block;                     /* additional blocking for low precision datatypes of ifm */
  unsigned int use_nts;                         /* non-zero if intent is to overwrite the output buffer using streaming stores */
  libxsmm_dnn_tensor_format format;
  libxsmm_dnn_conv_option option;
  libxsmm_dnn_datatype datatype;
  libxsmm_dnn_datatype datatype_itm;
  libxsmm_convolution_prefetch_type prefetch;   /* prefetch type, can be ORed vales of libxsmm_convolution_prefetch_type */
} libxsmm_convolution_forward_descriptor;

/** Backward convolution argument descriptor (similar to forward descriptor). */
typedef struct LIBXSMM_MAY_ALIAS libxsmm_convolution_backward_descriptor {
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
  unsigned int ofw;
  unsigned int fm_lp_block;                    /* additional blocking for low precision datatypes of ifm */
  libxsmm_dnn_tensor_format format;
  libxsmm_dnn_conv_option option;
  libxsmm_dnn_datatype datatype;
  libxsmm_dnn_datatype datatype_itm;
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
  libxsmm_dnn_tensor_format format;
  libxsmm_dnn_conv_option option;
  libxsmm_dnn_datatype datatype;
  libxsmm_dnn_datatype datatype_itm;
  libxsmm_convolution_prefetch_type prefetch;   /* prefetch type, can be ORed vales of libxsmm_convolution_prefetch_type */
} libxsmm_convolution_weight_update_descriptor;

/**
 * Structure storing the convolution winograd argument description.
 */
typedef struct LIBXSMM_MAY_ALIAS libxsmm_convolution_winograd_descriptor {
  /** alpha determines the tile size */
  unsigned int alpha;
  /** number of itiles */
  unsigned int itiles;
  /** number of jtiles */
  unsigned int jtiles;
  /** number of images in a block */
  unsigned int bimg;
  /** unroll factor */
  unsigned int ur;
  /** number of ifm blocks to unroll */
  unsigned int ur_ifm;
  /** number of ifm blocks. When ur_ifm equals to this, do additional optimizations */
  unsigned int blocks_ifm;
  /** prefetch type, can be ORed vales of libxsmm_convolution_prefetch_type */
  libxsmm_convolution_prefetch_type prefetch;
} libxsmm_convolution_winograd_descriptor;

/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (double-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_dmmfunction)(const double* a, const double* b, double* c, ...);
/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (single-precision). */
typedef LIBXSMM_RETARGETABLE void(*libxsmm_smmfunction)(const float* a, const float* b, float* c, ...);
/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (low-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_wmmfunction)(const short* a, const short* b, int* c, ...);
/** Function type which is either libxsmm_smmfunction or libxsmm_dmmfunction (weak-typed). */
typedef union LIBXSMM_RETARGETABLE libxsmm_xmmfunction {
  void (*xmm)(const void* a, const void* b, void* c, ...);
  libxsmm_dmmfunction dmm; libxsmm_smmfunction smm; libxsmm_wmmfunction wmm;
} libxsmm_xmmfunction;

/** Determines the kernel kind. */
typedef enum libxsmm_kernel_kind {
  /** Matrix multiplication kernel */
  LIBXSMM_KERNEL_KIND_MATMUL  = 0,
  /** Matcopy kernel kind */
  LIBXSMM_KERNEL_KIND_MCOPY   = 1,
  /** Transpose kernel kind */
  LIBXSMM_KERNEL_KIND_TCOPY   = 2,
  /** Not a JIT kernel */
  LIBXSMM_KERNEL_KIND_INVALID = 3
} libxsmm_kernel_kind;

/** Specialized function for matrix-copy (weak-typed). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_xmatcopyfunction)(const void* in, const unsigned int* ldi, void* out, const unsigned int* ldo, ...);

/** Specialized function for transpose (weak-typed). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_xtransfunction)(const void* in, const unsigned int* ldi, void* out, const unsigned int* ldo);

/** Structure to receive information about the code registry status (libxsmm_get_registry_info). */
typedef struct LIBXSMM_RETARGETABLE libxsmm_registry_info { size_t capacity, size, nbytes, nstatic, ncache; } libxsmm_registry_info;

#endif /*LIBXSMM_TYPEDEFS_H*/

