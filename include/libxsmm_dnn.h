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
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_DNN_H
#define LIBXSMM_DNN_H

#include "libxsmm_macros.h"
#include "libxsmm_typedefs.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/** Opaque handles which represents convolutions and LIBXSMM datatypes */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_layer libxsmm_dnn_layer;
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_tensor libxsmm_dnn_tensor;
typedef unsigned int libxsmm_dnn_err_t;

/** Define error and warning codes */
#define LIBXSMM_DNN_SUCCESS                             0
#define LIBXSMM_DNN_WARN_FALLBACK                   90000
#define LIBXSMM_DNN_ERR_GENERAL                    100000
#define LIBXSMM_DNN_ERR_CREATE_HANDLE              100001
#define LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE       100002
#define LIBXSMM_DNN_ERR_INVALID_BLOCKING           100003
#define LIBXSMM_DNN_ERR_INVALID_HANDLE             100004
#define LIBXSMM_DNN_ERR_DATA_NOT_BOUND             100005
#define LIBXSMM_DNN_ERR_CREATE_TENSOR              100006
#define LIBXSMM_DNN_ERR_INVALID_TENSOR             100007
#define LIBXSMM_DNN_ERR_MISMATCH_TENSOR            100008
#define LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR      100009
#define LIBXSMM_DNN_ERR_INVALID_KIND               100010
#define LIBXSMM_DNN_ERR_INVALID_FORMAT_NCHW        100011
#define LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT     100012
#define LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT     100013
#define LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE    100014
#define LIBXSMM_DNN_ERR_INVALID_FORMAT_KCRS        100015
#define LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL     100016
#define LIBXSMM_DNN_ERR_CREATE_LAYOUT              100017
#define LIBXSMM_DNN_ERR_INVALID_LAYOUT             100018
#define LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH           100019
#define LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED        100020
#define LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE        100021
#define LIBXSMM_DNN_ERR_INVALID_ALGO               100022
#define LIBXSMM_DNN_ERR_INVALID_PADDING            100023
#define LIBXSMM_DNN_ERR_UNKNOWN_BIAS_TYPE          100024
#define LIBXSMM_DNN_ERR_MISMATCH_BIAS              100025
#define LIBXSMM_DNN_ERR_INVALID_HANDLE_BIAS        100026

/** Kinds of supported compute flavor operations. */
typedef enum libxsmm_dnn_compute_kind {
  /** Forward path */
  LIBXSMM_DNN_COMPUTE_KIND_FWD,
  /** Backward path */
  LIBXSMM_DNN_COMPUTE_KIND_BWD,
  /** Updated weights. */
  LIBXSMM_DNN_COMPUTE_KIND_UPD,
  /** All routines, need for some init routines. */
  LIBXSMM_DNN_COMPUTE_KIND_ALL
} libxsmm_dnn_compute_kind;

/** type/meaning of dimension in a LIBXSMM DNN tensor */
typedef enum libxsmm_dnn_tensor_dimtype {
  /** Mini-batch */
  LIBXSMM_DNN_TENSOR_DIMTYPE_N,
  /** Image Height */
  LIBXSMM_DNN_TENSOR_DIMTYPE_H,
  /** Image Width */
  LIBXSMM_DNN_TENSOR_DIMTYPE_W,
  /** channels or input channels */
  LIBXSMM_DNN_TENSOR_DIMTYPE_C,
  /** output channels */
  LIBXSMM_DNN_TENSOR_DIMTYPE_K,
  /** kernel height */
  LIBXSMM_DNN_TENSOR_DIMTYPE_R,
  /** kernel width */
  LIBXSMM_DNN_TENSOR_DIMTYPE_S,
  /** general counter */
  LIBXSMM_DNN_TENSOR_DIMTYPE_X
} libxsmm_dnn_tensor_dimtype;

/** types of different buffers */
typedef enum libxsmm_dnn_tensor_type {
  /** regular input buffer */
  LIBXSMM_DNN_REGULAR_INPUT,
  /** regular input buffer, transpose */
  LIBXSMM_DNN_REGULAR_INPUT_TRANS,
  /** gradient input buffer */
  LIBXSMM_DNN_GRADIENT_INPUT,
  /** regular output buffer */
  LIBXSMM_DNN_REGULAR_OUTPUT,
  /** gradient output buffer */
  LIBXSMM_DNN_GRADIENT_OUTPUT,
  /** general input type */
  LIBXSMM_DNN_INPUT,
  /** general output type */
  LIBXSMM_DNN_OUTPUT,
  /** general activation type */
  LIBXSMM_DNN_ACTIVATION,
  /* regular filter */
  LIBXSMM_DNN_REGULAR_FILTER,
  /* regular filter */
  LIBXSMM_DNN_REGULAR_FILTER_TRANS,
  /* gradient filter */
  LIBXSMM_DNN_GRADIENT_FILTER,
  /** general filter type */
  LIBXSMM_DNN_FILTER,
  /* regular bias */
  LIBXSMM_DNN_REGULAR_BIAS,
  /* gradient bias */
  LIBXSMM_DNN_GRADIENT_BIAS,
  /** general bias type */
  LIBXSMM_DNN_BIAS,
  /** batch stats */
  LIBXSMM_DNN_BATCH_STATS,
  LIBXSMM_DNN_MAX_STATS_FWD,
  LIBXSMM_DNN_MAX_STATS_BWD,
  LIBXSMM_DNN_MAX_STATS_UPD,
   /** general type, if needed might cause API issues in copy in/out API */
  LIBXSMM_DNN_TENSOR
} libxsmm_dnn_tensor_type;

/** layout descriptor to allow external data handling
    outside of LIBXSMM */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_tensor_datalayout {
  libxsmm_dnn_tensor_dimtype* dim_type;
  unsigned int* dim_size;
  unsigned int num_dims;
  libxsmm_dnn_tensor_format format;                /* format of activation buffer */
  libxsmm_dnn_internal_format custom_format;       /* internal classifier of format, an internal subgroup */
  libxsmm_dnn_datatype datatype;                   /* data type */
  libxsmm_dnn_tensor_type tensor_type;             /* tensor type */
} libxsmm_dnn_tensor_datalayout;

typedef enum libxsmm_dnn_conv_fuse_op {
  /* we fuse nothing into convolution */
  LIBXSMM_DNN_CONV_FUSE_NONE = 0,
  /* we fuse bias addition into convolution */
  LIBXSMM_DNN_CONV_FUSE_BIAS = 1,
  /* we fuse ReLU calculation into fwd convolution op */
  LIBXSMM_DNN_CONV_FUSE_RELU_FWD = 2,
  /* we fuse ReLU calculation into bwd convolution op */
  LIBXSMM_DNN_CONV_FUSE_RELU_BWD = 4,
  /* we fuse batch stats */
  LIBXSMM_DNN_CONV_FUSE_BATCH_STATS = 8,
  LIBXSMM_DNN_CONV_FUSE_MAX_STATS = 16,
  LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_RELU_BWD = LIBXSMM_DNN_CONV_FUSE_RELU_BWD | LIBXSMM_DNN_CONV_FUSE_BATCH_STATS,
  LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_RELU_BWD_AND_MAX = LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_RELU_BWD | LIBXSMM_DNN_CONV_FUSE_MAX_STATS,
  LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_AND_MAX = LIBXSMM_DNN_CONV_FUSE_BATCH_STATS |  LIBXSMM_DNN_CONV_FUSE_MAX_STATS,
  LIBXSMM_DNN_CONV_FUSE_RELU_BWD_AND_MAX = LIBXSMM_DNN_CONV_FUSE_RELU_BWD | LIBXSMM_DNN_CONV_FUSE_MAX_STATS,
    /* we fuse bias addition and ReLU into convolution op */
  LIBXSMM_DNN_CONV_FUSE_RELU = LIBXSMM_DNN_CONV_FUSE_RELU_FWD | LIBXSMM_DNN_CONV_FUSE_RELU_BWD,
  LIBXSMM_DNN_CONV_FUSE_BIAS_RELU = LIBXSMM_DNN_CONV_FUSE_BIAS | LIBXSMM_DNN_CONV_FUSE_RELU
} libxsmm_dnn_conv_fuse_op;

/** Type of algorithm used for convolutions. */
typedef enum libxsmm_dnn_conv_algo {
  /** let the library decide */
  LIBXSMM_DNN_CONV_ALGO_AUTO,
  /** direct convolution. */
  LIBXSMM_DNN_CONV_ALGO_DIRECT,
  /** winograd convolution. */
  LIBXSMM_DNN_CONV_ALGO_WINOGRAD
} libxsmm_dnn_conv_algo;

/** Structure which describes the input and output of data (DNN). */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_conv_desc {
  int N;                                    /* number of images in mini-batch */
  int C;                                    /* number of input feature maps */
  int H;                                    /* height of input image */
  int W;                                    /* width of input image */
  int K;                                    /* number of output feature maps */
  int R;                                    /* height of filter kernel */
  int S;                                    /* width of filter kernel */
  int u;                                    /* vertical stride */
  int v;                                    /* horizontal stride */
  int pad_h;                                /* height of logical rim padding to input
                                               for adjusting output height */
  int pad_w;                                /* width of logical rim padding to input
                                               for adjusting output width */
  int pad_h_in;                             /* height of zero-padding in input buffer,
                                               must equal to pad_h for direct conv */
  int pad_w_in;                             /* width of zero-padding in input buffer,
                                               must equal to pad_w for direct conv */
  int pad_h_out;                            /* height of zero-padding in output buffer */
  int pad_w_out;                            /* width of zero-padding in output buffer */
  int threads;                              /* number of threads to use when running
                                               convolution */
  libxsmm_dnn_datatype datatype_in;         /* datatypes used for all input related buffer */
  libxsmm_dnn_datatype datatype_out;        /* datatypes used for all output related buffer */
  libxsmm_dnn_tensor_format buffer_format;  /* format which is for buffer buffers */
  libxsmm_dnn_tensor_format filter_format;  /* format which is for filter buffers */
  libxsmm_dnn_conv_algo algo;               /* convolution algorithm used */
  libxsmm_dnn_conv_option options;          /* additional options */
  libxsmm_dnn_conv_fuse_op fuse_ops;        /* used ops into convolutions */
} libxsmm_dnn_conv_desc;

/** these are some quatization definitions, not sure if we want to
    move them into some main part of LIBXSMM */
/* @TODO check position of these declarations and defines */
typedef union LIBXSMM_RETARGETABLE libxsmm_intfloat {
  unsigned int ui;
  float f;
} libxsmm_intfloat;

/* F32 masking defines */
#define LIBXSNN_DNN_MASK_SIGN_F32      0x80000000
#define LIBXSMM_DNN_MASK_EXP_F32       0x7f800000
#define LIBXSMM_DNN_MASK_MANT_F32      0x007fffff
#define LIBXSMM_DNN_MASK_ABS_F32       0x7fffffff
#define LIBXSMM_DNN_MASK_FULL_F32      0xffffffff
#define LIBXSMM_DNN_MANT_SZ_F32        23
#define LIBXSMM_DNN_SZ_F32             32

/* DFP16 masking defines */
#define LIBXSMM_DNN_MANT_DFP16         15
#define LIXSMMM_DNN_RES_DFP16          (pow(2,-(LIBXSMM_DNN_MANT_DFP16)))

/* Qunatization Rounding Defines */
#define LIBXSMM_DNN_QUANT_NO_ROUND       80000
#define LIBXSMM_DNN_QUANT_BIAS_ROUND     80001
#define LIBXSMM_DNN_QUANT_STOCH_ROUND    80002
#define LIBXSMM_DNN_QUANT_NEAREST_ROUND  80003
#define LIBXSMM_DNN_QUANT_FPHW_ROUND     80004

/** get string of error code */
LIBXSMM_API const char* libxsmm_dnn_get_error(libxsmm_dnn_err_t code);
LIBXSMM_API size_t libxsmm_dnn_typesize(libxsmm_dnn_datatype datatype);
LIBXSMM_API size_t libxsmm_dnn_get_simd_width(libxsmm_dnn_datatype datatype);

/** Create a layer handle (non-NULL if successful), and pre-build all JIT-code versions. */
LIBXSMM_API libxsmm_dnn_layer* libxsmm_dnn_create_conv_layer(libxsmm_dnn_conv_desc conv_desc, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_conv_layer(const libxsmm_dnn_layer* handle);

/** get layout description of buffers and fiters from handle */
LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_create_tensor_datalayout(const libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_duplicate_tensor_datalayout(const libxsmm_dnn_tensor_datalayout* layout, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_tensor_datalayout(libxsmm_dnn_tensor_datalayout* layout);
LIBXSMM_API unsigned int libxsmm_dnn_compare_tensor_datalayout(const libxsmm_dnn_tensor_datalayout* layout_a, const libxsmm_dnn_tensor_datalayout* layout_b, libxsmm_dnn_err_t* status);
LIBXSMM_API unsigned int libxsmm_dnn_get_tensor_size(const libxsmm_dnn_tensor_datalayout* layout, libxsmm_dnn_err_t* status);
LIBXSMM_API unsigned int libxsmm_dnn_get_tensor_elements(const libxsmm_dnn_tensor_datalayout* layout, libxsmm_dnn_err_t* status);

/** Create and manage buffers, filters and bias (non-NULL if successful) */
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_link_tensor(const libxsmm_dnn_tensor_datalayout* layout, const void* data, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_link_qtensor(const libxsmm_dnn_tensor_datalayout* layout, const void* data, const unsigned char exp, libxsmm_dnn_err_t* status);
LIBXSMM_API void* libxsmm_dnn_get_tensor_data_ptr(const libxsmm_dnn_tensor* tensor, libxsmm_dnn_err_t* status);
LIBXSMM_API unsigned char libxsmm_dnn_get_qtensor_scf(const libxsmm_dnn_tensor* tensor, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_set_qtensor_scf(libxsmm_dnn_tensor* tensor, const unsigned char scf);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_tensor(const libxsmm_dnn_tensor* tensor);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_zero_tensor(const libxsmm_dnn_tensor* tensor);

/**
 * Copy-in/out from a plain format such [N][C][H][W] or [N][H][W][C]
 */
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyin_tensor(const libxsmm_dnn_tensor* tensor, const void* data, const libxsmm_dnn_tensor_format in_format);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyout_tensor(const libxsmm_dnn_tensor* tensor, void* data, const libxsmm_dnn_tensor_format out_format);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_trans_reg_filter(const libxsmm_dnn_layer* handle);

/** scratch pad management */
LIBXSMM_API size_t libxsmm_dnn_get_scratch_size(const libxsmm_dnn_layer* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_bind_scratch(libxsmm_dnn_layer* handle, const libxsmm_dnn_compute_kind kind, const void* scratch);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_release_scratch(libxsmm_dnn_layer* handle, const libxsmm_dnn_compute_kind kind);

/** Bind/Release buffers, filters and bias to layer operation */
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_bind_tensor(libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type);
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_get_tensor(libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_release_tensor(libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor_type type);

/** Run the layer identified by the handle; may use threads internally. */
LIBXSMM_API void libxsmm_dnn_execute(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_execute_st(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

/** some helper functions for framework integration */
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_transpose_filter(libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor_type type);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_reduce_wu_filters(libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor_type type);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_get_codegen_success(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_get_parallel_tasks(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind, unsigned int* num_tasks);

/** some quantization helper functions,
    @TODO need to be integrated better for all different ways of quantizations */
#define _mm512_quantize_near_ps_epi16( A, B ) _mm512_cvtepi32_epi16( _mm512_cvt_roundps_epi32( _mm512_mul_ps( _mm512_load_ps(A), B), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) ) )
LIBXSMM_API void libxsmm_dnn_quantize( float* in_buffer, short* out_buffer, int length, unsigned char add_shift, unsigned char* scf, int round_mode );
LIBXSMM_API void libxsmm_dnn_quantize_act( float* in_buffer, short* out_buffer, unsigned int N, unsigned int C, unsigned int H, unsigned int W, unsigned int cblk_f32, unsigned int cblk_i16, unsigned int lp_blk, unsigned char add_shift, unsigned char* scf, int round_mode );
LIBXSMM_API void libxsmm_dnn_quantize_fil( float* in_buffer, short* out_buffer, unsigned int K, unsigned int C, unsigned int R, unsigned int S, unsigned int cblk_f32, unsigned int cblk_i16, unsigned int kblk_f32, unsigned int kblk_i16, unsigned int lp_blk, unsigned char add_shift, unsigned char* scf, int round_mode );
LIBXSMM_API void libxsmm_dnn_dequantize( short* in_buffer, float* out_buffer, int length, unsigned char scf );

#if defined(LIBXSMM_BUILD) || defined(LIBXSMM_DNN_INTERNAL_API) /* Internal API */

/** Function type used for convolutions (single-precision); the actual signature depends on the kind of convolution. */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_sconvfunction)(
  const float* input1, const float* input2, float* output,
  const float* ipf1, const float* ipf2, const float* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_wconvfunction)(
  const short* input1, const short* input2, int* output,
  const short* ipf1, const short* ipf2, const int* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_wsconvfunction)(
  const short* input1, const short* input2, float* output,
  const short* ipf1, const short* ipf2, const float* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_uwsconvfunction)(
  short* input1, float* input2, short* output,
  short* ipf1, float* ipf2, short* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_bdbconvfunction)(
  unsigned char* input1, int* input2, unsigned char* output,
  unsigned char* ipf1, int* ipf2, unsigned char* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_busconvfunction)(
  const unsigned char* input1, const char* input2, short* output,
  const unsigned char* ipf1, const char* ipf2, const short* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_budconvfunction)(
  const unsigned char* input1, const char* input2, int* output,
  const unsigned char* ipf1, const char* ipf2, const int* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_wconvfunction_bwd)(
  int* input1, const short* input2, const short* output,
  const int* ipf1, const short* ipf2, const short* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_busconvfunction_bwd)(
  const unsigned short* input1, const char* input2, const char* output,
  const unsigned short* ipf1, const char* ipf2, const char* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_budconvfunction_bwd)(
  const unsigned int* input1, const char* input2, const char* output,
  const unsigned int* ipf1, const char* ipf2, const char* opf, ...);

/** Function type which is either libxsmm_sconvfunction or libxsmm_wconvfunction (weak-typed). */
LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_xconvfunction {
  libxsmm_sconvfunction sconv;
  libxsmm_wsconvfunction wsconv;
  libxsmm_uwsconvfunction uwsconv;
  libxsmm_wconvfunction wconv;
  libxsmm_bdbconvfunction bdbconv;
  libxsmm_busconvfunction busconv;
  libxsmm_budconvfunction budconv;
  libxsmm_wconvfunction_bwd wconvb;
  libxsmm_busconvfunction_bwd busconvb;
  libxsmm_budconvfunction_bwd budconvb;
} libxsmm_xconvfunction;

/** Code generation routine for a forward-convolution kernel. Call libxsmm_release_kernel in order to deallocate the JIT'ted code. */
LIBXSMM_API libxsmm_sconvfunction libxsmm_create_sconv_forward(const libxsmm_convolution_forward_descriptor* descriptor);

/** Code generation routine for a backward-convolution kernel. Call libxsmm_release_kernel in order to deallocate the JIT'ted code. */
LIBXSMM_API libxsmm_sconvfunction libxsmm_create_sconv_backward(const libxsmm_convolution_backward_descriptor* descriptor);

/** Code generation routine for a convolution kernel as specified by descriptor. */
LIBXSMM_API libxsmm_sconvfunction libxsmm_create_sconv_update_weights(const libxsmm_convolution_weight_update_descriptor* descriptor);

/** Code generation routine for a forward-convolution kernel. Call libxsmm_release_kernel in order to deallocate the JIT'ted code. */
LIBXSMM_API void* libxsmm_create_xconv_forward(const libxsmm_convolution_forward_descriptor* descriptor);

/** Code generation routine for a backward-convolution kernel. Call libxsmm_release_kernel in order to deallocate the JIT'ted code. */
LIBXSMM_API void* libxsmm_create_xconv_backward(const libxsmm_convolution_backward_descriptor* descriptor);

/** Code generation routine for a convolution kernel as specified by descriptor. */
LIBXSMM_API void* libxsmm_create_xconv_update_weights(const libxsmm_convolution_weight_update_descriptor* descriptor);

/** Code generation routine for a forward-convolution winograd kernel. Call libxsmm_release_kernel in order to deallocate the JIT'ted code. */
LIBXSMM_API void* libxsmm_create_xconv_wino_forward(const libxsmm_convolution_winograd_descriptor* descriptor);

/** Code generation routine for a backward-convolution winograd kernel. Call libxsmm_release_kernel in order to deallocate the JIT'ted code. */
LIBXSMM_API void* libxsmm_create_xconv_wino_backward(const libxsmm_convolution_winograd_descriptor* descriptor);

/** Code generation routine for a weight-update-convolution winograd kernel as specified by descriptor. */
LIBXSMM_API void* libxsmm_create_xconv_wino_update_weights(const libxsmm_convolution_winograd_descriptor* descriptor);

#endif
#endif /*LIBXSMM_DNN_H*/

