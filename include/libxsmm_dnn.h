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
typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_layer libxsmm_dnn_layer;
typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_buffer libxsmm_dnn_buffer;
typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_bias libxsmm_dnn_bias;
typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_filter libxsmm_dnn_filter;
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
#define LIBXSMM_DNN_ERR_CREATE_BUFFER              100006
#define LIBXSMM_DNN_ERR_INVALID_BUFFER             100007
#define LIBXSMM_DNN_ERR_CREATE_FILTER              100008
#define LIBXSMM_DNN_ERR_INVALID_FILTER             100009
#define LIBXSMM_DNN_ERR_CREATE_BIAS                100010
#define LIBXSMM_DNN_ERR_INVALID_BIAS               100011
#define LIBXSMM_DNN_ERR_MISMATCH_BUFFER            100012
#define LIBXSMM_DNN_ERR_INVALID_HANDLE_BUFFER      100013
#define LIBXSMM_DNN_ERR_MISMATCH_FILTER            100014
#define LIBXSMM_DNN_ERR_INVALID_HANDLE_FILTER      100015
#define LIBXSMM_DNN_ERR_INVALID_KIND               100016
#define LIBXSMM_DNN_ERR_INVALID_FORMAT_NCHW        100017
#define LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT     100018
#define LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT     100019
#define LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE    100020
#define LIBXSMM_DNN_ERR_INVALID_FORMAT_KCRS        100021
#define LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL     100022
#define LIBXSMM_DNN_ERR_CREATE_LAYOUT              100023
#define LIBXSMM_DNN_ERR_INVALID_LAYOUT             100024
#define LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH           100025
#define LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED        100026
#define LIBXSMM_DNN_ERR_UNKNOWN_BUFFER_TYPE        100027
#define LIBXSMM_DNN_ERR_UNKNOWN_FILTER_TYPE        100028
#define LIBXSMM_DNN_ERR_INVALID_ALGO               100029
#define LIBXSMM_DNN_ERR_INVALID_PADDING            100030

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
  LIBXSMM_DNN_TENSOR_DIMTYPE_S
} libxsmm_dnn_tensor_dimtype;

/** types of different buffers */
typedef enum libxsmm_dnn_buffer_type {
  /** regular input buffer */
  LIBXSMM_DNN_REGULAR_INPUT,
  /** gradient input buffer */
  LIBXSMM_DNN_GRADIENT_INPUT,
  /** regular output buffer */
  LIBXSMM_DNN_REGULAR_OUTPUT,
  /** gradient output buffer */
  LIBXSMM_DNN_GRADIENT_OUTPUT,
  /** general input type */
  LIBXSMM_DNN_INPUT,
  /** general output type */
  LIBXSMM_DNN_OUTPUT
} libxsmm_dnn_buffer_type;

/** types of different filters */
typedef enum libxsmm_dnn_filter_type {
  /* regular filter */
  LIBXSMM_DNN_REGULAR_FILTER,
  /* gradient filter */
  LIBXSMM_DNN_GRADIENT_FILTER,
  /** general filter type */
  LIBXSMM_DNN_FILTER
} libxsmm_dnn_filter_type;

/** layout descriptor to allow external data allocation
    outside of LIBXSMM */
typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_tensor_datalayout {
  libxsmm_dnn_tensor_dimtype* dim_type;
  unsigned int* dim_size;
  unsigned int num_dims;
} libxsmm_dnn_tensor_datalayout;

typedef enum libxsmm_dnn_conv_fuse_op {
  /* we fuse nothing into convolution */
  LIBXSMM_DNN_CONV_FUSE_NONE = 0
#if 0
  ,
  /* we fuse fuse bias init into convolution */
  LIBXSMM_DNN_CONV_FUSE_BIAS = 1,
  /* we fase fase ReLU calculation into convolution Op */
  LIBXSMM_DNN_CONV_FUSE_RELU = 2
#endif
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
typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_conv_desc {
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
  libxsmm_dnn_datatype datatype;            /* datatypes use for all input and outputs */
  libxsmm_dnn_tensor_format buffer_format;  /* format which is for buffer buffers */
  libxsmm_dnn_tensor_format filter_format;  /* format which is for filter buffers */
  libxsmm_dnn_conv_algo algo;               /* convolution algorithm used */
  libxsmm_dnn_conv_option options;          /* additional options */
  libxsmm_dnn_conv_fuse_op fuse_ops;        /* used ops into convolutions */
} libxsmm_dnn_conv_desc;

/** get string of error code */
LIBXSMM_API const char* libxsmm_dnn_get_error(libxsmm_dnn_err_t code);
LIBXSMM_API size_t libxsmm_dnn_typesize(libxsmm_dnn_datatype datatype);
LIBXSMM_API size_t libxsmm_dnn_get_simd_width(libxsmm_dnn_datatype datatype);

/** Create a layer handle (non-NULL if successful), and pre-build all JIT-code versions. */
LIBXSMM_API libxsmm_dnn_layer* libxsmm_dnn_create_conv_layer(libxsmm_dnn_conv_desc conv_desc, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_conv_layer(const libxsmm_dnn_layer* handle);

/** get layout description of buffers and fiters from handle */
LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_get_buffer_datalayout(const libxsmm_dnn_layer* handle, const libxsmm_dnn_buffer_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_get_filter_datalayout(const libxsmm_dnn_layer* handle, const libxsmm_dnn_filter_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_datalayout(libxsmm_dnn_tensor_datalayout* layout);

/** Create and manage buffers, filters and bias (non-NULL if successful) */
LIBXSMM_API libxsmm_dnn_buffer* libxsmm_dnn_link_buffer(const libxsmm_dnn_layer* handle, const libxsmm_dnn_buffer_type type, const void* data, libxsmm_dnn_tensor_format in_format, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_filter* libxsmm_dnn_link_filter(const libxsmm_dnn_layer* handle, const libxsmm_dnn_filter_type type, const void* data, libxsmm_dnn_tensor_format in_format, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_buffer* libxsmm_dnn_link_qbuffer(const libxsmm_dnn_layer* handle, const libxsmm_dnn_buffer_type type, const void* data, const char exp, libxsmm_dnn_tensor_format in_format, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_filter* libxsmm_dnn_link_qfilter(const libxsmm_dnn_layer* handle, const libxsmm_dnn_filter_type type, const void* data, const char exp, libxsmm_dnn_tensor_format in_format, libxsmm_dnn_err_t* status);
LIBXSMM_API void* libxsmm_dnn_get_buffer_data_ptr(const libxsmm_dnn_buffer* buffer, libxsmm_dnn_err_t* status);
LIBXSMM_API void* libxsmm_dnn_get_filter_data_ptr(const libxsmm_dnn_filter* filter, libxsmm_dnn_err_t* status);
LIBXSMM_API char libxsmm_dnn_get_qbuffer_exp(const libxsmm_dnn_buffer* buffer, libxsmm_dnn_err_t* status);
LIBXSMM_API char libxsmm_dnn_get_qfilter_exp(const libxsmm_dnn_filter* filter, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_set_qbuffer_exp(libxsmm_dnn_buffer* buffer, const char exp);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_set_qfilter_exp(libxsmm_dnn_filter* filter, const char exp);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_buffer(const libxsmm_dnn_buffer* buffer);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_filter(const libxsmm_dnn_filter* filter);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_bias(const libxsmm_dnn_bias* bias);
/**
 * Copy-in from a plain format such as input := [N][C][H][W] or [N][H][W][C]
 * The index specifies the actual channel number, and an eventual
 * padding is defined by the handle (pitch/stride).
 */
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyin_buffer(const libxsmm_dnn_buffer* buffer, const void* data, libxsmm_dnn_tensor_format in_format);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyin_filter(const libxsmm_dnn_filter* filter, const void* data, libxsmm_dnn_tensor_format in_format);
/*LIBXSMM_API libxsmm_dnn_err_t libxsmm_conv_copyin_bias(const libxsmm_dnn_bias* bias, const void* data);*/
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_zero_buffer(const libxsmm_dnn_buffer* buffer);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_zero_filter(const libxsmm_dnn_filter* filter);
/**
 * Copy-out into a plain format such as output := [N][C][H][W] or [N][H][W][C]
 * The index specifies the actual channel number, and an eventual
 * padding is defined by the handle (pitch/stride).
 */
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyout_buffer(const libxsmm_dnn_buffer* buffer, void* data, libxsmm_dnn_tensor_format out_format);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyout_filter(const libxsmm_dnn_filter* filter, void* data, libxsmm_dnn_tensor_format out_format);
/*LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyout_bias(const libxsmm_dnn_bias* bias, void* data);*/

/** scratch pad management */
LIBXSMM_API size_t libxsmm_dnn_get_scratch_size(const libxsmm_dnn_layer* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_bind_scratch(libxsmm_dnn_layer* handle, const libxsmm_dnn_compute_kind kind, const void* scratch);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_release_scratch(libxsmm_dnn_layer* handle, const libxsmm_dnn_compute_kind kind);

/** Bind/Release buffers, filters and bias to layer operation */
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_bind_buffer(libxsmm_dnn_layer* handle, const libxsmm_dnn_buffer* buffer, const libxsmm_dnn_buffer_type type);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_bind_filter(libxsmm_dnn_layer* handle, const libxsmm_dnn_filter* filter, const libxsmm_dnn_filter_type type);
LIBXSMM_API libxsmm_dnn_buffer* libxsmm_dnn_get_buffer(libxsmm_dnn_layer* handle, const libxsmm_dnn_buffer_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_filter* libxsmm_dnn_get_filter(libxsmm_dnn_layer* handle, const libxsmm_dnn_filter_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_release_buffer(libxsmm_dnn_layer* handle, const libxsmm_dnn_buffer_type type);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_release_filter(libxsmm_dnn_layer* handle, const libxsmm_dnn_filter_type type);

/** Run the layer identified by the handle; may use threads internally. */
LIBXSMM_API void libxsmm_dnn_execute(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_execute_st(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

/** some helper functions for framework integration */
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_transpose_filter(libxsmm_dnn_layer* handle, const libxsmm_dnn_filter_type type);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_reduce_wu_filters(libxsmm_dnn_layer* handle, const libxsmm_dnn_filter_type type);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_get_codegen_success(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_get_parallel_tasks(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind, unsigned int* num_tasks);

#if defined(LIBXSMM_BUILD) || defined(LIBXSMM_DNN_INTERNAL_API) /* Internal API */

/** Function type used for convolutions (single-precision); the actual signature depends on the kind of convolution. */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_sconvfunction)(const float* input1, const float* input2, float* output,
                                                           const float* ipf1, const float* ipf2, const float* opf);

typedef LIBXSMM_RETARGETABLE void (*libxsmm_wconvfunction)(const short* input1, const short* input2, int* output,
                                                           const short* ipf1, const short* ipf2, const int* opf);

typedef LIBXSMM_RETARGETABLE void (*libxsmm_busconvfunction)(const unsigned char* input1, const char* input2, short* output,
                                                             const unsigned char* ipf1, const char* ipf2, const short* opf);

typedef LIBXSMM_RETARGETABLE void (*libxsmm_budconvfunction)(const unsigned char* input1, const char* input2, int* output,
                                                             const unsigned char* ipf1, const char* ipf2, const int* opf);

typedef LIBXSMM_RETARGETABLE void (*libxsmm_wconvfunction_bwd)(int* input1, const short* input2, const short* output,
                                                           const int* ipf1, const short* ipf2, const short* opf);

typedef LIBXSMM_RETARGETABLE void (*libxsmm_busconvfunction_bwd)(const unsigned short* input1, const char* input2, const char* output,
                                                             const unsigned short* ipf1, const char* ipf2, const char* opf);

typedef LIBXSMM_RETARGETABLE void (*libxsmm_budconvfunction_bwd)(const unsigned int* input1, const char* input2, const char* output,
                                                             const unsigned int* ipf1, const char* ipf2, const char* opf);

/** Function type which is either libxsmm_sconvfunction or libxsmm_wconvfunction (weak-typed). */
typedef union LIBXSMM_RETARGETABLE libxsmm_xconvfunction { libxsmm_sconvfunction sconv; libxsmm_wconvfunction wconv; libxsmm_busconvfunction busconv; libxsmm_budconvfunction budconv; libxsmm_wconvfunction_bwd wconvb; libxsmm_busconvfunction_bwd busconvb; libxsmm_budconvfunction_bwd budconvb; } libxsmm_xconvfunction;

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

