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
typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_conv_handle libxsmm_dnn_conv_handle;
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

/** Kinds of supported convolution operations. */
typedef enum libxsmm_dnn_conv_kind {
  /** Forward convolution. */
  LIBXSMM_DNN_CONV_KIND_FWD,
  /** Backward convolution. */
  LIBXSMM_DNN_CONV_KIND_BWD,
  /** Updated weights. */
  LIBXSMM_DNN_CONV_KIND_UPD
} libxsmm_dnn_conv_kind;

/** type/meaning of dimension in a LIBXSMM DNN tensor */
typedef enum libxsmm_dnn_conv_dimtype {
  /** Mini-batch */
  LIBXSMM_DNN_CONV_DIMTYPE_N,
  /** Image Height */
  LIBXSMM_DNN_CONV_DIMTYPE_H,
  /** Image Width */
  LIBXSMM_DNN_CONV_DIMTYPE_W,
  /** channles or input channels */
  LIBXSMM_DNN_CONV_DIMTYPE_C,
  /** output channels */
  LIBXSMM_DNN_CONV_DIMTYPE_K,
  /** kernel height */
  LIBXSMM_DNN_CONV_DIMTYPE_R,
  /** kernel width */
  LIBXSMM_DNN_CONV_DIMTYPE_S
} libxsmm_dnn_conv_dimtype;

/** layout descriptor to allow external data allocation
    outside of LIBXSMM */
typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_conv_datalayout {
  libxsmm_dnn_conv_dimtype* dim_type;
  unsigned int* dim_size;
  unsigned int num_dims;
} libxsmm_dnn_conv_datalayout;

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
  LIBXSMM_DNN_CONV_ALGO_AUTO,   /* ignored for now */
  /** direct convolution. */
  LIBXSMM_DNN_CONV_ALGO_DIRECT
} libxsmm_dnn_conv_algo;

/** Structure which describes the input and output of data (DNN). */
typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_conv_desc {
  int N;                                       /* number of images in mini-batch */
  int C;                                       /* number of input feature maps */
  int H;                                       /* height of input image */
  int W;                                       /* width of input image */
  int K;                                       /* number of output feature maps */
  int R;                                       /* height of filter kernel */
  int S;                                       /* width of filter kernel */
  int u;                                       /* vertical stride */
  int v;                                       /* horizontal stride */
  int pad_h_in;                                /* height of zero-padding in input buffer, ignored */
  int pad_w_in;                                /* width of zero-padding in input buffer, ignored */
  int pad_h_out;                               /* height of zero-padding in output buffer */
  int pad_w_out;                               /* width of zero-padding in output buffer */
  int threads;                                 /* number of threads to use when running convolution */
  libxsmm_dnn_conv_algo algo;                  /* convolution algorithm used */
  libxsmm_dnn_conv_format buffer_format;       /* format which is for buffer buffers */
  libxsmm_dnn_conv_format filter_format;       /* format which is for filter buffers */
  libxsmm_dnn_conv_fuse_op fuse_ops;           /* used ops into convolutions */
  libxsmm_dnn_conv_option options;             /* additional options */
  libxsmm_dnn_datatype datatype_in;            /* datatypes use for all input-related data such as activations, filter */
  libxsmm_dnn_datatype datatype_out;           /* datatypes use for all input-related data such as activations, bias */
} libxsmm_dnn_conv_desc;

/** get string of error code */
LIBXSMM_API const char* libxsmm_dnn_get_error(libxsmm_dnn_err_t code);
LIBXSMM_API size_t libxsmm_dnn_typesize(libxsmm_dnn_datatype datatype);

/** Create a handle (non-NULL if successful), and pre-build all JIT-code versions. */
LIBXSMM_API libxsmm_dnn_conv_handle* libxsmm_dnn_create_conv_handle(
  libxsmm_dnn_conv_desc     conv_desc );

LIBXSMM_API libxsmm_dnn_conv_handle* libxsmm_dnn_create_conv_handle_check(
  libxsmm_dnn_conv_desc     conv_desc,
  libxsmm_dnn_err_t*        status );

/** Release the given convolution handle. */
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_conv_handle(const libxsmm_dnn_conv_handle* handle);

/** Create buffers, filters and bias (non-NULL if successful) */
LIBXSMM_API libxsmm_dnn_buffer* libxsmm_dnn_create_input_buffer(const libxsmm_dnn_conv_handle* handle);
LIBXSMM_API libxsmm_dnn_buffer* libxsmm_dnn_create_output_buffer(const libxsmm_dnn_conv_handle* handle);
LIBXSMM_API libxsmm_dnn_filter* libxsmm_dnn_create_filter(const libxsmm_dnn_conv_handle* handle);
LIBXSMM_API libxsmm_dnn_bias*   libxsmm_dnn_create_bias(const libxsmm_dnn_conv_handle* handle);
LIBXSMM_API libxsmm_dnn_buffer* libxsmm_dnn_link_input_buffer(const libxsmm_dnn_conv_handle* handle, const void* data, libxsmm_dnn_conv_format in_format);
LIBXSMM_API libxsmm_dnn_buffer* libxsmm_dnn_link_output_buffer(const libxsmm_dnn_conv_handle* handle, const void* data, libxsmm_dnn_conv_format in_format);
LIBXSMM_API libxsmm_dnn_filter* libxsmm_dnn_link_filter(const libxsmm_dnn_conv_handle* handle, const void* data, libxsmm_dnn_conv_format in_format);

LIBXSMM_API libxsmm_dnn_buffer* libxsmm_dnn_create_input_buffer_check(const libxsmm_dnn_conv_handle* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_buffer* libxsmm_dnn_create_output_buffer_check(const libxsmm_dnn_conv_handle* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_filter* libxsmm_dnn_create_filter_check(const libxsmm_dnn_conv_handle* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_bias*   libxsmm_dnn_create_bias_check(const libxsmm_dnn_conv_handle* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_buffer* libxsmm_dnn_link_input_buffer_check(const libxsmm_dnn_conv_handle* handle, const void* data, libxsmm_dnn_conv_format in_format, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_buffer* libxsmm_dnn_link_output_buffer_check(const libxsmm_dnn_conv_handle* handle, const void* data, libxsmm_dnn_conv_format in_format, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_filter* libxsmm_dnn_link_filter_check(const libxsmm_dnn_conv_handle* handle, const void* data, libxsmm_dnn_conv_format in_format, libxsmm_dnn_err_t* status);

/** get layout description of buffers and fiters from handle */
LIBXSMM_API libxsmm_dnn_conv_datalayout* libxsmm_dnn_get_input_buffer_datalayout(const libxsmm_dnn_conv_handle* handle);
LIBXSMM_API libxsmm_dnn_conv_datalayout* libxsmm_dnn_get_output_buffer_datalayout(const libxsmm_dnn_conv_handle* handle);
LIBXSMM_API libxsmm_dnn_conv_datalayout* libxsmm_dnn_get_filter_datalayout(const libxsmm_dnn_conv_handle* handle);
LIBXSMM_API libxsmm_dnn_conv_datalayout* libxsmm_dnn_get_input_buffer_datalayout_check(const libxsmm_dnn_conv_handle* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_conv_datalayout* libxsmm_dnn_get_output_buffer_datalayout_check(const libxsmm_dnn_conv_handle* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_conv_datalayout* libxsmm_dnn_get_filter_datalayout_check(const libxsmm_dnn_conv_handle* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_datalayout(libxsmm_dnn_conv_datalayout* layout);

/** Bind buffers, filters and bias to convolutions operation */
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_bind_input_buffer(libxsmm_dnn_conv_handle* handle, const libxsmm_dnn_buffer* input);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_bind_output_buffer(libxsmm_dnn_conv_handle* handle, const libxsmm_dnn_buffer* output);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_bind_filter(libxsmm_dnn_conv_handle* handle, const libxsmm_dnn_filter* filter);

/** Release buffers, filters and bias from convolutions operation */
#if 0
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_release_input_buffer(libxsmm_dnn_conv_handle* handle);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_release_output_buffer(libxsmm_dnn_conv_handle* handle);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_release_filter(libxsmm_dnn_conv_handle* handle);
#endif

/** Release the given layer, filters, bias handle. */
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_buffer(const libxsmm_dnn_buffer* buffer);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_filter(const libxsmm_dnn_filter* filter);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_bias(const libxsmm_dnn_bias* bias);

/**
 * Copy-in from a plain format such as input := [N][C][H][W] or [N][H][W][C]
 * The index specifies the actual channel number, and an eventual
 * padding is defined by the handle (pitch/stride).
 */
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyin_buffer(const libxsmm_dnn_buffer* buffer, const void* data, libxsmm_dnn_conv_format in_format);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyin_filter(const libxsmm_dnn_filter* filter, const void* data, libxsmm_dnn_conv_format in_format);
/*LIBXSMM_API libxsmm_dnn_err_t libxsmm_conv_copyin_bias(const libxsmm_dnn_bias* bias, const void* data);*/
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_zero_buffer(const libxsmm_dnn_buffer* layer);

/**
 * Copy-out into a plain format such as output := [N][C][H][W] or [N][H][W][C]
 * The index specifies the actual channel number, and an eventual
 * padding is defined by the handle (pitch/stride).
 */
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyout_buffer(const libxsmm_dnn_buffer* buffer, void* data, libxsmm_dnn_conv_format out_format);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyout_filter(const libxsmm_dnn_filter* filter, void* data, libxsmm_dnn_conv_format out_format);
/*LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyout_bias(const libxsmm_dnn_bias* bias, void* data);*/

/** Run the convolution identified by the handle; may use threads internally. */
LIBXSMM_API void libxsmm_dnn_convolve(libxsmm_dnn_conv_handle* handle, libxsmm_dnn_conv_kind kind);

/** Run the convolution identified by the handle; takes a thread id. */
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_convolve_st(libxsmm_dnn_conv_handle* handle, libxsmm_dnn_conv_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid );

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

/** Function type which is either libxsmm_sconvfunction or libxsmm_wconvfunction (weak-typed). */
typedef union LIBXSMM_RETARGETABLE libxsmm_xconvfunction { libxsmm_sconvfunction sconv; libxsmm_wconvfunction wconv; libxsmm_busconvfunction busconv; libxsmm_busconvfunction budconv; } libxsmm_xconvfunction;

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

#endif
#endif /*LIBXSMM_DNN_H*/

