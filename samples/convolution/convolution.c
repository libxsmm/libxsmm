/******************************************************************************
** Copyright (c) 2017, Intel Corporation                                     **
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
#include <libxsmm_mhd.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if 1
# define MALLOC(SIZE) libxsmm_aligned_malloc(SIZE, 0/*auto*/)
# define FREE(POINTER) libxsmm_free(POINTER)
#else
# define MALLOC(SIZE) malloc(SIZE)
# define FREE(POINTER) free(POINTER)
#endif


#if 0
void convolution(
  const float* in_x, int in_xpitch,
  const float* in_y, int in_ypitch,
  int n, float* out)
{
}
#endif


int main(int argc, char* argv[])
{
  const char *const filename_in   = (1 < argc ? argv[1] : "convolution_in.mhd");
  const char *const filename_out  = (2 < argc ? argv[2] : "convolution_out.mhd");
  const int kh = (3 < argc ? atoi(argv[3]) : 39);
  const int kw = (4 < argc ? atoi(argv[4]) : kh);
  int result = 0 != strcmp(filename_in, filename_out) ? EXIT_SUCCESS : EXIT_FAILURE;
  size_t ndims = 3, size[3], pitch[2], ncomponents = 0, header_size = 0, extension_size;
  void *conv_input_buffer = 0, *conv_filter_buffer = 0, *conv_output_buffer = 0;
  libxsmm_dnn_buffer *conv_input = 0, *conv_output = 0;
  libxsmm_dnn_filter *conv_filter = 0;
  libxsmm_dnn_datatype type_out = LIBXSMM_DNN_DATATYPE_F32;
  libxsmm_mhd_elemtype type_in = LIBXSMM_MHD_ELEMTYPE_UNKNOWN;
  libxsmm_dnn_conv_desc descriptor = { 0 };
  libxsmm_dnn_layer* handle = 0;
  libxsmm_dnn_err_t status;
  size_t size1 = 0, typesize = 0;
  static int error_once = 0;
  char filename_data[1024];
  void *filter = 0;
  void *image = 0;

  /* Read MHD-header information; function includes various sanity checks. */
  if (EXIT_SUCCESS == result) {
    result = libxsmm_mhd_read_header(filename_in, sizeof(filename_data),
      filename_data, &ndims, size, &ncomponents, &type_in,
      &header_size, &extension_size);
  }

  /* Only accept 2-d images or a single slice of a 3d-image. */
  if (2 == ndims || (3 == ndims && 1 == size[2])) {
    size1 = size[0] * size[1];
    pitch[0] = size[0];
    pitch[1] = size[1];
  }
  else {
    result = EXIT_FAILURE;
  }

  /* Allocate image data according to the MHD-header information. */
  if (EXIT_SUCCESS == result) {
    /* DNN type: assume that MHD I/O provides a super-set of types */
    if (0 != libxsmm_mhd_typename((libxsmm_mhd_elemtype)type_out, &typesize)) {
      const size_t filter_size = ncomponents * kh * kw;
      image = MALLOC(size1 * ncomponents * typesize);
      if (0 == image) result = EXIT_FAILURE;
      filter = MALLOC(filter_size * typesize);
      if (0 != filter) {
        size_t i;
        switch (type_out) {
          case LIBXSMM_DNN_DATATYPE_F32: {
            float *const f = (float*)filter;
            for (i = 0; i < filter_size; ++i) {
              f[i] = (float)(0.05 - ((double)rand() / RAND_MAX) * 0.1);
            }
          } break;
          default: result = EXIT_FAILURE;
        }
      }
      else {
        result = EXIT_FAILURE;
      }
    }
    else {
      result = EXIT_FAILURE;
    }
  }

  /* Read the image data according to the header into the allocated buffer. */
  if (EXIT_SUCCESS == result) {
    result = libxsmm_mhd_read(filename_data,
      size, pitch, ndims, ncomponents, header_size, type_in,
      /* eventually perform a type-conversion (type_in != type_out) */
      (const libxsmm_mhd_elemtype*)&type_out, image,
      0/*handle_element*/, 0/*extension*/, 0/*extension_size*/);
  }

  /* Setup convolution descriptor. */
  if (EXIT_SUCCESS == result) {
    descriptor.threads = 1;
    descriptor.N = 1; /* number of images */
    descriptor.R = kh; /* kernel height */
    descriptor.S = kw; /* kernel width */
    descriptor.C = (int)ncomponents; /* in */
    descriptor.K = descriptor.C; /* no reduction */
    descriptor.u = 1; /* H-stride */
    descriptor.v = 1; /* W-stride */
    descriptor.pad_h = 19; /* H-pad */
    descriptor.pad_w = 19; /* W-pad */
    descriptor.pad_h_in = 0;
    descriptor.pad_w_in = 0;
    descriptor.pad_h_out = 0;
    descriptor.pad_w_out = 0;
    descriptor.H = (int)((size[1] + 2 * descriptor.pad_h - descriptor.R) / descriptor.u + 1);
    descriptor.W = (int)((size[0] + 2 * descriptor.pad_w - descriptor.S) / descriptor.v + 1);
    descriptor.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT/*LIBXSMM_DNN_CONV_ALGO_AUTO*/;
    descriptor.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    descriptor.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    descriptor.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
    descriptor.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;
    descriptor.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
    descriptor.datatype = LIBXSMM_DNN_DATATYPE_F32;
    handle = libxsmm_dnn_create_conv_layer(descriptor, &status);
    if (LIBXSMM_DNN_SUCCESS != status) {
      const char *const error_message = libxsmm_dnn_get_error(status);
      fprintf(stderr, "%s\n", error_message);
      if (LIBXSMM_DNN_WARN_FALLBACK != status) result = EXIT_FAILURE;
    }
  }

  /* Link buffers and convert NCHW-image and KCRS-filter to internal format. */
  if (EXIT_SUCCESS == result) {
    /* Input buffer */
    conv_input_buffer = MALLOC(descriptor.N * descriptor.C * (descriptor.H + 2 * descriptor.pad_h_in) * (descriptor.W + 2 * descriptor.pad_w_in) * typesize);
    if (0 == conv_input_buffer) result = EXIT_FAILURE;
    conv_input = libxsmm_dnn_link_buffer(handle, LIBXSMM_DNN_INPUT, conv_input_buffer, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxsmm_dnn_bind_buffer(handle, conv_input, LIBXSMM_DNN_REGULAR_INPUT);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxsmm_dnn_copyin_buffer(conv_input, image, LIBXSMM_DNN_TENSOR_FORMAT_NCHW);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    /* Filter buffer */
    conv_filter_buffer = MALLOC(descriptor.K + descriptor.C * descriptor.R + descriptor.S * typesize);
    if (0 == conv_filter_buffer) result = EXIT_FAILURE;
    conv_filter = libxsmm_dnn_link_filter(handle, LIBXSMM_DNN_FILTER, conv_filter_buffer, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxsmm_dnn_bind_filter(handle, conv_filter, LIBXSMM_DNN_REGULAR_FILTER);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxsmm_dnn_copyin_filter(conv_filter, filter, LIBXSMM_DNN_TENSOR_FORMAT_KCRS);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    /* Output buffer */
    conv_output_buffer = MALLOC(descriptor.N * descriptor.K * (descriptor.H + 2 * descriptor.pad_h_out) * (descriptor.W + 2 * descriptor.pad_w_out) * typesize);
    if (0 == conv_output_buffer) result = EXIT_FAILURE;
    conv_output = libxsmm_dnn_link_buffer(handle, LIBXSMM_DNN_OUTPUT, conv_output_buffer, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxsmm_dnn_bind_buffer(handle, conv_output, LIBXSMM_DNN_REGULAR_OUTPUT);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
  }

  /* Attempt to run the convolution. */
  if (EXIT_SUCCESS == result) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      const libxsmm_dnn_err_t r = libxsmm_dnn_execute_st(handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid);
      if (LIBXSMM_DNN_SUCCESS != r && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
        const char *const error_message = libxsmm_dnn_get_error(r);
        fprintf(stderr, "%s\n", error_message);
        result = EXIT_FAILURE;
      }
    }
  }

  /* Copy-out image into original format. */
  if (EXIT_SUCCESS == result) {
    status = libxsmm_dnn_copyout_buffer(conv_output, image, LIBXSMM_DNN_TENSOR_FORMAT_NCHW);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
  }

  /* Write the image into a different file. */
  if (EXIT_SUCCESS == result) {
    result = libxsmm_mhd_write(filename_out, size, pitch, 2, ncomponents,
      /* DNN type: assume that MHD I/O provides a super-set of types */
      (libxsmm_mhd_elemtype)type_out, image,
      0/*extension_header*/,
      0/*extension*/,
      0/*extension_size*/);
  }

  /* Release resources. */
  if (LIBXSMM_DNN_SUCCESS != libxsmm_dnn_destroy_conv_layer(handle)) result = EXIT_FAILURE;
  if (LIBXSMM_DNN_SUCCESS != libxsmm_dnn_destroy_buffer(conv_input)) result = EXIT_FAILURE;
  if (LIBXSMM_DNN_SUCCESS != libxsmm_dnn_destroy_buffer(conv_output)) result = EXIT_FAILURE;
  if (LIBXSMM_DNN_SUCCESS != libxsmm_dnn_destroy_filter(conv_filter)) result = EXIT_FAILURE;
  FREE(conv_output_buffer);
  FREE(conv_filter_buffer);
  FREE(conv_input_buffer);
  FREE(filter);
  FREE(image);

  return result;
}

