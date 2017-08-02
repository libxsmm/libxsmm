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
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
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
  int result = 0 != strcmp(filename_in, filename_out) ? EXIT_SUCCESS : EXIT_FAILURE;
  size_t ndims = 3, size[3], pitch[2], ncomponents = 0, header_size = 0, extension_size;
  libxsmm_dnn_datatype type_out = LIBXSMM_DNN_DATATYPE_F32;
  libxsmm_mhd_elemtype type_in = LIBXSMM_MHD_ELEMTYPE_UNKNOWN;
  libxsmm_dnn_conv_desc descriptor;
  libxsmm_dnn_layer* layer = 0;
  libxsmm_dnn_err_t status;
  char filename_data[1024];
  void* data = 0;

  /* Read MHD-header information; function includes various sanity checks. */
  if (EXIT_SUCCESS == result) {
    result = libxsmm_mhd_read_header(filename_in, sizeof(filename_data),
      filename_data, &ndims, size, &ncomponents, &type_in,
      &header_size, &extension_size);
  }

  /* Only accept 2-d images or a single slice of a 3d-image. */
  if (2 == ndims || (3 == ndims && 1 == size[2])) {
    pitch[0] = size[0];
    pitch[1] = size[1];
  }
  else {
    result = EXIT_FAILURE;
  }

  /* Allocate data according to the MHD-header information. */
  if (EXIT_SUCCESS == result) {
    size_t typesize;
    /* DNN type: assume that MHD I/O provides a super-set of types */
    if (0 != libxsmm_mhd_typename((libxsmm_mhd_elemtype)type_out, &typesize)) {
      const size_t nelements = pitch[0] * pitch[1];
      data = malloc(ncomponents * typesize * nelements);
    }
    else {
      result = EXIT_FAILURE;
    }
  }
#if 1
  /* Read the data according to the header into the allocated buffer. */
  if (EXIT_SUCCESS == result) {
    result = libxsmm_mhd_read(filename_data,
      size, pitch, ndims, ncomponents, header_size, type_in,
      /* eventually perform a type-conversion (type_in != type_out) */
      (const libxsmm_mhd_elemtype*)&type_out, data,
      0/*handle_element*/, 0/*extension*/, 0/*extension_size*/);
  }
#endif
#if 0
  if (EXIT_SUCCESS == result) {
    descriptor.N = 1;
    descriptor.H = (int)size[1];
    descriptor.W = (int)size[0];
    descriptor.C = (int)ncomponents;
    descriptor.K = (int)ncomponents;
    descriptor.R = 39;
    descriptor.S = 39;
    descriptor.u = 1;
    descriptor.v = 1;
    descriptor.pad_h = 0;
    descriptor.pad_w = 0;
    descriptor.pad_h_in = 0;
    descriptor.pad_w_in = 0;
    descriptor.pad_h_out = 0;
    descriptor.pad_w_out = 0;
    descriptor.threads = 1;
    descriptor.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT/*LIBXSMM_DNN_CONV_ALGO_AUTO*/;
    descriptor.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    descriptor.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    descriptor.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
    descriptor.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;
    descriptor.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
    descriptor.datatype = LIBXSMM_DNN_DATATYPE_F32;
    layer = libxsmm_dnn_create_conv_layer(descriptor, &status);
    if (LIBXSMM_DNN_SUCCESS != status) result = EXIT_FAILURE;
  }
#endif
  /* Write the data into a different file. */
  if (EXIT_SUCCESS == result) {
    result = libxsmm_mhd_write(filename_out, size, pitch, 2, ncomponents,
      /* DNN type: assume that MHD I/O provides a super-set of types */
      (libxsmm_mhd_elemtype)type_out, data,
      0/*extension_header*/,
      0/*extension*/,
      0/*extension_size*/);
  }

  /* Release resources. */
  if (LIBXSMM_DNN_SUCCESS != libxsmm_dnn_destroy_conv_layer(layer)) result = EXIT_FAILURE;
  free(data);

  return result;
}

