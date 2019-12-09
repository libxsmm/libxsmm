/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm_mhd.h>


int main(int argc, char* argv[])
{
  const char *const filename = (1 < argc ? argv[1] : "mhd_image.mhd");
  /* take some block-sizes, which are used to test leading dimensions */
  const int bw = LIBXSMM_MAX(2 < argc ? atoi(argv[2]) : 64, 1);
  const int bh = LIBXSMM_MAX(3 < argc ? atoi(argv[3]) : 64, 1);
  size_t ndims = 3, size[3], pitch[3], offset[3], ncomponents, header_size, extension_size;
  libxsmm_mhd_elemtype type;
  char data_filename[1024];
  void* data = NULL;
  int result;

  /* Read header information; function includes various sanity checks. */
  result = libxsmm_mhd_read_header(filename, sizeof(data_filename),
    data_filename, &ndims, size, &ncomponents, &type,
    &header_size, &extension_size);

  /* Allocate data according to the header information. */
  if (EXIT_SUCCESS == result) {
    size_t typesize;
    pitch[0] = (size[0] + bw - 1) / bw * bw;
    pitch[1] = (size[1] + bh - 1) / bh * bh;
    pitch[2] = size[2];
    /* center the image inside of the (pitched) buffer */
    offset[0] = (pitch[0] - size[0]) / 2;
    offset[1] = (pitch[1] - size[1]) / 2;
    offset[2] = 0;
    if (0 != libxsmm_mhd_typename(type, &typesize, NULL/*ctypename*/)) {
      const size_t nelements = pitch[0] * (1 < ndims ? (pitch[1] * (2 < ndims ? pitch[2] : 1)) : 1);
      data = malloc(ncomponents * nelements * typesize);
    }
    else {
      result = EXIT_FAILURE;
    }
  }

  /* perform tests with libxsmm_mhd_element_conversion (int2signed) */
  if (EXIT_SUCCESS == result) {
    short src = 2507, src_min = 0, src_max = 5000;
    float dst_f32; /* destination range is implicit due to type */
    signed char dst_i8; /* destination range is implicit due to type */
    result = libxsmm_mhd_element_conversion(
      &dst_f32, LIBXSMM_MHD_ELEMTYPE_F32/*dst_type*/, LIBXSMM_MHD_ELEMTYPE_I16/*src_type*/,
      &src, NULL/*src_min*/, NULL/*src_max*/);
    if (EXIT_SUCCESS == result && src != dst_f32) result = EXIT_FAILURE;
    if (EXIT_SUCCESS == result) {
      result = libxsmm_mhd_element_conversion(
        &dst_f32, LIBXSMM_MHD_ELEMTYPE_F32/*dst_type*/, LIBXSMM_MHD_ELEMTYPE_I16/*src_type*/,
        &src, &src_min, &src_max);
      if (EXIT_SUCCESS == result && src != dst_f32) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      result = libxsmm_mhd_element_conversion(
        &dst_i8, LIBXSMM_MHD_ELEMTYPE_I8/*dst_type*/, LIBXSMM_MHD_ELEMTYPE_I16/*src_type*/,
        &src, NULL/*src_min*/, NULL/*src_max*/);
      if (EXIT_SUCCESS == result && LIBXSMM_MIN(127, src) != dst_i8) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      result = libxsmm_mhd_element_conversion(
        &dst_i8, LIBXSMM_MHD_ELEMTYPE_I8/*dst_type*/, LIBXSMM_MHD_ELEMTYPE_I16/*src_type*/,
        &src, &src_min, &src_max);
      if (EXIT_SUCCESS == result && 64 != dst_i8) result = EXIT_FAILURE;
    }
  }

  /* perform tests with libxsmm_mhd_element_conversion (float2int) */
  if (EXIT_SUCCESS == result) {
    double src = 1975, src_min = -25071975, src_max = 1981;
    short dst_i16; /* destination range is implicit due to type */
    unsigned char dst_u8; /* destination range is implicit due to type */
    result = libxsmm_mhd_element_conversion(
      &dst_i16, LIBXSMM_MHD_ELEMTYPE_I16/*dst_type*/, LIBXSMM_MHD_ELEMTYPE_F64/*src_type*/,
      &src, NULL/*src_min*/, NULL/*src_max*/);
    if (EXIT_SUCCESS == result && src != dst_i16) result = EXIT_FAILURE;
    if (EXIT_SUCCESS == result) {
      result = libxsmm_mhd_element_conversion(
        &dst_i16, LIBXSMM_MHD_ELEMTYPE_I16/*dst_type*/, LIBXSMM_MHD_ELEMTYPE_F64/*src_type*/,
        &src, &src_min, &src_max);
      if (EXIT_SUCCESS == result && 2 != dst_i16) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      result = libxsmm_mhd_element_conversion(
        &dst_u8, LIBXSMM_MHD_ELEMTYPE_U8/*dst_type*/, LIBXSMM_MHD_ELEMTYPE_F64/*src_type*/,
        &src, NULL/*src_min*/, NULL/*src_max*/);
      if (EXIT_SUCCESS == result && LIBXSMM_MIN(255, src) != dst_u8) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      result = libxsmm_mhd_element_conversion(
        &dst_u8, LIBXSMM_MHD_ELEMTYPE_U8/*dst_type*/, LIBXSMM_MHD_ELEMTYPE_F64/*src_type*/,
        &src, &src_min, &src_max);
      if (EXIT_SUCCESS == result && 255 != dst_u8) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      src = -src;
      result = libxsmm_mhd_element_conversion(
        &dst_u8, LIBXSMM_MHD_ELEMTYPE_U8/*dst_type*/, LIBXSMM_MHD_ELEMTYPE_F64/*src_type*/,
        &src, &src_min, &src_max);
      if (EXIT_SUCCESS == result && 0 != dst_u8) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      result = libxsmm_mhd_element_conversion(
        &dst_i16, LIBXSMM_MHD_ELEMTYPE_I16/*dst_type*/, LIBXSMM_MHD_ELEMTYPE_F64/*src_type*/,
        &src, &src_min, &src_max);
      if (EXIT_SUCCESS == result && -3 != dst_i16) result = EXIT_FAILURE;
    }
  }

  /* Read the data according to the header into the allocated buffer. */
  if (EXIT_SUCCESS == result) {
    result = libxsmm_mhd_read(data_filename,
      offset, size, pitch, ndims, ncomponents, header_size, type, NULL/*type_data*/, data,
      NULL/*handle_element*/, NULL/*extension*/, 0/*extension_size*/);
  }

  /* Write the data into a new file; update header_size. */
  if (EXIT_SUCCESS == result) {
    result = libxsmm_mhd_write("mhd_test.mhd", NULL/*offset*/, pitch, pitch,
      ndims, ncomponents, type, NULL/*no conversion*/, data, &header_size,
      NULL/*extension_header*/,
      NULL/*extension*/,
      0/*extension_size*/);
  }

  /* Check the written data against the buffer. */
  if (EXIT_SUCCESS == result) {
    result = libxsmm_mhd_read(data_filename,
      offset, size, pitch, ndims, ncomponents, 0/*header_size*/,
      type, NULL/*type*/, data, libxsmm_mhd_element_comparison,
      NULL/*extension*/, 0/*extension_size*/);
  }

  /* Check the written data against the buffer with conversion. */
  if (EXIT_SUCCESS == result) {
    const libxsmm_mhd_elemtype tcomp = LIBXSMM_MHD_ELEMTYPE_F64;
    void* buffer = NULL;
    size_t typesize;
    if (0 != libxsmm_mhd_typename(tcomp, &typesize, NULL/*ctypename*/)) {
      const size_t nelements = pitch[0] * (1 < ndims ? (pitch[1] * (2 < ndims ? pitch[2] : 1)) : 1);
      buffer = malloc(ncomponents * nelements * typesize);
    }
    result = libxsmm_mhd_read(data_filename,
      offset, size, pitch, ndims, ncomponents, 0/*header_size*/,
      type, &tcomp, buffer, NULL/*libxsmm_mhd_element_comparison*/,
      NULL/*extension*/, 0/*extension_size*/);
    free(buffer);
  }

  /* Deallocate the buffer. */
  free(data);

  return result;
}

