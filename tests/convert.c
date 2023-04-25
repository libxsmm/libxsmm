/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <utils/libxsmm_utils.h>
#include <libxsmm.h>

#if !defined(CONVERT_SCALE)
# define CONVERT_SCALE(SIZE) (1.0/(SIZE))
#endif
#if !defined(CONVERT_SEED)
# define CONVERT_SEED 1
#endif
#if !defined(CONVERT_M)
# define CONVERT_M 256
#endif
#if !defined(CONVERT_N)
# define CONVERT_N 256
#endif


int main(/*int argc, char* argv[]*/void)
{
  int result = EXIT_SUCCESS;
  libxsmm_datatype type = LIBXSMM_DATATYPE_F32;
  size_t size[] = { CONVERT_M, CONVERT_N }, size1 = 0;
  size_t ndims = 2, channels = 1, header = 0;
  void *data = NULL, *data_lp = NULL;
  char filename[1024];

#if defined(__linux__)
  /* seed RNG for reproducible stochastic rounding */
  libxsmm_rng_set_seed(CONVERT_SEED);
#endif

  /* libxsmm_truncate_convert_f32_bf16 */
  if (EXIT_SUCCESS == libxsmm_mhd_read_header("convert_bf16_truncate.mhd", sizeof(filename), filename,
    &ndims, size, &channels, (libxsmm_mhd_elemtype*)&type, &header, NULL/*extension_size*/))
  { /* check against gold data */
    const size_t s = size[0] * size[1];
    if (size1 < s) {
      free(data_lp); free(data);
      data_lp = malloc(LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_BF16) * s);
      data = malloc(LIBXSMM_TYPESIZE(type) * s);
      size1 = s;
    }
    if (NULL != data && NULL != data_lp) {
      LIBXSMM_MATINIT(float, CONVERT_SEED, data, size[0], size[1], size[0], CONVERT_SCALE(s));
      libxsmm_truncate_convert_f32_bf16((const float*)data, (libxsmm_bfloat16*)data_lp, s);
      libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)data_lp, data, s);
      result = libxsmm_mhd_read(filename,
        NULL/*offset*/, size, NULL/*pitch*/, ndims, channels, header,
        (libxsmm_mhd_elemtype)type, NULL/*type*/, data, libxsmm_mhd_element_comparison,
        NULL/*extension*/, 0/*extension_size*/);
    }
    else result = EXIT_FAILURE;
  }
  else { /* write gold data */
    const size_t s = size[0] * size[1];
    if (size1 < s) {
      free(data_lp); free(data);
      data_lp = malloc(LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_BF16) * s);
      data = malloc(LIBXSMM_TYPESIZE(type) * s);
      size1 = s;
    }
    if (NULL != data && NULL != data_lp) {
      LIBXSMM_MATINIT(float, CONVERT_SEED, data, size[0], size[1], size[0], CONVERT_SCALE(s));
      libxsmm_truncate_convert_f32_bf16((const float*)data, (libxsmm_bfloat16*)data_lp, s);
      libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)data_lp, data, s);
      result = libxsmm_mhd_write("convert_bf16_truncate.mhd", NULL/*offset*/, size, NULL/*pitch*/, ndims, channels,
        (libxsmm_mhd_elemtype)type, NULL/*no conversion*/, data, NULL/*header_size*/,
        NULL/*extension_header*/, NULL/*extension*/, 0/*extension_size*/);
    }
    else result = EXIT_FAILURE;
  }

  /* libxsmm_rnaz_convert_fp32_bf16 */
  if (EXIT_SUCCESS == libxsmm_mhd_read_header("convert_bf16_rnaz.mhd", sizeof(filename), filename,
    &ndims, size, &channels, (libxsmm_mhd_elemtype*)&type, &header, NULL/*extension_size*/))
  { /* check against gold data */
    const size_t s = size[0] * size[1];
    if (size1 < s) {
      free(data_lp); free(data);
      data_lp = malloc(LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_BF16) * s);
      data = malloc(LIBXSMM_TYPESIZE(type) * s);
      size1 = s;
    }
    if (NULL != data && NULL != data_lp) {
      LIBXSMM_MATINIT(float, CONVERT_SEED, data, size[0], size[1], size[0], CONVERT_SCALE(s));
      libxsmm_rnaz_convert_fp32_bf16((const float*)data, (libxsmm_bfloat16*)data_lp, s);
      libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)data_lp, data, s);
      result = libxsmm_mhd_read(filename,
        NULL/*offset*/, size, NULL/*pitch*/, ndims, channels, header,
        (libxsmm_mhd_elemtype)type, NULL/*type*/, data, libxsmm_mhd_element_comparison,
        NULL/*extension*/, 0/*extension_size*/);
    }
    else result = EXIT_FAILURE;
  }
  else { /* write gold data */
    const size_t s = size[0] * size[1];
    if (size1 < s) {
      free(data_lp); free(data);
      data_lp = malloc(LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_BF16) * s);
      data = malloc(LIBXSMM_TYPESIZE(type) * s);
      size1 = s;
    }
    if (NULL != data && NULL != data_lp) {
      LIBXSMM_MATINIT(float, CONVERT_SEED, data, size[0], size[1], size[0], CONVERT_SCALE(s));
      libxsmm_rnaz_convert_fp32_bf16((const float*)data, (libxsmm_bfloat16*)data_lp, s);
      libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)data_lp, data, s);
      result = libxsmm_mhd_write("convert_bf16_rnaz.mhd", NULL/*offset*/, size, NULL/*pitch*/, ndims, channels,
        (libxsmm_mhd_elemtype)type, NULL/*no conversion*/, data, NULL/*header_size*/,
        NULL/*extension_header*/, NULL/*extension*/, 0/*extension_size*/);
    }
    else result = EXIT_FAILURE;
  }

  /* libxsmm_rne_convert_fp32_bf16 */
  if (EXIT_SUCCESS == libxsmm_mhd_read_header("convert_bf16_rne.mhd", sizeof(filename), filename,
    &ndims, size, &channels, (libxsmm_mhd_elemtype*)&type, &header, NULL/*extension_size*/))
  { /* check against gold data */
    const size_t s = size[0] * size[1];
    if (size1 < s) {
      free(data_lp); free(data);
      data_lp = malloc(LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_BF16) * s);
      data = malloc(LIBXSMM_TYPESIZE(type) * s);
      size1 = s;
    }
    if (NULL != data && NULL != data_lp) {
      LIBXSMM_MATINIT(float, CONVERT_SEED, data, size[0], size[1], size[0], CONVERT_SCALE(s));
      libxsmm_rne_convert_fp32_bf16((const float*)data, (libxsmm_bfloat16*)data_lp, s);
      libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)data_lp, data, s);
      result = libxsmm_mhd_read(filename,
        NULL/*offset*/, size, NULL/*pitch*/, ndims, channels, header,
        (libxsmm_mhd_elemtype)type, NULL/*type*/, data, libxsmm_mhd_element_comparison,
        NULL/*extension*/, 0/*extension_size*/);
    }
    else result = EXIT_FAILURE;
  }
  else { /* write gold data */
    const size_t s = size[0] * size[1];
    if (size1 < s) {
      free(data_lp); free(data);
      data_lp = malloc(LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_BF16) * s);
      data = malloc(LIBXSMM_TYPESIZE(type) * s);
      size1 = s;
    }
    if (NULL != data && NULL != data_lp) {
      LIBXSMM_MATINIT(float, CONVERT_SEED, data, size[0], size[1], size[0], CONVERT_SCALE(s));
      libxsmm_rne_convert_fp32_bf16((const float*)data, (libxsmm_bfloat16*)data_lp, s);
      libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)data_lp, data, s);
      result = libxsmm_mhd_write("convert_bf16_rne.mhd", NULL/*offset*/, size, NULL/*pitch*/, ndims, channels,
        (libxsmm_mhd_elemtype)type, NULL/*no conversion*/, data, NULL/*header_size*/,
        NULL/*extension_header*/, NULL/*extension*/, 0/*extension_size*/);
    }
    else result = EXIT_FAILURE;
  }

  /* libxsmm_rne_convert_fp32_f16 */
  if (EXIT_SUCCESS == libxsmm_mhd_read_header("convert_f16_rne.mhd", sizeof(filename), filename,
    &ndims, size, &channels, (libxsmm_mhd_elemtype*)&type, &header, NULL/*extension_size*/))
  { /* check against gold data */
    const size_t s = size[0] * size[1];
    if (size1 < s) {
      free(data_lp); free(data);
      data_lp = malloc(LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F16) * s);
      data = malloc(LIBXSMM_TYPESIZE(type) * s);
      size1 = s;
    }
    if (NULL != data && NULL != data_lp) {
      LIBXSMM_MATINIT(float, CONVERT_SEED, data, size[0], size[1], size[0], CONVERT_SCALE(s));
      libxsmm_rne_convert_fp32_f16((const float*)data, (libxsmm_float16*)data_lp, s);
      libxsmm_convert_f16_f32((const libxsmm_float16*)data_lp, data, s);
      result = libxsmm_mhd_read(filename,
        NULL/*offset*/, size, NULL/*pitch*/, ndims, channels, header,
        (libxsmm_mhd_elemtype)type, NULL/*type*/, data, libxsmm_mhd_element_comparison,
        NULL/*extension*/, 0/*extension_size*/);
    }
    else result = EXIT_FAILURE;
  }
  else { /* write gold data */
    const size_t s = size[0] * size[1];
    if (size1 < s) {
      free(data_lp); free(data);
      data_lp = malloc(LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F16) * s);
      data = malloc(LIBXSMM_TYPESIZE(type) * s);
      size1 = s;
    }
    if (NULL != data && NULL != data_lp) {
      LIBXSMM_MATINIT(float, CONVERT_SEED, data, size[0], size[1], size[0], CONVERT_SCALE(s));
      libxsmm_rne_convert_fp32_f16((const float*)data, (libxsmm_float16*)data_lp, s);
      libxsmm_convert_f16_f32((const libxsmm_float16*)data_lp, data, s);
      result = libxsmm_mhd_write("convert_f16_rne.mhd", NULL/*offset*/, size, NULL/*pitch*/, ndims, channels,
        (libxsmm_mhd_elemtype)type, NULL/*no conversion*/, data, NULL/*header_size*/,
        NULL/*extension_header*/, NULL/*extension*/, 0/*extension_size*/);
    }
    else result = EXIT_FAILURE;
  }

  /* libxsmm_rne_convert_fp32_bf8 */
  if (EXIT_SUCCESS == libxsmm_mhd_read_header("convert_bf8_rne.mhd", sizeof(filename), filename,
    &ndims, size, &channels, (libxsmm_mhd_elemtype*)&type, &header, NULL/*extension_size*/))
  { /* check against gold data */
    const size_t s = size[0] * size[1];
    if (size1 < s) {
      free(data_lp); free(data);
      data_lp = malloc(LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_BF8) * s);
      data = malloc(LIBXSMM_TYPESIZE(type) * s);
      size1 = s;
    }
    if (NULL != data && NULL != data_lp) {
      LIBXSMM_MATINIT(float, CONVERT_SEED, data, size[0], size[1], size[0], CONVERT_SCALE(s));
      libxsmm_rne_convert_fp32_bf8((const float*)data, (libxsmm_bfloat8*)data_lp, s);
      libxsmm_convert_bf8_f32((const libxsmm_bfloat8*)data_lp, data, s);
      result = libxsmm_mhd_read(filename,
        NULL/*offset*/, size, NULL/*pitch*/, ndims, channels, header,
        (libxsmm_mhd_elemtype)type, NULL/*type*/, data, libxsmm_mhd_element_comparison,
        NULL/*extension*/, 0/*extension_size*/);
    }
    else result = EXIT_FAILURE;
  }
  else { /* write gold data */
    const size_t s = size[0] * size[1];
    if (size1 < s) {
      free(data_lp); free(data);
      data_lp = malloc(LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_BF8) * s);
      data = malloc(LIBXSMM_TYPESIZE(type) * s);
      size1 = s;
    }
    if (NULL != data && NULL != data_lp) {
      LIBXSMM_MATINIT(float, CONVERT_SEED, data, size[0], size[1], size[0], CONVERT_SCALE(s));
      libxsmm_rne_convert_fp32_bf8((const float*)data, (libxsmm_bfloat8*)data_lp, s);
      libxsmm_convert_bf8_f32((const libxsmm_bfloat8*)data_lp, data, s);
      result = libxsmm_mhd_write("convert_bf8_rne.mhd", NULL/*offset*/, size, NULL/*pitch*/, ndims, channels,
        (libxsmm_mhd_elemtype)type, NULL/*no conversion*/, data, NULL/*header_size*/,
        NULL/*extension_header*/, NULL/*extension*/, 0/*extension_size*/);
    }
    else result = EXIT_FAILURE;
  }

#if defined(__linux__)
  /* libxsmm_stochastic_convert_fp32_bf8 */
  if (EXIT_SUCCESS == libxsmm_mhd_read_header("convert_bf8_stochastic.mhd", sizeof(filename), filename,
    &ndims, size, &channels, (libxsmm_mhd_elemtype*)&type, &header, NULL/*extension_size*/))
  { /* check against gold data */
    const size_t s = size[0] * size[1];
    if (size1 < s) {
      free(data_lp); free(data);
      data_lp = malloc(LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_BF8) * s);
      data = malloc(LIBXSMM_TYPESIZE(type) * s);
      size1 = s;
    }
    if (NULL != data && NULL != data_lp) {
      LIBXSMM_MATINIT(float, CONVERT_SEED, data, size[0], size[1], size[0], CONVERT_SCALE(s));
      libxsmm_stochastic_convert_fp32_bf8((const float*)data, (libxsmm_bfloat8*)data_lp, s);
      libxsmm_convert_bf8_f32((const libxsmm_bfloat8*)data_lp, data, s);
      result = libxsmm_mhd_read(filename,
        NULL/*offset*/, size, NULL/*pitch*/, ndims, channels, header,
        (libxsmm_mhd_elemtype)type, NULL/*type*/, data, libxsmm_mhd_element_comparison,
        NULL/*extension*/, 0/*extension_size*/);
    }
    else result = EXIT_FAILURE;
  }
  else { /* write gold data */
    const size_t s = size[0] * size[1];
    if (size1 < s) {
      free(data_lp); free(data);
      data_lp = malloc(LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_BF8) * s);
      data = malloc(LIBXSMM_TYPESIZE(type) * s);
      size1 = s;
    }
    if (NULL != data && NULL != data_lp) {
      LIBXSMM_MATINIT(float, CONVERT_SEED, data, size[0], size[1], size[0], CONVERT_SCALE(s));
      libxsmm_stochastic_convert_fp32_bf8((const float*)data, (libxsmm_bfloat8*)data_lp, s);
      libxsmm_convert_bf8_f32((const libxsmm_bfloat8*)data_lp, data, s);
      result = libxsmm_mhd_write("convert_bf8_stochastic.mhd", NULL/*offset*/, size, NULL/*pitch*/, ndims, channels,
        (libxsmm_mhd_elemtype)type, NULL/*no conversion*/, data, NULL/*header_size*/,
        NULL/*extension_header*/, NULL/*extension*/, 0/*extension_size*/);
    }
    else result = EXIT_FAILURE;
  }
#endif

  /* libxsmm_rne_convert_fp32_hf8 */
  if (EXIT_SUCCESS == libxsmm_mhd_read_header("convert_hf8_rne.mhd", sizeof(filename), filename,
    &ndims, size, &channels, (libxsmm_mhd_elemtype*)&type, &header, NULL/*extension_size*/))
  { /* check against gold data */
    const size_t s = size[0] * size[1];
    if (size1 < s) {
      free(data_lp); free(data);
      data_lp = malloc(LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_HF8) * s);
      data = malloc(LIBXSMM_TYPESIZE(type) * s);
      size1 = s;
    }
    if (NULL != data && NULL != data_lp) {
      LIBXSMM_MATINIT(float, CONVERT_SEED, data, size[0], size[1], size[0], CONVERT_SCALE(s));
      libxsmm_rne_convert_fp32_hf8((const float*)data, (libxsmm_hfloat8*)data_lp, s);
      libxsmm_convert_hf8_f32((const libxsmm_hfloat8*)data_lp, data, s);
      result = libxsmm_mhd_read(filename,
        NULL/*offset*/, size, NULL/*pitch*/, ndims, channels, header,
        (libxsmm_mhd_elemtype)type, NULL/*type*/, data, libxsmm_mhd_element_comparison,
        NULL/*extension*/, 0/*extension_size*/);
    }
    else result = EXIT_FAILURE;
  }
  else { /* write gold data */
    const size_t s = size[0] * size[1];
    if (size1 < s) {
      free(data_lp); free(data);
      data_lp = malloc(LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_HF8) * s);
      data = malloc(LIBXSMM_TYPESIZE(type) * s);
      size1 = s;
    }
    if (NULL != data && NULL != data_lp) {
      LIBXSMM_MATINIT(float, CONVERT_SEED, data, size[0], size[1], size[0], CONVERT_SCALE(s));
      libxsmm_rne_convert_fp32_hf8((const float*)data, (libxsmm_hfloat8*)data_lp, s);
      libxsmm_convert_hf8_f32((const libxsmm_hfloat8*)data_lp, data, s);
      result = libxsmm_mhd_write("convert_hf8_rne.mhd", NULL/*offset*/, size, NULL/*pitch*/, ndims, channels,
        (libxsmm_mhd_elemtype)type, NULL/*no conversion*/, data, NULL/*header_size*/,
        NULL/*extension_header*/, NULL/*extension*/, 0/*extension_size*/);
    }
    else result = EXIT_FAILURE;
  }

  free(data_lp);
  free(data);

  return result;
}
