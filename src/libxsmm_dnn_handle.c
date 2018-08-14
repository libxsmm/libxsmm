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
/* Alexander Heinecke, Hans Pabst, Rajkishore Barik,
 * Ankush Mandal, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_handle.h"
#include "libxsmm_main.h"
#include <libxsmm.h>
#include "libxsmm_dnn_dryruns.h"
#include "libxsmm_dnn_setup.h"

#if !defined(LIBXSMM_DNN_HANDLE_DEBUG) && 0
# define LIBXSMM_DNN_HANDLE_DEBUG
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
# include <stdio.h>
#endif
#if defined(_OPENMP)
# include <omp.h>
#endif
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_internal_create_conv_handle_direct( libxsmm_dnn_layer* handle ) {
  /* flag to test if we found an architecture which is supported */
  int noarch = 1;
  int internal_format_type;
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  const char *const env = getenv("LIBXSMM_DNN_INTERNAL_FORMAT");
  LIBXSMM_ASSERT(0 != handle);

  if ( 0 == env || 0 == *env) {
    /* Default internal format type */
    handle->custom_format_type = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1;
  } else {
    internal_format_type = atoi(env);
    if (internal_format_type == 1) {
      handle->custom_format_type = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1;
    } else if ( internal_format_type == 2) {
      handle->custom_format_type = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2;
    } else {
      status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
      free(handle);
      handle = 0;
      return status;
    }
  }

  /* let's enable generic code path by default */
  handle->use_fwd_generic = 1;
  handle->use_bwd_generic = 1;
  handle->use_upd_generic = 1;

  handle->use_thread_private_jit = 0;
  /* If we have AVX512 arch consider kernel streams  */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if (/* If we use any options/fuse ops, keep kernel streams disabled */
    0 >= (handle->desc.fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS)
    /* If we do not run on custom/custom format, keep kernel streams disabled */
    && handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM
    && handle->filter_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM)
  {
# if (LIBXSMM_X86_AVX512 > LIBXSMM_STATIC_TARGET_ARCH)
    if (LIBXSMM_X86_AVX512 <= libxsmm_target_archid)
# endif
    {
      handle->use_thread_private_jit = 1;
    }
  }
#endif

  /* If we have AVX512 and kernel streams is enabled, then we generate specialized code */
  if (handle->use_thread_private_jit != 0) {
    LIBXSMM_ASSERT(LIBXSMM_X86_AVX512 <= libxsmm_target_archid);

    /* This is basically a decision pertaining for all three passes: FWD, BWD and UPD */
    /* Initialize fields that control layer fusion */
    noarch = 0;
    handle->compute_batch_stats_in_kernel = 0;
    handle->compute_max_in_kernel_fwd = 0;
    handle->compute_max_in_kernel_bwd = 0;
    handle->perform_relu_in_kernel = 0;

    /* Calculate feature map blocking factors based on precision and datatypes */
    status = libxsmm_dnn_setup_feature_map_blocks(handle, &noarch);
    if ( status ==  LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE) {
      free(handle);
      handle = 0;
      return status;
    }

    /* Forward path setup */
    status = libxsmm_dnn_setup_fwd(handle, &noarch);

    /* Backward path setup */
    status = libxsmm_dnn_setup_bwd(handle, &noarch);

    /* Weight update path setup */
    status = libxsmm_dnn_setup_upd(handle, &noarch);

    /* Calculate scratch requirements */
    libxsmm_dnn_setup_scratch(handle);
  }

  if (0 != noarch) { /* Setup generic code generation */
    status = libxsmm_dnn_setup_generic(handle);
  }

#if !(defined(LIBXSMM_DNN_VLA_TLS1) && defined(LIBXSMM_DNN_VLA_TLS2) && defined(LIBXSMM_DNN_VLA_TLS3))
# if 0 /* TODO: Bf16 currently triggers error 90005 before but we want to continue */
  if (LIBXSMM_DNN_SUCCESS == status)
# endif
  {
# if !defined(LIBXSMM_DNN_VLA_TLS1)
    if (0 != handle->use_fwd_generic || 0 != handle->use_bwd_generic || 0 != handle->use_upd_generic) {
      const size_t padded_h = ((size_t)2 * handle->desc.pad_h) + handle->desc.H, padded_w = ((size_t)2 * handle->desc.pad_w) + handle->desc.W;
      const size_t size5_tensor = padded_h * padded_w * handle->ifmblock * libxsmm_dnn_typesize(handle->datatype_in);
      const size_t size5 = LIBXSMM_UP2(size5_tensor, LIBXSMM_CACHELINE) * handle->desc.threads;
      if (handle->max_scratch5_size < size5) handle->max_scratch5_size = size5;
    }
    handle->scratch5 = 0;
# endif
# if !defined(LIBXSMM_DNN_VLA_TLS2)
    handle->scratch6_size = 0;
#   if 0 /* make float-accumulation scratch always available as it is referenced even if below property is false */
    if (handle->use_accumulation_scratch)
#   endif
    {
      const size_t size6a = (size_t)handle->ofmblock * handle->ofw * handle->ofh * sizeof(float);
      const size_t size6b = (size_t)handle->ifmblock * handle->fm_lp_block *  handle->desc.W * handle->desc.H * sizeof(float);
      const size_t size6 = ( size6a > size6b ) ? size6a : size6b;
      handle->scratch6_size = LIBXSMM_UP2(size6, LIBXSMM_CACHELINE) * handle->desc.threads;
    }
    if (0 != handle->use_upd_generic) {
      const size_t output_typesize = libxsmm_dnn_typesize(handle->datatype_out);
      const size_t size6_tensor = (size_t)handle->ofhp * handle->ofwp * handle->ofmblock * output_typesize;
      const size_t size6 = LIBXSMM_UP2(size6_tensor, LIBXSMM_CACHELINE) * handle->desc.threads;
      if (handle->scratch6_size < size6) handle->scratch6_size = size6;
    }
    handle->scratch6 = 0;
# endif
# if !defined(LIBXSMM_DNN_VLA_TLS3)
    if (0 != handle->use_upd_generic) {
      /* FIXME: currently filter data-type is always smaller/equal output type */
      const size_t filter_typesize = libxsmm_dnn_typesize(handle->datatype_out);
      const size_t size7 = (size_t)handle->desc.R * handle->desc.S * handle->ifmblock * handle->ofmblock * filter_typesize;
      handle->scratch7_size = LIBXSMM_UP2(size7, LIBXSMM_CACHELINE) * handle->desc.threads;
    }
    else {
      handle->scratch7_size = 0;
    }
    handle->scratch7 = 0;
# endif
  }
#endif
  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_internal_create_conv_handle_winograd_check( libxsmm_dnn_layer* handle ) {
  /* flag to test if we found an architecture which is supported */
  int noarch = 1;
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  const char *const env = getenv("LIBXSMM_DNN_INTERNAL_FORMAT");
  int internal_format_type;
  if ( 0 == env || 0 == *env) {
    /* Default internal format type */
    handle->custom_format_type = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1;
  } else {
    internal_format_type = atoi(env);
    if (internal_format_type == 1) {
      handle->custom_format_type = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1;
    } else if ( internal_format_type == 2) {
      handle->custom_format_type = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2;
    } else {
      status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
      free(handle);
      handle = 0;
      return status;
    }
  }

  /* now architecture specific */
  if ((libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
        libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
        libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM  ||
        libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL    ) &&
      (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) &&
      (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) &&
      (0 == (handle->desc.C % 16) && 0 == (handle->desc.K % 16)) &&
      (3 == handle->desc.R && 3 == handle->desc.S) &&
      (1 == handle->desc.u && 1 == handle->desc.v))
  {
    noarch = 0;
    /* calculate blockings */
    handle->ifmblock = 16;
    handle->ofmblock = 16;
    handle->blocksifm = handle->desc.C / 16;
    handle->blocksofm = handle->desc.K / 16;
    handle->fm_lp_block = 1;

  } else {
    status = LIBXSMM_DNN_WARN_FALLBACK;
  }

  if (noarch == 0) {
    libxsmm_convolution_winograd_descriptor wino_desc_fp;
    libxsmm_convolution_winograd_descriptor wino_desc_bp;
    libxsmm_convolution_winograd_descriptor wino_desc_wu;
    const int alpha = 6;
    const int tileSize = alpha - 2;
    int allowed_unroll = 0;
    int max_acc = 0;
    int temp_ur;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
    int flagBenchmark = 0;
#endif
    /* Forward path */
    { wino_desc_fp.alpha = alpha;
      wino_desc_fp.jtiles = (handle->ofh + tileSize - 1) / tileSize;
      wino_desc_fp.itiles = (handle->ofw + tileSize - 1) / tileSize;

      /* LUT for DeepBench */
      if ((240 == handle->ofw) && (24 == handle->ofh) && (16 == handle->desc.N) && (16 == handle->desc.C) && (32 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 1;
        wino_desc_fp.ur = 6;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((120 == handle->ofw) && (12 == handle->ofh) && (16 == handle->desc.N) && (32 == handle->desc.C) && (64 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 1;
        wino_desc_fp.ur = 6;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((60 == handle->ofw) && (6 == handle->ofh) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 1;
        wino_desc_fp.ur = 6;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((54 == handle->ofw) && (54 == handle->ofh) && (8 == handle->desc.N) && (64 == handle->desc.C) && (64 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 1;
        wino_desc_fp.ur = 7;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((27 == handle->ofw) && (27 == handle->ofh) && (8 == handle->desc.N) && (128 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 1;
        wino_desc_fp.ur = 7;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 8;
        wino_desc_fp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((7 == handle->ofw) && (7 == handle->ofh) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 8;
        wino_desc_fp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((112 == handle->ofw) && (112 == handle->ofh) && (8 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 1;
        wino_desc_fp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((56 == handle->ofw) && (56 == handle->ofh) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 1;
        wino_desc_fp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->ofw) && (28 == handle->ofh) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 8;
        wino_desc_fp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((7 == handle->ofw) && (7 == handle->ofh) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 8;
        wino_desc_fp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((112 == handle->ofw) && (112 == handle->ofh) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 1;
        wino_desc_fp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((56 == handle->ofw) && (56 == handle->ofh) && (16 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 1;
        wino_desc_fp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->ofw) && (28 == handle->ofh) && (16 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 4;
        wino_desc_fp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((7 == handle->ofw) && (7 == handle->ofh) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 16;
        wino_desc_fp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      }

      /* LUT for AlexNet */
      else if ((13 == handle->ofw) && (13 == handle->ofh) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((13 == handle->ofw) && (13 == handle->ofh) && (64 <= handle->desc.N) && (384 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((13 == handle->ofw) && (13 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      }

      /* LUT for GoogLenetV1 */
      else if ((56 == handle->ofw) && (56 == handle->ofh) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (192 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 1;
        wino_desc_fp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = (0 == wino_desc_fp.bimg % 2) ? 14 : 7;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (192 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = (0 == wino_desc_fp.bimg % 2) ? 14 : 7;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (208 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (112 == handle->desc.C) && (224 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (144 == handle->desc.C) && (288 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((7 == handle->ofw) && (7 == handle->ofh) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = 4;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((7 == handle->ofw) && (7 == handle->ofh) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = 4;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      }

      /* LUT for Overfeat */
      else if ((12 == handle->ofw) && (12 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = (0 == wino_desc_fp.bimg % 4) ? 12 :
          (0 == wino_desc_fp.bimg % 2) ? 6 : 3;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((12 == handle->ofw) && (12 == handle->ofh) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (1024 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = (0 == wino_desc_fp.bimg % 4) ? 12 :
          (0 == wino_desc_fp.bimg % 2) ? 6 : 3;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((12 == handle->ofw) && (12 == handle->ofh) && (64 <= handle->desc.N) && (1024 == handle->desc.C) && (1024 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = (0 == wino_desc_fp.bimg % 4) ? 12 :
          (0 == wino_desc_fp.bimg % 2) ? 6 : 3;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      }

      /* LUT for VGGA */
      else if ((112 == handle->ofw) && (112 == handle->ofh) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 1;
        wino_desc_fp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((56 == handle->ofw) && (56 == handle->ofh) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 1;
        wino_desc_fp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((56 == handle->ofw) && (56 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = 1;
        wino_desc_fp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = (0 == wino_desc_fp.bimg % 2) ? 14 : 7;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = (0 == wino_desc_fp.bimg % 2) ? 14 : 7;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_fp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_fp.ur = 4;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      }

      /* General scenario */
      else {
        if ((handle->desc.N % 4) == 0) {
          wino_desc_fp.bimg = 4;
        } else if ((handle->desc.N % 2) == 0) {
          wino_desc_fp.bimg = 2;
        } else {
          wino_desc_fp.bimg = 1;
        }
        if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
          max_acc = 24;
        } else {
          max_acc = 26;
        }
        wino_desc_fp.ur = libxsmm_product_limit(wino_desc_fp.itiles*wino_desc_fp.jtiles*wino_desc_fp.bimg, max_acc, 0);
        /* ur should be at least 14 to hide qfma latency */
        temp_ur = LIBXSMM_MIN(LIBXSMM_MAX(wino_desc_fp.ur, 14), wino_desc_fp.itiles*wino_desc_fp.jtiles*wino_desc_fp.bimg);
        if (0 == wino_desc_fp.itiles*wino_desc_fp.jtiles*wino_desc_fp.bimg % temp_ur) {
          wino_desc_fp.ur = temp_ur;
        }
      }

      /* The following condition checks whether we have encountered an input which is listed in our benchmark LUT */
#if defined(LIBXSMM_DNN_HANDLE_DEBUG) && 0
      if (flagBenchmark) printf("In benchmark\n");
#elif defined(LIBXSMM_DNN_HANDLE_DEBUG)
      LIBXSMM_UNUSED(flagBenchmark);
#endif
      /* ur_ifm = blocksifm so that we don't need to zero-initialize M and use streaming store */
      wino_desc_fp.ur_ifm = handle->blocksifm;
      wino_desc_fp.blocks_ifm = handle->blocksifm;

      handle->cwino_fwd = wino_desc_fp;

      /* TODO check JIT errors */
      if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM  ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL    )
      {
        wino_desc_fp.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
        handle->code_fwd[0].pmm = libxsmm_create_xconv_wino_forward(&wino_desc_fp);
        wino_desc_fp.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1;
        handle->code_fwd[1].pmm = libxsmm_create_xconv_wino_forward(&wino_desc_fp);
        wino_desc_fp.prefetch = (libxsmm_convolution_prefetch_type)(LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2);
        handle->code_fwd[2].pmm = libxsmm_create_xconv_wino_forward(&wino_desc_fp);
      } else {
        assert(0/*should not happen*/);
      }
    }
    /* Backward path */
    { wino_desc_bp.alpha = alpha;
      wino_desc_bp.jtiles = (handle->desc.H + tileSize - 1) / tileSize;
      wino_desc_bp.itiles = (handle->desc.W + tileSize - 1) / tileSize;

      /* LUT for DeepBench */
      if ((240 == handle->desc.W) && (24 == handle->desc.H) && (16 == handle->desc.N) && (16 == handle->desc.C) && (32 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 1;
        wino_desc_bp.ur = 6;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((120 == handle->desc.W) && (12 == handle->desc.H) && (16 == handle->desc.N) && (32 == handle->desc.C) && (64 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 1;
        wino_desc_bp.ur = 6;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((60 == handle->desc.W) && (6 == handle->desc.H) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 1;
        wino_desc_bp.ur = 6;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((54 == handle->desc.W) && (54 == handle->desc.H) && (8 == handle->desc.N) && (64 == handle->desc.C) && (64 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 1;
        wino_desc_bp.ur = 7;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((27 == handle->desc.W) && (27 == handle->desc.H) && (8 == handle->desc.N) && (128 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 1;
        wino_desc_bp.ur = 7;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 8;
        wino_desc_bp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 8;
        wino_desc_bp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((112 == handle->desc.W) && (112 == handle->desc.H) && (8 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 1;
        wino_desc_bp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 1;
        wino_desc_bp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 8;
        wino_desc_bp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 8;
        wino_desc_bp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((112 == handle->desc.W) && (112 == handle->desc.H) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 1;
        wino_desc_bp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (16 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 1;
        wino_desc_bp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (16 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 4;
        wino_desc_bp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 16;
        wino_desc_bp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      }

      /* LUT for AlexNet */
      else if ((13 == handle->desc.W) && (13 == handle->desc.H) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((13 == handle->desc.W) && (13 == handle->desc.H) && (64 <= handle->desc.N) && (384 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((13 == handle->desc.W) && (13 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      }

      /* LUT for GoogLenetV1 */
      else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (192 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 1;
        wino_desc_bp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = (0 == wino_desc_bp.bimg % 2) ? 14 : 7;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (192 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = (0 == wino_desc_bp.bimg % 2) ? 14 : 7;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (208 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (112 == handle->desc.C) && (224 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (144 == handle->desc.C) && (288 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = 16;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = 4;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = 4;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      }

      /* LUT for Overfeat */
      else if ((12 == handle->desc.W) && (12 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = (0 == wino_desc_bp.bimg % 4) ? 12 :
          (0 == wino_desc_bp.bimg % 2) ? 6 : 3;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((12 == handle->desc.W) && (12 == handle->desc.H) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (1024 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = (0 == wino_desc_bp.bimg % 4) ? 12 :
          (0 == wino_desc_bp.bimg % 2) ? 6 : 3;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((12 == handle->desc.W) && (12 == handle->desc.H) && (64 <= handle->desc.N) && (1024 == handle->desc.C) && (1024 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = (0 == wino_desc_bp.bimg % 4) ? 12 :
          (0 == wino_desc_bp.bimg % 2) ? 6 : 3;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      }

      /* LUT for VGGA */
      else if ((112 == handle->desc.W) && (112 == handle->desc.H) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 1;
        wino_desc_bp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 1;
        wino_desc_bp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = 1;
        wino_desc_bp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = 14;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = (0 == wino_desc_bp.bimg % 2) ? 14 : 7;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_bp.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_bp.ur = 4;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      }

      /* General scenario */
      else {
        wino_desc_bp.bimg = wino_desc_fp.bimg;
        if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
          max_acc = 24;
        } else {
          max_acc = 26;
        }
        wino_desc_bp.ur = libxsmm_product_limit(wino_desc_bp.itiles*wino_desc_bp.jtiles*wino_desc_bp.bimg, max_acc, 0);
        temp_ur = LIBXSMM_MIN(LIBXSMM_MAX(wino_desc_bp.ur, 14), wino_desc_bp.itiles*wino_desc_bp.jtiles*wino_desc_bp.bimg);
        if (0 == wino_desc_bp.itiles*wino_desc_bp.jtiles*wino_desc_bp.bimg % temp_ur) {
          wino_desc_bp.ur = temp_ur;
        }
      }

      wino_desc_bp.ur_ifm = handle->blocksofm;
      wino_desc_bp.blocks_ifm = handle->blocksofm;

      handle->cwino_bwd = wino_desc_bp;

      /* TODO check JIT errors */
      if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM  ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL    )
      {
        wino_desc_bp.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
        handle->code_bwd[0].pmm = libxsmm_create_xconv_wino_backward(&wino_desc_bp);
        wino_desc_bp.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1;
        handle->code_bwd[1].pmm = libxsmm_create_xconv_wino_backward(&wino_desc_bp);
        wino_desc_bp.prefetch = (libxsmm_convolution_prefetch_type)(LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2);
        handle->code_bwd[2].pmm = libxsmm_create_xconv_wino_backward(&wino_desc_bp);
      } else {
        assert(0/*should not happen*/);
      }
    } /* End of backward */
    /* Weight update path */
    { wino_desc_wu.alpha = alpha;
      wino_desc_wu.jtiles = wino_desc_fp.jtiles;
      wino_desc_wu.itiles = wino_desc_fp.itiles;

      /* LUT for DeepBench */
      if ((240 == handle->ofw) && (24 == handle->ofh) && (16 == handle->desc.N) && (16 == handle->desc.C) && (32 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 1;
        wino_desc_wu.ur = 1;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((120 == handle->ofw) && (12 == handle->ofh) && (16 == handle->desc.N) && (32 == handle->desc.C) && (64 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 1;
        wino_desc_wu.ur = 1;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((60 == handle->ofw) && (6 == handle->ofh) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 1;
        wino_desc_wu.ur = 1;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((54 == handle->ofw) && (54 == handle->ofh) && (8 == handle->desc.N) && (64 == handle->desc.C) && (64 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 1;
        if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
          wino_desc_wu.ur = 1;
        } else {
          wino_desc_wu.ur = 2;
        }
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((27 == handle->ofw) && (27 == handle->ofh) && (8 == handle->desc.N) && (128 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 1; /*8;*/
        wino_desc_wu.ur = 1; /*2;*/
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 8;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((7 == handle->ofw) && (7 == handle->ofh) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 8;
        wino_desc_wu.ur = 4;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((112 == handle->ofw) && (112 == handle->ofh) && (8 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 1;
        wino_desc_wu.ur = 1;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((56 == handle->ofw) && (56 == handle->ofh) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 1; /*2;*/
        if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
          wino_desc_wu.ur = 1;
        } else {
          wino_desc_wu.ur = 2;
        }
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->ofw) && (28 == handle->ofh) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 2; /*4;*/
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 8;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((7 == handle->ofw) && (7 == handle->ofh) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 8;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((112 == handle->ofw) && (112 == handle->ofh) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 1;
        wino_desc_wu.ur = 1;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((56 == handle->ofw) && (56 == handle->ofh) && (16 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 1;
        if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
          wino_desc_wu.ur = 1;
        } else {
          wino_desc_wu.ur = 2;
        }
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->ofw) && (28 == handle->ofh) && (16 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 2; /*16;*/
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 4; /*16;*/
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((7 == handle->ofw) && (7 == handle->ofh) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 16;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      }

      /* LUT for AlexNet */
      else if ((13 == handle->ofw) && (13 == handle->ofh) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((13 == handle->ofw) && (13 == handle->ofh) && (64 <= handle->desc.N) && (384 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((13 == handle->ofw) && (13 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      }

      /* LUT for GoogLenetV1 */
      else if ((56 == handle->ofw) && (56 == handle->ofh) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (192 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 1;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = (0 == wino_desc_wu.bimg % 2) ? 2 : 1;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (192 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = (0 == wino_desc_wu.bimg % 2) ? 2 : 1;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (208 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (112 == handle->desc.C) && (224 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (144 == handle->desc.C) && (288 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = 1;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((7 == handle->ofw) && (7 == handle->ofh) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 32) ? 32 :
          (0 == handle->desc.N % 16) ? 16 :
          (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((7 == handle->ofw) && (7 == handle->ofh) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 32) ? 32 :
          (0 == handle->desc.N % 16) ? 16 :
          (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      }

      /* LUT for Overfeat */
      else if ((12 == handle->ofw) && (12 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 32) ? 32 :
          (0 == handle->desc.N % 16) ? 16 :
          (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = (0 == wino_desc_wu.bimg % 2) ? 2 : 1;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((12 == handle->ofw) && (12 == handle->ofh) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (1024 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 32) ? 32 :
          (0 == handle->desc.N % 16) ? 16 :
          (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = (0 == wino_desc_wu.bimg % 2) ? 2 : 1;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((12 == handle->ofw) && (12 == handle->ofh) && (64 <= handle->desc.N) && (1024 == handle->desc.C) && (1024 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 32) ? 32 :
          (0 == handle->desc.N % 16) ? 16 :
          (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = (0 == wino_desc_wu.bimg % 2) ? 2 : 1;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      }

      /* LUT for VGGA */
      else if ((112 == handle->ofw) && (112 == handle->ofh) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 1;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((56 == handle->ofw) && (56 == handle->ofh) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 1;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((56 == handle->ofw) && (56 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = 1;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = (0 == wino_desc_wu.bimg % 2) ? 2 : 1;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = (0 == wino_desc_wu.bimg % 2) ? 2 : 1;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
        wino_desc_wu.bimg = (0 == handle->desc.N % 8) ? 8 :
          (0 == handle->desc.N % 4) ? 4 :
          (0 == handle->desc.N % 2) ? 2 : 1;
        wino_desc_wu.ur = 2;
#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        flagBenchmark = 1;
#endif
      }

      /* General scenario */
      else {
        if ((handle->desc.N % 4) == 0) {
          wino_desc_wu.bimg = 4;
        } else if ((handle->desc.N % 2) == 0) {
          wino_desc_wu.bimg = 2;
        } else {
          wino_desc_wu.bimg = 1;
        }
        allowed_unroll = 512 / (wino_desc_wu.bimg*wino_desc_wu.itiles*wino_desc_wu.jtiles);
        allowed_unroll = (allowed_unroll > 26) ? 26 : allowed_unroll;
        if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM && (wino_desc_wu.itiles*wino_desc_wu.jtiles*wino_desc_wu.bimg % 4) == 0) {
          wino_desc_wu.ur = libxsmm_product_limit(wino_desc_wu.itiles*wino_desc_wu.jtiles*wino_desc_wu.bimg/4, allowed_unroll, 0);
        } else {
          wino_desc_wu.ur = libxsmm_product_limit(wino_desc_wu.itiles*wino_desc_wu.jtiles*wino_desc_wu.bimg, allowed_unroll, 0);
        }
      }

      wino_desc_wu.ur_ifm = 1;

      handle->cwino_upd = wino_desc_wu;
      /* TODO check JIT errors */
      if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM  ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL    )
      {
        /* NONE */
        wino_desc_wu.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
        handle->code_upd[0].pmm = libxsmm_create_xconv_wino_update_weights(&wino_desc_wu);
        /* ALL */
        wino_desc_wu.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
        handle->code_upd[1].pmm = libxsmm_create_xconv_wino_update_weights(&wino_desc_wu);
      } else {
        assert(0/*should not happen*/);
      }
    } /* end of weight-update handle */
    {
      /* Populating scratch registers for U V and M */
      int ijtiles;
      if (wino_desc_bp.itiles * wino_desc_bp.jtiles >= wino_desc_fp.itiles * wino_desc_fp.jtiles) {
        ijtiles = wino_desc_bp.itiles * wino_desc_bp.jtiles;
      } else {
        ijtiles = wino_desc_fp.itiles * wino_desc_fp.jtiles;
      }

      handle->scratch1 = 0;
      handle->scratch1_size = (size_t)alpha*alpha*handle->desc.C*handle->desc.K*libxsmm_dnn_typesize(handle->datatype_in);
      handle->scratch3 = 0;
      handle->scratch3_size = (size_t)alpha*alpha*ijtiles*handle->desc.N * handle->desc.C * libxsmm_dnn_typesize(handle->datatype_in);
      handle->scratch4 = 0;
      handle->scratch4_size = (size_t)alpha*alpha*ijtiles*handle->desc.N * handle->desc.K * libxsmm_dnn_typesize(handle->datatype_out);
      handle->scratch2 = 0;
      handle->scratch2_size = 0;
      handle->scratchIw = 0;
      handle->scratchIw_size = (size_t)ijtiles*alpha*alpha*16*libxsmm_dnn_typesize(handle->datatype_in)*handle->desc.threads;
      handle->scratchOw = 0;
      handle->scratchOw_size = (size_t)ijtiles*alpha*alpha*16*libxsmm_dnn_typesize(handle->datatype_out)*handle->desc.threads;
      handle->scratchVk = 0;
      handle->scratchVk_size = handle->scratch3_size;
      handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);
    }
  } else {
    handle->code_fwd[0].xconv.sconv = 0;
    handle->code_fwd[1].xconv.sconv = 0;
    handle->code_fwd[2].xconv.sconv = 0;
    /* Backward path */
    handle->code_bwd[0].xconv.sconv = 0;
    handle->code_bwd[1].xconv.sconv = 0;
    handle->code_bwd[2].xconv.sconv = 0;
    /* weight update path */
    handle->code_upd[0].xconv.sconv = 0;
    handle->code_upd[1].xconv.sconv = 0;
    handle->barrier  = 0;
    handle->scratch1 = 0;
    handle->scratch1_size = 0;
    handle->scratch3 = 0;
    handle->scratch3_size = 0;
    handle->scratch4 = 0;
    handle->scratch4_size = 0;
    handle->scratch2 = 0;
    handle->scratch2_size = 0;
  }

  return status;
}

