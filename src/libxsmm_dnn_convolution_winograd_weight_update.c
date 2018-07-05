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
/* Kunal Banerjee (Intel Corp.), Rajkishore Barik (Intel Corp.),
 * Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include "libxsmm_dnn_convolution_winograd_weight_update.h"
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if !defined(NDEBUG)
# include <assert.h>
# include <stdio.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_INLINE void internal_upd_input_transform_custom_custom(
                                           float *inp,
                                           float *tinp,
                                           float *Iwp,
                                           const libxsmm_dnn_layer* handle )
{
  if (handle->cwino_upd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_custom_custom_input_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_upd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_custom_custom_input_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXSMM error: Unsupported alpha %u\n", handle->cwino_upd.alpha);
    assert(0);
  }
#endif
}

LIBXSMM_API_INLINE void internal_upd_input_transform_nhwc_custom(
                                         float *inp,
                                         float *tinp,
                                         float *Iwp,
                                         const libxsmm_dnn_layer* handle )
{
  if (handle->cwino_upd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_nhwc_custom_input_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_upd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_nhwc_custom_input_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXSMM error: Unsupported alpha %u\n", handle->cwino_upd.alpha);
    assert(0);
  }
#endif
}

LIBXSMM_API_INLINE void internal_upd_deloutput_transform_custom_custom(
                                               float *inp,
                                               float *tinp,
                                               float *Owp,
                                               const libxsmm_dnn_layer* handle )
{
  if (handle->cwino_upd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_custom_custom_deloutput_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_upd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_custom_custom_deloutput_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXSMM error: Unsupported alpha %u\n", handle->cwino_upd.alpha);
    assert(0);
  }
#endif
}

LIBXSMM_API_INLINE void internal_upd_deloutput_transform_nhwc_custom(
                                             float *inp,
                                             float *tinp,
                                             float *Owp,
                                             const libxsmm_dnn_layer* handle )
{
  if (handle->cwino_upd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_nhwc_custom_deloutput_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_upd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_nhwc_custom_deloutput_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXSMM error: Unsupported alpha %u\n", handle->cwino_upd.alpha);
    assert(0);
  }
#endif
}

LIBXSMM_API_INLINE void internal_upd_delweight_transform(
                                 float *wp,
                                 float *twp,
                                 const libxsmm_dnn_layer* handle )
{
  if (handle->cwino_upd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_delweight_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_upd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_delweight_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXSMM error: Unsupported alpha %u\n", handle->cwino_upd.alpha);
    assert(0);
  }
#endif
}

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_winograd_st_upd_custom_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->grad_output == 0 || handle->grad_filter == 0 || handle->scratch1 == 0 || handle->scratch3 == 0 || handle->scratch4 == 0 || handle->scratchIw == 0 || handle->scratchOw == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if ( handle->use_upd_generic != 0 ) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) {
      const libxsmm_blasint ldx = (libxsmm_blasint)(handle->desc.W+(2*handle->desc.pad_w));
      const libxsmm_blasint ldx_alt = (libxsmm_blasint)(handle->desc.v*handle->ifmblock);
      const libxsmm_blasint ldb_alt = (libxsmm_blasint)handle->ofwp;
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxsmm_smmfunction gemm_function;
      /* let's do a ofmblock x ifmblock x ofw_rb GEMM :-) or in other words M=nbOfm, N=nbIfm, K=ofw (col-major) */
      gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->ifmblock, handle->ofw, NULL, &ldx, NULL, NULL, NULL, NULL, NULL);
      /* for strided convolutions with kernel size bigger than 1 the above GEMM doesn't work and we need to switch to more transposes and an
         alternative GEMM:
         let's do a ifmblock x ofmblock x ofw_rb GEMM :-) or in other words M=nbIfm, N=nbOfm, K=ofw (col-major) */
      gemm_function gemm_kernel_alt = libxsmm_smmdispatch(handle->ifmblock, handle->ofmblock, handle->ofw, &ldx_alt, &ldb_alt, NULL, NULL, NULL, NULL, NULL);
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom_generic.tpl.c"
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) {
      if (handle->cwino_upd.alpha == 6  && libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM && (handle->cwino_upd.itiles*handle->cwino_upd.jtiles*handle->cwino_upd.bimg % 4) == 0) {
        if (handle->scratchVk == 0) {
          status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
          return status;
        } else {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_custom_custom_inlined_knm.tpl.c"
#undef TDVLEN
#undef ALPHA
        }
      } else if (handle->cwino_upd.alpha == 4  && libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM && (handle->cwino_upd.itiles*handle->cwino_upd.jtiles*handle->cwino_upd.bimg % 4) == 0) {
        if (handle->scratchVk == 0) {
          status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
          return status;
        } else {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_custom_custom_inlined_knm.tpl.c"
#undef TDVLEN
#undef ALPHA
        }
      } else if (handle->cwino_upd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_custom_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
      } else if (handle->cwino_upd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_custom_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
      }
#if !defined(NDEBUG)
      else {
        fprintf(stderr, "LIBXSMM error: Unsupported alpha %u\n", handle->cwino_upd.alpha);
        assert(0);
      }
#endif
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_winograd_st_upd_nhwc_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->grad_output == 0 || handle->grad_filter == 0 || handle->scratch1 == 0 || handle->scratch3 == 0 || handle->scratch4 == 0 || handle->scratchIw == 0 || handle->scratchOw == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if ( handle->use_upd_generic != 0 ) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) {
      const libxsmm_blasint lda     = (libxsmm_blasint)(handle->blocksofm*handle->ofmblock);
      const libxsmm_blasint ldb     = (libxsmm_blasint)(handle->desc.W+(2*handle->desc.pad_w));
      const libxsmm_blasint ldc     = (libxsmm_blasint)(handle->ofmblock);
      const libxsmm_blasint lda_alt = (libxsmm_blasint)((handle->desc.pad_h == handle->desc.pad_h_in && handle->desc.pad_w == handle->desc.pad_w_in)
                            ? (handle->desc.v*handle->blocksifm*handle->ifmblock) : (handle->desc.v*handle->ifmblock));
      const libxsmm_blasint ldb_alt = (libxsmm_blasint)(handle->ofwp);
      const libxsmm_blasint ldc_alt = (libxsmm_blasint)(handle->ifmblock);
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxsmm_smmfunction gemm_function;
      /* let's do a ofmblock x ifmblock x ofw_rb GEMM :-) or in other words M=nbOfm, N=nbIfm, K=ofw (col-major) */
      gemm_function gemm_kernel     = libxsmm_smmdispatch(handle->ofmblock, handle->ifmblock, handle->ofw, &lda, &ldb, &ldc, NULL, NULL, NULL, NULL);
      /* for strided convolutions with kernel size bigger than 1 the above GEMM doesn't work and we need to switch to more transposes and an
         alternative GEMM:
         let's do a ifmblock x ofmblock x ofw_rb GEMM :-) or in other words M=nbIfm, N=nbOfm, K=ofw (col-major) */
      gemm_function gemm_kernel_alt = libxsmm_smmdispatch(handle->ifmblock, handle->ofmblock, handle->ofw, &lda_alt, &ldb_alt, &ldc_alt, NULL, NULL, NULL, NULL);
#define LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) {
      if (handle->cwino_upd.alpha == 6 && libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM && (handle->cwino_upd.itiles*handle->cwino_upd.jtiles*handle->cwino_upd.bimg % 4) == 0) {
        if (handle->scratchVk == 0) {
          status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
          return status;
        } else {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_nhwc_custom_inlined_knm.tpl.c"
#undef TDVLEN
#undef ALPHA
        }
      } else if (handle->cwino_upd.alpha == 4 && libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM && (handle->cwino_upd.itiles*handle->cwino_upd.jtiles*handle->cwino_upd.bimg % 4) == 0) {
        if (handle->scratchVk == 0) {
          status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
          return status;
        } else {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_nhwc_custom_inlined_knm.tpl.c"
#undef TDVLEN
#undef ALPHA
        }
      } else if (handle->cwino_upd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_nhwc_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
      } else if (handle->cwino_upd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_weight_update_nhwc_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
      }
#if !defined(NDEBUG)
      else {
        fprintf(stderr, "LIBXSMM error: Unsupported alpha %u\n", handle->cwino_upd.alpha);
        assert(0);
      }
#endif
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}
