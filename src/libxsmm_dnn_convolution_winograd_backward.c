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
/* Kunal Banerjee (Intel Corp.), Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_convolution_winograd_backward.h"
#include "libxsmm_main.h"
#include <libxsmm_intrinsics_x86.h>
#include <libxsmm.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <assert.h>
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/* Enable/disable specific code paths */
#if defined(LIBXSMM_INTRINSICS_AVX512) && !defined(LIBXSMM_DNN_CONVOLUTION_WINOGRAD_BACKWARD_AVX512)
# define LIBXSMM_DNN_CONVOLUTION_WINOGRAD_BACKWARD_AVX512
#endif


/* function pointer for the CPUID-dispatched implementation */
LIBXSMM_APIVAR(void (*internal_bwd_input_transform_custom_custom_alpha6)(const float*, float*, float*, const libxsmm_dnn_layer*));
LIBXSMM_APIVAR(void (*internal_bwd_input_transform_nhwc_custom_alpha6)(const float*, float*, float*, const libxsmm_dnn_layer*));
LIBXSMM_APIVAR(void (*internal_bwd_output_transform_custom_custom_alpha6)(float*, float*, float*, const libxsmm_dnn_layer*));
LIBXSMM_APIVAR(void (*internal_bwd_output_transform_nhwc_custom_alpha6)(float*, float*, float*, const libxsmm_dnn_layer*));
LIBXSMM_APIVAR(void (*internal_dnn_convolve_winograd_st_bwd_custom_custom_alpha6)(libxsmm_dnn_layer*, int, int));
LIBXSMM_APIVAR(void (*internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6)(libxsmm_dnn_layer*, int, int));


LIBXSMM_API_INLINE void internal_bwd_input_transform_custom_custom_alpha6_default(
  const float* inp, float* tinp, float* Iwp, const libxsmm_dnn_layer* handle)
{
#define ALPHA 6
#define TDVLEN 16
#include "template/libxsmm_dnn_convolution_winograd_backward_custom_custom_input_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
LIBXSMM_ATTRIBUTE_UNUSED void internal_bwd_input_transform_custom_custom_alpha6_avx512(
  const float* inp, float* tinp, float* Iwp, const libxsmm_dnn_layer* handle)
{
#if defined(LIBXSMM_DNN_CONVOLUTION_WINOGRAD_BACKWARD_AVX512)
# define ALPHA 6
# define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_custom_custom_input_trans_alpha6_avx512.tpl.c"
# undef TDVLEN
# undef ALPHA
  LIBXSMM_UNUSED(Iwp);
#else /* next lower/available code path (fall-back chain) */
  internal_bwd_input_transform_custom_custom_alpha6_default(inp, tinp, Iwp, handle);
#endif
}

LIBXSMM_API_INLINE void internal_bwd_input_transform_custom_custom(
  const float *inp, float *tinp, float *Iwp, const libxsmm_dnn_layer* handle)
{
  if (handle->cwino_bwd.alpha == 6) {
    /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH)
    internal_bwd_input_transform_custom_custom_alpha6_avx512(inp, tinp, Iwp, handle);
#else /* pointer based function call */
    assert(0 != internal_bwd_input_transform_custom_custom_alpha6);
    internal_bwd_input_transform_custom_custom_alpha6(inp, tinp, Iwp, handle);
#endif
  } else if (handle->cwino_bwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_custom_custom_input_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXSMM error: Unsupported alpha %u\n", handle->cwino_bwd.alpha);
    assert(0);
  }
#endif
}


LIBXSMM_API_INLINE void internal_bwd_input_transform_nhwc_custom_alpha6_default(
  const float* inp, float* tinp, float* Iwp, const libxsmm_dnn_layer* handle)
{
#define ALPHA 6
#define TDVLEN 16
#include "template/libxsmm_dnn_convolution_winograd_backward_nhwc_custom_input_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
LIBXSMM_ATTRIBUTE_UNUSED void internal_bwd_input_transform_nhwc_custom_alpha6_avx512(
  const float* inp, float* tinp, float* Iwp, const libxsmm_dnn_layer* handle)
{
#if defined(LIBXSMM_DNN_CONVOLUTION_WINOGRAD_BACKWARD_AVX512)
# define ALPHA 6
# define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_nhwc_custom_input_trans_alpha6_avx512.tpl.c"
# undef TDVLEN
# undef ALPHA
  LIBXSMM_UNUSED(Iwp);
#else /* next lower/available code path (fall-back chain) */
  internal_bwd_input_transform_nhwc_custom_alpha6_default(inp, tinp, Iwp, handle);
#endif
}


LIBXSMM_API_INLINE void internal_bwd_input_transform_nhwc_custom(
  const float* inp, float* tinp, float* Iwp, const libxsmm_dnn_layer* handle)
{
  if (handle->cwino_bwd.alpha == 6) {
    /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH)
    internal_bwd_input_transform_nhwc_custom_alpha6_avx512(inp, tinp, Iwp, handle);
#else /* pointer based function call */
    assert(0 != internal_bwd_input_transform_nhwc_custom_alpha6);
    internal_bwd_input_transform_nhwc_custom_alpha6(inp, tinp, Iwp, handle);
#endif
  } else if (handle->cwino_bwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_nhwc_custom_input_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXSMM error: Unsupported alpha %u\n", handle->cwino_bwd.alpha);
    assert(0);
  }
#endif
}


LIBXSMM_API_INLINE void internal_bwd_weight_transform(float *wp, float *twp, const libxsmm_dnn_layer* handle)
{
  if (handle->cwino_bwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_weight_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_bwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_weight_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXSMM error: Unsupported alpha %u\n", handle->cwino_bwd.alpha);
    assert(0);
  }
#endif
}


LIBXSMM_API_INLINE void internal_bwd_output_transform_custom_custom_alpha6_default(
  float *toutp, float *outp, float *Owp, const libxsmm_dnn_layer* handle)
{
#define ALPHA 6
#define TDVLEN 16
#include "template/libxsmm_dnn_convolution_winograd_backward_custom_custom_output_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
LIBXSMM_ATTRIBUTE_UNUSED void internal_bwd_output_transform_custom_custom_alpha6_avx512(
  float *toutp, float *outp, float *Owp, const libxsmm_dnn_layer* handle)
{
#if defined(LIBXSMM_DNN_CONVOLUTION_WINOGRAD_BACKWARD_AVX512)
  if ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) {
# define USE_OVERWRITE
# define ALPHA 6
# define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_custom_custom_output_trans_alpha6_avx512.tpl.c"
# undef TDVLEN
# undef ALPHA
# undef USE_OVERWRITE
    LIBXSMM_UNUSED(Owp);
  } else {
# define ALPHA 6
# define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_custom_custom_output_trans_alpha6_avx512.tpl.c"
# undef TDVLEN
# undef ALPHA
    LIBXSMM_UNUSED(Owp);
  }
#else /* next lower/available code path (fall-back chain) */
  internal_bwd_output_transform_custom_custom_alpha6_default(toutp, outp, Owp, handle);
#endif
}


LIBXSMM_API_INLINE void internal_bwd_output_transform_custom_custom(
  float *toutp, float *outp, float *Owp, const libxsmm_dnn_layer* handle)
{
  if (handle->cwino_bwd.alpha == 6) {
    /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH)
    internal_bwd_output_transform_custom_custom_alpha6_avx512(toutp, outp, Owp, handle);
#else /* pointer based function call */
    assert(0 != internal_bwd_output_transform_custom_custom_alpha6);
    internal_bwd_output_transform_custom_custom_alpha6(toutp, outp, Owp, handle);
#endif
  } else if (handle->cwino_bwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_custom_custom_output_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXSMM error: Unsupported alpha %u\n", handle->cwino_bwd.alpha);
    assert(0);
  }
#endif
}


LIBXSMM_API_INLINE void internal_bwd_output_transform_nhwc_custom_alpha6_default(
  float *toutp, float *outp, float *Owp, const libxsmm_dnn_layer* handle)
{
#define ALPHA 6
#define TDVLEN 16
#include "template/libxsmm_dnn_convolution_winograd_backward_nhwc_custom_output_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
LIBXSMM_ATTRIBUTE_UNUSED void internal_bwd_output_transform_nhwc_custom_alpha6_avx512(
  float *toutp, float *outp, float *Owp, const libxsmm_dnn_layer* handle)
{
#if defined(LIBXSMM_DNN_CONVOLUTION_WINOGRAD_BACKWARD_AVX512)
  if ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) {
# define USE_OVERWRITE
# define ALPHA 6
# define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_nhwc_custom_output_trans_alpha6_avx512.tpl.c"
# undef TDVLEN
# undef ALPHA
# undef USE_OVERWRITE
    LIBXSMM_UNUSED(Owp);
  } else {
# define ALPHA 6
# define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_nhwc_custom_output_trans_alpha6_avx512.tpl.c"
# undef TDVLEN
# undef ALPHA
    LIBXSMM_UNUSED(Owp);
  }
#else /* next lower/available code path (fall-back chain) */
  internal_bwd_output_transform_nhwc_custom_alpha6_default(toutp, outp, Owp, handle);
#endif
}


LIBXSMM_API_INLINE void internal_bwd_output_transform_nhwc_custom(
  float *toutp, float *outp, float *Owp, const libxsmm_dnn_layer* handle)
{
  if (handle->cwino_bwd.alpha == 6) {
    /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH)
    internal_bwd_output_transform_nhwc_custom_alpha6_avx512(toutp, outp, Owp, handle);
#else /* pointer based function call */
    assert(0 != internal_bwd_output_transform_nhwc_custom_alpha6);
    internal_bwd_output_transform_nhwc_custom_alpha6(toutp, outp, Owp, handle);
#endif
  } else if (handle->cwino_bwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_nhwc_custom_output_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXSMM error: Unsupported alpha %u\n", handle->cwino_bwd.alpha);
    assert(0);
  }
#endif
}


LIBXSMM_API_INLINE void internal_dnn_convolve_winograd_st_bwd_custom_custom_alpha6_default(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
#define ALPHA 6
#define TDVLEN 16
#include "template/libxsmm_dnn_convolution_winograd_backward_custom_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
LIBXSMM_ATTRIBUTE_UNUSED void internal_dnn_convolve_winograd_st_bwd_custom_custom_alpha6_avx512(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
#if defined(LIBXSMM_DNN_CONVOLUTION_WINOGRAD_BACKWARD_AVX512)
# define ALPHA 6
# define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_custom_custom_inlined_avx512.tpl.c"
# undef TDVLEN
# undef ALPHA
#else /* next lower/available code path (fall-back chain) */
  internal_dnn_convolve_winograd_st_bwd_custom_custom_alpha6_default(handle, start_thread, tid);
#endif
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_winograd_st_bwd_custom_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0 || handle->scratch3 == 0 || handle->scratch4 == 0 || handle->scratchIw == 0 || handle->scratchOw == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if ( handle->use_bwd_generic != 0 ) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) {
      const libxsmm_blasint ldx = (libxsmm_blasint)(handle->desc.v*handle->ifmblock);
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxsmm_smmfunction gemm_function;
      /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
      gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, NULL, NULL, &ldx, NULL, NULL, NULL, NULL);
# include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_generic.tpl.c"
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) {
      if (handle->cwino_bwd.alpha == 6) {
#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH)
        internal_dnn_convolve_winograd_st_bwd_custom_custom_alpha6_avx512(handle, start_thread, tid);
#else /* pointer based function call */
        assert(0 != internal_dnn_convolve_winograd_st_bwd_custom_custom_alpha6);
        internal_dnn_convolve_winograd_st_bwd_custom_custom_alpha6(handle, start_thread, tid);
#endif
      } else if (handle->cwino_bwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_custom_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
      }
#if !defined(NDEBUG)
      else {
        fprintf(stderr, "LIBXSMM error: Unsupported alpha %u\n", handle->cwino_bwd.alpha);
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


LIBXSMM_API_INLINE void internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6_default(
  libxsmm_dnn_layer* handle, int start_thread, int tid)
{
#define ALPHA 6
#define TDVLEN 16
#include "template/libxsmm_dnn_convolution_winograd_backward_nhwc_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
LIBXSMM_ATTRIBUTE_UNUSED void internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6_avx512(
  libxsmm_dnn_layer* handle, int start_thread, int tid)
{
#if defined(LIBXSMM_DNN_CONVOLUTION_WINOGRAD_BACKWARD_AVX512)
# define ALPHA 6
# define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_nhwc_custom_inlined_avx512.tpl.c"
# undef TDVLEN
# undef ALPHA
#else /* next lower/available code path (fall-back chain) */
  internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6_default(handle, start_thread, tid);
#endif
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_winograd_st_bwd_nhwc_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0 || handle->scratch3 == 0 || handle->scratch4 == 0 || handle->scratchIw == 0 || handle->scratchOw == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if ( handle->use_bwd_generic != 0 ) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) {
      const libxsmm_blasint lda = (libxsmm_blasint)(handle->ifmblock);
      const libxsmm_blasint ldb = (libxsmm_blasint)(handle->blocksofm*handle->ofmblock);
      const libxsmm_blasint ldc = (libxsmm_blasint)((handle->desc.pad_h == handle->desc.pad_h_in && handle->desc.pad_w == handle->desc.pad_w_in)
                        ? (handle->desc.v*handle->blocksifm*handle->ifmblock) : (handle->desc.v*handle->ifmblock));
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxsmm_smmfunction gemm_function;
      /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
      gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, &lda, &ldb, &ldc, NULL, NULL, NULL, NULL);
#define LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) {
      if (handle->cwino_bwd.alpha == 6) {
        /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH)
        internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6_avx512(handle, start_thread, tid);
#else /* pointer based function call */
        assert(0 != internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6);
        internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6(handle, start_thread, tid);
#endif
      } else if (handle->cwino_bwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxsmm_dnn_convolution_winograd_backward_nhwc_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
      }
#if !defined(NDEBUG)
      else {
        fprintf(stderr, "LIBXSMM error: Unsupported alpha %u\n", handle->cwino_bwd.alpha);
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


LIBXSMM_API_INTERN void libxsmm_dnn_convolve_winograd_bwd_init(int target_arch)
{
  if (LIBXSMM_X86_AVX512 <= target_arch) {
    internal_bwd_input_transform_custom_custom_alpha6 = internal_bwd_input_transform_custom_custom_alpha6_avx512;
    internal_bwd_input_transform_nhwc_custom_alpha6 = internal_bwd_input_transform_nhwc_custom_alpha6_avx512;
    internal_bwd_output_transform_custom_custom_alpha6 = internal_bwd_output_transform_custom_custom_alpha6_avx512;
    internal_bwd_output_transform_nhwc_custom_alpha6 = internal_bwd_output_transform_nhwc_custom_alpha6_avx512;
    internal_dnn_convolve_winograd_st_bwd_custom_custom_alpha6 = internal_dnn_convolve_winograd_st_bwd_custom_custom_alpha6_avx512;
    internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6 = internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6_avx512;
  }
  else {
    internal_bwd_input_transform_custom_custom_alpha6 = internal_bwd_input_transform_custom_custom_alpha6_default;
    internal_bwd_input_transform_nhwc_custom_alpha6 = internal_bwd_input_transform_nhwc_custom_alpha6_default;
    internal_bwd_output_transform_custom_custom_alpha6 = internal_bwd_output_transform_custom_custom_alpha6_default;
    internal_bwd_output_transform_nhwc_custom_alpha6 = internal_bwd_output_transform_nhwc_custom_alpha6_default;
    internal_dnn_convolve_winograd_st_bwd_custom_custom_alpha6 = internal_dnn_convolve_winograd_st_bwd_custom_custom_alpha6_default;
    internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6 = internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6_default;
  }
  assert(0 != internal_bwd_input_transform_custom_custom_alpha6);
  assert(0 != internal_bwd_input_transform_nhwc_custom_alpha6);
  assert(0 != internal_bwd_output_transform_custom_custom_alpha6);
  assert(0 != internal_bwd_output_transform_nhwc_custom_alpha6);
  assert(0 != internal_dnn_convolve_winograd_st_bwd_custom_custom_alpha6);
  assert(0 != internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6);
}


LIBXSMM_API_INTERN void libxsmm_dnn_convolve_winograd_bwd_finalize(void)
{
}

