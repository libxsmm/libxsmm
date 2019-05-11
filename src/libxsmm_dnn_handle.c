/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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

#if 0
  /* If we have AVX512 and kernel streams is enabled, and we use libxsmm's custom format, then we generate specialized code */
  if ( (LIBXSMM_X86_AVX512 <= libxsmm_target_archid)                &&
       (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) &&
       (handle->filter_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM)    ) {
    /* This is basically a decision pertaining for all three passes: FWD, BWD and UPD */
    /* Initialize fields that control layer fusion */
    noarch = 0;
    handle->compute_batch_stats_in_kernel_fwd = 0;
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

    /* only continue if we could block data in LIBXSMM's custom format, otherwise use generic code */
    if ( noarch == 0 ) {
      /* Forward path setup, @TODO check status */
      status = libxsmm_dnn_setup_fwd(handle, &noarch);

      /* Backward path setup, @TODO check status */
      status = libxsmm_dnn_setup_bwd(handle, &noarch);

      /* Weight update path setup, @TODO check status */
      status = libxsmm_dnn_setup_upd(handle, &noarch);

      /* Calculate scratch requirements */
      libxsmm_dnn_setup_scratch(handle);
    }
  }
#endif

  if (0 != noarch) { /* Setup generic code generation */
    status = libxsmm_dnn_setup_generic(handle);
  }

# if 0 /* TODO: Bf16 currently triggers error 90005 before but we want to continue */
  if (LIBXSMM_DNN_SUCCESS == status)
# endif
  {
    if (0 != handle->use_fwd_generic || 0 != handle->use_bwd_generic || 0 != handle->use_upd_generic) {
      const size_t padded_h = ((size_t)2 * handle->desc.pad_h) + handle->desc.H, padded_w = ((size_t)2 * handle->desc.pad_w) + handle->desc.W;
      const size_t size5_tensor = padded_h * padded_w * handle->ifmblock * libxsmm_dnn_typesize(handle->datatype_in);
      const size_t size5 = LIBXSMM_UP2(size5_tensor, LIBXSMM_CACHELINE) * handle->desc.threads;
      if (handle->max_scratch5_size < size5) handle->max_scratch5_size = size5;
    }
    handle->scratch5 = 0;
#   if 0 /* make float-accumulation scratch always available as it is referenced even if below property is false */
    if (handle->use_accumulation_scratch)
#   endif
    {
      const size_t size6a = (size_t)handle->ofmblock * handle->ofw * handle->ofh * sizeof(float);
      const size_t size6b = (size_t)handle->ifmblock * handle->fm_lp_block *  handle->desc.W * handle->desc.H * sizeof(float);
      const size_t size6 = ( size6a > size6b ) ? size6a : size6b;
      handle->scratch6_size = LIBXSMM_MAX(LIBXSMM_UP2(size6, LIBXSMM_CACHELINE) * handle->desc.threads, handle->scratch6_size);
    }
    if (0 != handle->use_upd_generic) {
      const size_t output_typesize = libxsmm_dnn_typesize(handle->datatype_out);
      const size_t size6_tensor = (size_t)handle->ofhp * handle->ofwp * handle->ofmblock * output_typesize;
      const size_t size6 = LIBXSMM_UP2(size6_tensor, LIBXSMM_CACHELINE) * handle->desc.threads;
      if (handle->scratch6_size < size6) handle->scratch6_size = size6;
    }
    handle->scratch6 = 0;
    if (0 != handle->use_upd_generic) {
      /* FIXME: currently filter data-type is always smaller/equal output type */
      const size_t filter_typesize = libxsmm_dnn_typesize(handle->datatype_out);
      const size_t size7 = (size_t)handle->desc.R * handle->desc.S * handle->desc.C * handle->desc.K * filter_typesize + handle->ifmblock * handle->ofmblock * sizeof(float);
      handle->scratch7_size = LIBXSMM_UP2(size7, LIBXSMM_CACHELINE) * handle->desc.threads;
    }
  }
  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_internal_free_structs_code_conv_handle( const libxsmm_dnn_layer* handle ) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    /* deallocate data components; not an error to deallocate a NULL-pointer
       deallocate code known to be not registered; no index attached
       do not use libxsmm_release_kernel here! */

    /* deallocate forward pass */
    if ( handle->use_fwd_generic == 0 ) {
      int loop;

      if (handle->custom_format_type != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) {
        libxsmm_free(handle->code_fwd[0].pmm);
      }
      libxsmm_free(handle->code_fwd[1].pmm);
      libxsmm_free(handle->code_fwd[2].pmm);

      if ( handle->algo == LIBXSMM_DNN_CONV_ALGO_DIRECT ) {
        for (loop = 0; loop < handle->desc.threads; loop++) {
          libxsmm_free( handle->compute_fwd_indices_ptrs[loop] );
          libxsmm_free( handle->bn_stats_indices_ptrs[loop] );
          libxsmm_free( handle->kernel_fwd_variant_ptrs[loop] );
          libxsmm_free( handle->fwd_code_segments[loop] );
        }
      }

      free( handle->n_entries_fwd );
      free( handle->compute_fwd_indices_ptrs );
      free( handle->bn_stats_indices_ptrs );
      free( handle->kernel_fwd_variant_ptrs );
      free( handle->n_fwd_code_segments );
      free( handle->fwd_code_segments );
      free( handle->ofh_fwd_start );
      free( handle->ofh_fwd_end );
    }

    /* deallocate backward pass */
    if ( handle->use_bwd_generic == 0 ) {
      int loop;

      if (handle->custom_format_type != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) {
        libxsmm_free(handle->code_bwd[0].pmm);
      }
      libxsmm_free(handle->code_bwd[1].pmm);
      libxsmm_free(handle->code_bwd[2].pmm);

      if ( handle->algo == LIBXSMM_DNN_CONV_ALGO_DIRECT ) {
        for (loop = 0; loop < handle->desc.threads; loop++) {
          libxsmm_free( handle->compute_bwd_indices_ptrs[loop] );
          libxsmm_free( handle->kernel_bwd_variant_ptrs[loop] );
          libxsmm_free( handle->bwd_code_segments[loop] );
          libxsmm_free( handle->transpose_bwd_indices_ptrs[loop]);
        }
      }

      free( handle->n_entries_bwd );
      free( handle->compute_bwd_indices_ptrs );
      free( handle->kernel_bwd_variant_ptrs );
      free( handle->n_bwd_code_segments );
      free( handle->bwd_code_segments );
      free( handle->n_entries_trans_bwd );
      free( handle->transpose_bwd_indices_ptrs );
      free( handle->ofh_bwd_start );
      free( handle->ofh_bwd_end );
    }

    /* deallocate update pass */
    if ( handle->use_upd_generic == 0 ) {
      int loop;

      if (handle->custom_format_type != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) {
        libxsmm_free(handle->code_upd[0].pmm);
      }
      libxsmm_free(handle->code_upd[1].pmm);

      if ( handle->algo == LIBXSMM_DNN_CONV_ALGO_DIRECT ) {
        for (loop = 0; loop < handle->desc.threads; loop++) {
          libxsmm_free( handle->compute_upd_indices_ptrs[loop] );
          libxsmm_free( handle->kernel_upd_variant_ptrs[loop] );
          libxsmm_free( handle->upd_code_segments[loop] );
          libxsmm_free( handle->init_upd_indices_ptrs[loop] );
          libxsmm_free( handle->copy_upd_indices_ptrs[loop] );
        }
      }

      free( handle->n_entries_upd );
      free( handle->compute_upd_indices_ptrs );
      free( handle->kernel_upd_variant_ptrs );
      free( handle->n_upd_code_segments );
      free( handle->upd_code_segments );
      free( handle->n_entries_init_upd );
      free( handle->init_upd_indices_ptrs );
      free( handle->n_entries_copy_upd );
      free( handle->copy_upd_indices_ptrs );
    }
  }

  return status;
}


