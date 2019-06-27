/******************************************************************************
** Copyright (c) 2018-2019, Intel Corporation                                **
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
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_setup.h"
#include "generator_common.h"
#include "libxsmm_main.h"
#include <libxsmm.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#define MIXED 0
#define KHWC 1
#define HWKC 2
#define CHWK 3
#define HWCK 4


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_get_feature_map_blocks( int C, int K, int* C_block, int* C_block_hp, int* K_block, int* K_block_lp, int* fm_lp_block, libxsmm_dnn_datatype datatype_in, libxsmm_dnn_datatype datatype_out, int* noarch) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  int ifmblock;
  int ofmblock;
  int lp_block;
  int ifmblock_hp;
  int ofmblock_lp;
  LIBXSMM_UNUSED(K);

  if ( (datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
    ifmblock = (C >=64) ? 64 : C;
    ofmblock = 64;
    lp_block = 1;
    ifmblock_hp = ifmblock * lp_block;
    ofmblock_lp = ofmblock / lp_block;
  } else if ( (datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
    ifmblock = (C >=64) ? 32 : C/2;
    ofmblock = 64;
    lp_block = 2;
    if (C == 3) {
      ifmblock = C;
      lp_block = 1;
    }
    ifmblock_hp = ifmblock * lp_block;
    ofmblock_lp = ofmblock / lp_block;
  } else if ( (datatype_in == LIBXSMM_DNN_DATATYPE_I16) && ((datatype_out == LIBXSMM_DNN_DATATYPE_I32) || (datatype_out == LIBXSMM_DNN_DATATYPE_F32)) ) {
    ifmblock = (C >=64) ? 32 : (C/2);
    ofmblock = 64;
    lp_block = 2;
    ifmblock_hp = ifmblock * lp_block;
    ofmblock_lp = ofmblock / lp_block;
    if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC ) {
      *noarch = 1;
    }
  } else if ( (datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (datatype_out == LIBXSMM_DNN_DATATYPE_I32)) {
    ifmblock = (C >=64) ? 16 : (C/4);
    ofmblock = 64;
    lp_block = 4;
    ifmblock_hp = ifmblock * lp_block;
    ofmblock_lp = ofmblock / lp_block;
    if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC || libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM ) {
      *noarch = 1;
    }
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
    return status;
  }

  *C_block = ifmblock;
  *K_block = ofmblock;
  *fm_lp_block = lp_block;
  *C_block_hp = ifmblock_hp;
  *K_block_lp = ofmblock_lp;

  return status;
}


LIBXSMM_API_INTERN void libxsmm_dnn_setup_scratch( libxsmm_dnn_layer* handle ) {
  handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);
  /* backward transpose filters */
  handle->scratch1 = 0;
  handle->scratch1_size = (size_t)handle->blocksifm * handle->ifmblock * handle->blocksofm * handle->ofmblock
    * handle->desc.R * handle->desc.S * libxsmm_dnn_typesize(handle->datatype_in);
  if (handle->fm_lp_block > 1) {
    /* If low precision, we need extra buffer to store intermediate weight tensor */
    handle->scratch1_size *= 2;
  }

  /* weight update transpose of minibatch */
  handle->scratch3 = 0;
  handle->scratch3_size = (size_t)handle->desc.N * handle->blocksifm * handle->ifmblock * handle->ifhp * ((size_t)handle->ifwp + 8)
    * libxsmm_dnn_typesize(handle->datatype_in);

  /* minibatch parallel execution of weight update kernel */
  if ( ((handle->blocksifm * handle->blocksofm) < handle->desc.threads) || ( handle->use_upd_generic == 0 ) ) {
    handle->upd_use_thread_fil = 1;
    handle->scratch4 = 0;
    handle->scratch4_size = (size_t)2 * handle->desc.threads * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S * libxsmm_dnn_typesize(handle->datatype_out);
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) {
      /* Allocate twice as much since the out datatype is BF16 while the intermediate output is in float  */
      handle->scratch4_size = 2 * handle->scratch4_size;
    }
    /* enable external reduce of filter scratch */
    if ( (handle->options & LIBXSMM_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE) > 0 ) {
      handle->upd_use_external_reduce = 1;
    }
  } else {
    handle->scratch4 = 0;
    handle->scratch4_size = 0;
    handle->upd_use_thread_fil = 0;
  }

  /* Allocate scratch for additional output transpose */
  if (handle->use_lp_kernel == 1) {
    handle->scratch2 = 0;
    handle->scratch2_size = (size_t)handle->desc.N * handle->blocksofm * handle->ofmblock
      * ((size_t)handle->ofhp     + (size_t)2 * handle->desc.pad_h)
      * ((size_t)handle->ofwp + 8 + (size_t)2 * handle->desc.pad_w)
      * libxsmm_dnn_typesize(handle->datatype_in);
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) {
      /* Allocate scratch to dump results before down-convert  */
      handle->scratch2_size += (size_t)handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S * sizeof(float);
    }
  } else {
    handle->scratch2 = 0;
    handle->scratch2_size = 0;
  }

  /* Allocate scratch for auxiliary batchstats (sum and sum^2 for FWD)  */
  if ( handle->fuse_batchstats_fwd == 1 || handle->fuse_batchstats_bwd == 1 ) {
    handle->scratch7 = 0;
    handle->scratch7_size = (size_t) 2 * LIBXSMM_MAX(handle->desc.K, handle->desc.C) * handle->desc.N * sizeof(float);
  } else {
    handle->scratch7 = 0;
    handle->scratch7_size = 0;
  }

}

/**********************************************************/
/* Helper functions for convolutions' general param setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_ifmblock( libxsmm_dnn_layer* handle ) {
  int result = 1, tmp_max_c_block = 32, tmp_block;
  if (libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE) {
    tmp_max_c_block = 64;
  }
  if ( handle->desc.C < tmp_max_c_block ) {
    result = handle->desc.C;
  } else {
    for ( tmp_block = 1; tmp_block <= tmp_max_c_block; tmp_block *= 2 ) {
      if ( handle->desc.C % tmp_block == 0 ) result = tmp_block;
    }
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_ofmblock( libxsmm_dnn_layer* handle ) {
  int result = 1, tmp_max_k_block = 32, tmp_block;
  if (libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE) {
    tmp_max_k_block = 64;
  }
  if ( handle->desc.K < tmp_max_k_block ) {
    result = handle->desc.K;
  } else {
    for ( tmp_block = 1; tmp_block <= tmp_max_k_block; tmp_block *= 2 ) {
      if ( handle->desc.K % tmp_block == 0 ) result = tmp_block;
    }
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_fm_lp_block( libxsmm_dnn_layer* handle ) {
  int result = 1;
  if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) {
    result = 2;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_fallback_loops_fwd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  /* FIXME: For now fallback only if MB is not divisible by number of threads */
  if (handle->desc.N % handle->desc.threads != 0) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_blocksifm( libxsmm_dnn_layer* handle ) {
  int result = handle->desc.C / handle->ifmblock;
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_blocksofm( libxsmm_dnn_layer* handle ) {
  int result = handle->desc.K / handle->ofmblock;
  return result;
}

/**********************************************************/
/* Helper functions for FWD convolutions' parameter setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_fwd_ofw_rb( libxsmm_dnn_layer* handle ) {
  int result = 0;
  result = handle->ofw;
  if (handle->ofw == 56) {
    result = 28;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_pack_input_fwd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  /* Pack only for small images and when having large K to amortize */
  if ((handle->ofw <= 14) && (handle->desc.K > 512) && (handle->desc.R == 1) && (handle->desc.S == 1) && (handle->desc.u == 2) && (handle->desc.v == 2)) {
    result = 1;
  }
  /* Make sure we don't pack when minibatch is not divisible by number of threads since H is used potentially for parallelism */
  if (handle->desc.N != handle->desc.threads) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_fwd_ofh_rb( libxsmm_dnn_layer* handle ) {
  int result = 1;
  /* Multiple rows for "small" images and 1x1 convolutions */
  if ((handle->ofh <= 14) && (handle->desc.R == 1) && (handle->desc.S == 1)) {
    result = handle->ofh;
  }
  /*  Make sure we don't use multiple rows when we don't pack input and convolutions are strided*/
  if ((handle->pack_input == 0) && ((handle->desc.u !=1 ) || (handle->desc.v != 1))) {
    result = 1;
  }
  /* In this case we will be using fallback generic loops, thus ofh_rb should be 1 */
  if (handle->desc.N % handle->desc.threads != 0) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_fwd_block_H( libxsmm_dnn_layer* handle ) {
  int result = 14;
  /* Block H only for large images  */
  if (handle->ofh >= 28) {
    result = 4;
  }
  if (handle->ofh == 28 && handle->desc.R == 3 ) {
    result = 14;
  }
  /* Make sure it is divisible bu the ofh_rb factor in the kernel */
  while ( result % handle->fwd_ofh_rb != 0 ) {
    result--;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_blocksifm_blocking( libxsmm_dnn_layer* handle ) {
  int result = 1;
  /* For 1x1 Convolutions bring in kernel all IFMs unless filters are huge*/
  if ((handle->desc.R == 1) && (handle->desc.S == 1) ) {
    result = handle->blocksifm;
    if ((handle->desc.C >= 2048) && (handle->desc.K >= 512)) {
      result = 1;
    }
    if ((libxsmm_target_archid < LIBXSMM_X86_AVX512) && (handle->desc.C >= 512) && (handle->desc.K >= 512) ) {
      result = 2;
    }
  } else {
    result = 1;
    /* If small image can bring in more IFMS even if NOT 1x1 convolution */
    if (handle->ofw <= 7) {
      result = 2;
    }
  }
  if (handle->blocksifm % result != 0) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_loop_order_fwd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  /* Switch to loop order 1 only if 1x1 convolution with "large" input image and "small" K */
  if ((handle->desc.H >= 28) && (handle->desc.R == 1) && (handle->desc.S == 1) && (handle->desc.C >=512) && (handle->desc.K <=512)) {
    result = 1;
  }
  if (handle->ofw == 56 && handle->desc.R == 1 && handle->desc.C == 256 && handle->desc.K == 64 ) {
    result = 1;
  }
  if (handle->ofw == 28 && handle->desc.R == 1) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_block_fwd_IFM( libxsmm_dnn_layer* handle ) {
  int result = 8;
  if (handle->ofw == 7 && handle->desc.C == 2048 && handle->desc.K == 512) {
    result = 4;
  }
  /* Make sure it is divisible by ifms in the kernel  */
  while (result % handle->blocksifm_blocking != 0) {
    result++;
  }
  result = LIBXSMM_MIN(handle->blocksifm, result);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_block_fwd_OFM( libxsmm_dnn_layer* handle ) {
  int result = 8;
  if (handle->ofw == 14 && handle->desc.K == 1024) {
    result = 16;
  }
  if (handle->ofw == 7) {
    result = 16;
  }
  result = LIBXSMM_MIN(handle->blocksofm, result);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_use_ofm_parallelization( libxsmm_dnn_layer* handle ) {
  int result = 0;
#if 0
  /* Use "hybrid" minibatch/ofm parallelization if we have huge filters */
  if ((handle->desc.R >= 3) && (handle->desc.S >= 3) && (handle->desc.C >= 512) && (handle->desc.K >= 512)) {
    result = 1;
  }
#endif
  if ((handle->ofw <= 7) && (handle->desc.C == 1024) && (handle->desc.K == 512)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_avoid_rim_fmas_fwd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  /* Avoid rim FMA if the convolution is 3x3 (non-strided) and the image is "small" */
  if ((handle->desc.R == 3) && (handle->desc.S == 3) &&
      (handle->desc.u  == 1) && (handle->desc.v == 1) &&
      (handle->desc.pad_h_in == 1) && (handle->desc.pad_w_in == 1) &&
      (handle->desc.H == handle->desc.W) ) {
    if (handle->ofw <= 28) {
      result = 1;
    }
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_shuffle_filter_accesses( libxsmm_dnn_layer* handle ) {
  int result = 0;
  /* Shuffle filter accesses only if "pure minibatch" parallelization and large filters are involved */
  if ((handle->use_ofm_parallelization == 0) && (handle->desc.C > 512) && (handle->desc.K > 512)) {
    result = 1;
  }
  if (handle->ofw == 7 && handle->desc.R == 3 && handle->desc.C == 512) {
    result = 1;
  }
  if (handle->ofw == 7 && handle->desc.R == 1 && handle->desc.C == 512 && handle->desc.K == 2048) {
    result = 1;
  }
  if (handle->ofw == 7 && handle->desc.R == 1 && handle->desc.C == 2048 && handle->desc.K == 512) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_avoid_acc_load( libxsmm_dnn_layer* handle ) {
  int result = 0;
  if ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) {
    if ((handle->desc.R == 1) && (handle->desc.S == 1)) {
      if (handle->blocksifm_blocking == handle->blocksifm) {
        result = 1;
      }
    } else {
      if ((handle->blocksifm_blocking == handle->blocksifm) && (handle->avoid_fmas_in_rim == 0)) {
        result = 1;
      }
    }
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_init_fwd_gemm_flags( libxsmm_dnn_layer* handle ) {
  int result = 0;
  /* If large image and NOT already loaded in accumulators, tnen use streaming stores */
  if ((handle->ofw >= 56) && (handle->desc.K >= 256) && (handle->avoid_acc_load == 1) && (handle->desc.R == 1) && (handle->desc.S == 1)) {
    result = LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT;
  }
  if (handle->ofw == 56 && handle->desc.C == 64 && handle->desc.K == 64 && handle->desc.R == 1) {
    result = LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT;
  }
  if (handle->ofw == 56 && handle->desc.C == 256 && handle->desc.K == 64 && handle->desc.R == 1) {
    result = LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT;
  }
  return result;
}

/**********************************************************/
/* Helper functions for BWD convolutions' parameter setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_fallback_loops_bwd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  /* FIXME: Fallback if MB is not divisible by number of threads */
  if (handle->desc.N % handle->desc.threads != 0) {
    result = 1;
  }
  if (handle->desc.R == 1 && handle->desc.S == 1 && (handle->desc.pad_h != 0 ||  handle->desc.pad_w != 0)) {
    result = 1;
  }
  if ((handle->desc.R > 1 && handle->desc.pad_h == 0) || (handle->desc.S > 1 && handle->desc.pad_w == 0)) {
    result = 1;
  }
  if ((handle->desc.R > 1 && handle->desc.u > 1) || (handle->desc.S > 1 && handle->desc.v > 1)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_bwd_ofw_rb( libxsmm_dnn_layer* handle ) {
  int result = libxsmm_dnn_setup_generic_fwd_ofw_rb(handle);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_bwd_ofh_rb( libxsmm_dnn_layer* handle ) {
  int result = libxsmm_dnn_setup_generic_fwd_ofh_rb(handle);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_bwd_block_H( libxsmm_dnn_layer* handle ) {
  int result = 0;
  result = libxsmm_dnn_setup_generic_fwd_block_H(handle);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_loop_order_bwd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  result = libxsmm_dnn_setup_generic_loop_order_fwd(handle);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_block_bwd_IFM( libxsmm_dnn_layer* handle ) {
  int result = 0;
  result = LIBXSMM_MIN(handle->blocksifm, 16);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_block_bwd_OFM( libxsmm_dnn_layer* handle ) {
  int result = 8;
  while (result % handle->blocksofm_blocking != 0) {
    result++;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_pack_input_bwd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  if ((handle->desc.u != 1) && (handle->bwd_ofh_rb != 1)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_use_ifm_parallelization( libxsmm_dnn_layer* handle ) {
  int result = 0;
  if (handle->ofw <= 7) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_avoid_rim_fmas_bwd( libxsmm_dnn_layer* handle ) {
  int result = libxsmm_dnn_setup_generic_avoid_rim_fmas_fwd(handle);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_blocksofm_blocking( libxsmm_dnn_layer* handle ) {
  int result = 0;
  if (handle->desc.R == 1 && handle->desc.S == 1) {
    result = handle->blocksofm;
  } else {
    result = 1;
    if (handle->desc.R == 3 && handle->desc.S == 3 && handle->ofh == 7 && handle->ofw == 7) {
      result = 2;
    }
  }
  if (handle->blocksofm % result != 0) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_init_bwd_gemm_flags( libxsmm_dnn_layer* handle ) {
  int result = 0;

  /* TODO: May want to experiment with streaming stores */
  LIBXSMM_UNUSED( handle );

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_spread_input_bwd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  LIBXSMM_UNUSED(handle);
  if (((handle->desc.u != 1) || (handle->desc.v != 1)) && (handle->bwd_ofh_rb == 1)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_avoid_acc_load_bwd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  if ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) {
    if ((handle->desc.R == 1) && (handle->desc.S == 1)) {
      if (handle->blocksofm_blocking == handle->blocksofm) {
        result = 1;
      }
    } else {
      if ((handle->blocksofm_blocking == handle->blocksofm) && (handle->avoid_fmas_in_rim == 0)) {
        result = 1;
      }
    }
  }
  return result;
}

/**********************************************************/
/* Helper functions for UPD convolutions' parameter setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_loop_order_upd( libxsmm_dnn_layer* handle ) {
  int result = 1;
  if (handle->ofh == 28 && handle->desc.R == 1 && handle->desc.u == 1 && handle->desc.C == 128 && handle->desc.K == 512) {
    result = 0;
  }
  if (handle->ofh == 28 && handle->desc.R == 3 && handle->desc.u == 1 && handle->desc.C == 128 && handle->desc.K == 128) {
    result = 0;
  }
  if (handle->ofw == 28 && handle->desc.R == 1 && handle->desc.C == 256 && handle->desc.K == 512) {
    result = 0;
  }
  if (handle->ofw == 14 && !(handle->desc.R == 1 && handle->desc.C == 1024 && handle->desc.K == 256)) {
    result = 0;
  }
  if (handle->ofw == 7) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_pack_input_upd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  /* Pack input only for very small images, 1x1 convs, with large K to amortize the relevant overhead */
  if ((handle->ofh <= 7) && (handle->desc.R == 1) && (handle->desc.S == 1) && (handle->desc.u != 1) && (handle->desc.v != 1) && (handle->desc.K >= 2048)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_avoid_rim_fmas_upd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  /* Avoid rim FMAs only for small images  */
  if ( (handle->ofh <= 7) && (handle->desc.R == 3) && (handle->desc.S == 3) && (handle->desc.pad_w == 1) && (handle->desc.pad_h == 1)) {
    result = 1;
  }
  if (handle->desc.N != handle->desc.threads) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_upd_ofw_rb( libxsmm_dnn_layer* handle ) {
  int result = 1;
  result = handle->ofw;
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_upd_ofh_rb( libxsmm_dnn_layer* handle ) {
  int result = 1;
  /* Restrict the reduction chain which is ofw_rb*ofh_rb*/
  if (handle->ofh <= 28 ) {
    result = handle->ofh;
  }
  /* In the following scenario with strided convolutions and non batch reduce kernel make sure we have ofh_rb = 1  */
  if ((handle->desc.u != 1) && (handle->desc.v != 1) && (handle->upd_use_batchreduce == 0) && (handle->upd_pack_input == 0)) {
    result = 1;
  }
  /* If using linearized taskview and have strided convs, make sure ofh_rb is 1.. */
  if (handle->upd_linearized_tasklist == 1 && handle->upd_avoid_rim_fmas == 0 && handle->upd_pack_input == 0 && handle->desc.u != 1) {
    result = 1;
  }
  if (handle->upd_linearized_tasklist == 1 && handle->upd_use_batchreduce == 0 && (handle->desc.R != 1 || handle->desc.S != 1)) {
    result = 1;
  }
  if (handle->upd_linearized_tasklist == 0 && handle->upd_use_batchreduce == 0 && (handle->desc.R != 1 || handle->desc.S != 1)) {
    result = 1;
  }
  if (handle->ofw == 56 && handle->desc.R == 1) {
    result = 2;
  }
  if (handle->upd_linearized_tasklist == 1 && handle->upd_use_batchreduce == 1 && handle->upd_avoid_rim_fmas == 1) {
    result = handle->ofh;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_block_upd_IFM( libxsmm_dnn_layer* handle ) {
  int result = 1;
  if (handle->ofh == 56 && handle->desc.R == 1 && handle->desc.S == 1 && handle->desc.u == 1 && handle->desc.v == 1) {
    result = 4;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_block_upd_OFM( libxsmm_dnn_layer* handle ) {
  int result = 1;
  LIBXSMM_UNUSED(handle);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_img_batchreduce_block( libxsmm_dnn_layer* handle ) {
  int result = 1;
  LIBXSMM_UNUSED(handle);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_use_batchreduce_upd( libxsmm_dnn_layer* handle ) {
  int result = 1;
  /* If W is large, no need for batchreduce kernel */
  if (handle->ofw >= 56) {
    result = 0;
  }
  /* If we have packed the input, then disable batch-reduce GEMM */
  if (handle->upd_pack_input == 1) {
    result = 0;
  }
  if (handle->upd_linearized_tasklist == 1 && handle->upd_avoid_rim_fmas == 0) {
    result = 0;
  }
  if (handle->upd_linearized_tasklist == 1 && handle->upd_avoid_rim_fmas == 1) {
    result = 1;
  }

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_weight_copies_upd( libxsmm_dnn_layer* handle ) {
  int result = handle->desc.threads;
  if (handle->ofw <= 14) {
    result = 9;
  }
  while (handle->desc.threads % result != 0) {
    result--;
  }
  /* FIXME: Hardcoded logic for N=27, N=26 */
  if (handle->desc.N == 27 && handle->desc.threads == 27 && handle->desc.R == 1 && handle->ofw == 14 && handle->desc.u == 1) {
    result = 7;
  }
  if (handle->ofh == 14 && handle->desc.R == 3 && handle->desc.S == 3) {
    if (handle->desc.N == 26) {
      result = 13;
    }
  }
  if ((handle->desc.N != handle->desc.threads) && !(handle->upd_linearized_tasklist == 0 && handle->upd_use_batchreduce == 0)) {
    result = handle->desc.N;
  }
  /* Make sure a single copy when we use linearized-task view */
  if (handle->upd_linearized_tasklist == 1) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_linearized_tasklist_upd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  /* Use linearized task-list (i.e. no reduction) only if small images and large filters */
  if (handle->ofh <= 10 && handle->ofw <= 10) {
    result = 1;
  }
#if 0
  if ((handle->blocksofm * handle->blocksifm * handle->desc.R * handle->desc.S > (handle->desc.threads * 4)) && (handle->ofh <= 56)) {
    result = 1;
  }
#endif
  if (handle->desc.u == 2 && handle->desc.v == 2 && handle->desc.K == 512) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_setup_generic_init_upd_gemm_flags( libxsmm_dnn_layer* handle ) {
  int result = 0;
  LIBXSMM_UNUSED(handle);
  return result;
}

LIBXSMM_API_INLINE void libxsmm_dnn_setup_generic_bf16_upd( libxsmm_dnn_layer* handle ) {
  int remainder_pixels, max_init_offset, max_compute_offset_input, input_compute_pad, accum_length_pixels, compute_pixels;
  const int multiple_target = 2;
  handle->upd_linearized_pixels = 1;
  if (handle->desc.S != 1 && handle->desc.v != 1) {
    handle->upd_linearized_pixels = 0;
  }

  handle->use_lp_kernel = 1;
  handle->on_the_fly_input_packing = 0;
  handle->upd_pack_input_upfront = 0;
  handle->use_hybrid_imgofm_parallelization = 0;
  handle->upd_linearized_tasklist = 0;

  if (handle->upd_linearized_pixels == 1) {
    /* Logistics to pad accumulation chainlength */
    compute_pixels = handle->ofw * handle->ofh + 2 * handle->desc.pad_w * (handle->ofh-1);
    remainder_pixels = (compute_pixels % multiple_target == 0) ? 0 : (compute_pixels/multiple_target+1)*multiple_target - compute_pixels;
    accum_length_pixels = compute_pixels + remainder_pixels;

    /* In this case compact input upfront */
    if (handle->desc.R == 1 && handle->desc.S == 1 && (handle->desc.u != 1 || handle->desc.v != 1)) {
      handle->upd_pack_input_upfront = 1;
    }

    /* Logistics for input transpose and additional pixel padding */
    max_init_offset = 2 * handle->desc.pad_h * handle->ifwp + 2 * handle->desc.pad_w;
    max_compute_offset_input = max_init_offset + accum_length_pixels;
    input_compute_pad = (max_compute_offset_input > handle->ifwp*handle->ifhp) ? max_compute_offset_input - handle->ifwp*handle->ifhp : 0;
    handle->input_pixels = handle->ifwp * handle->ifhp + input_compute_pad;
    if (handle->upd_pack_input_upfront) {
      handle->input_pixels = accum_length_pixels;
    }
    handle->output_pixels = accum_length_pixels;
    handle->pixel_blocking = accum_length_pixels;
    handle->n_used_pixels = accum_length_pixels;

    handle->use_intermediate_f32_wt_tensor = (handle->pixel_blocking == handle->n_used_pixels) ? 0 : 1;
    handle->scratch2_size = (size_t) (handle->desc.N * handle->output_pixels * handle->desc.K * sizeof(float)/2);
    if (handle->use_intermediate_f32_wt_tensor) {
      handle->scratch2_size += (size_t) handle->desc.R * handle->desc.S * handle->desc.C * handle->desc.K * handle->desc.threads * sizeof(float);
    }
    handle->scratch3_size = (size_t) (handle->desc.N * handle->input_pixels * handle->desc.C * sizeof(float)/2);

    if (handle->ofw <= 14) {
      handle->use_hybrid_imgofm_parallelization = 1;
    } else {
      handle->weight_copies = handle->desc.threads;
    }
  }

  if (handle->upd_linearized_pixels == 0) {
    handle->weight_copies = handle->desc.threads;
    if (handle->desc.v !=1) {
      handle->on_the_fly_input_packing = 1;
    }
    remainder_pixels = (handle->ofw % multiple_target == 0) ? 0 : (handle->ofw/multiple_target+1)*multiple_target - handle->ofw;
    handle->ofwp_extended = handle->ofwp + remainder_pixels;
    handle->ifwp_extended = handle->ifwp + remainder_pixels;
    handle->batchreduce_h_pixels = handle->ofh;
    handle->use_intermediate_f32_wt_tensor = (handle->batchreduce_h_pixels == handle->ofh) ? 0 : 1;
    handle->scratch2_size = (size_t) (handle->desc.N * handle->ofhp*handle->ofwp_extended * handle->desc.K * sizeof(float)/2);
    if (handle->use_intermediate_f32_wt_tensor) {
      handle->scratch2_size += (size_t) handle->desc.R * handle->desc.S * handle->desc.C * handle->desc.K * handle->desc.threads * sizeof(float);
    }
    handle->scratch3_size = (size_t) (handle->desc.N * handle->ifhp * handle->ifwp_extended * handle->desc.C * sizeof(float)/2);
  }

}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_setup_generic( libxsmm_dnn_layer* handle ) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  const libxsmm_trans_descriptor* tr_desc = 0;
  libxsmm_descriptor_blob blob;

  /* Generic parameter setup  */
  handle->ifmblock = libxsmm_dnn_setup_generic_ifmblock(handle);
  handle->ofmblock = libxsmm_dnn_setup_generic_ofmblock(handle);
  handle->fm_lp_block = libxsmm_dnn_setup_generic_fm_lp_block(handle);
  handle->blocksifm = libxsmm_dnn_setup_generic_blocksifm(handle);
  handle->blocksofm = libxsmm_dnn_setup_generic_blocksofm(handle);

  /* FWD parameter setup  */
  handle->fwd_ofw_rb = libxsmm_dnn_setup_generic_fwd_ofw_rb(handle);
  handle->pack_input = libxsmm_dnn_setup_generic_pack_input_fwd(handle);
  handle->fwd_ofh_rb = libxsmm_dnn_setup_generic_fwd_ofh_rb(handle);
  handle->block_fwd_oj = libxsmm_dnn_setup_generic_fwd_block_H(handle);
  handle->loop_order = libxsmm_dnn_setup_generic_loop_order_fwd(handle);
  handle->blocksifm_blocking = libxsmm_dnn_setup_generic_blocksifm_blocking(handle);
  handle->block_fwd_ofm = libxsmm_dnn_setup_generic_block_fwd_OFM(handle);
  handle->block_fwd_ifm = libxsmm_dnn_setup_generic_block_fwd_IFM(handle);;
  handle->avoid_fmas_in_rim = libxsmm_dnn_setup_generic_avoid_rim_fmas_fwd(handle);
  handle->use_ofm_parallelization = libxsmm_dnn_setup_generic_use_ofm_parallelization(handle);
  handle->shuffle_filter_accesses = libxsmm_dnn_setup_generic_shuffle_filter_accesses(handle);
  handle->avoid_acc_load = libxsmm_dnn_setup_generic_avoid_acc_load(handle);
  handle->fwd_flags = libxsmm_dnn_setup_generic_init_fwd_gemm_flags(handle);
  handle->use_fallback_fwd_loops = libxsmm_dnn_setup_generic_fallback_loops_fwd(handle);
  handle->code_fwd[0].xconv.sconv = 0;
  handle->code_fwd[1].xconv.sconv = 0;
  handle->code_fwd[2].xconv.sconv = 0;

#if 0
  /* Spit out FWD parameters that are selected...  */
  printf("FWD params...\n");
  printf("Fwd_ofw_rb = %d\n", handle->fwd_ofw_rb);
  printf("Fwd_ofh_rb = %d\n", handle->fwd_ofh_rb);
  printf("Pack input = %d\n", handle->pack_input);
  printf("Block oj = %d\n", handle->block_fwd_oj);
  printf("Loop order = %d\n", handle->loop_order);
  printf("Blocksifm_blocking = %d\n", handle->blocksifm_blocking);
  printf("Block fwd ofm = %d\n", handle->block_fwd_ofm);
  printf("Block fwd ifm = %d\n", handle->block_fwd_ifm);
  printf("Avoid rim fmas = %d\n", handle->avoid_fmas_in_rim);
  printf("Ofm parallelization = %d\n", handle->use_ofm_parallelization);
  printf("Shuffle filter accesses = %d\n", handle->shuffle_filter_accesses);
  printf("Avoid acc load = %d\n", handle->avoid_acc_load);
  printf("Fwd GEMM flags = %d\n", handle->fwd_flags);
#endif

  /* BWD parameter setup  */
  handle->bwd_ofw_rb = libxsmm_dnn_setup_generic_bwd_ofw_rb(handle);
  handle->bwd_ofh_rb = libxsmm_dnn_setup_generic_bwd_ofh_rb(handle);
  handle->pack_input_bwd = libxsmm_dnn_setup_generic_pack_input_bwd(handle);
  handle->spread_input_bwd = libxsmm_dnn_setup_generic_spread_input_bwd(handle);
  handle->blocksofm_blocking = libxsmm_dnn_setup_generic_blocksofm_blocking(handle);
  handle->avoid_acc_load_bwd = libxsmm_dnn_setup_generic_avoid_acc_load_bwd(handle);
  handle->use_ifm_parallelization = libxsmm_dnn_setup_generic_use_ifm_parallelization(handle);
  handle->block_bwd_ofm = libxsmm_dnn_setup_generic_block_bwd_OFM(handle);
  handle->block_bwd_ifm = libxsmm_dnn_setup_generic_block_bwd_IFM(handle);
  handle->block_bwd_oj = libxsmm_dnn_setup_generic_bwd_block_H(handle);
  handle->use_fallback_bwd_loops = libxsmm_dnn_setup_generic_fallback_loops_bwd(handle);

#if 0
  /* Spit out BWD parameters that are selected...  */
  printf("BWD params...\n");
  printf("Bwd_ofw_rb = %d\n", handle->bwd_ofw_rb);
  printf("Bwd_ofh_rb = %d\n", handle->bwd_ofh_rb);
  printf("Pack input = %d\n", handle->pack_input_bwd);
  printf("Spread input = %d\n", handle->spread_input_bwd);
  printf("Blocksofm_blocking = %d\n", handle->blocksofm_blocking);
  printf("Avoid acc load = %d\n", handle->avoid_acc_load_bwd);
  printf("Ifm parallelization = %d\n", handle->use_ifm_parallelization);
  printf("Block bwd ofm = %d\n", handle->block_bwd_ofm);
  printf("Block bwd ifm = %d\n", handle->block_bwd_ifm);
  printf("Block oj = %d\n", handle->block_bwd_oj);
#endif

  handle->code_bwd[0].xconv.sconv = 0;
  handle->code_bwd[1].xconv.sconv = 0;
  handle->code_bwd[2].xconv.sconv = 0;
  /* Transpose kernel used for filter transpose in bwd pass  */
  tr_desc = libxsmm_trans_descriptor_init(&blob, sizeof(float), 64, 16, 64);
  handle->tr_kernel = libxsmm_dispatch_trans(tr_desc);

  /* UPD parameter setup */
  handle->upd_linearized_tasklist = libxsmm_dnn_setup_generic_linearized_tasklist_upd(handle);
  handle->upd_avoid_rim_fmas = libxsmm_dnn_setup_generic_avoid_rim_fmas_upd(handle);
  handle->upd_pack_input = libxsmm_dnn_setup_generic_pack_input_upd(handle);
  handle->upd_use_batchreduce = libxsmm_dnn_setup_generic_use_batchreduce_upd(handle);
  handle->upd_ofw_rb = libxsmm_dnn_setup_generic_upd_ofw_rb(handle);
  handle->upd_ofh_rb = libxsmm_dnn_setup_generic_upd_ofh_rb(handle);
  handle->upd_loop_order = libxsmm_dnn_setup_generic_loop_order_upd(handle);
  handle->weight_copies = libxsmm_dnn_setup_generic_weight_copies_upd(handle);
  handle->block_upd_ofm = libxsmm_dnn_setup_generic_block_upd_OFM(handle);
  handle->block_upd_ifm = libxsmm_dnn_setup_generic_block_upd_IFM(handle);
  handle->upd_loop_order = libxsmm_dnn_setup_generic_loop_order_upd(handle);

  if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) {
    libxsmm_dnn_setup_generic_bf16_upd(handle);
  }

#if 0
  /* Spit out UPD parameters that are selected...  */
  printf("UPD params...\n");
  printf("UPD linearized tasks = %d\n", handle->upd_linearized_tasklist);
  printf("UPD avoid rim fmas = %d\n", handle->upd_avoid_rim_fmas);
  printf("UPD Pack input = %d\n", handle->upd_pack_input);
  printf("UPD use batch-reduce GEMM = %d\n", handle->upd_use_batchreduce);
  printf("Upd_ofw_rb = %d\n", handle->upd_ofw_rb);
  printf("Upd_ofh_rb = %d\n", handle->upd_ofh_rb);
  printf("UPD loop order = %d\n", handle->upd_loop_order);
  printf("UPD weight_copies = %d\n", handle->weight_copies);
  printf("Block upd ofm = %d\n", handle->block_upd_ofm);
  printf("Block upd ifm = %d\n", handle->block_upd_ifm);
#endif

  handle->code_upd[0].xconv.sconv = 0;
  handle->code_upd[1].xconv.sconv = 0;

  /*****************************/
  /* Barrier and scratch setup */
  /*****************************/
  /* prepare barrier */
  handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);
  /* backward transpose filters, as we want to call small GEMMs we need that scratch AND also scratch to potentially pack input if requested*/
  handle->scratch1 = 0;
  handle->scratch1_size = (size_t)handle->blocksifm * handle->ifmblock * handle->blocksofm * handle->ofmblock
    * handle->desc.R * handle->desc.S * libxsmm_dnn_typesize(handle->datatype_in) + (size_t)handle->desc.N * handle->ofwp * handle->ofhp * handle->desc.C;
  if (handle->fm_lp_block > 1) {
    /* If low precision, we need extra buffer to store intermediate weight tensor */
    handle->scratch1_size *= 2;
  }
  handle->scratch3 = 0;
  handle->scratch3_size = 0;
  handle->scratch4 = 0;
  handle->scratch4_size = 0;
  handle->scratch6 = 0;
  handle->scratch6_size = 0;

  /* In this case, allocate scratch for output in fp32 precision (to use when we don't fully accumulate) + a scratchpad (when we fully accumulate)  */
  if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) {
    handle->scratch6_size = (size_t) (handle->desc.N * LIBXSMM_MAX(handle->ofwp * handle->ofhp * handle->desc.K, handle->desc.W * handle->desc.H * handle->desc.C) + handle->desc.threads * LIBXSMM_MAX(handle->fwd_ofw_rb * handle->fwd_ofh_rb * handle->ofmblock, handle->bwd_ofw_rb * handle->bwd_ofh_rb * handle->ifmblock))* sizeof(float);
  }

  return status;
}

#undef MIXED
#undef KHWC
#undef HWKC
#undef CHWK
#undef HWCK

