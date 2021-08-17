/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst, Alexander Heinecke, Evangelos Georganas, Rajkishore Barik (Intel Corp.)
******************************************************************************/
#include <libxsmm_sync.h>
#include "libxsmm_main.h"
#include "libxsmm_dnn_convolution_forward.h"
#include "libxsmm_dnn_convolution_backward.h"
#include "libxsmm_dnn_convolution_weight_update.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#define MIXED 0
#define KHWC 1
#define HWKC 2
#define CHWK 3
#define HWCK 4

/**********************************************************/
/* Helper functions for convolutions' general param setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_ifmblock( libxsmm_dnn_layer* handle ) {
  int result = 1;
  int ofm, lp;

  libxsmm_dnn_get_feature_map_blocks( handle->desc.C, handle->desc.K, &result, &ofm, &lp, handle->desc.datatype_in, handle->desc.datatype_out );

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_ofmblock( libxsmm_dnn_layer* handle ) {
  int result = 1;
  int ifm, lp;

  libxsmm_dnn_get_feature_map_blocks( handle->desc.C, handle->desc.K, &ifm, &result, &lp, handle->desc.datatype_in, handle->desc.datatype_out );

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_fm_lp_block( libxsmm_dnn_layer* handle ) {
  int result = 1;
  int ifm, ofm;

  libxsmm_dnn_get_feature_map_blocks( handle->desc.C, handle->desc.K, &ifm, &ofm, &result, handle->desc.datatype_in, handle->desc.datatype_out );

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_fallback_loops_fwd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  /* FIXME: For now fallback only if MB is not divisible by number of threads */
  if (handle->desc.N % handle->desc.threads != 0) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_blocksifm( libxsmm_dnn_layer* handle ) {
  int result = handle->desc.C / handle->ifmblock;
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_blocksofm( libxsmm_dnn_layer* handle ) {
  int result = handle->desc.K / handle->ofmblock;
  return result;
}

/**********************************************************/
/* Helper functions for FWD convolutions' parameter setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_fwd_ofw_rb( libxsmm_dnn_layer* handle ) {
  int result = 0;
  result = handle->ofw;
  if (handle->ofw == 56) {
    result = 28;
  }
  if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) {
    if (handle->ofw % 2 == 0) {
      result = handle->ofw/2;
    }
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_pack_input_fwd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  /* Pack only for small images and when having large K to amortize, and we can only pack for 1x1 convolutions */
  if ((handle->ofw <= 14) && (handle->desc.K > 512) && (handle->desc.R == 1) && (handle->desc.S == 1) && (handle->desc.u == 2) && (handle->desc.v == 2)) {
    result = 1;
  }

  /* For SPR we allow packing more aggressively to generate more efficient BRGEMMs */
  if ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) && ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8)) ) {
    if ((handle->ofw <= 14) && (handle->desc.R == 1) && (handle->desc.S == 1) && (handle->desc.u == 2) && (handle->desc.v == 2)) {
      result = 1;
    }
  }

  /* Make sure we don't pack when minibatch is not divisible by number of threads since H is used potentially for parallelism */
  if (handle->desc.N != handle->desc.threads) {
    result = 0;
  }
  /* we don't pack for int8 */
  if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_fwd_ofh_rb( libxsmm_dnn_layer* handle ) {
  int result = 1;
  /* Multiple rows for "small" images and 1x1 convolutions */
  if ((handle->ofh <= 14) && (handle->desc.R == 1) && (handle->desc.S == 1)) {
    result = handle->ofh;
  }

  /* In this case we will be using fallback generic loops, thus ofh_rb should be 1 */
  if ((handle->desc.N % handle->desc.threads != 0) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8)) {
    result = 1;
  }

  if ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) && ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8)) ) {
    if (handle->ofw == 7 && handle->ofh == 7 && handle->desc.R == 3 && handle->desc.S == 3) {
      result = 7;
    }
    if (handle->ofw == 14 && handle->ofh == 14 /*&& handle->desc.R == 3 && handle->desc.S == 3*/) {
      result = 2;
    }
  }

  /*  Make sure we don't use multiple rows when we don't pack input and convolutions are strided*/
  if ((handle->pack_input == 0) && ((handle->desc.u !=1 ) || (handle->desc.v != 1))) {
    result = 1;
  }

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_fwd_pixels_gemm( libxsmm_dnn_layer* handle ) {
  int result = handle->fwd_ofw_rb * handle->fwd_ofh_rb;
  /* In the case below we calculate redundantly pixels in order to efficiently use AMX */
  if ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) && ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8)) ) {
    if (handle->desc.R != 1 || handle->desc.R != 1) {
      if (handle->ofw < 24) {
        result = (handle->fwd_ofw_rb+2*handle->desc.pad_w) * (handle->fwd_ofh_rb-2) + 2 * (handle->fwd_ofw_rb+handle->desc.pad_w);
      }
    }
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_fwd_block_H( libxsmm_dnn_layer* handle ) {
  int result = 14;

  if ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) && ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8)) ) {
    /* Spatial dimension block tuning for SPR */
    if ((handle->ofh == 7 && handle->desc.u == 2) || (handle->ofh == 14 && handle->desc.R != 3 ) ||  handle->ofh == 27 || (handle->ofh == 28 && handle->desc.R == 1) || handle->ofh == 48 || handle->ofh == 54 || handle->ofh == 56 || handle->ofh == 112 ) {
      result = 4;
    }
  } else {
    /* Block H only for large images  */
    if (handle->ofh >= 28) {
      result = 4;
    }
    if (handle->ofh == 28 && handle->desc.R == 3 ) {
      result = 14;
    }
  }
  /* Make sure it is divisible bu the ofh_rb factor in the kernel */
  while ( result % handle->fwd_ofh_rb != 0 ) {
    result--;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_blocksifm_blocking( libxsmm_dnn_layer* handle ) {
  int result = 1;
  /* For 1x1 Convolutions bring in kernel all IFMs unless filters are huge*/
  if ((handle->desc.R == 1) && (handle->desc.S == 1) ) {
    result = handle->blocksifm;
    if ((handle->desc.C >= 2048) && (handle->desc.K >= 512)) {
      result = 1;
    }
    if ( (handle->target_archid < LIBXSMM_X86_AVX512_VL256) && (handle->desc.C >= 512) ) {
      result = 2;
    }
    if ( (handle->target_archid < LIBXSMM_X86_AVX512_VL256) && (handle->desc.C >= 1024) ) {
      result = 4;
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

  /* In case of SPR bring always in all accumulation */
  if ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) && ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8))) {
    result = handle->blocksifm;
  }

  if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) {
    result = handle->blocksifm;
  }

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_loop_order_fwd( libxsmm_dnn_layer* handle ) {
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

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_block_fwd_IFM( libxsmm_dnn_layer* handle ) {
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

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_block_fwd_OFM( libxsmm_dnn_layer* handle ) {
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

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_use_ofm_parallelization( libxsmm_dnn_layer* handle ) {
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
  if ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) && ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8))) {
    if (handle->ofw == 7) {
      result = 1;
    }
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_avoid_rim_fmas_fwd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  /* Avoid rim FMA if the convolution is 3x3 (non-strided) and the image is "small" */
  if ((handle->desc.R == 3) && (handle->desc.S == 3) &&
      (handle->desc.u  == 1) && (handle->desc.v == 1) &&
      (handle->desc.pad_h_in == 1) && (handle->desc.pad_w_in == 1) &&
      (handle->desc.H == handle->desc.W) ) {
    if (handle->ofw <= 28) {
      result = 1;
    }
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) {
      result = 0;
    }
  }
  if ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) && ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8))) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_shuffle_filter_accesses( libxsmm_dnn_layer* handle ) {
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
  if ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) && ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8)) )  {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_avoid_acc_load( libxsmm_dnn_layer* handle ) {
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

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_init_fwd_gemm_flags( libxsmm_dnn_layer* handle ) {
  int result = 0;

#if defined(LIBXSMM_DNN_CONVOLUTION_SETUP_USE_NTS)
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
  /* Disable since the GEMM output is going to f32 scratch  */
  if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16 || handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) {
    result = 0;
  }
#else
  LIBXSMM_UNUSED(handle);
#endif

  if ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) && ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8))) {
    result = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  }

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_fwd_padding_copy( libxsmm_dnn_layer* handle ) {
  int result = 0;
  if ( (handle->desc.pad_h != handle->desc.pad_h_in) && (handle->desc.pad_w != handle->desc.pad_w_in) ) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE void libxsmm_dnn_convolution_setup_fwd_scratch( libxsmm_dnn_layer* handle ) {
  handle->fwd_packing_padding_scratch_size = 0;
  /* packing of input */
  if ( handle->pack_input != 0 ) {
    handle->fwd_packing_padding_scratch_size = (size_t)handle->desc.N * handle->desc.C *
      handle->desc.H/handle->desc.u *
      handle->desc.W/handle->desc.v *
      libxsmm_dnn_typesize(handle->datatype_in);
  }
  /* logical padding with copying in the fly */
  if ( handle->fwd_padding_copy != 0 ) {
    handle->fwd_packing_padding_scratch_size = (size_t)handle->desc.N * handle->desc.C *
      (handle->desc.H + 2*handle->desc.pad_h) *
      (handle->desc.W + 2*handle->desc.pad_w) *
      libxsmm_dnn_typesize(handle->datatype_in);
  }
  /* output buffer in high precision when we use BF16 */
  if ( ( handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16 ) ||
      ( handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8 )      ) {
    handle->fwd_lp_output_full_scratch_size = (size_t) LIBXSMM_MAX(handle->desc.threads * handle->fwd_gemm_pixels * handle->ofmblock * libxsmm_dnn_typesize(LIBXSMM_DNN_DATATYPE_F32), handle->desc.N * handle->desc.K * handle->ofwp * handle->ofhp * libxsmm_dnn_typesize(LIBXSMM_DNN_DATATYPE_F32));
    handle->fwd_lp_output_block_scratch_size = (size_t)handle->desc.threads * handle->fwd_ofw_rb *
      handle->fwd_ofh_rb * handle->ofmblock *
      libxsmm_dnn_typesize(LIBXSMM_DNN_DATATYPE_F32);
  } else {
    handle->fwd_lp_output_full_scratch_size = 0;
    handle->fwd_lp_output_block_scratch_size = 0;
  }
  /* align sizes to full cacheline */
  handle->fwd_packing_padding_scratch_size += ( handle->fwd_packing_padding_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (handle->fwd_packing_padding_scratch_size % LIBXSMM_CACHELINE);
  handle->fwd_lp_output_full_scratch_size += ( handle->fwd_lp_output_full_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (handle->fwd_lp_output_full_scratch_size % LIBXSMM_CACHELINE);
  handle->fwd_lp_output_block_scratch_size += ( handle->fwd_lp_output_block_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (handle->fwd_lp_output_block_scratch_size % LIBXSMM_CACHELINE);

  /* set offsets */
  handle->fwd_packing_padding_scratch_offset = 0;
  handle->fwd_lp_output_full_scratch_offset = handle->fwd_packing_padding_scratch_size;
  handle->fwd_lp_output_block_scratch_offset = handle->fwd_lp_output_full_scratch_offset +
    handle->fwd_lp_output_full_scratch_size;

  /* set overall scratch size for forward */
  handle->fwd_scratch_size = handle->fwd_packing_padding_scratch_size +
    handle->fwd_lp_output_full_scratch_size +
    handle->fwd_lp_output_block_scratch_size;
}

/**********************************************************/
/* Helper functions for BWD convolutions' parameter setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_fallback_loops_bwd( libxsmm_dnn_layer* handle ) {
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
  if ((handle->desc.R > 1 && (handle->desc.pad_h_out == 0 || handle->desc.pad_h_in == 0)) ||
      (handle->desc.S > 1 && (handle->desc.pad_w_out == 0 || handle->desc.pad_w_in == 0))    ) {
    result = 1;
  }
  if ((handle->desc.R > 1 && handle->desc.u > 1) || (handle->desc.S > 1 && handle->desc.v > 1)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_bwd_ofw_rb( libxsmm_dnn_layer* handle ) {
  int result = libxsmm_dnn_convolution_setup_fwd_ofw_rb(handle);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_bwd_ofh_rb( libxsmm_dnn_layer* handle ) {
  int result = libxsmm_dnn_convolution_setup_fwd_ofh_rb(handle);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_bwd_pixels_gemm( libxsmm_dnn_layer* handle ) {
  int result = handle->bwd_ofw_rb * handle->bwd_ofh_rb;
  /* In the case below we calculate redundantly pixels in order to efficiently use AMX */
  if ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) && ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8)) ) {
    if (handle->desc.R != 1 || handle->desc.R != 1) {
      if (handle->ofw < 24) {
        result = (handle->bwd_ofw_rb+2*handle->desc.pad_w) * (handle->bwd_ofh_rb-2) + 2 * (handle->bwd_ofw_rb+handle->desc.pad_w);
      }
    }
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_bwd_block_H( libxsmm_dnn_layer* handle ) {
  int result = 0;
  result = libxsmm_dnn_convolution_setup_fwd_block_H(handle);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_loop_order_bwd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  result = libxsmm_dnn_convolution_setup_loop_order_fwd(handle);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_block_bwd_IFM( libxsmm_dnn_layer* handle ) {
  int result = 0;
  result = LIBXSMM_MIN(handle->blocksifm, 16);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_block_bwd_OFM( libxsmm_dnn_layer* handle ) {
  int result = 8;
  while (result % handle->blocksofm_blocking != 0) {
    result++;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_pack_input_bwd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  if ((handle->desc.u != 1) && (handle->bwd_ofh_rb != 1)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_use_ifm_parallelization( libxsmm_dnn_layer* handle ) {
  int result = 0;
  if (handle->ofw <= 7) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_avoid_rim_fmas_bwd( libxsmm_dnn_layer* handle ) {
  int result = libxsmm_dnn_convolution_setup_avoid_rim_fmas_fwd(handle);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_blocksofm_blocking( libxsmm_dnn_layer* handle ) {
  int result = 0;
  if (handle->desc.R == 1 && handle->desc.S == 1) {
    result = handle->blocksofm;
  } else {
    result = 1;
    if (handle->desc.R == 3 && handle->desc.S == 3 && handle->ofh == 7 && handle->ofw == 7) {
      result = 2;
    }
  }

  if ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) && ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8)) ) {
    result = handle->blocksofm;
  }

  if (handle->blocksofm % result != 0) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_init_bwd_gemm_flags( libxsmm_dnn_layer* handle ) {
  int result = 0;
  LIBXSMM_UNUSED( handle );
  if ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) && ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8)) ) {
    result = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_spread_input_bwd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  LIBXSMM_UNUSED(handle);
  if (((handle->desc.u != 1) || (handle->desc.v != 1)) && (handle->bwd_ofh_rb == 1)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_avoid_acc_load_bwd( libxsmm_dnn_layer* handle ) {
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

LIBXSMM_API_INLINE void libxsmm_dnn_convolution_setup_bwd_scratch( libxsmm_dnn_layer* handle ) {
  /* transpose of weights */
  handle->bwd_filter_trans_scratch_size = (size_t)handle->desc.C * handle->desc.K *
    handle->desc.R * handle->desc.S *
    libxsmm_dnn_typesize(handle->datatype_in);

  handle->bwd_packing_padding_scratch_size = 0;
  /* packing of input */
  if ( handle->pack_input_bwd != 0 ) {
    handle->bwd_packing_padding_scratch_size = (size_t)handle->desc.N * handle->desc.C *
      handle->ofhp * handle->ofwp *
      libxsmm_dnn_typesize(handle->datatype_in);
  }
  /* logical padding with copying in the fly */
  if ( handle->use_fallback_bwd_loops != 0 ) {
    handle->bwd_packing_padding_scratch_size = (size_t)handle->desc.threads * handle->ifmblock *
      (handle->desc.H + 2*handle->desc.pad_h) *
      (handle->desc.W + 2*handle->desc.pad_w) *
      libxsmm_dnn_typesize(handle->datatype_in);
  }
  /* input bufffer in high precision when we use BF16 */
  if ( handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16 ) {
    handle->bwd_lp_input_full_scratch_size = (size_t) LIBXSMM_MAX(handle->desc.threads * handle->bwd_gemm_pixels * handle->ifmblock * libxsmm_dnn_typesize(LIBXSMM_DNN_DATATYPE_F32), handle->desc.N * handle->desc.C * handle->ifwp * handle->ifhp * libxsmm_dnn_typesize(LIBXSMM_DNN_DATATYPE_F32));
    /* logical padding with copying in the fly */
    if ( handle->use_fallback_bwd_loops != 0 ) {
      handle->bwd_packing_padding_scratch_size = (size_t)handle->desc.threads * handle->ifmblock *
        (handle->desc.H + 2*handle->desc.pad_h) *
        (handle->desc.W + 2*handle->desc.pad_w) *
        libxsmm_dnn_typesize(LIBXSMM_DNN_DATATYPE_F32);
    }
  } else {
    handle->bwd_lp_input_full_scratch_size = 0;
  }
  /* align sizes to full cacheline */
  handle->bwd_filter_trans_scratch_size += ( handle->bwd_filter_trans_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (handle->bwd_filter_trans_scratch_size % LIBXSMM_CACHELINE);
  handle->bwd_packing_padding_scratch_size += ( handle->bwd_packing_padding_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (handle->bwd_packing_padding_scratch_size % LIBXSMM_CACHELINE);
  handle->bwd_lp_input_full_scratch_size += ( handle->bwd_lp_input_full_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (handle->bwd_lp_input_full_scratch_size % LIBXSMM_CACHELINE);

  /* set offsets */
  handle->bwd_filter_trans_scratch_offset = 0;
  handle->bwd_packing_padding_scratch_offset = handle->bwd_filter_trans_scratch_size;
  handle->bwd_lp_input_full_scratch_offset = handle->bwd_packing_padding_scratch_offset +
    handle->bwd_packing_padding_scratch_size;

  /* set overall scratch size for forward */
  handle->bwd_scratch_size = handle->bwd_filter_trans_scratch_size +
    handle->bwd_packing_padding_scratch_size +
    handle->bwd_lp_input_full_scratch_size;
}

/**********************************************************/
/* Helper functions for UPD convolutions' parameter setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_loop_order_upd( libxsmm_dnn_layer* handle ) {
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

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_pack_input_upd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  /* Pack input only for very small images, 1x1 convs, with large K to amortize the relevant overhead */
  if ((handle->ofh <= 7) && (handle->desc.R == 1) && (handle->desc.S == 1) && (handle->desc.u != 1) && (handle->desc.v != 1) && (handle->desc.K >= 2048)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_avoid_rim_fmas_upd( libxsmm_dnn_layer* handle ) {
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

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_upd_ofw_rb( libxsmm_dnn_layer* handle ) {
  int result = 1;
  result = handle->ofw;
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_upd_ofh_rb( libxsmm_dnn_layer* handle ) {
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

  if ((handle->desc.N != handle->desc.threads) && (handle->desc.R > 1 || handle->desc.S > 1 ) && (handle->desc.u > 1 || handle->desc.v > 1 )) {
    result = 1;
  }

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_block_upd_IFM( libxsmm_dnn_layer* handle ) {
  int result = 1;
  if (handle->ofh == 56 && handle->desc.R == 1 && handle->desc.S == 1 && handle->desc.u == 1 && handle->desc.v == 1) {
    result = 4;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_block_upd_OFM( libxsmm_dnn_layer* handle ) {
  int result = 1;
  LIBXSMM_UNUSED(handle);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_img_batchreduce_block( libxsmm_dnn_layer* handle ) {
  int result = 1;
  LIBXSMM_UNUSED(handle);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_use_batchreduce_upd( libxsmm_dnn_layer* handle ) {
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

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_weight_copies_upd( libxsmm_dnn_layer* handle ) {
  int result = handle->desc.threads;
  if (handle->ofw <= 14) {
    result = 9;
  }
  if (handle->ofw == 14 && handle->desc.N == 92 && handle->desc.threads == 92) {
    result = 23;
  }
  if (handle->ofw == 7 && handle->desc.N == 92 && handle->desc.threads == 92 && handle->desc.R == 3 && handle->desc.S == 3 && handle->desc.u == 1 && handle->desc.v == 1) {
    result = 23;
  }
  while (handle->desc.threads % result != 0) {
    result--;
  }
  /* FIXME: Hardcoded logic for N=27, N=26 */
  if (handle->desc.N == 27 && handle->desc.threads == 27 && handle->desc.R == 1 && handle->ofw == 14 && handle->desc.u == 1) {
    result = 7;
  }
  if (((handle->ofh == 14) || (handle->ofw == 7 && handle->desc.u == 2)) && handle->desc.N == 26 && handle->desc.threads == 26) {
    result = 13;
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

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_linearized_tasklist_upd( libxsmm_dnn_layer* handle ) {
  int result = 0;
  /* Use linearized task-list (i.e. no reduction) only if small images and large filters */
  if (handle->ofh <= 10 && handle->ofw <= 10) {
    result = 1;
  }
  if (handle->ofw == 7 && handle->desc.N == 92 && handle->desc.threads == 92 && handle->desc.R == 3 && handle->desc.S == 3 && handle->desc.u == 1 && handle->desc.v == 1) {
    result = 0;
  }
  if (handle->ofh == 14  && handle->ofw == 14 && handle->desc.N == 23 && handle->desc.threads == 23) {
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

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_init_upd_gemm_flags( libxsmm_dnn_layer* handle ) {
  int result = 0;
  LIBXSMM_UNUSED(handle);
  return result;
}

LIBXSMM_API_INLINE void libxsmm_dnn_convolution_setup_bf16_upd( libxsmm_dnn_layer* handle ) {
  int remainder_pixels, max_init_offset, max_compute_offset_input, input_compute_pad, accum_length_pixels, compute_pixels;
  const int multiple_target = 2;
  int IFHP = (handle->upd_padding_copy == 1) ? handle->ifhp + 2 * handle->desc.pad_h : handle->ifhp;
  int IFWP = (handle->upd_padding_copy == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
  int OFHP = (handle->upd_padding_copy == 1) ? handle->ofhp + 2 * handle->desc.pad_h : handle->ofhp;
  int OFWP = (handle->upd_padding_copy == 1) ? handle->ofwp + 2 * handle->desc.pad_w : handle->ofwp;

  handle->upd_linearized_pixels = 1;
  if (handle->desc.S != 1 && handle->desc.v != 1) {
    handle->upd_linearized_pixels = 0;
    handle->upd_trans_w_only = 0;
  }
  /* For large images facilitate the "large" transposes by blocking the pixel/reduction domains  */
  if (handle->ofw >= 56 && handle->ofh >=56 && handle->desc.R == 1 && handle->desc.S == 1 && handle->desc.u == 1 && handle->desc.v == 1) {
    handle->upd_linearized_pixels = 0;
    handle->upd_trans_w_only = 1;
  }

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
    max_init_offset = 2 * handle->desc.pad_h * IFWP + 2 * handle->desc.pad_w;
    max_compute_offset_input = max_init_offset + accum_length_pixels;
    input_compute_pad = (max_compute_offset_input > IFWP*IFHP) ? max_compute_offset_input - IFWP*IFHP : 0;
    handle->input_pixels = IFWP * IFHP + input_compute_pad;
    if (handle->upd_pack_input_upfront) {
      handle->input_pixels = accum_length_pixels;
    }
    handle->output_pixels = accum_length_pixels;
    handle->pixel_blocking = accum_length_pixels;
    handle->n_used_pixels = accum_length_pixels;
    handle->compute_pixels = compute_pixels;

    handle->use_intermediate_f32_wt_tensor = (handle->pixel_blocking == handle->n_used_pixels) ? 0 : 1;

    if (handle->ofw <= 14) {
      handle->use_hybrid_imgofm_parallelization = 1;
      handle->weight_copies = libxsmm_dnn_convolution_setup_weight_copies_upd(handle);
      if (handle->ofw == 14 && handle->desc.K >= 1024) {
        handle->use_hybrid_imgofm_parallelization = 0;
        handle->weight_copies = handle->desc.threads;
      }
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
    handle->ofwp_extended = OFWP + remainder_pixels;
    handle->ifwp_extended = IFWP + remainder_pixels;
    handle->output_pixels = OFHP * handle->ofwp_extended;
    /* coverity[identical_branches] */
    handle->batchreduce_h_pixels = (handle->upd_trans_w_only) ? 1 : 1; /* TODO: identical_branches */
    handle->use_intermediate_f32_wt_tensor = (handle->batchreduce_h_pixels == handle->ofh) ? 0 : 1;
  }

  if (handle->desc.N != handle->desc.threads) {
    handle->use_intermediate_f32_wt_tensor = 1;
    handle->use_hybrid_imgofm_parallelization = 0;
    handle->weight_copies = LIBXSMM_MIN(handle->desc.N, handle->desc.threads);
  }

}

LIBXSMM_API_INLINE void libxsmm_dnn_convolution_setup_bf16_upd_amx( libxsmm_dnn_layer* handle ) {
  /* JIT related variables...  */
  libxsmm_blasint LDA = handle->ofmblock;
  libxsmm_blasint LDB = handle->input_pixels;
  libxsmm_blasint LDC = handle->ofmblock;
  int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
  int l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  int l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
  size_t stride_a, stride_b;
  int unroll_hint;
  float beta;

  int remainder_pixels, max_init_offset, max_compute_offset_input, input_compute_pad, accum_length_pixels, compute_pixels;
  const int multiple_target = 32;
  int IFHP = (handle->upd_padding_copy == 1) ? handle->ifhp + 2 * handle->desc.pad_h : handle->ifhp;
  int IFWP = (handle->upd_padding_copy == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
  int OFWP = (handle->upd_padding_copy == 1) ? handle->ofwp + 2 * handle->desc.pad_w : handle->ofwp;

  handle->upd_linearized_pixels = 1;
  if (handle->desc.S != 1 && handle->desc.v != 1) {
    handle->upd_linearized_pixels = 0;
  }
  handle->fuse_upd_transposes = 1;
  handle->pack_to_cnhw = 0;
  handle->on_the_fly_input_packing = 0;
  handle->upd_pack_input_upfront = 0;
  handle->use_hybrid_imgofm_parallelization = 0;
  handle->upd_linearized_tasklist = 0;
  if (((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT)) && (handle->ofw == 7) && (handle->desc.R == 1) && (handle->desc.S == 1) ) {
    handle->pack_to_cnhw= 1;
  }

  if (handle->upd_linearized_pixels == 1) {
    if (handle->pack_to_cnhw == 0) {
      handle->fuse_upd_transposes = 1;
      /* Logistics to pad accumulation chainlength */
      compute_pixels = handle->ofw * handle->ofh + 2 * handle->desc.pad_w * (handle->ofh-1);
      remainder_pixels = (compute_pixels % multiple_target == 0) ? 0 : (compute_pixels/multiple_target+1)*multiple_target - compute_pixels;
      accum_length_pixels = compute_pixels + remainder_pixels;

      /* In this case compact input upfront */
      if (handle->desc.R == 1 && handle->desc.S == 1 && (handle->desc.u != 1 || handle->desc.v != 1)) {
        handle->upd_pack_input_upfront = 1;
      }

      /* Logistics for input transpose and additional pixel padding */
      max_init_offset = 2 * handle->desc.pad_h * IFWP + 2 * handle->desc.pad_w;
      max_compute_offset_input = max_init_offset + accum_length_pixels;
      input_compute_pad = (max_compute_offset_input > IFWP*IFHP) ? max_compute_offset_input - IFWP*IFHP : 0;
      handle->input_pixels = IFWP*IFHP+ input_compute_pad;
      if (handle->upd_pack_input_upfront) {
        handle->input_pixels = accum_length_pixels;
      }
      handle->output_pixels = accum_length_pixels;
      handle->pixel_blocking = accum_length_pixels;
      handle->n_used_pixels = accum_length_pixels;
      handle->compute_pixels = compute_pixels;

      handle->use_intermediate_f32_wt_tensor = (handle->pixel_blocking == handle->n_used_pixels) ? 0 : 1;
#if 0
      handle->scratch2_size = (size_t) (handle->desc.N * handle->output_pixels * handle->desc.K * sizeof(float)/2);
      if (handle->use_intermediate_f32_wt_tensor) {
        handle->scratch2_size += (size_t) handle->desc.R * handle->desc.S * handle->desc.C * handle->desc.K * handle->desc.threads * sizeof(float);
      }
      handle->scratch3_size = (size_t) (handle->desc.N * handle->input_pixels * handle->desc.C * sizeof(float)/2);
#endif

      if (handle->ofw <= 14) {
        handle->use_hybrid_imgofm_parallelization = 1;
        handle->fuse_upd_transposes = 0;
      } else {
        handle->weight_copies = handle->desc.threads;
      }

      if ((handle->ofmblock % 32 != 0) || (handle->ifmblock % 32 != 0)) {
        handle->fuse_upd_transposes = 0;
      }
    } else {
      /* Logistics to pad accumulation chainlength */
      handle->use_hybrid_imgofm_parallelization = 1;
      handle->weight_copies = 7;
      while (handle->desc.threads % handle->weight_copies != 0) {
        handle->weight_copies--;
      }
      compute_pixels = handle->ofw * handle->ofh * (handle->desc.N/handle->weight_copies);
      remainder_pixels = (compute_pixels % multiple_target == 0) ? 0 : (compute_pixels/multiple_target+1)*multiple_target - compute_pixels;
      handle->remainder_pixels = remainder_pixels;
      accum_length_pixels = compute_pixels + remainder_pixels;
      handle->output_pixels = accum_length_pixels * handle->weight_copies;
      handle->input_pixels = accum_length_pixels * handle->weight_copies;
      handle->pixel_blocking = accum_length_pixels;
      handle->n_used_pixels = accum_length_pixels;

      handle->use_intermediate_f32_wt_tensor = 0;
#if 0
      handle->scratch2_size = (size_t) (handle->weight_copies * handle->output_pixels * handle->desc.K * sizeof(float)/2);
      handle->scratch3_size = (size_t) (handle->weight_copies * handle->input_pixels * handle->desc.C * sizeof(float)/2);
#endif
    }
  }

  if (handle->upd_linearized_pixels == 0) {
    handle->weight_copies = handle->desc.threads;
    if (handle->desc.v !=1) {
      handle->on_the_fly_input_packing = 1;
    }
    remainder_pixels = (handle->ofw % multiple_target == 0) ? 0 : (handle->ofw/multiple_target+1)*multiple_target - handle->ofw;
    handle->remainder_pixels = remainder_pixels;
    handle->ofwp_extended = OFWP + remainder_pixels;
    handle->ifwp_extended = IFWP + remainder_pixels;
    handle->batchreduce_h_pixels = handle->ofh;
    handle->use_intermediate_f32_wt_tensor = (handle->batchreduce_h_pixels == handle->ofh) ? 0 : 1;
#if 0
    handle->scratch2_size = (size_t) (handle->desc.N * handle->ofhp*handle->ofwp_extended * handle->desc.K * sizeof(float)/2);
    if (handle->use_intermediate_f32_wt_tensor) {
      handle->scratch2_size += (size_t) handle->desc.R * handle->desc.S * handle->desc.C * handle->desc.K * handle->desc.threads * sizeof(float);
    }
    handle->scratch3_size = (size_t) (handle->desc.N * handle->ifhp * handle->ifwp_extended * handle->desc.C * sizeof(float)/2);
#endif
  }

  /* Now that all decisions have been made, JIT the proper kernel...  */
  beta = (handle->use_intermediate_f32_wt_tensor) ? (float)1.0 : (float)0.0;
  if (handle->upd_linearized_pixels == 0) {
    LDA = handle->ofmblock;
    LDB = IFHP*handle->ifwp_extended;
    LDC = handle->ofmblock;
    prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
    unroll_hint = handle->batchreduce_h_pixels;
    stride_a = handle->ofwp_extended * handle->ofmblock * libxsmm_dnn_typesize(handle->datatype_in);
    stride_b = handle->desc.u * handle->ifwp_extended * libxsmm_dnn_typesize(handle->datatype_in);
    handle->upd_config_kernel = libxsmm_bsmmdispatch(handle->ofmblock, handle->ifmblock, handle->ofw+handle->remainder_pixels, &LDA, &LDB, &LDC, NULL, &beta, &l_tc_flags, NULL);
    handle->upd_compute_kernel_brgemm_no_linearized_pixels = libxsmm_bsmmdispatch_reducebatch_strd_unroll(handle->ofmblock, handle->ifmblock, handle->ofw+handle->remainder_pixels,
      (libxsmm_blasint)stride_a, (libxsmm_blasint)stride_b, unroll_hint, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);
  } else {
    LDA = handle->ofmblock;
    LDB = handle->input_pixels;
    LDC = handle->ofmblock;
    prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
    if (handle->use_hybrid_imgofm_parallelization == 0) {
      handle->upd_config_kernel = libxsmm_bsmmdispatch(handle->ofmblock, handle->ifmblock, handle->pixel_blocking, &LDA, &LDB, &LDC, NULL, &beta, &l_tc_flags, NULL);
      handle->upd_compute_kernel_gemm_linearized_pixels_no_hybrid_par = libxsmm_bsmmdispatch(handle->ofmblock, handle->ifmblock, handle->pixel_blocking, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);
    } else {
      if (handle->pack_to_cnhw == 1) {
        handle->upd_config_kernel = libxsmm_bsmmdispatch(handle->ofmblock, handle->ifmblock, handle->pixel_blocking, &LDA, &LDB, &LDC, NULL, &beta, &l_tc_flags, NULL);
        handle->upd_compute_kernel_gemm_linearized_pixels_hybrid_par_cnhw = libxsmm_bsmmdispatch(handle->ofmblock, handle->ifmblock, handle->pixel_blocking, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);
      } else {
        /* TODO: Hoist here hybrid parallelization logic and then we should be able to also provide unroll hint in the BRGEMM call */
        stride_a = handle->blocksofm * handle->output_pixels * handle->ofmblock * libxsmm_dnn_typesize(handle->datatype_in);
        stride_b = handle->blocksifm * handle->ifmblock * handle->input_pixels * libxsmm_dnn_typesize(handle->datatype_in);
        handle->upd_config_kernel = libxsmm_bsmmdispatch(handle->ofmblock, handle->ifmblock, handle->pixel_blocking, &LDA, &LDB, &LDC, NULL, &beta, &l_tc_flags, NULL);
        handle->upd_compute_kernel_brgemm_linearized_pixels_hybrid_par_no_cnhw = libxsmm_bsmmdispatch_reducebatch_strd(handle->ofmblock, handle->ifmblock, handle->pixel_blocking,
          (libxsmm_blasint)stride_a, (libxsmm_blasint)stride_b, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);
      }
    }
  }

  if (handle->desc.N != handle->desc.threads) {
    handle->use_intermediate_f32_wt_tensor = 1;
    handle->use_hybrid_imgofm_parallelization = 0;
    handle->weight_copies = LIBXSMM_MIN(handle->desc.N, handle->desc.threads);
  }

}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_upd_padding_copy( libxsmm_dnn_layer* handle ) {
  int result = 0;
  if ( (handle->desc.pad_h != handle->desc.pad_h_in) && (handle->desc.pad_w != handle->desc.pad_w_in) ) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE void libxsmm_dnn_convolution_setup_upd_scratch( libxsmm_dnn_layer* handle ) {
  handle->upd_packing_padding_scratch_size = 0;
  /* packing of input */
  if ( handle->upd_pack_input != 0 ) {
    handle->upd_packing_padding_scratch_size = (size_t)handle->desc.N * handle->desc.C *
      handle->desc.H/handle->desc.u *
      handle->desc.W/handle->desc.v *
      libxsmm_dnn_typesize(handle->datatype_in);
  }
  /* logical padding with copying in the fly */
  if ( handle->upd_padding_copy != 0 ) {
    handle->upd_packing_padding_scratch_size = (size_t)handle->desc.N * handle->desc.C *
      (handle->desc.H + 2*handle->desc.pad_h) *
      (handle->desc.W + 2*handle->desc.pad_w) *
      libxsmm_dnn_typesize(handle->datatype_in);
  }
  /* output/input buffer to transpose when we use bf16 */
  if ( handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16 ) {
    if  (handle->target_archid >= LIBXSMM_X86_AVX512_SPR) {
      int OFHP = (handle->upd_padding_copy == 1) ? handle->ofhp + 2 * handle->desc.pad_h : handle->ofhp;
      int IFHP = (handle->upd_padding_copy == 1) ? handle->ifhp + 2 * handle->desc.pad_h : handle->ifhp;

      if (handle->upd_linearized_pixels == 1) {
        handle->upd_lp_output_full_scratch_size = (size_t) (handle->desc.N * handle->output_pixels * handle->desc.K * sizeof(handle->datatype_in));
        handle->upd_lp_input_full_scratch_size = (size_t) (handle->desc.N * handle->input_pixels * handle->desc.C * sizeof(handle->datatype_in));
      }

      if (handle->upd_linearized_pixels == 0) {
        handle->upd_lp_output_full_scratch_size = (size_t) (handle->desc.N * OFHP * handle->ofwp_extended * handle->desc.K * sizeof(handle->datatype_in));
        handle->upd_lp_input_full_scratch_size = (size_t) (handle->desc.N * IFHP * handle->ifwp_extended * handle->desc.C * sizeof(handle->datatype_in));
      }
    } else {
      const int multiple_target = 2;
      int IFHP = (handle->upd_padding_copy == 1) ? handle->ifhp + 2 * handle->desc.pad_h : handle->ifhp;
      int IFWP = (handle->upd_padding_copy == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
      int OFHP = (handle->upd_padding_copy == 1) ? handle->ofhp + 2 * handle->desc.pad_h : handle->ofhp;
      int OFWP = (handle->upd_padding_copy == 1) ? handle->ofwp + 2 * handle->desc.pad_w : handle->ofwp;

      if (handle->upd_linearized_pixels == 1) {
        int compute_pixels = handle->ofw * handle->ofh + 2 * handle->desc.pad_w * (handle->ofh-1);
        int remainder_pixels = (compute_pixels % multiple_target == 0) ? 0 : (compute_pixels/multiple_target+1)*multiple_target - compute_pixels;
        int accum_length_pixels = compute_pixels + remainder_pixels;

        int max_init_offset = 2 * handle->desc.pad_h * IFWP + 2 * handle->desc.pad_w;
        int max_compute_offset_input = max_init_offset + accum_length_pixels;
        int input_compute_pad = (max_compute_offset_input > IFWP*IFHP) ? max_compute_offset_input - IFWP*IFHP : 0;
        int input_pixels = IFWP * IFHP + input_compute_pad;

        if (handle->upd_pack_input_upfront == 1) {
          input_pixels = accum_length_pixels;
        }

        handle->upd_lp_output_full_scratch_size = (size_t) (handle->desc.N * accum_length_pixels * handle->desc.K * sizeof(handle->datatype_in));
        handle->upd_lp_input_full_scratch_size = (size_t) (handle->desc.N * input_pixels * handle->desc.C * sizeof(handle->datatype_in));
      }

      if (handle->upd_linearized_pixels == 0) {
        int remainder_pixels = (handle->ofw % multiple_target == 0) ? 0 : (handle->ofw/multiple_target+1)*multiple_target - handle->ofw;
        int ofwp_extended = OFWP + remainder_pixels;
        int ifwp_extended = IFWP + remainder_pixels;

        handle->upd_lp_output_full_scratch_size = (size_t) (handle->desc.N * OFHP * ofwp_extended * handle->desc.K * sizeof(handle->datatype_in));
        handle->upd_lp_input_full_scratch_size = (size_t) (handle->desc.N * IFHP * ifwp_extended * handle->desc.C * sizeof(handle->datatype_in));
      }
    }
    handle->upd_lp_filter_full_scratch_size = (size_t)handle->desc.R * handle->desc.S * handle->desc.C * handle->desc.K * handle->desc.threads *
      libxsmm_dnn_typesize(LIBXSMM_DNN_DATATYPE_F32);
  } else {
    handle->upd_lp_output_full_scratch_size = 0;
    handle->upd_lp_input_full_scratch_size = 0;
    handle->upd_lp_filter_full_scratch_size = 0;
  }
  /* filter scratch */
  handle->upd_filter_scratch_size = (size_t) handle->desc.R * handle->desc.S * handle->desc.C * handle->desc.K * LIBXSMM_MAX(handle->desc.threads, handle->desc.N) * sizeof(float);

  /* align sizes to full cacheline */
  handle->upd_packing_padding_scratch_size += ( handle->upd_packing_padding_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (handle->upd_packing_padding_scratch_size % LIBXSMM_CACHELINE);
  handle->upd_lp_output_full_scratch_size += ( handle->upd_lp_output_full_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (handle->upd_lp_output_full_scratch_size % LIBXSMM_CACHELINE);
  handle->upd_lp_input_full_scratch_size += ( handle->upd_lp_input_full_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (handle->upd_lp_input_full_scratch_size % LIBXSMM_CACHELINE);
  handle->upd_filter_scratch_size += ( handle->upd_filter_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (handle->upd_filter_scratch_size % LIBXSMM_CACHELINE);
  handle->upd_lp_filter_full_scratch_size += ( handle->upd_lp_filter_full_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (handle->upd_lp_filter_full_scratch_size % LIBXSMM_CACHELINE);

  /* calculate offsets */
  handle->upd_packing_padding_scratch_offset = 0;
  handle->upd_lp_output_full_scratch_offset = handle->upd_packing_padding_scratch_size;
  handle->upd_lp_input_full_scratch_offset = handle->upd_lp_output_full_scratch_offset + handle->upd_lp_output_full_scratch_size;
  handle->upd_filter_scratch_offset = handle->upd_lp_input_full_scratch_offset + handle->upd_lp_input_full_scratch_size;
  handle->upd_lp_filter_full_scratch_offset = handle->upd_filter_scratch_offset + handle->upd_filter_scratch_size;

  /* set overall scratch size for update */
  handle->upd_scratch_size = handle->upd_packing_padding_scratch_size +
    handle->upd_lp_output_full_scratch_size +
    handle->upd_lp_input_full_scratch_size +
    handle->upd_filter_scratch_size +
    handle->upd_lp_filter_full_scratch_size;
}

LIBXSMM_API_INLINE libxsmm_dnn_err_t libxsmm_dnn_convolution_setup( libxsmm_dnn_layer* handle ) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  libxsmm_blasint _ldi = 64, _ldo = 64;
  libxsmm_blasint ldx;
  libxsmm_blasint ldA;
  libxsmm_blasint ldC;
  int  beta_int;
  float beta;
  int l_flags;
  int l_tc_flags;

  /* init libxsmm */
  LIBXSMM_INIT

    /* Generic parameter setup  */
  handle->target_archid = libxsmm_target_archid;
  if ( ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT)) && (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && ((handle->desc.C % 16 != 0) || (handle->desc.K % 16 != 0)) ) {
    handle->target_archid = LIBXSMM_X86_AVX512_CPX;
  }
  handle->ifmblock = libxsmm_dnn_convolution_setup_ifmblock(handle);
  handle->ofmblock = libxsmm_dnn_convolution_setup_ofmblock(handle);
  handle->fm_lp_block = libxsmm_dnn_convolution_setup_fm_lp_block(handle);
  handle->blocksifm = libxsmm_dnn_convolution_setup_blocksifm(handle);
  handle->blocksofm = libxsmm_dnn_convolution_setup_blocksofm(handle);

  /* If in SPR, generate tilerelease kernel */
  if (handle->target_archid >= LIBXSMM_X86_AVX512_SPR) {
    int l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
    handle->tilerelease_kernel = libxsmm_bsmmdispatch(handle->ifmblock, handle->ifmblock, handle->ifmblock, NULL, NULL, NULL, NULL, NULL, &l_tr_flags, NULL);
  }

  /* FWD parameter setup  */
  handle->fwd_ofw_rb = libxsmm_dnn_convolution_setup_fwd_ofw_rb(handle);
  handle->pack_input = libxsmm_dnn_convolution_setup_pack_input_fwd(handle);
  handle->fwd_ofh_rb = libxsmm_dnn_convolution_setup_fwd_ofh_rb(handle);
  handle->fwd_gemm_pixels = libxsmm_dnn_convolution_setup_fwd_pixels_gemm(handle);
  handle->block_fwd_oj = libxsmm_dnn_convolution_setup_fwd_block_H(handle);
  handle->loop_order = libxsmm_dnn_convolution_setup_loop_order_fwd(handle);
  handle->blocksifm_blocking = libxsmm_dnn_convolution_setup_blocksifm_blocking(handle);
  handle->block_fwd_ofm = libxsmm_dnn_convolution_setup_block_fwd_OFM(handle);
  handle->block_fwd_ifm = libxsmm_dnn_convolution_setup_block_fwd_IFM(handle);
  handle->avoid_fmas_in_rim = libxsmm_dnn_convolution_setup_avoid_rim_fmas_fwd(handle);
  handle->use_ofm_parallelization = libxsmm_dnn_convolution_setup_use_ofm_parallelization(handle);
  handle->shuffle_filter_accesses = libxsmm_dnn_convolution_setup_shuffle_filter_accesses(handle);
  handle->avoid_acc_load = libxsmm_dnn_convolution_setup_avoid_acc_load(handle);
  handle->fwd_flags = libxsmm_dnn_convolution_setup_init_fwd_gemm_flags(handle);
  handle->use_fallback_fwd_loops = libxsmm_dnn_convolution_setup_fallback_loops_fwd(handle);
  handle->fwd_padding_copy = libxsmm_dnn_convolution_setup_fwd_padding_copy(handle);

#if 0
  if ( handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 ) {
    int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
    int brgemm_pf_oob = 0;
    const char *const env_brgemm_pf_oob = getenv("BRGEMM_PF_OOB");
    handle->block_fwd_ofm = 1;
    handle->block_fwd_oj = handle->fwd_ofh_rb;
    ldx = (handle->pack_input == 1) ? (libxsmm_blasint)handle->ifmblock : (libxsmm_blasint)handle->desc.v*handle->ifmblock;
    ldA = handle->ofmblock;
    ldC = handle->ofmblock;
    beta = (handle->avoid_acc_load) ? (float)0.0 : (float)1.0;
    l_flags = ( LIBXSMM_GEMM_FLAGS('N', 'N') ) | handle->fwd_flags;
    if ( 0 == env_brgemm_pf_oob ) {
    } else {
      brgemm_pf_oob = atoi(env_brgemm_pf_oob);
    }
    if (brgemm_pf_oob > 0) {
      prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB);
    }
    handle->fwd_compute_kernel_offs_f32 = NULL;
    handle->fwd_compute_kernel_strd_f32 = NULL;
    handle->fwd_compute_kernel_addr_a_f32 = NULL;
    handle->fwd_compute_kernel_addr_b_f32 = NULL;
    if (handle->desc.R == 1 && handle->desc.S == 1) {
      const int IFW = (handle->pack_input == 1) ? handle->ofwp : handle->ifwp;
      const int IFH = (handle->pack_input == 1) ? handle->ofhp : handle->ifhp;
      int stride_a = handle->desc.R * handle->desc.S * handle->ifmblock * handle->ofmblock * libxsmm_dnn_typesize(handle->datatype_in);
      int stride_b = IFW * IFH * handle->ifmblock * libxsmm_dnn_typesize(handle->datatype_in);
      handle->fwd_compute_kernel_strd_f32 = libxsmm_smmdispatch_reducebatch_strd_unroll(handle->ofmblock, handle->fwd_gemm_pixels, handle->ifmblock, stride_a, stride_b, handle->blocksifm_blocking, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
    } else {
      const int IFW = (handle->fwd_padding_copy == 1) ? handle->ifwp + 2*handle->desc.pad_w : ( (handle->pack_input == 1) ? handle->ofwp : handle->ifwp );
      const int IFH = (handle->fwd_padding_copy == 1) ? handle->ifhp + 2*handle->desc.pad_h : ( (handle->pack_input == 1) ? handle->ofhp : handle->ifhp );
      int n_blocks = handle->desc.R * handle->desc.S * handle->blocksifm_blocking;
      int i = 0, ifm, ki, kj;
      handle->A_offsets = (unsigned long long*) malloc(n_blocks * sizeof(unsigned long long));
      handle->B_offsets = (unsigned long long*) malloc(n_blocks * sizeof(unsigned long long));
      for (ifm = 0; ifm < handle->blocksifm_blocking; ifm++) {
        for (kj = 0; kj < handle->desc.R; kj++) {
          for (ki = 0; ki < handle->desc.S; ki++) {
            handle->A_offsets[i] = (ifm * handle->desc.R * handle->desc.S * handle->ifmblock * handle->ofmblock +
                kj * handle->desc.S * handle->ifmblock * handle->ofmblock +
                ki * handle->ifmblock * handle->ofmblock) * libxsmm_dnn_typesize(handle->datatype_in);
            handle->B_offsets[i] = (ifm * IFH * IFW * handle->ifmblock +
                kj * IFW * handle->ifmblock +
                ki * handle->ifmblock) * libxsmm_dnn_typesize(handle->datatype_in);
            i++;
          }
        }
      }
      handle->fwd_compute_kernel_offs_f32 = libxsmm_smmdispatch_reducebatch_offs(handle->ofmblock, handle->fwd_gemm_pixels, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
    }
    handle->fwd_compute_kernel_addr_a_f32 = libxsmm_smmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, &prefetch_mode);
    handle->fwd_compute_kernel_addr_b_f32 = libxsmm_smmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*(handle->fwd_ofw_rb-1), handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, &prefetch_mode);
  }
#endif

  if ( ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT)) && (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) ) {
    handle->block_fwd_ofm = 1;
    handle->block_fwd_oj = handle->fwd_ofh_rb;
    ldx = (handle->pack_input == 1) ? (libxsmm_blasint)handle->ifmblock : (libxsmm_blasint)handle->desc.v*handle->ifmblock;
    ldA = handle->ofmblock;
    ldC = handle->ofmblock;
    beta = (handle->avoid_acc_load) ? (float)0.0 : (float)1.0;
    l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
    l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
    handle->fwd_compute_kernel_addr = NULL;
    handle->fwd_compute_kernel_offs_a = NULL;
    handle->fwd_compute_kernel_offs_b = NULL;
    handle->fwd_compute_kernel_strd = NULL;
    if (handle->desc.R == 1 && handle->desc.S == 1) {
      const int IFW = (handle->pack_input == 1) ? handle->ofwp : handle->ifwp;
      const int IFH = (handle->pack_input == 1) ? handle->ofhp : handle->ifhp;
      size_t stride_a = handle->desc.R * handle->desc.S * handle->ifmblock * handle->ofmblock * libxsmm_dnn_typesize(handle->datatype_in);
      size_t stride_b = IFW * IFH * handle->ifmblock * libxsmm_dnn_typesize(handle->datatype_in);
      handle->fwd_compute_kernel_strd = libxsmm_bmmdispatch_reducebatch_strd_unroll(handle->ofmblock, handle->fwd_gemm_pixels, handle->ifmblock,
        (libxsmm_blasint)stride_a, (libxsmm_blasint)stride_b, handle->blocksifm_blocking, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
    } else {
      const int IFW = (handle->fwd_padding_copy == 1) ? handle->ifwp + 2*handle->desc.pad_w : ( (handle->pack_input == 1) ? handle->ofwp : handle->ifwp );
      const int IFH = (handle->fwd_padding_copy == 1) ? handle->ifhp + 2*handle->desc.pad_h : ( (handle->pack_input == 1) ? handle->ofhp : handle->ifhp );
      int n_blocks = handle->desc.R * handle->desc.S * handle->blocksifm_blocking;
      int i = 0, ifm, ki, kj;
      handle->A_offsets = (unsigned long long*) malloc(n_blocks * sizeof(unsigned long long));
      handle->B_offsets = (unsigned long long*) malloc(n_blocks * sizeof(unsigned long long));
      for (ifm = 0; ifm < handle->blocksifm_blocking; ifm++) {
        for (kj = 0; kj < handle->desc.R; kj++) {
          for (ki = 0; ki < handle->desc.S; ki++) {
            handle->A_offsets[i] = (ifm * handle->desc.R * handle->desc.S * handle->ifmblock * handle->ofmblock +
                kj * handle->desc.S * handle->ifmblock * handle->ofmblock +
                ki * handle->ifmblock * handle->ofmblock) * libxsmm_dnn_typesize(handle->datatype_in);
            handle->B_offsets[i] = (ifm * IFH * IFW * handle->ifmblock +
                kj * IFW * handle->ifmblock +
                ki * handle->ifmblock) * libxsmm_dnn_typesize(handle->datatype_in);
            i++;
          }
        }
      }
      handle->fwd_compute_kernel_offs_a = libxsmm_bmmdispatch_reducebatch_offs(handle->ofmblock, handle->fwd_gemm_pixels, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
      handle->fwd_compute_kernel_offs_b = libxsmm_bsmmdispatch_reducebatch_offs(handle->ofmblock, handle->fwd_gemm_pixels, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
    }
    handle->fwd_config_kernel = libxsmm_bsmmdispatch(handle->ofmblock, handle->fwd_gemm_pixels, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_tc_flags, NULL);
  }

  handle->code_fwd[0].ptr = 0;
  handle->code_fwd[1].ptr = 0;
  handle->code_fwd[2].ptr = 0;

  /* JIT cvt eltwise functions for fwd convolutions */
  if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) {
    _ldi = handle->ofmblock * handle->ofwp;
    _ldo = handle->ofmblock * handle->ofwp;
    handle->fwd_cvtfp32bf16_kernel = libxsmm_dispatch_meltw_unary(handle->ofmblock * handle->fwd_ofw_rb, handle->fwd_ofh_rb, &_ldi, &_ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  }

  /* Create strided BRGEMMs for i8i32 convolutions  */
  if ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32)) {
    ldx = (handle->pack_input == 1) ? (libxsmm_blasint)handle->ifmblock : (libxsmm_blasint)handle->desc.v*handle->ifmblock;
    ldA = handle->ofmblock;
    ldC = handle->ofmblock;
    beta_int = (handle->avoid_acc_load) ? 0 : 1;
    l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | handle->fwd_flags;
    if (handle->desc.R == 1 && handle->desc.S == 1) {
      const int IFW = (handle->pack_input == 1) ? handle->ofwp : handle->ifwp;
      const int IFH = (handle->pack_input == 1) ? handle->ofhp : handle->ifhp;
      libxsmm_blasint stride_A = handle->ifmblock * handle->ofmblock * sizeof(char);
      libxsmm_blasint stride_B = handle->ifmblock * IFW * IFH * sizeof(char) ;
      handle->gemm_fwd.xgemm.subimrs = libxsmm_subimmdispatch_reducebatch_strd(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, stride_A, stride_B, &ldA, &ldx, &ldC, NULL, &beta_int, &l_flags, NULL);
    } else {
      const int IFW = (handle->pack_input == 1) ? handle->ofwp : handle->ifwp;
      const int IFH = (handle->pack_input == 1) ? handle->ofhp : handle->ifhp;
      if (handle->avoid_fmas_in_rim == 0) {
        int n_blocks = handle->desc.R * handle->desc.S * handle->blocksifm_blocking;
        int i = 0, ifm, ki, kj;
        handle->A_offsets = (unsigned long long*) malloc(n_blocks * sizeof(unsigned long long));
        handle->B_offsets = (unsigned long long*) malloc(n_blocks * sizeof(unsigned long long));
        for (ifm = 0; ifm < handle->blocksifm_blocking; ifm++) {
          for (kj = 0; kj < handle->desc.R; kj++) {
            for (ki = 0; ki < handle->desc.S; ki++) {
              handle->A_offsets[i] = (ifm * handle->desc.R * handle->desc.S * handle->ifmblock * handle->ofmblock +
                  kj * handle->desc.S * handle->ifmblock * handle->ofmblock +
                  ki * handle->ifmblock * handle->ofmblock) * sizeof(char);
              handle->B_offsets[i] = (ifm * IFH * IFW * handle->ifmblock +
                  kj * IFW * handle->ifmblock +
                  ki * handle->ifmblock) * sizeof(char);
              i++;
            }
          }
        }
        handle->gemm_fwd.xgemm.subimro = libxsmm_subimmdispatch_reducebatch_offs(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta_int, &l_flags, NULL);
      } else {
        libxsmm_blasint stride_A = handle->ifmblock * handle->desc.R * handle->desc.S * handle->ofmblock * sizeof(char);
        libxsmm_blasint stride_B = handle->ifmblock * IFW * IFH * sizeof(char) ;
        handle->gemm_fwd.xgemm.subimrs = libxsmm_subimmdispatch_reducebatch_strd(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, stride_A, stride_B, &ldA, &ldx, &ldC, NULL, &beta_int, &l_flags, NULL);
        handle->gemm_fwd2.xgemm.subimrs = libxsmm_subimmdispatch_reducebatch_strd(handle->ofmblock, handle->fwd_ofh_rb*(handle->fwd_ofw_rb-1), handle->ifmblock, stride_A, stride_B, &ldA, &ldx, &ldC, NULL, &beta_int, &l_flags, NULL);
      }
    }
  } else if ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I8)) {
    ldx = (libxsmm_blasint)handle->desc.v*handle->ifmblock;
    ldA = handle->ofmblock;
    ldC = handle->ofmblock;
    beta_int = 0;
    l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | handle->fwd_flags;
    if (handle->desc.R == 1 && handle->desc.S == 1) {
      const int IFW = handle->ifwp;
      const int IFH = handle->ifhp;
      libxsmm_blasint stride_A = handle->ifmblock * handle->ofmblock * sizeof(char);
      libxsmm_blasint stride_B = handle->ifmblock * IFW * IFH * sizeof(char) ;
      handle->gemm_fwd.xgemm.sububmrs = libxsmm_sububmmdispatch_reducebatch_strd(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, stride_A, stride_B, &ldA, &ldx, &ldC, NULL, &beta_int, &l_flags, NULL);
    } else {
      const int IFW = handle->ifwp;
      const int IFH = handle->ifhp;
      int n_blocks = handle->desc.R * handle->desc.S * handle->blocksifm_blocking;
      int i = 0, ifm, ki, kj;
      handle->A_offsets = (unsigned long long*) malloc(n_blocks * sizeof(unsigned long long));
      handle->B_offsets = (unsigned long long*) malloc(n_blocks * sizeof(unsigned long long));
      for (ifm = 0; ifm < handle->blocksifm_blocking; ifm++) {
        for (kj = 0; kj < handle->desc.R; kj++) {
          for (ki = 0; ki < handle->desc.S; ki++) {
            handle->A_offsets[i] = (ifm * handle->desc.R * handle->desc.S * handle->ifmblock * handle->ofmblock +
                kj * handle->desc.S * handle->ifmblock * handle->ofmblock +
                ki * handle->ifmblock * handle->ofmblock) * sizeof(char);
            handle->B_offsets[i] = (ifm * IFH * IFW * handle->ifmblock +
                kj * IFW * handle->ifmblock +
                ki * handle->ifmblock) * sizeof(char);
            i++;
          }
        }
      }
      handle->gemm_fwd.xgemm.sububmro = libxsmm_sububmmdispatch_reducebatch_offs(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta_int, &l_flags, NULL);
    }
  }

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
  handle->bwd_ofw_rb = libxsmm_dnn_convolution_setup_bwd_ofw_rb(handle);
  handle->bwd_ofh_rb = libxsmm_dnn_convolution_setup_bwd_ofh_rb(handle);
  handle->bwd_gemm_pixels = libxsmm_dnn_convolution_setup_bwd_pixels_gemm(handle);
  handle->pack_input_bwd = libxsmm_dnn_convolution_setup_pack_input_bwd(handle);
  handle->spread_input_bwd = libxsmm_dnn_convolution_setup_spread_input_bwd(handle);
  handle->blocksofm_blocking = libxsmm_dnn_convolution_setup_blocksofm_blocking(handle);
  handle->avoid_acc_load_bwd = libxsmm_dnn_convolution_setup_avoid_acc_load_bwd(handle);
  handle->use_ifm_parallelization = libxsmm_dnn_convolution_setup_use_ifm_parallelization(handle);
  handle->block_bwd_ofm = libxsmm_dnn_convolution_setup_block_bwd_OFM(handle);
  handle->block_bwd_ifm = libxsmm_dnn_convolution_setup_block_bwd_IFM(handle);
  handle->block_bwd_oj = libxsmm_dnn_convolution_setup_bwd_block_H(handle);
  handle->use_fallback_bwd_loops = libxsmm_dnn_convolution_setup_fallback_loops_bwd(handle);
  handle->bwd_flags = libxsmm_dnn_convolution_setup_init_bwd_gemm_flags(handle);

  if ( ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT)) && (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) ) {
    handle->block_bwd_ifm = 1;
    handle->block_bwd_oj = handle->bwd_ofh_rb ;
    ldx = ((libxsmm_blasint)handle->ofmblock);
    ldA = handle->ifmblock;
    ldC = (handle->spread_input_bwd == 1) ? handle->ifmblock * handle->desc.v : handle->ifmblock;
    beta = (handle->avoid_acc_load_bwd) ? (float)0.0 : (float)1.0;
    l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
    l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
    handle->bwd_compute_kernel_addr = NULL;
    handle->bwd_compute_kernel_offs = NULL;
    handle->bwd_compute_kernel_strd = NULL;
    if (handle->desc.R == 1 && handle->desc.S == 1) {
      size_t stride_a = handle->desc.R * handle->desc.S * handle->ifmblock * handle->ofmblock * libxsmm_dnn_typesize(handle->datatype_in);
      size_t stride_b = handle->ofwp * handle->ofhp * handle->ofmblock * libxsmm_dnn_typesize(handle->datatype_in);
      handle->bwd_compute_kernel_strd = libxsmm_bsmmdispatch_reducebatch_strd_unroll(handle->ifmblock, handle->bwd_gemm_pixels, handle->ofmblock,
        (libxsmm_blasint)stride_a, (libxsmm_blasint)stride_b, handle->blocksofm_blocking, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
    } else {
      int n_blocks = handle->desc.R * handle->desc.S * handle->blocksofm_blocking;
      int i = 0, ofm, ki, kj;
      handle->A_offsets_bwd = (unsigned long long*) malloc(n_blocks * sizeof(unsigned long long));
      handle->B_offsets_bwd = (unsigned long long*) malloc(n_blocks * sizeof(unsigned long long));
      for (ofm = 0; ofm < handle->blocksofm_blocking; ofm++) {
        for (kj = 0; kj < handle->desc.R; kj++) {
          for (ki = 0; ki < handle->desc.S; ki++) {
            handle->A_offsets_bwd[i] = (ofm * handle->desc.R * handle->desc.S * handle->ifmblock * handle->ofmblock +
                kj * handle->desc.S * handle->ifmblock * handle->ofmblock +
                ki * handle->ifmblock * handle->ofmblock) * libxsmm_dnn_typesize(handle->datatype_in);
            handle->B_offsets_bwd[i] = (ofm * handle->ofhp * handle->ofwp * handle->ofmblock +
                kj * handle->ofwp * handle->ofmblock +
                ki * handle->ofmblock) * libxsmm_dnn_typesize(handle->datatype_in);
            i++;
          }
        }
      }
      handle->bwd_compute_kernel_offs = libxsmm_bsmmdispatch_reducebatch_offs(handle->ifmblock, handle->bwd_gemm_pixels, handle->ofmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
    }
    handle->bwd_config_kernel = libxsmm_bsmmdispatch(handle->ifmblock, handle->bwd_gemm_pixels, handle->ofmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_tc_flags, NULL);
  }

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

  handle->code_bwd[0].ptr = 0;
  handle->code_bwd[1].ptr = 0;
  handle->code_bwd[2].ptr = 0;

  /* Transpose kernel used for filter transpose in bwd pass  */
  handle->tr_kernel = libxsmm_dispatch_meltw_unary(64, 16, &(_ldi), &(_ldo), LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);

  /* UPD parameter setup */
  handle->upd_linearized_tasklist = libxsmm_dnn_convolution_setup_linearized_tasklist_upd(handle);
  handle->upd_avoid_rim_fmas = libxsmm_dnn_convolution_setup_avoid_rim_fmas_upd(handle);
  handle->upd_pack_input = libxsmm_dnn_convolution_setup_pack_input_upd(handle);
  handle->upd_use_batchreduce = libxsmm_dnn_convolution_setup_use_batchreduce_upd(handle);
  handle->upd_ofw_rb = libxsmm_dnn_convolution_setup_upd_ofw_rb(handle);
  handle->upd_ofh_rb = libxsmm_dnn_convolution_setup_upd_ofh_rb(handle);
  handle->upd_loop_order = libxsmm_dnn_convolution_setup_loop_order_upd(handle);
  handle->weight_copies = libxsmm_dnn_convolution_setup_weight_copies_upd(handle);
  handle->block_upd_ofm = libxsmm_dnn_convolution_setup_block_upd_OFM(handle);
  handle->block_upd_ifm = libxsmm_dnn_convolution_setup_block_upd_IFM(handle);
  handle->upd_loop_order = libxsmm_dnn_convolution_setup_loop_order_upd(handle);
  handle->upd_padding_copy = libxsmm_dnn_convolution_setup_upd_padding_copy(handle);

  if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) {
    if  ((handle->target_archid == LIBXSMM_X86_AVX512_SPR) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT)) {
      libxsmm_dnn_convolution_setup_bf16_upd_amx(handle);
    } else {
      libxsmm_dnn_convolution_setup_bf16_upd(handle);
    }
  }

#if 0
  /* Spit out UPD parameters that are selected...  */
  printf("UPD params...\n");
  if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) {
    printf("BF16 path...\n");
    printf("UPD use_hybrid_imgofm_parallelization = %d\n", handle->use_hybrid_imgofm_parallelization);
    printf("UPD linearized_pixels = %d\n", handle->upd_linearized_pixels);
    printf("UPD upd_trans_w_only = %d\n", handle->upd_trans_w_only);
    printf("UPD on_the_fly_input_packing = %d\n", handle->on_the_fly_input_packing);
    printf("UPD use_intermediate_f32_wt_tensor = %d\n", handle->use_intermediate_f32_wt_tensor);
    printf("UPD pack to CNHW format = %d\n", handle->pack_to_cnhw);
    printf("UPD batchreduce H pixels = %d\n", handle->batchreduce_h_pixels);
  }
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

  handle->code_upd[0].ptr = 0;
  handle->code_upd[1].ptr = 0;

  /* prepare barrier */
  handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);

  /* setup up scratch */
  libxsmm_dnn_convolution_setup_fwd_scratch( handle );
  libxsmm_dnn_convolution_setup_bwd_scratch( handle );
  libxsmm_dnn_convolution_setup_upd_scratch( handle );
  handle->scratch = 0;
  handle->scratch_size = LIBXSMM_MAX( handle->fwd_scratch_size, LIBXSMM_MAX( handle->bwd_scratch_size, handle->upd_scratch_size ) );

  return status;
}

#undef MIXED
#undef KHWC
#undef HWKC
#undef CHWK
#undef HWCK


LIBXSMM_API libxsmm_dnn_layer* libxsmm_dnn_create_conv_layer(
    libxsmm_dnn_conv_desc     conv_desc,
    libxsmm_dnn_err_t*        status)
{
  libxsmm_dnn_layer* handle = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  /* currently we don't support NCHW */
  if ( (conv_desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NCHW) > 0 ) {
    *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_NCHW;
    return 0;
  }
  /* currently we don't support KCRS */
  if ( (conv_desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_KCRS) > 0 ) {
    *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_KCRS;
    return 0;
  }
  /* we only support physical paddind in these days */
  /* @TODO: add logical padding support for other datatypes than FP32 */
  if ( ( ( conv_desc.pad_h != conv_desc.pad_h_in )  ||
        ( conv_desc.pad_w != conv_desc.pad_w_in )  ||
        ( conv_desc.pad_h != conv_desc.pad_h_out ) ||
        ( conv_desc.pad_w != conv_desc.pad_w_out )    ) &&
      ( conv_desc.datatype_in != LIBXSMM_DNN_DATATYPE_F32 ) && (conv_desc.datatype_in != LIBXSMM_DNN_DATATYPE_BF16) ) {
    *status = LIBXSMM_DNN_ERR_INVALID_PADDING;
    return 0;
  }

  /* zero entire content; not only safer but also sets data and code pointers to NULL */
  handle = (libxsmm_dnn_layer*)calloc(1, sizeof(libxsmm_dnn_layer));

  if (0 != handle) {
    /* initialize known handle components */
    handle->desc = conv_desc;
    handle->datatype_in = conv_desc.datatype_in;
    handle->datatype_out = conv_desc.datatype_out;
    /* select the intermediate format, only applicable for integer types */
    if ( (conv_desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (conv_desc.datatype_out != LIBXSMM_DNN_DATATYPE_F32) ) {
      /* error */
    } else if ( (conv_desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (conv_desc.datatype_out != LIBXSMM_DNN_DATATYPE_BF16) ) {
      /* error */
    } else if ( (conv_desc.datatype_in == LIBXSMM_DNN_DATATYPE_I16) && (conv_desc.datatype_out != LIBXSMM_DNN_DATATYPE_F32) ) {
      /* error */
    } else if ( (conv_desc.datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (conv_desc.datatype_out != LIBXSMM_DNN_DATATYPE_I32) ) {
      /* error */
    } else if ( (conv_desc.datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (conv_desc.datatype_out != LIBXSMM_DNN_DATATYPE_I8) ) {
      /* error */
    } else if ( (conv_desc.datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (conv_desc.datatype_out != LIBXSMM_DNN_DATATYPE_F32) ) {
      /* error */
    } else {
      /* fine, no error */
    }
    handle->buffer_format = conv_desc.buffer_format;
    handle->filter_format = conv_desc.filter_format;
    handle->fuse_ops = conv_desc.fuse_ops;
    handle->options = conv_desc.options;

    /* derive additional values */
    handle->ifhp = conv_desc.H + 2*conv_desc.pad_h_in;
    handle->ifwp = conv_desc.W + 2*conv_desc.pad_w_in;
    handle->ofh = (conv_desc.H + 2*conv_desc.pad_h - conv_desc.R) / conv_desc.u + 1;
    handle->ofw = (conv_desc.W + 2*conv_desc.pad_w - conv_desc.S) / conv_desc.v + 1;
    handle->ofhp = handle->ofh + 2*conv_desc.pad_h_out;
    handle->ofwp = handle->ofw + 2*conv_desc.pad_w_out;
    handle->ifmblock = 1;
    handle->ofmblock = 1;
    handle->blocksifm = conv_desc.C;
    handle->blocksofm = conv_desc.K;
    handle->fwd_ofw_rb = 1;
    handle->fwd_ofh_rb = 1;
    handle->bwd_ofw_rb = 1;
    handle->bwd_ofh_rb = 1;
    handle->upd_ofw_rb = 1;
    handle->upd_ofh_rb = 1;
    handle->fm_lp_block = 1;
    handle->blocksifm_blocking = 1;
    handle->blocksofm_blocking = 1;
    /* Set algorithm to use */
    if (conv_desc.algo == LIBXSMM_DNN_CONV_ALGO_AUTO) {
      handle->algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
    } else {
      handle->algo = conv_desc.algo;
    }
    if ( handle->algo != LIBXSMM_DNN_CONV_ALGO_DIRECT ) {
      *status = LIBXSMM_DNN_ERR_INVALID_ALGO;
      free(handle);
      handle = 0;
      return 0;
    }

    *status = libxsmm_dnn_convolution_setup(handle);
  }
  else {
    *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
  }
  /* account for eventually deallocated handle */
  if ( LIBXSMM_DNN_SUCCESS != *status ) {
    handle = 0;
  }
  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_conv_layer(const libxsmm_dnn_layer* handle)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxsmm_barrier_release((const libxsmm_barrier*)handle->barrier); }

    /* deallocate handle structure itself */
    free(/*remove constness*/(libxsmm_dnn_layer*)handle);
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_create_tensor_datalayout(const libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor_datalayout* layout;

  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    layout = (libxsmm_dnn_tensor_datalayout*)calloc(1, sizeof(libxsmm_dnn_tensor_datalayout));

    if (layout != 0) {
      if ( (type == LIBXSMM_DNN_REGULAR_INPUT)  || (type == LIBXSMM_DNN_GRADIENT_INPUT)  || (type == LIBXSMM_DNN_INPUT)  ||
          (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT)    ) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXSMM_DNN_ACTIVATION;

        if ((handle->buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 5;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) || (type == LIBXSMM_DNN_INPUT) ) {
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = handle->ifwp;
                layout->dim_size[2] = handle->ifhp;
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.N;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = handle->ofwp;
                layout->dim_size[2] = handle->ofhp;
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.N;
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
            }
            /* @TODO this need to change */
          } else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32) ) {
            if ( ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_INPUT) )  ) {
              layout->datatype = handle->datatype_in;
            } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
              layout->datatype = handle->datatype_out;
            }
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 5;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) || (type == LIBXSMM_DNN_INPUT) )   {
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = handle->ifwp;
                layout->dim_size[2] = handle->ifhp;
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.N;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = handle->ofwp;
                layout->dim_size[2] = handle->ofhp;
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.N;
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            }
          } else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_BF16;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(6*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 5;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) || (type == LIBXSMM_DNN_INPUT) )   {
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = handle->ifwp;
                layout->dim_size[2] = handle->ifhp;
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.N;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = handle->ofwp;
                layout->dim_size[2] = handle->ofhp;
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.N;
              } else { /* coverity[dead_error_begin] */
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            }
          } else if ( ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32)) ||  (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8)  ) {
            if ( ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_INPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT)  )  ) {
              layout->datatype = handle->datatype_in;
            } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) ) {
              layout->datatype = handle->datatype_out;
            }
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_INPUT) )   {
                layout->num_dims = 5;
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = handle->ifwp;
                layout->dim_size[2] = handle->ifhp;
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.N;
              } else if ( type == LIBXSMM_DNN_GRADIENT_OUTPUT )   {
                layout->num_dims = 5;
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = handle->ofwp;
                layout->dim_size[2] = handle->ofhp;
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.N;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                layout->num_dims = 5;
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = handle->ofwp;
                layout->dim_size[2] = handle->ofhp;
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.N;
              } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT ) {
                layout->num_dims = 5;
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = handle->ifwp;
                layout->dim_size[2] = handle->ifhp;
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.N;
              } else { /* coverity[dead_error_begin] */
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0) {
          if ( ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) || (type == LIBXSMM_DNN_INPUT) )   {
                layout->dim_size[0] = handle->ifmblock * handle->blocksifm;
                layout->dim_size[1] = handle->ifwp;
                layout->dim_size[2] = handle->ifhp;
                layout->dim_size[3] = handle->desc.N;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
                layout->dim_size[1] = handle->ofwp;
                layout->dim_size[2] = handle->ofhp;
                layout->dim_size[3] = handle->desc.N;
              } else { /* coverity[dead_error_begin] */
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXSMM_DNN_REGULAR_FILTER) || (type == LIBXSMM_DNN_GRADIENT_FILTER) || (type == LIBXSMM_DNN_FILTER) ) {
        layout->format = handle->filter_format;
        layout->tensor_type = LIBXSMM_DNN_FILTER;

        if ((handle->filter_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(6*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 6;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_size[0] = handle->ofmblock;
              layout->dim_size[1] = handle->ifmblock;
              layout->dim_size[2] = handle->desc.S;
              layout->dim_size[3] = handle->desc.R;
              layout->dim_size[4] = handle->blocksifm;
              layout->dim_size[5] = handle->blocksofm;
            }
          } else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_BF16;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(7*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(7*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 7;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
              layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[6] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_size[0] = handle->fm_lp_block;
              layout->dim_size[1] = handle->ofmblock;
              layout->dim_size[2] = handle->ifmblock/handle->fm_lp_block;
              layout->dim_size[3] = handle->desc.S;
              layout->dim_size[4] = handle->desc.R;
              layout->dim_size[5] = handle->blocksifm;
              layout->dim_size[6] = handle->blocksofm;
            }
          } else if ( ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32)) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8 ) ) {
            if ( (type == LIBXSMM_DNN_REGULAR_FILTER) || (type == LIBXSMM_DNN_FILTER) ) {
              layout->datatype = handle->datatype_in;
            } else if (type == LIBXSMM_DNN_GRADIENT_FILTER) {
              layout->datatype = handle->datatype_out;
            }
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(7*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(7*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              if ((type == LIBXSMM_DNN_REGULAR_FILTER) || (type == LIBXSMM_DNN_FILTER)) {
                layout->num_dims = 7;
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
                layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
                layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[6] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_size[0] = handle->fm_lp_block;
                layout->dim_size[1] = handle->ofmblock;
                layout->dim_size[2] = handle->ifmblock/handle->fm_lp_block;
                layout->dim_size[3] = handle->desc.S;
                layout->dim_size[4] = handle->desc.R;
                layout->dim_size[5] = handle->blocksifm;
                layout->dim_size[6] = handle->blocksofm;
              }
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->filter_format & LIBXSMM_DNN_TENSOR_FORMAT_RSCK) > 0) {
          if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
              layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
              layout->dim_size[1] = handle->ifmblock * handle->blocksifm;
              layout->dim_size[2] = handle->desc.S;
              layout->dim_size[3] = handle->desc.R;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( type == LIBXSMM_DNN_REGULAR_FILTER_TRANS ) {
        layout->format = handle->filter_format;
        layout->tensor_type = LIBXSMM_DNN_REGULAR_FILTER_TRANS;

        if ((handle->filter_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(6*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 6;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->ifmblock;
              layout->dim_size[1] = handle->ofmblock;
              layout->dim_size[2] = handle->desc.S;
              layout->dim_size[3] = handle->desc.R;
              layout->dim_size[4] = handle->blocksofm;
              layout->dim_size[5] = handle->blocksifm;
            }
          } else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_BF16;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(7*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(7*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 7;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
              layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[6] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->fm_lp_block;
              layout->dim_size[1] = handle->ifmblock;
              layout->dim_size[2] = handle->ofmblock/handle->fm_lp_block;
              layout->dim_size[3] = handle->desc.S;
              layout->dim_size[4] = handle->desc.R;
              layout->dim_size[5] = handle->blocksofm;
              layout->dim_size[6] = handle->blocksifm;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
#if 0
        } else if ((handle->filter_format & LIBXSMM_DNN_TENSOR_FORMAT_RSCK) > 0) {
          if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
              layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
              layout->dim_size[1] = handle->ifmblock * handle->blocksifm;
              layout->dim_size[2] = handle->desc.S;
              layout->dim_size[3] = handle->desc.K;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
#endif
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXSMM_DNN_REGULAR_CHANNEL_BIAS) || (type == LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS) || (type == LIBXSMM_DNN_CHANNEL_BIAS) ) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXSMM_DNN_CHANNEL_SCALAR;

        if ((handle->buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
            layout->datatype = handle->datatype_out;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(2*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 2;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->ofmblock;
              layout->dim_size[1] = handle->blocksofm;
            }
#if 0
          } else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) ) {
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(3*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(3*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 3;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->fm_lp_block;
              layout->dim_size[1] = handle->ofmblock;
              layout->dim_size[2] = handle->blocksofm;
            }
#endif
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0) {
          layout->datatype = handle->datatype_out;
          if ( handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 ) {
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(1*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(1*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 1;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->ofmblock*handle->blocksofm;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXSMM_DNN_BATCH_STATS) ) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXSMM_DNN_BATCH_STATS;

        if ((handle->buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) || (handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 2;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_X;
              layout->dim_size[0] = handle->ofmblock;
              layout->dim_size[1] = handle->desc.N;
              layout->dim_size[2] = handle->blocksofm;
              layout->dim_size[3] = 2;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if (type == LIBXSMM_DNN_MAX_STATS_FWD) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXSMM_DNN_MAX_STATS_FWD;
        layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
        layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(2*sizeof(libxsmm_dnn_tensor_dimtype));
        layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));
        if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
          layout->num_dims = 2;
          layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
          layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
          layout->dim_size[0] = handle->ifmblock;
          layout->dim_size[1] = handle->desc.N;
        }
      } else if (type == LIBXSMM_DNN_MAX_STATS_BWD) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXSMM_DNN_MAX_STATS_BWD;
        layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
        layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(2*sizeof(libxsmm_dnn_tensor_dimtype));
        layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));
        if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
          layout->num_dims = 2;
          layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
          layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
          layout->dim_size[0] = handle->ifmblock;
          layout->dim_size[1] = handle->desc.N;
        }
      } else if (type == LIBXSMM_DNN_MAX_STATS_UPD) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXSMM_DNN_MAX_STATS_UPD;
        layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
        layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(2*sizeof(libxsmm_dnn_tensor_dimtype));
        layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));
        if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
          layout->num_dims = 2;
          layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
          layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
          layout->dim_size[0] = handle->ifmblock;
          layout->dim_size[1] = handle->desc.N;
        }
      } else {
        free(layout);
        layout = 0; /* make sure a NULL is returned */
        *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
      }
    } else {
      *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT;
    }
  }
  else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return layout;
}

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_trans_reg_bf16_filter(const libxsmm_dnn_layer* handle) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0) {
    if ( (handle->reg_filter != 0) && (handle->reg_filter_tr != 0) ) {
      /* TODO handle more datatypes */
      int ifm1, ifm2, kj, ki, ofm1, ofm2;
      int ofmblock_lp = handle->ofmblock/handle->fm_lp_block;
      int ifmblock_lp = handle->ifmblock/handle->fm_lp_block;
      int lpb = handle->fm_lp_block;
      LIBXSMM_VLA_DECL(7, libxsmm_bfloat16, wt, (libxsmm_bfloat16*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, lpb);
      LIBXSMM_VLA_DECL(7, libxsmm_bfloat16, tr_wt, (libxsmm_bfloat16*)handle->reg_filter_tr->data, handle->blocksofm, handle->desc.R, handle->desc.S, ofmblock_lp, handle->ifmblock, lpb);

      /* TODO we might want to do this in parallel.... */
      for ( ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1 ) {
        for ( ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1 ) {
          for (kj=0; kj < handle->desc.R; ++kj) {
            for (ki=0; ki < handle->desc.S; ++ki) {
              for ( ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2 ) {
                for ( ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2 ) {
                  LIBXSMM_VLA_ACCESS(7, tr_wt, ifm1, ofm1, handle->desc.R-1-kj , handle->desc.S-1-ki, ofm2/lpb, ifm2, ofm2%lpb, handle->blocksofm, handle->desc.R, handle->desc.S, ofmblock_lp, handle->ifmblock, lpb) =
                    LIBXSMM_VLA_ACCESS(7, wt, ofm1, ifm1, kj, ki, ifm2/lpb, ofm2, ifm2%lpb, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, lpb);
                }
              }
            }
          }
        }
      }
    } else {
      status = LIBXSMM_DNN_ERR_INVALID_TENSOR;
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_trans_reg_filter(const libxsmm_dnn_layer* handle) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0) {
    if ( (handle->reg_filter != 0) && (handle->reg_filter_tr != 0) ) {
      /* TODO handle more datatypes */
      int ifm1, ifm2, kj, ki, ofm1, ofm2;
      LIBXSMM_VLA_DECL(6, float, wt, (float*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
      LIBXSMM_VLA_DECL(6, float, tr_wt, (float*)handle->reg_filter_tr->data, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);

      /* TODO we might want to do this in parallel.... */
      for ( ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1 ) {
        for ( ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1 ) {
          for (kj=0; kj < handle->desc.R; ++kj) {
            for (ki=0; ki < handle->desc.S; ++ki) {
              for ( ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2 ) {
                for ( ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2 ) {
                  LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, handle->desc.R-1-kj, handle->desc.S-1-ki, ofm2, ifm2, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock) =
                    LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                }
              }
            }
          }
        }
      }
    } else {
      status = LIBXSMM_DNN_ERR_INVALID_TENSOR;
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_bind_tensor(libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)        && (type != LIBXSMM_DNN_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_REGULAR_OUTPUT)       && (type != LIBXSMM_DNN_GRADIENT_OUTPUT) &&
      (type != LIBXSMM_DNN_REGULAR_FILTER)       && (type != LIBXSMM_DNN_GRADIENT_FILTER) &&
      (type != LIBXSMM_DNN_REGULAR_CHANNEL_BIAS)         && (type != LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS)   &&
      (type != LIBXSMM_DNN_REGULAR_FILTER_TRANS) && (type != LIBXSMM_DNN_BATCH_STATS) && (type != LIBXSMM_DNN_MAX_STATS_FWD) && (type != LIBXSMM_DNN_MAX_STATS_BWD)  && (type != LIBXSMM_DNN_MAX_STATS_UPD)  ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxsmm_dnn_tensor_datalayout* handle_layout = libxsmm_dnn_create_tensor_datalayout(handle, type, &status);

    if ( libxsmm_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXSMM_DNN_REGULAR_INPUT ) {
        handle->reg_input = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT ) {
        handle->grad_input = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_REGULAR_OUTPUT ) {
        handle->reg_output = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_OUTPUT ) {
        handle->grad_output = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_REGULAR_FILTER ) {
        handle->reg_filter = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_FILTER ) {
        handle->grad_filter = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_REGULAR_CHANNEL_BIAS ) {
        handle->reg_bias = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS ) {
        handle->grad_bias = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_REGULAR_FILTER_TRANS ) {
        handle->reg_filter_tr = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_BATCH_STATS ) {
        handle->batch_stats = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_MAX_STATS_FWD ) {
        handle->maxstats_fwd = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_MAX_STATS_BWD ) {
        handle->maxstats_bwd = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_MAX_STATS_UPD ) {
        handle->maxstats_upd = (libxsmm_dnn_tensor*)tensor;
      } else {
        /* cannot happen */
      }
    } else {
      status = LIBXSMM_DNN_ERR_MISMATCH_TENSOR;
    }

    libxsmm_dnn_destroy_tensor_datalayout( handle_layout );
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_get_tensor(libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_tensor* return_tensor = 0;

  *status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)        && (type != LIBXSMM_DNN_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_REGULAR_OUTPUT)       && (type != LIBXSMM_DNN_GRADIENT_OUTPUT) &&
      (type != LIBXSMM_DNN_REGULAR_FILTER)       && (type != LIBXSMM_DNN_GRADIENT_FILTER) &&
      (type != LIBXSMM_DNN_REGULAR_CHANNEL_BIAS)         && (type != LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS)   &&
      (type != LIBXSMM_DNN_REGULAR_FILTER_TRANS) && (type != LIBXSMM_DNN_BATCH_STATS) && (type != LIBXSMM_DNN_MAX_STATS_FWD) && (type != LIBXSMM_DNN_MAX_STATS_BWD)  && (type != LIBXSMM_DNN_MAX_STATS_UPD)  ) {
    *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return return_tensor;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_REGULAR_INPUT ) {
      return_tensor = handle->reg_input;
    } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT ) {
      return_tensor = handle->grad_input;
    } else if ( type == LIBXSMM_DNN_REGULAR_OUTPUT ) {
      return_tensor = handle->reg_output;
    } else if ( type == LIBXSMM_DNN_GRADIENT_OUTPUT ) {
      return_tensor = handle->grad_output;
    } else if ( type == LIBXSMM_DNN_REGULAR_FILTER ) {
      return_tensor = handle->reg_filter;
    } else if ( type == LIBXSMM_DNN_GRADIENT_FILTER ) {
      return_tensor = handle->grad_filter;
    } else if ( type == LIBXSMM_DNN_REGULAR_CHANNEL_BIAS ) {
      return_tensor = handle->reg_bias;
    } else if ( type == LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS ) {
      return_tensor = handle->grad_bias;
    } else if ( type == LIBXSMM_DNN_REGULAR_FILTER_TRANS ) {
      return_tensor = handle->reg_filter_tr;
    } else if ( type == LIBXSMM_DNN_BATCH_STATS ) {
      return_tensor = handle->batch_stats;
    } else if ( type == LIBXSMM_DNN_MAX_STATS_FWD ) {
      return_tensor = handle->maxstats_fwd;
    } else if ( type == LIBXSMM_DNN_MAX_STATS_BWD ) {
      return_tensor = handle->maxstats_bwd;
    } else if ( type == LIBXSMM_DNN_MAX_STATS_UPD ) {
      return_tensor = handle->maxstats_upd;
    } else {
      /* cannot happen */
    }
  }
  else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return return_tensor;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_release_tensor(libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)        && (type != LIBXSMM_DNN_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_REGULAR_OUTPUT)       && (type != LIBXSMM_DNN_GRADIENT_OUTPUT) &&
      (type != LIBXSMM_DNN_REGULAR_FILTER)       && (type != LIBXSMM_DNN_GRADIENT_FILTER) &&
      (type != LIBXSMM_DNN_REGULAR_CHANNEL_BIAS)         && (type != LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS)   &&
      (type != LIBXSMM_DNN_REGULAR_FILTER_TRANS) && (type != LIBXSMM_DNN_BATCH_STATS) && (type != LIBXSMM_DNN_MAX_STATS_FWD) && (type != LIBXSMM_DNN_MAX_STATS_BWD)  && (type != LIBXSMM_DNN_MAX_STATS_UPD)  ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_REGULAR_INPUT ) {
      handle->reg_input = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT ) {
      handle->grad_input = 0;
    } else if ( type == LIBXSMM_DNN_REGULAR_OUTPUT ) {
      handle->reg_output = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_OUTPUT ) {
      handle->grad_output = 0;
    } else if ( type == LIBXSMM_DNN_REGULAR_FILTER ) {
      handle->reg_filter = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_FILTER ) {
      handle->grad_filter = 0;
    } else if ( type == LIBXSMM_DNN_REGULAR_CHANNEL_BIAS ) {
      handle->reg_bias = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS ) {
      handle->grad_bias = 0;
    } else if ( type == LIBXSMM_DNN_REGULAR_FILTER_TRANS ) {
      handle->reg_filter_tr = 0;
    } else if ( type == LIBXSMM_DNN_BATCH_STATS ) {
      handle->batch_stats = 0;
    } else if ( type == LIBXSMM_DNN_MAX_STATS_FWD ) {
      handle->maxstats_fwd = 0;
    } else if ( type == LIBXSMM_DNN_MAX_STATS_BWD ) {
      handle->maxstats_bwd = 0;
    } else if ( type == LIBXSMM_DNN_MAX_STATS_UPD ) {
      handle->maxstats_upd = 0;
    } else {
      /* cannot happen */
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXSMM_API size_t libxsmm_dnn_get_scratch_size(const libxsmm_dnn_layer* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status)
{
  size_t l_scratch_size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: break;
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: break;
      default: {
        *status = LIBXSMM_DNN_ERR_INVALID_KIND;
      }
    }
    l_scratch_size += handle->scratch_size + 64;
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return l_scratch_size;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_bind_scratch(libxsmm_dnn_layer* handle, const libxsmm_dnn_compute_kind kind, const void* scratch)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  uintptr_t address = (uintptr_t)scratch;
  size_t offset = 0;

  if (scratch == 0) {
    status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  if (0 != handle) {
    if (address % 64 == 0) {
      handle->scratch = (void*)address;
    } else {
      offset = (64 - address % 64);
      handle->scratch = (void*)(address+offset);
    }
    address += handle->scratch_size + 64;

    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: break;
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: break;
      default: {
        status = LIBXSMM_DNN_ERR_INVALID_KIND;
      }
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_release_scratch(libxsmm_dnn_layer* handle, const libxsmm_dnn_compute_kind kind)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    handle->scratch = 0;
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: break;
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: break;
      default: {
        status = LIBXSMM_DNN_ERR_INVALID_KIND;
      }
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API_INLINE libxsmm_dnn_err_t internal_execute_st(libxsmm_dnn_layer* handle,
    libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (handle->algo) {
      case LIBXSMM_DNN_CONV_ALGO_DIRECT: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            switch (handle->buffer_format) {
              case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_st_fwd_custom_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              case LIBXSMM_DNN_TENSOR_FORMAT_NHWC: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_RSCK: {
                    status = libxsmm_dnn_convolve_st_fwd_nhwc_rsck(handle, start_thread, tid);
                  } break;
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_st_fwd_nhwc_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
            switch (handle->buffer_format) {
              case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_st_bwd_custom_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              case LIBXSMM_DNN_TENSOR_FORMAT_NHWC: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_RSCK: {
                    status = libxsmm_dnn_convolve_st_bwd_nhwc_rsck(handle, start_thread, tid);
                  } break;
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_st_bwd_nhwc_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
            switch (handle->buffer_format) {
              case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_st_upd_custom_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              case LIBXSMM_DNN_TENSOR_FORMAT_NHWC: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_RSCK: {
                    status = libxsmm_dnn_convolve_st_upd_nhwc_rsck(handle, start_thread, tid);
                  } break;
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_st_upd_nhwc_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD: {
            switch (handle->buffer_format) {
              case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_st_upd_custom_custom(handle, start_thread, tid);
                    status = libxsmm_dnn_convolve_st_bwd_custom_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              case LIBXSMM_DNN_TENSOR_FORMAT_NHWC: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_RSCK: {
                    status = libxsmm_dnn_convolve_st_upd_nhwc_rsck(handle, start_thread, tid);
                    status = libxsmm_dnn_convolve_st_bwd_nhwc_rsck(handle, start_thread, tid);
                  } break;
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_st_upd_nhwc_custom(handle, start_thread, tid);
                    status = libxsmm_dnn_convolve_st_bwd_nhwc_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      default: {
        status = LIBXSMM_DNN_ERR_INVALID_ALGO;
      }
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_execute_st(libxsmm_dnn_layer* handle,
    libxsmm_dnn_compute_kind kind, /*unsigned*/int start_thread, /*unsigned*/int tid)
{
  return internal_execute_st(handle, kind, start_thread, tid);
}


LIBXSMM_API void libxsmm_dnn_execute(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind)
{
#if defined(_OPENMP)
# pragma omp parallel num_threads(handle->desc.threads)
  {
    const int tid = omp_get_thread_num();
    internal_execute_st(handle, kind, 0, tid);
  }
#else
  internal_execute_st(handle, kind, 0/*start_thread*/, 0/*tid*/);
#endif
}

