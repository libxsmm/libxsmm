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
  /* Pack only for small images and when having large K to amortize */
  if ((handle->ofw <= 14) && (handle->desc.K > 512) && (handle->desc.R == 1) && (handle->desc.S == 1) && (handle->desc.u == 2) && (handle->desc.v == 2)) {
    result = 1;
  }
  /* Make sure we don't pack when minibatch is not divisible by number of threads since H is used potentially for parallelism */
  if (handle->desc.N != handle->desc.threads) {
    result = 0;
  }

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
  /*  Make sure we don't use multiple rows when we don't pack input and convolutions are strided*/
  if ((handle->pack_input == 0) && ((handle->desc.u !=1 ) || (handle->desc.v != 1))) {
    result = 1;
  }
  /* In this case we will be using fallback generic loops, thus ofh_rb should be 1 */
  if ((handle->desc.N % handle->desc.threads != 0) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_fwd_block_H( libxsmm_dnn_layer* handle ) {
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

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_blocksifm_blocking( libxsmm_dnn_layer* handle ) {
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

  return result;
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
  if (handle->blocksofm % result != 0) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_convolution_setup_init_bwd_gemm_flags( libxsmm_dnn_layer* handle ) {
  int result = 0;

  /* TODO: May want to experiment with streaming stores */
  LIBXSMM_UNUSED( handle );

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
    handle->compute_pixels = compute_pixels;

    handle->use_intermediate_f32_wt_tensor = (handle->pixel_blocking == handle->n_used_pixels) ? 0 : 1;
    handle->scratch2_size = (size_t) (handle->desc.N * handle->output_pixels * handle->desc.K * sizeof(float)/2);
    if (handle->use_intermediate_f32_wt_tensor) {
      handle->scratch2_size += (size_t) handle->desc.R * handle->desc.S * handle->desc.C * handle->desc.K * handle->desc.threads * sizeof(float);
    }
    handle->scratch3_size = (size_t) (handle->desc.N * handle->input_pixels * handle->desc.C * sizeof(float)/2);

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
    handle->ofwp_extended = handle->ofwp + remainder_pixels;
    handle->ifwp_extended = handle->ifwp + remainder_pixels;
    handle->output_pixels = handle->ofwp * handle->ofwp_extended;
    /* coverity[identical_branches] */
    handle->batchreduce_h_pixels = (handle->upd_trans_w_only) ? 1 : 1; /* TODO: identical_branches */
    handle->use_intermediate_f32_wt_tensor = (handle->batchreduce_h_pixels == handle->ofh) ? 0 : 1;
    handle->scratch2_size = (size_t) (handle->desc.N * handle->ofhp*handle->ofwp_extended * handle->desc.K * sizeof(float)/2);
    if (handle->use_intermediate_f32_wt_tensor) {
      handle->scratch2_size += (size_t) handle->desc.R * handle->desc.S * handle->desc.C * handle->desc.K * handle->desc.threads * sizeof(float);
    }
    handle->scratch3_size = (size_t) (handle->desc.N * handle->ifhp * handle->ifwp_extended * handle->desc.C * sizeof(float)/2);
  }

}


LIBXSMM_API_INLINE libxsmm_dnn_err_t libxsmm_dnn_convolution_setup( libxsmm_dnn_layer* handle ) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  const libxsmm_trans_descriptor* tr_desc = 0;
  libxsmm_descriptor_blob blob;

  /* Generic parameter setup  */
  handle->ifmblock = libxsmm_dnn_convolution_setup_ifmblock(handle);
  handle->ofmblock = libxsmm_dnn_convolution_setup_ofmblock(handle);
  handle->fm_lp_block = libxsmm_dnn_convolution_setup_fm_lp_block(handle);
  handle->blocksifm = libxsmm_dnn_convolution_setup_blocksifm(handle);
  handle->blocksofm = libxsmm_dnn_convolution_setup_blocksofm(handle);

  /* FWD parameter setup  */
  handle->fwd_ofw_rb = libxsmm_dnn_convolution_setup_fwd_ofw_rb(handle);
  handle->pack_input = libxsmm_dnn_convolution_setup_pack_input_fwd(handle);
  handle->fwd_ofh_rb = libxsmm_dnn_convolution_setup_fwd_ofh_rb(handle);
  handle->block_fwd_oj = libxsmm_dnn_convolution_setup_fwd_block_H(handle);
  handle->loop_order = libxsmm_dnn_convolution_setup_loop_order_fwd(handle);
  handle->blocksifm_blocking = libxsmm_dnn_convolution_setup_blocksifm_blocking(handle);
  handle->block_fwd_ofm = libxsmm_dnn_convolution_setup_block_fwd_OFM(handle);
  handle->block_fwd_ifm = libxsmm_dnn_convolution_setup_block_fwd_IFM(handle);;
  handle->avoid_fmas_in_rim = libxsmm_dnn_convolution_setup_avoid_rim_fmas_fwd(handle);
  handle->use_ofm_parallelization = libxsmm_dnn_convolution_setup_use_ofm_parallelization(handle);
  handle->shuffle_filter_accesses = libxsmm_dnn_convolution_setup_shuffle_filter_accesses(handle);
  handle->avoid_acc_load = libxsmm_dnn_convolution_setup_avoid_acc_load(handle);
  handle->fwd_flags = libxsmm_dnn_convolution_setup_init_fwd_gemm_flags(handle);
  handle->use_fallback_fwd_loops = libxsmm_dnn_convolution_setup_fallback_loops_fwd(handle);
  handle->code_fwd[0].ptr = 0;
  handle->code_fwd[1].ptr = 0;
  handle->code_fwd[2].ptr = 0;

  /* Create strided BRGEMMs for i8i32 convolutions  */
  if ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32)) {
    const libxsmm_blasint ldx = (handle->pack_input == 1) ? (libxsmm_blasint)handle->ifmblock : (libxsmm_blasint)handle->desc.v*handle->ifmblock;
    const libxsmm_blasint ldA = handle->ofmblock;
    const libxsmm_blasint ldC = handle->ofmblock;
    const int beta = (handle->avoid_acc_load) ? 0 : 1;
    int l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | handle->fwd_flags;
    if (handle->desc.R == 1 && handle->desc.S == 1) {
      const int IFW = (handle->pack_input == 1) ? handle->ofwp : handle->ifwp;
      const int IFH = (handle->pack_input == 1) ? handle->ofhp : handle->ifhp;
      libxsmm_blasint stride_A = handle->ifmblock * handle->ofmblock * sizeof(char);
      libxsmm_blasint stride_B = handle->ifmblock * IFW * IFH * sizeof(char) ;
      handle->gemm_fwd.xgemm.subimrs = libxsmm_subimmdispatch_reducebatch_strd(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, stride_A, stride_B, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
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
        handle->gemm_fwd.xgemm.subimro = libxsmm_subimmdispatch_reducebatch_offs(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
      } else {
        libxsmm_blasint stride_A = handle->ifmblock * handle->desc.R * handle->desc.S * handle->ofmblock * sizeof(char);
        libxsmm_blasint stride_B = handle->ifmblock * IFW * IFH * sizeof(char) ;
        handle->gemm_fwd.xgemm.subimrs = libxsmm_subimmdispatch_reducebatch_strd(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, stride_A, stride_B, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
        handle->gemm_fwd2.xgemm.subimrs = libxsmm_subimmdispatch_reducebatch_strd(handle->ofmblock, handle->fwd_ofh_rb*(handle->fwd_ofw_rb-1), handle->ifmblock, stride_A, stride_B, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
      }
    }
  } else if ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I8)) {
    const libxsmm_blasint ldx = (libxsmm_blasint)handle->desc.v*handle->ifmblock;
    const libxsmm_blasint ldA = handle->ofmblock;
    const libxsmm_blasint ldC = handle->ofmblock;
    const int beta = 0;
    int l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | handle->fwd_flags;
    if (handle->desc.R == 1 && handle->desc.S == 1) {
      const int IFW = handle->ifwp;
      const int IFH = handle->ifhp;
      libxsmm_blasint stride_A = handle->ifmblock * handle->ofmblock * sizeof(char);
      libxsmm_blasint stride_B = handle->ifmblock * IFW * IFH * sizeof(char) ;
      handle->gemm_fwd.xgemm.sububmrs = libxsmm_sububmmdispatch_reducebatch_strd(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, stride_A, stride_B, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
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
      handle->gemm_fwd.xgemm.sububmro = libxsmm_sububmmdispatch_reducebatch_offs(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
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
  handle->pack_input_bwd = libxsmm_dnn_convolution_setup_pack_input_bwd(handle);
  handle->spread_input_bwd = libxsmm_dnn_convolution_setup_spread_input_bwd(handle);
  handle->blocksofm_blocking = libxsmm_dnn_convolution_setup_blocksofm_blocking(handle);
  handle->avoid_acc_load_bwd = libxsmm_dnn_convolution_setup_avoid_acc_load_bwd(handle);
  handle->use_ifm_parallelization = libxsmm_dnn_convolution_setup_use_ifm_parallelization(handle);
  handle->block_bwd_ofm = libxsmm_dnn_convolution_setup_block_bwd_OFM(handle);
  handle->block_bwd_ifm = libxsmm_dnn_convolution_setup_block_bwd_IFM(handle);
  handle->block_bwd_oj = libxsmm_dnn_convolution_setup_bwd_block_H(handle);
  handle->use_fallback_bwd_loops = libxsmm_dnn_convolution_setup_fallback_loops_bwd(handle);

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
  tr_desc = libxsmm_trans_descriptor_init(&blob, sizeof(float), 64, 16, 64);
  handle->tr_kernel = libxsmm_dispatch_trans(tr_desc);

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

  if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) {
    libxsmm_dnn_convolution_setup_bf16_upd(handle);
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

  handle->code_upd[0].ptr = 0;
  handle->code_upd[1].ptr = 0;

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
  if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16 || handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) {
    handle->scratch6_size = (size_t) (handle->desc.N * LIBXSMM_MAX(handle->ofwp * handle->ofhp * handle->desc.K, handle->desc.W * handle->desc.H * handle->desc.C) + handle->desc.threads * LIBXSMM_MAX(handle->fwd_ofw_rb * handle->fwd_ofh_rb * handle->ofmblock, handle->bwd_ofw_rb * handle->desc.v * handle->bwd_ofh_rb * handle->ifmblock))* sizeof(float);
  }

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
  /* @TODO: add logical padding support */
  if ( ( conv_desc.pad_h != conv_desc.pad_h_in )  ||
      ( conv_desc.pad_w != conv_desc.pad_w_in )  ||
      ( conv_desc.pad_h != conv_desc.pad_h_out ) ||
      ( conv_desc.pad_w != conv_desc.pad_w_out )    ) {
    *status = LIBXSMM_DNN_ERR_INVALID_PADDING;
    return 0;
  }

  handle = (libxsmm_dnn_layer*)malloc(sizeof(libxsmm_dnn_layer));

  if (0 != handle) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
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
    handle->post_bn = handle->desc.post_bn;
    handle->pre_bn = handle->desc.pre_bn;
    handle->fuse_batchstats_fwd = 0;
    handle->fuse_batchstats_bwd = 0;
    handle->fuse_eltwise_bwd = 0;
    handle->fuse_relu_bwd = 0;

    /* TODO: This check should be removed when this fuse flag is deprecated */
    if (handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_FWD) {
      handle->fuse_batchstats_fwd = 1;
    }

    if (handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) {
      handle->fuse_relu_bwd = 1;
    }

    /* Enable batchnorm fusion depending on the input */
    if (handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCHNORM_STATS) {
      if (handle->desc.post_bn != NULL) {
        handle->fuse_batchstats_fwd = 1;
      }
      if (handle->desc.pre_bn != NULL) {
        handle->fuse_batchstats_bwd = 1;
      }
    }

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
    handle->fwd_ofw_rb_2 = 0;
    handle->fwd_ofh_rb = 1;
    handle->bwd_ofw_rb = 1;
    handle->bwd_ofh_rb = 1;
    handle->upd_ofw_rb = 1;
    handle->upd_ofh_rb = 1;
    handle->fm_lp_block = 1;
    handle->blocksifm_blocking = 1;
    handle->blocksofm_blocking = 1;
    handle->upd_use_thread_fil = 0;
    handle->upd_use_external_reduce = 0;
    handle->filter_transposed = 0;
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

    /* Fix up scratch */
    /* @TODO move all scratch calculation into one place */
    {
      {
        const size_t padded_h = ((size_t)2 * handle->desc.pad_h) + handle->desc.H, padded_w = ((size_t)2 * handle->desc.pad_w) + handle->desc.W;
        const size_t size5_tensor = padded_h * padded_w * handle->ifmblock * libxsmm_dnn_typesize(handle->datatype_in);
        const size_t size5 = LIBXSMM_UP2(size5_tensor, LIBXSMM_CACHELINE) * handle->desc.threads;
        if (handle->max_scratch5_size < size5) handle->max_scratch5_size = size5;
        handle->scratch5 = 0;
      }
      {
        const size_t size6a = (size_t)handle->ofmblock * handle->ofw * handle->ofh * sizeof(float);
        const size_t size6b = (size_t)handle->ifmblock * handle->fm_lp_block *  handle->desc.W * handle->desc.H * sizeof(float);
        const size_t size6 = ( size6a > size6b ) ? size6a : size6b;
        handle->scratch6_size = LIBXSMM_MAX(LIBXSMM_UP2(size6, LIBXSMM_CACHELINE) * handle->desc.threads, handle->scratch6_size);
      }
      {
        const size_t output_typesize = libxsmm_dnn_typesize(handle->datatype_out);
        const size_t size6_tensor = (size_t)handle->ofhp * handle->ofwp * handle->ofmblock * output_typesize;
        const size_t size6 = LIBXSMM_UP2(size6_tensor, LIBXSMM_CACHELINE) * handle->desc.threads;
        if (handle->scratch6_size < size6) handle->scratch6_size = size6;
      }
      handle->scratch6 = 0;
      {
        /* FIXME: currently filter data-type is always smaller/equal output type */
        const size_t filter_typesize = libxsmm_dnn_typesize(handle->datatype_out);
        const size_t size7 = (size_t)handle->desc.R * handle->desc.S * handle->desc.C * handle->desc.K * filter_typesize + handle->ifmblock * handle->ofmblock * sizeof(float);
        handle->scratch7_size = LIBXSMM_UP2(size7, LIBXSMM_CACHELINE) * LIBXSMM_MAX(handle->desc.threads, handle->desc.N);
      }
    }
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
    layout = (libxsmm_dnn_tensor_datalayout*) malloc(sizeof(libxsmm_dnn_tensor_datalayout));

    if (layout != 0) {
      memset(layout, 0, sizeof(libxsmm_dnn_tensor_datalayout));
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
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           if (handle->padding_flag == 1) {
                                             l_scratch_size = handle->fwdbwd_scratch_size + 64;
                                           }
                                           l_scratch_size += handle->max_scratch5_size + 64;
                                           l_scratch_size += handle->scratch6_size + 64;
                                           l_scratch_size += handle->scratch7_size + 64;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
                                           /* we need filter for transpose, + 64 to do alignment while performing bind, scratch1 */
                                           l_scratch_size = handle->scratch1_size + 64;
                                           l_scratch_size += handle->fwdbwd_scratch_size + 64;
                                           l_scratch_size += handle->max_scratch5_size + 64;
                                           l_scratch_size += handle->scratch7_size + 64;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
                                           if (handle->use_lp_kernel == 1) {
                                             l_scratch_size += handle->scratch2_size + 64;
                                           }
                                           /* we need a minibatch copy for transpose of input, scratch3 */
                                           l_scratch_size += handle->scratch3_size + 64;
                                           /* potentially we need thread-local filter copies, scratch4 */
                                           if (handle->upd_use_thread_fil == 1) {
                                             l_scratch_size += handle->scratch4_size + 64;
                                           }
                                           l_scratch_size += handle->max_scratch5_size + 64;
                                           l_scratch_size += handle->minibatch_scratch_size + 64;
                                           l_scratch_size += handle->scratch6_size + 64;
                                           if (handle->scratch7_size != 0) {
                                             l_scratch_size += handle->scratch7_size + 64;
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           /* we need filter for transpose, + 64 to do alignment while performing bind, scratch1 */
                                           l_scratch_size += handle->scratch1_size + 64;
                                           if (handle->use_lp_kernel == 1) {
                                             l_scratch_size += handle->scratch2_size + 64;
                                           }
                                           /* we need a minibatch copy for transpose of input, scratch3 */
                                           l_scratch_size += handle->scratch3_size + 64;
                                           /* potentially we need thread-local filter copies, scratch4 */
                                           if (handle->upd_use_thread_fil == 1) {
                                             l_scratch_size += handle->scratch4_size + 64;
                                           }
                                           l_scratch_size += handle->max_scratch5_size + 64;
                                           if (handle->scratch6_size != 0) {
                                             l_scratch_size += handle->scratch6_size + 64;
                                           }
                                           if (handle->scratch7_size != 0) {
                                             l_scratch_size += handle->scratch7_size + 64;
                                           }
                                         } break;
      default: {
                 *status = LIBXSMM_DNN_ERR_INVALID_KIND;
               }
    }
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
    /* check this if, this is bogus, not sure why there */
#if 0
    if ( (kind == LIBXSMM_DNN_COMPUTE_KIND_FWD) && (handle->datatype_in == handle->datatype_out) ) {
      status = LIBXSMM_DNN_SUCCESS;
    }
#endif
    return status;
  }

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           if (address % 64 == 0) {
                                             handle->scratch1 = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->scratch1 = (void*)(address+offset);
                                           }
                                           address += handle->scratch1_size + 64;
                                           if (address % 64 == 0) {
                                             handle->scratch5 = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->scratch5 = (void*)(address+offset);
                                           }
                                           address += handle->max_scratch5_size + 64;
                                           if (handle->scratch6_size != 0) {
                                             if (address % 64 == 0) {
                                               handle->scratch6 = (void*)address;
                                             }
                                             else {
                                               offset = (64 - address % 64);
                                               handle->scratch6 = (void*)(address + offset);
                                             }
                                             address += handle->scratch6_size + 64;
                                           }
                                           if (handle->scratch7_size != 0) {
                                             if (address % 64 == 0) {
                                               handle->scratch7 = (void*)address;
                                             } else {
                                               offset = (64 - address % 64);
                                               handle->scratch7 = (void*)(address+offset);
                                             }
                                             address += handle->scratch7_size + 64;
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
                                           /* we need filter for transpose, + 64 to do alignment while performing bind, scratch1 */
                                           if (address % 64 == 0) {
                                             handle->scratch1 = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->scratch1 = (void*)(address+offset);
                                           }
                                           if (address % 64 == 0) {
                                             handle->scratch5 = (void*)address;
                                           }
                                           else {
                                             offset = (64 - address % 64);
                                             handle->scratch5 = (void*)(address + offset);
                                           }
                                           address += handle->max_scratch5_size + 64;
                                           if (handle->scratch7_size != 0) {
                                             if (address % 64 == 0) {
                                               handle->scratch7 = (void*)address;
                                             } else {
                                               offset = (64 - address % 64);
                                               handle->scratch7 = (void*)(address+offset);
                                             }
                                             address += handle->scratch7_size + 64;
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
                                           /* we need a minibatch copy for transpose of input, scratch3 */
                                           if (handle->use_lp_kernel == 1) {
                                             if (address % 64 == 0) {
                                               handle->scratch2 = (void*)address;
                                             }
                                             else {
                                               offset = (64 - address % 64);
                                               handle->scratch2 = (void*)(address + offset);
                                             }
                                             address += handle->scratch2_size + 64;
                                           }
                                           if (address % 64 == 0) {
                                             handle->scratch3 = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->scratch3 = (void*)(address+offset);
                                           }
                                           /* potentially we need thread-local filter copies, scratch4 */
                                           if (handle->upd_use_thread_fil == 1) {
                                             address += handle->scratch3_size + 64;
                                             if (address % 64 == 0) {
                                               handle->scratch4 = (void*)address;
                                             } else {
                                               offset = (64 - address % 64);
                                               handle->scratch4 = (void*)(address+offset);
                                             }
                                             address += handle->scratch4_size + 64;
                                           }
                                           if (address % 64 == 0) {
                                             handle->scratch5 = (void*)address;
                                           }
                                           else {
                                             offset = (64 - address % 64);
                                             handle->scratch5 = (void*)(address + offset);
                                           }
                                           address += handle->max_scratch5_size + 64;
                                           if (handle->scratch6_size != 0) {
                                             if (address % 64 == 0) {
                                               handle->scratch6 = (void*)address;
                                             }
                                             else {
                                               offset = (64 - address % 64);
                                               handle->scratch6 = (void*)(address + offset);
                                             }
                                             address += handle->scratch6_size + 64;
                                           }
                                           if (handle->scratch7_size != 0) {
                                             if (address % 64 == 0) {
                                               handle->scratch7 = (void*)address;
                                             } else {
                                               offset = (64 - address % 64);
                                               handle->scratch7 = (void*)(address+offset);
                                             }
                                             address += handle->scratch7_size + 64;
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           /* we need filter for transpose, + 64 to do alignment while performing bind, scratch1 */
                                           if (address % 64 == 0) {
                                             handle->scratch1 = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->scratch1 = (void*)(address+offset);
                                           }
                                           address += handle->scratch1_size + 64;
                                           if (handle->use_lp_kernel == 1) {
                                             if (address % 64 == 0) {
                                               handle->scratch2 = (void*)address;
                                             }
                                             else {
                                               offset = (64 - address % 64);
                                               handle->scratch2 = (void*)(address + offset);
                                             }
                                             address += handle->scratch2_size + 64;
                                           }
                                           /* we need a minibatch copy for transpose of input, scratch3 */
                                           if (address % 64 == 0) {
                                             handle->scratch3 = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->scratch3 = (void*)(address+offset);
                                           }
                                           address += handle->scratch3_size + 64;
                                           /* potentially we need thread-local filter copies, scratch4 */
                                           if (handle->upd_use_thread_fil == 1) {
                                             if (address % 64 == 0) {
                                               handle->scratch4 = (void*)address;
                                             } else {
                                               offset = (64 - address % 64);
                                               handle->scratch4 = (void*)(address+offset);
                                             }
                                             address += handle->scratch4_size + 64;
                                           }
                                           if (address % 64 == 0) {
                                             handle->scratch5 = (void*)address;
                                           }
                                           else {
                                             offset = (64 - address % 64);
                                             handle->scratch5 = (void*)(address + offset);
                                           }
                                           address += handle->max_scratch5_size + 64;
                                           if (handle->scratch6_size != 0) {
                                             if (address % 64 == 0) {
                                               handle->scratch6 = (void*)address;
                                             }
                                             else {
                                               offset = (64 - address % 64);
                                               handle->scratch6 = (void*)(address + offset);
                                             }
                                             address += handle->scratch6_size + 64;
                                           }
                                           if (handle->scratch7_size != 0) {
                                             if (address % 64 == 0) {
                                               handle->scratch7 = (void*)address;
                                             } else {
                                               offset = (64 - address % 64);
                                               handle->scratch7 = (void*)(address+offset);
                                             }
                                             address += handle->scratch7_size + 64;
                                           }
                                         } break;
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
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           handle->scratch5 = 0;
                                           handle->scratch6 = 0;
                                           handle->scratch7 = 0;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
                                           handle->scratch1 = 0;
                                           handle->scratch5 = 0;
                                           handle->scratch7 = 0;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
                                           handle->scratch2 = 0;
                                           handle->scratch3 = 0;
                                           handle->scratch4 = 0;
                                           handle->scratch5 = 0;
                                           handle->scratch6 = 0;
                                           handle->scratch7 = 0;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           handle->scratch1 = 0;
                                           handle->scratch2 = 0;
                                           handle->scratch3 = 0;
                                           handle->scratch4 = 0;
                                           handle->scratch5 = 0;
                                           handle->scratch6 = 0;
                                           handle->scratch7 = 0;
                                         } break;
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_transpose_filter(libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  int ofm1, ifm1, kj, ki, ifm2, ofm2;

  /* check for filter type */
  if ( (type != LIBXSMM_DNN_REGULAR_FILTER) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  /* check if we have input, output and filter */
  if (handle->reg_filter == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have scratch */
  if (handle->scratch1 == 0) {
    status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  /* check that filter is in RSCK storage */
  if ( (handle->filter_format & LIBXSMM_DNN_TENSOR_FORMAT_RSCK) == 0 ) {
    status = LIBXSMM_DNN_ERR_MISMATCH_TENSOR;
    return status;
  }

  /* check that we are in FP32 */
  if ( handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 ) {
    LIBXSMM_VLA_DECL(6, float, wt, (float*)handle->reg_filter->data, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
    LIBXSMM_VLA_DECL(6, float, tr_wt, (float*)handle->scratch1, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);

    for (ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1) {
      for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
        for (kj=0; kj < handle->desc.R; ++kj) {
          for (ki=0; ki < handle->desc.S; ++ki) {
            for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
              for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kj, ki, ofm2, ifm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock) =
                  LIBXSMM_VLA_ACCESS(6, wt,  kj, ki, ifm1, ifm2, ofm1, ofm2, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
              }
            }
          }
        }
      }
    }
    handle->filter_transposed = 1;
    return status;
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
    return status;
  }
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_reduce_wu_filters(libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  int i, j, filter_size;

  /* check for filter type */
  if ( (type != LIBXSMM_DNN_GRADIENT_FILTER) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  /* check if we have input, output and filter */
  if (handle->grad_filter == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check that we are in FP32 */
  if (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
    if (handle->upd_use_external_reduce != 0) {
      float* filter_ptr = (float*)handle->grad_filter->data;
      /* calculate filter size */
      filter_size = handle->blocksofm * handle->blocksifm * handle->desc.R * handle->desc.S * handle->ofmblock * handle->ifmblock;

      for ( i = 0; i < handle->desc.threads; ++i ) {
        float* tmp_filter_ptr = ((float*)handle->scratch4) + ((size_t)i * filter_size);
        for ( j = 0; j < filter_size; ++j ) {
          filter_ptr[j] += tmp_filter_ptr[j];
        }
      }
    }
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_get_codegen_success(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           if (handle->code_fwd[0].ptr == 0) {
                                             status = LIBXSMM_DNN_WARN_FALLBACK;
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
                                           if (handle->code_bwd[0].ptr == 0) {
                                             status = LIBXSMM_DNN_WARN_FALLBACK;
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
                                           if (handle->code_upd[0].ptr == 0) {
                                             status = LIBXSMM_DNN_WARN_FALLBACK;
                                           }
                                         } break;
      default: {
                 status = LIBXSMM_DNN_ERR_INVALID_KIND;
               }
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_get_parallel_tasks(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind, unsigned int* num_tasks) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           *num_tasks = handle->desc.N * handle->blocksofm;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
                                           *num_tasks = handle->desc.N * handle->blocksifm;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
                                           if (handle->upd_use_thread_fil > 0) {
                                             *num_tasks = handle->desc.N * handle->blocksifm * handle->blocksofm;
                                           } else {
                                             *num_tasks = handle->blocksifm * handle->blocksofm;
                                           }
                                         } break;
      default: {
                 status = LIBXSMM_DNN_ERR_INVALID_KIND;
               }
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}

