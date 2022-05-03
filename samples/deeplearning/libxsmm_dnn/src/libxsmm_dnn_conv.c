/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/

#include <libxsmm_dnn_conv.h>
#include "libxsmm_dnn_conv_setup.h"

#define MIXED 0
#define KHWC 1
#define HWKC 2
#define CHWK 3
#define HWCK 4
#define  LIBXSMM_DNN_CONV_SETUP_USE_NTS

#define LIBXSMM_BLOCK64
#if defined LIBXSMM_BLOCK64
# define LIBXSMM_BLOCK_SIZE 64
#else
# define LIBXSMM_BLOCK_SIZE 32
#endif

/***********************************************************/
/* Helper functions for convolutions' general param setup */
/**********************************************************/

LIBXSMM_API_INLINE void  libxsmm_dnn_conv_get_feature_map_blocks( int C, int K, int* C_block, int* K_block, int* fm_lp_block, libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_blasint bc, libxsmm_blasint bk ) {
  int ifmblock = 0;
  int ofmblock = 0;
  int lp_block = 0;
  int tmp_max_c_block = bc;
  int tmp_max_k_block = bk;
  int tmp_block = 0;

  /* init libxsmm */
  LIBXSMM_INIT

  /* C */
  if ((libxsmm_target_archid == LIBXSMM_X86_AVX512_VL256_CLX) || (libxsmm_target_archid >= LIBXSMM_X86_AVX512_VL256_CPX)
          || (libxsmm_target_archid == LIBXSMM_X86_AVX512_VL256)
        ){
    tmp_max_c_block = LIBXSMM_BLOCK_SIZE;
  } else if ( /*((libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR) && (datatype_in == LIBXSMM_DATATYPE_BF16)) ||*/
       (libxsmm_target_archid < LIBXSMM_X86_AVX512 ) ) {
    tmp_max_c_block = 32;
  } else if ( libxsmm_target_archid == LIBXSMM_AARCH64_V81 ) {
    tmp_max_c_block = 16;
  }
  if ( C <= tmp_max_c_block ) {
    ifmblock = C;
  } else if (C % tmp_max_c_block == 0) {
    ifmblock = tmp_max_c_block;
  } else {
    for ( tmp_block = 1; tmp_block <= tmp_max_c_block; tmp_block *= 2 ) {
      if ( C % tmp_block == 0 ) ifmblock = tmp_block;
    }
  }

  /* K */
  if ((libxsmm_target_archid == LIBXSMM_X86_AVX512_VL256_CLX) || (libxsmm_target_archid >= LIBXSMM_X86_AVX512_VL256_CPX)
        || (libxsmm_target_archid == LIBXSMM_X86_AVX512_VL256)
      ){
    tmp_max_k_block = LIBXSMM_BLOCK_SIZE;
  } else if ( /*((libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR) && (datatype_in == LIBXSMM_DATATYPE_BF16)) ||*/
       (libxsmm_target_archid < LIBXSMM_X86_AVX512 ) ) {
    tmp_max_k_block = 32;
  } else if ( libxsmm_target_archid == LIBXSMM_AARCH64_V81 ) {
    tmp_max_k_block = 16;
  }
  if ( K <= tmp_max_k_block ) {
    ofmblock = K;
  } else if (K % tmp_max_k_block == 0) {
    ofmblock = tmp_max_k_block;
  } else {
    for ( tmp_block = 1; tmp_block <= tmp_max_k_block; tmp_block *= 2 ) {
      if ( K % tmp_block == 0 ) ofmblock = tmp_block;
    }
  }

  /* when do we need VNNI format? */
  if ( (datatype_in == LIBXSMM_DATATYPE_F32) && (datatype_out == LIBXSMM_DATATYPE_F32) ) {
    lp_block = 1;
  } else if ( (datatype_in == LIBXSMM_DATATYPE_BF16) && (datatype_out == LIBXSMM_DATATYPE_BF16) ) {
    lp_block = 2;
  } else if ( (datatype_in == LIBXSMM_DATATYPE_I16) && ((datatype_out == LIBXSMM_DATATYPE_I32) || (datatype_out == LIBXSMM_DATATYPE_F32)) ) {
    lp_block = 2;
  } else if (datatype_in == LIBXSMM_DATATYPE_I8) {
    lp_block = 4;
  } else {
    return;
  }

  *C_block = ifmblock;
  *K_block = ofmblock;
  *fm_lp_block = lp_block;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_ifmblock( libxsmm_dnn_conv_config* cfg ) {
  int result = 1;
  int ofm, lp;

  libxsmm_dnn_conv_get_feature_map_blocks( cfg->C, cfg->K, &result, &ofm, &lp, cfg->datatype_in, cfg->datatype_out, cfg->bc, cfg->bk );

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_ofmblock( libxsmm_dnn_conv_config* cfg ) {
  int result = 1;
  int ifm, lp;

  libxsmm_dnn_conv_get_feature_map_blocks( cfg->C, cfg->K, &ifm, &result, &lp, cfg->datatype_in, cfg->datatype_out, cfg->bc, cfg->bk );

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fm_lp_block( libxsmm_dnn_conv_config* cfg ) {
  int result = 1;
  int ifm, ofm;

  libxsmm_dnn_conv_get_feature_map_blocks( cfg->C, cfg->K, &ifm, &ofm, &result, cfg->datatype_in, cfg->datatype_out, cfg->bc, cfg->bk);

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fallback_loops_fwd( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  /* FIXME: For now fallback only if MB is not divisible by number of threads */
  if (cfg->N % cfg->threads != 0) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_blocksifm( libxsmm_dnn_conv_config* cfg ) {
  int result = cfg->C / cfg->ifmblock;
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_blocksofm( libxsmm_dnn_conv_config* cfg ) {
  int result = cfg->K / cfg->ofmblock;
  return result;
}

/**********************************************************/
/* Helper functions for FWD convolutions' parameter setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fwd_ofw_rb( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  result = cfg->ofw;
  if (cfg->ofw == 56) {
    result = 28;
  }
  if (cfg->datatype_in == LIBXSMM_DATATYPE_I8) {
    if (cfg->ofw % 2 == 0) {
      result = cfg->ofw/2;
    }
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_pack_input_fwd( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  /* Pack only for small images and when having large K to amortize, and we can only pack for 1x1 convolutions */
  if ((cfg->ofw <= 14) && (cfg->K > 512) && (cfg->R == 1) && (cfg->S == 1) && (cfg->u == 2) && (cfg->v == 2)) {
    result = 1;
  }

#if 0
  /* For SPR we allow packing more aggressively to generate more efficient BRGEMMs */
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) ) {
    if ((cfg->ofw <= 14) && (cfg->R == 1) && (cfg->S == 1) && (cfg->u == 2) && (cfg->v == 2)) {
      result = 1;
    }
  }
#endif

  /* Make sure we don't pack when minibatch is not divisible by number of threads since H is used potentially for parallelism */
  if (cfg->N != cfg->threads) {
    result = 0;
  }
  /* we don't pack for int8 */
  if (cfg->datatype_in == LIBXSMM_DATATYPE_I8) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fwd_ofh_rb( libxsmm_dnn_conv_config* cfg ) {
  int result = 1;
  /* Multiple rows for "small" images and 1x1 convolutions */
  if ((cfg->ofh <= 14) && (cfg->R == 1) && (cfg->S == 1) && (cfg->pad_w_out == 0) && (cfg->pad_h_out == 0)) {
    result = cfg->ofh;
  }

  /* In this case we will be using fallback generic loops, thus ofh_rb should be 1 */
  if ((cfg->N % cfg->threads != 0) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) {
    result = 1;
  }

#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) ) {
    if (cfg->ofw == 7 && cfg->ofh == 7 && cfg->R == 3 && cfg->S == 3) {
      result = 7;
    }
    if (cfg->ofw == 14 && cfg->ofh == 14 /*&& cfg->R == 3 && cfg->S == 3*/) {
      result = 2;
    }
  }
#endif

  /*  Make sure we don't use multiple rows when we don't pack input and convolutions are strided*/
  if ((cfg->pack_input == 0) && ((cfg->u !=1 ) || (cfg->v != 1))) {
    result = 1;
  }

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fwd_pixels_gemm( libxsmm_dnn_conv_config* cfg ) {
  int result = cfg->fwd_ofw_rb * cfg->fwd_ofh_rb;
  /* In the case below we calculate redundantly pixels in order to efficiently use AMX */
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) ) {
    if (cfg->R != 1 || cfg->S != 1) {
      if (cfg->ofw < 24) {
        result = (cfg->fwd_ofw_rb+2*cfg->pad_w) * (cfg->fwd_ofh_rb-2) + 2 * (cfg->fwd_ofw_rb+cfg->pad_w);
      }
    }
  }
#endif
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fwd_block_H( libxsmm_dnn_conv_config* cfg ) {
  int result = 14;

#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) ) {
    /* Spatial dimension block tuning for SPR */
    if ((cfg->ofh == 7 && cfg->u == 2) || (cfg->ofh == 14 && cfg->R != 3 ) ||  cfg->ofh == 27 || (cfg->ofh == 28 && cfg->R == 1) || cfg->ofh == 48 || cfg->ofh == 54 || cfg->ofh == 56 || cfg->ofh == 112 ) {
      result = 4;
    }
  } else {
    /* Block H only for large images  */
    if (cfg->ofh >= 28) {
      result = 4;
    }
    if (cfg->ofh == 28 && cfg->R == 3 ) {
      result = 14;
    }
  }
#else
  /* Block H only for large images  */
  if (cfg->ofh >= 28) {
    result = 4;
  }
  if (cfg->ofh == 28 && cfg->R == 3 ) {
    result = 14;
  }
#endif
  /* Make sure it is divisible bu the ofh_rb factor in the kernel */
  while ( result % cfg->fwd_ofh_rb != 0 ) {
    result--;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_blocksifm_blocking( libxsmm_dnn_conv_config* cfg ) {
  int result = 1;
  /* For 1x1 Convolutions bring in kernel all IFMs unless filters are huge*/
  if ((cfg->R == 1) && (cfg->S == 1) ) {
    result = cfg->blocksifm;
    if ((cfg->C >= 2048) && (cfg->K >= 512)) {
      result = 1;
    }
    if ( (cfg->target_archid < LIBXSMM_X86_AVX512_VL256) && (cfg->C >= 512) ) {
      result = 2;
    }
    if ( (cfg->target_archid < LIBXSMM_X86_AVX512_VL256) && (cfg->C >= 1024) ) {
      result = 4;
    }
  } else {
    result = 1;
    /* If small image can bring in more IFMS even if NOT 1x1 convolution */
    if (cfg->ofw <= 7) {
      result = 2;
    }
  }
  if (cfg->blocksifm % result != 0) {
    result = 1;
  }

  /* In case of SPR bring always in all accumulation */
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8))) {
    result = cfg->blocksifm;
  }
#endif

  if (cfg->datatype_in == LIBXSMM_DATATYPE_I8) {
    result = cfg->blocksifm;
  }

  if (cfg->datatype_in == LIBXSMM_DATATYPE_BF16) {
    result = cfg->blocksifm;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_loop_order_fwd( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  /* Switch to loop order 1 only if 1x1 convolution with "large" input image and "small" K */
  if ((cfg->H >= 28) && (cfg->R == 1) && (cfg->S == 1) && (cfg->C >=512) && (cfg->K <=512)) {
    result = 1;
  }
  if (cfg->ofw == 56 && cfg->R == 1 && cfg->C == 256 && cfg->K == 64 ) {
    result = 1;
  }
  if (cfg->ofw == 28 && cfg->R == 1) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_fwd_IFM( libxsmm_dnn_conv_config* cfg ) {
  int result = 8;
  if (cfg->ofw == 7 && cfg->C == 2048 && cfg->K == 512) {
    result = 4;
  }
  /* Make sure it is divisible by ifms in the kernel  */
  while (result % cfg->blocksifm_blocking != 0) {
    result++;
  }
  result = LIBXSMM_MIN(cfg->blocksifm, result);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_fwd_OFM( libxsmm_dnn_conv_config* cfg ) {
  int result = 8;
  if (cfg->ofw == 14 && cfg->K == 1024) {
    result = 16;
  }
  if (cfg->ofw == 7) {
    result = 16;
  }
  result = LIBXSMM_MIN(cfg->blocksofm, result);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_use_ofm_parallelization( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
#if 0
  /* Use "hybrid" minibatch/ofm parallelization if we have huge filters */
  if ((cfg->R >= 3) && (cfg->S >= 3) && (cfg->C >= 512) && (cfg->K >= 512)) {
    result = 1;
  }
#endif
  if ((cfg->ofw <= 7) && (cfg->C == 1024) && (cfg->K == 512)) {
    result = 1;
  }
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8))) {
    if (cfg->ofw == 7) {
      result = 1;
    }
  }
#endif
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_avoid_rim_fmas_fwd( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  /* Avoid rim FMA if the convolution is 3x3 (non-strided) and the image is "small" */
  if ((cfg->R == 3) && (cfg->S == 3) &&
      (cfg->u  == 1) && (cfg->v == 1) &&
      (cfg->pad_h_in == 1) && (cfg->pad_w_in == 1) &&
      (cfg->H == cfg->W) ) {
    if (cfg->ofw <= 28) {
      result = 1;
    }
    if (cfg->datatype_in == LIBXSMM_DATATYPE_I8) {
      result = 0;
    }
  }
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8))) {
    result = 0;
  }
#endif

  if (cfg->datatype_in == LIBXSMM_DATATYPE_BF16) {
    result = 0;
  }

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_shuffle_filter_accesses( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  /* Shuffle filter accesses only if "pure minibatch" parallelization and large filters are involved */
  if ((cfg->use_ofm_parallelization == 0) && (cfg->C > 512) && (cfg->K > 512)) {
    result = 1;
  }
  if (cfg->ofw == 7 && cfg->R == 3 && cfg->C == 512) {
    result = 1;
  }
  if (cfg->ofw == 7 && cfg->R == 1 && cfg->C == 512 && cfg->K == 2048) {
    result = 1;
  }
  if (cfg->ofw == 7 && cfg->R == 1 && cfg->C == 2048 && cfg->K == 512) {
    result = 1;
  }
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) )  {
    result = 0;
  }
#endif
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_avoid_acc_load( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  if ((cfg->overwrite_output) > 0) {
    if ((cfg->R == 1) && (cfg->S == 1)) {
      if (cfg->blocksifm_blocking == cfg->blocksifm) {
        result = 1;
      }
    } else {
      if ((cfg->blocksifm_blocking == cfg->blocksifm) && (cfg->avoid_fmas_in_rim == 0)) {
        result = 1;
      }
    }
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_init_fwd_gemm_flags( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;

#if defined(LIBXSMM_DNN_CONV_SETUP_USE_NTS)
  /* If large image and NOT already loaded in accumulators, tnen use streaming stores */
  if ((cfg->ofw >= 56) && (cfg->K >= 256) && (cfg->avoid_acc_load == 1) && (cfg->R == 1) && (cfg->S == 1)) {
    result = LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT;
  }
  if (cfg->ofw == 56 && cfg->C == 64 && cfg->K == 64 && cfg->R == 1) {
    result = LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT;
  }
  if (cfg->ofw == 56 && cfg->C == 256 && cfg->K == 64 && cfg->R == 1) {
    result = LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT;
  }
  /* Disable since the GEMM output is going to f32 scratch  */
  if (cfg->datatype_in == LIBXSMM_DATATYPE_BF16 || cfg->datatype_in == LIBXSMM_DATATYPE_I8) {
    result = 0;
  }
#else
  LIBXSMM_UNUSED(cfg);
#endif
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8))) {
    result = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  }
#endif

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fwd_padding_copy( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  if ( (cfg->pad_h != cfg->pad_h_in) || (cfg->pad_w != cfg->pad_w_in) ) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE void libxsmm_dnn_conv_setup_fwd_scratch( libxsmm_dnn_conv_config* cfg ) {
  cfg->fwd_packing_padding_scratch_size = 0;
  /* packing of input */
  if ( cfg->pack_input != 0 ) {
    cfg->fwd_packing_padding_scratch_size = (size_t)cfg->N * cfg->C *
      cfg->H/cfg->u *
      cfg->W/cfg->v *
      LIBXSMM_TYPESIZE(cfg->datatype_in);
  }
  /* logical padding with copying in the fly */
  if ( cfg->fwd_padding_copy != 0 ) {
    cfg->fwd_packing_padding_scratch_size = (size_t)cfg->N * cfg->C *
      (cfg->H + 2*cfg->pad_h) *
      (cfg->W + 2*cfg->pad_w) *
      LIBXSMM_TYPESIZE(cfg->datatype_in);
  }
  /* output buffer in high precision when we use BF16 */
  if ( ( cfg->datatype_in == LIBXSMM_DATATYPE_BF16 ) ||
      ( cfg->datatype_in == LIBXSMM_DATATYPE_I8 )      ) {
    cfg->fwd_lp_output_full_scratch_size = (size_t) LIBXSMM_MAX(cfg->threads * cfg->fwd_gemm_pixels * cfg->ofmblock * LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32), cfg->N * cfg->K * cfg->ofwp * cfg->ofhp * LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32));
    cfg->fwd_lp_output_block_scratch_size = (size_t)cfg->threads * cfg->fwd_ofw_rb *
      cfg->fwd_ofh_rb * cfg->ofmblock *
      LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32);
  } else {
    cfg->fwd_lp_output_full_scratch_size = 0;
    cfg->fwd_lp_output_block_scratch_size = 0;
  }
  /* align sizes to full cacheline */
  cfg->fwd_packing_padding_scratch_size += ( cfg->fwd_packing_padding_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->fwd_packing_padding_scratch_size % LIBXSMM_CACHELINE);
  cfg->fwd_lp_output_full_scratch_size += ( cfg->fwd_lp_output_full_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->fwd_lp_output_full_scratch_size % LIBXSMM_CACHELINE);
  cfg->fwd_lp_output_block_scratch_size += ( cfg->fwd_lp_output_block_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->fwd_lp_output_block_scratch_size % LIBXSMM_CACHELINE);

  /* set offsets */
  cfg->fwd_packing_padding_scratch_offset = 0;
  cfg->fwd_lp_output_full_scratch_offset = cfg->fwd_packing_padding_scratch_size;
  cfg->fwd_lp_output_block_scratch_offset = cfg->fwd_lp_output_full_scratch_offset +
    cfg->fwd_lp_output_full_scratch_size;

  /* set overall scratch size for forward */
  cfg->fwd_scratch_size = cfg->fwd_packing_padding_scratch_size +
    cfg->fwd_lp_output_full_scratch_size +
    cfg->fwd_lp_output_block_scratch_size;
}

/**********************************************************/
/* Helper functions for BWD convolutions' parameter setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fallback_loops_bwd( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  /* FIXME: Fallback if MB is not divisible by number of threads */
  if (cfg->N % cfg->threads != 0) {
    result = 1;
  }
  if (cfg->R == 1 && cfg->S == 1 && (cfg->pad_h != 0 ||  cfg->pad_w != 0)) {
    result = 1;
  }
  if ((cfg->R > 1 && cfg->pad_h == 0) || (cfg->S > 1 && cfg->pad_w == 0)) {
    result = 1;
  }
  if ((cfg->R > 1 && (cfg->pad_h_out == 0 || cfg->pad_h_in == 0)) ||
      (cfg->S > 1 && (cfg->pad_w_out == 0 || cfg->pad_w_in == 0))    ) {
    result = 1;
  }
  if ((cfg->R > 1 && cfg->u > 1) || (cfg->S > 1 && cfg->v > 1)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_bwd_ofw_rb( libxsmm_dnn_conv_config* cfg ) {
  int result = libxsmm_dnn_conv_setup_fwd_ofw_rb(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_bwd_ofh_rb( libxsmm_dnn_conv_config* cfg ) {
  int result = libxsmm_dnn_conv_setup_fwd_ofh_rb(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_bwd_pixels_gemm( libxsmm_dnn_conv_config* cfg ) {
  int result = cfg->bwd_ofw_rb * cfg->bwd_ofh_rb;
  /* In the case below we calculate redundantly pixels in order to efficiently use AMX */
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) ) {
    if (cfg->R != 1 || cfg->S != 1) {
      if (cfg->ofw < 24) {
        result = (cfg->bwd_ofw_rb+2*cfg->pad_w) * (cfg->bwd_ofh_rb-2) + 2 * (cfg->bwd_ofw_rb+cfg->pad_w);
      }
    }
  }
#endif
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_bwd_block_H( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  result = libxsmm_dnn_conv_setup_fwd_block_H(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_loop_order_bwd( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  result = libxsmm_dnn_conv_setup_loop_order_fwd(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_bwd_IFM( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  result = LIBXSMM_MIN(cfg->blocksifm, 16);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_bwd_OFM( libxsmm_dnn_conv_config* cfg ) {
  int result = 8;
  while (result % cfg->blocksofm_blocking != 0) {
    result++;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_pack_input_bwd( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  if ((cfg->u != 1) && (cfg->bwd_ofh_rb != 1)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_use_ifm_parallelization( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  if (cfg->ofw <= 7) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_avoid_rim_fmas_bwd( libxsmm_dnn_conv_config* cfg ) {
  int result = libxsmm_dnn_conv_setup_avoid_rim_fmas_fwd(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_blocksofm_blocking( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  if (cfg->R == 1 && cfg->S == 1) {
    result = cfg->blocksofm;
  } else {
    result = 1;
    if (cfg->R == 3 && cfg->S == 3 && cfg->ofh == 7 && cfg->ofw == 7) {
      result = 2;
    }
  }
#if  0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) ) {
    result = cfg->blocksofm;
  }
#endif

  if (cfg->blocksofm % result != 0) {
    result = 1;
  }

  if (cfg->datatype_in == LIBXSMM_DATATYPE_BF16) {
    result = cfg->blocksofm;
  }

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_init_bwd_gemm_flags( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
#if 0
  if ((cfg->target_archid == LIBXSMM_X86_AVX512_SPR) && (cfg->target_archid <= LIBXSMM_X86_ALLFEAT) && ((cfg->datatype_in == LIBXSMM_DATATYPE_BF16) || (cfg->datatype_in == LIBXSMM_DATATYPE_I8)) ) {
    result = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  }
#endif
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_spread_input_bwd( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  if (((cfg->u != 1) || (cfg->v != 1)) && (cfg->bwd_ofh_rb == 1)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_avoid_acc_load_bwd( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  if (cfg->overwrite_output > 0) {
    if ((cfg->R == 1) && (cfg->S == 1)) {
      if (cfg->blocksofm_blocking == cfg->blocksofm) {
        result = 1;
      }
    } else {
      if ((cfg->blocksofm_blocking == cfg->blocksofm) && (cfg->avoid_fmas_in_rim == 0)) {
        result = 1;
      }
    }
  }
  return result;
}

LIBXSMM_API_INLINE void libxsmm_dnn_conv_setup_bwd_scratch( libxsmm_dnn_conv_config* cfg ) {
  /* transpose of weights */
  cfg->bwd_filter_trans_scratch_size = (size_t)cfg->C * cfg->K *
    cfg->R * cfg->S *
    LIBXSMM_TYPESIZE(cfg->datatype_in);

  cfg->bwd_packing_padding_scratch_size = 0;
  /* packing of input */
  if ( cfg->pack_input_bwd != 0 ) {
    cfg->bwd_packing_padding_scratch_size = (size_t)cfg->N * cfg->C *
      cfg->ofhp * cfg->ofwp *
      LIBXSMM_TYPESIZE(cfg->datatype_in);
  }
  /* logical padding with copying in the fly */
  if ( cfg->use_fallback_bwd_loops != 0 ) {
    cfg->bwd_packing_padding_scratch_size = (size_t)cfg->threads * cfg->ifmblock *
      (cfg->H + 2*cfg->pad_h) *
      (cfg->W + 2*cfg->pad_w) *
      LIBXSMM_TYPESIZE(cfg->datatype_in);
  }
  /* input bufffer in high precision when we use BF16 */
  if ( cfg->datatype_in == LIBXSMM_DATATYPE_BF16 ) {
    cfg->bwd_lp_input_full_scratch_size = (size_t) LIBXSMM_MAX(cfg->threads * cfg->bwd_gemm_pixels * cfg->ifmblock * LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32), cfg->N * cfg->C * cfg->ifwp * cfg->ifhp * LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32));
    /* logical padding with copying in the fly */
    if ( cfg->use_fallback_bwd_loops != 0 ) {
      cfg->bwd_packing_padding_scratch_size = (size_t)cfg->threads * cfg->ifmblock *
        (cfg->H + 2*cfg->pad_h) *
        (cfg->W + 2*cfg->pad_w) *
        LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32);
    }
  } else {
    cfg->bwd_lp_input_full_scratch_size = 0;
  }
  /* align sizes to full cacheline */
  cfg->bwd_filter_trans_scratch_size += ( cfg->bwd_filter_trans_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->bwd_filter_trans_scratch_size % LIBXSMM_CACHELINE);
  cfg->bwd_packing_padding_scratch_size += ( cfg->bwd_packing_padding_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->bwd_packing_padding_scratch_size % LIBXSMM_CACHELINE);
  cfg->bwd_lp_input_full_scratch_size += ( cfg->bwd_lp_input_full_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->bwd_lp_input_full_scratch_size % LIBXSMM_CACHELINE);

  /* set offsets */
  cfg->bwd_filter_trans_scratch_offset = 0;
  cfg->bwd_packing_padding_scratch_offset = cfg->bwd_filter_trans_scratch_size;
  cfg->bwd_lp_input_full_scratch_offset = cfg->bwd_packing_padding_scratch_offset +
    cfg->bwd_packing_padding_scratch_size;

  /* set overall scratch size for forward */
  cfg->bwd_scratch_size = cfg->bwd_filter_trans_scratch_size +
    cfg->bwd_packing_padding_scratch_size +
    cfg->bwd_lp_input_full_scratch_size;
}

/**********************************************************/
/* Helper functions for UPD convolutions' parameter setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_weight_copies_upd( libxsmm_dnn_conv_config* cfg ) {
  int result = cfg->threads;
  if (cfg->ofw <= 14) {
    result = 9;
  }
  if (cfg->ofw == 14 && cfg->N == 92 && cfg->threads == 92) {
    result = 23;
  }
  if (cfg->ofw == 7 && cfg->N == 92 && cfg->threads == 92 && cfg->R == 3 && cfg->S == 3 && cfg->u == 1 && cfg->v == 1) {
    result = 23;
  }
  while (cfg->threads % result != 0) {
    result--;
  }
  /* FIXME: Hardcoded logic for N=27, N=26 */
  if (cfg->N == 27 && cfg->threads == 27 && cfg->R == 1 && cfg->ofw == 14 && cfg->u == 1) {
    result = 7;
  }
  if (((cfg->ofh == 14) || (cfg->ofw == 7 && cfg->u == 2)) && cfg->N == 26 && cfg->threads == 26) {
    result = 13;
  }
  if ((cfg->N != cfg->threads) && !(cfg->upd_linearized_tasklist == 0 && cfg->upd_use_batchreduce == 0)) {
    result = cfg->N;
  }
  /* Make sure a single copy when we use linearized-task view */
  if (cfg->upd_linearized_tasklist == 1) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE void libxsmm_dnn_conv_setup_bf16_upd_algorithms( libxsmm_dnn_conv_config* inout_cfg ) {
  libxsmm_dnn_conv_config res = *inout_cfg;
  int remainder_pixels, max_init_offset, max_compute_offset_input, input_compute_pad, accum_length_pixels, compute_pixels;
  const int multiple_target = 2;
  int IFHP = (res.upd_padding_copy == 1) ? res.ifhp + 2 * res.pad_h : res.ifhp;
  int IFWP = (res.upd_padding_copy == 1) ? res.ifwp + 2 * res.pad_w : res.ifwp;
  int OFHP = (res.upd_padding_copy == 1) ? res.ofhp + 2 * res.pad_h : res.ofhp;
  int OFWP = (res.upd_padding_copy == 1) ? res.ofwp + 2 * res.pad_w : res.ofwp;
  res.ifwp_extended = IFWP;
  res.upd_linearized_pixels = 1;
  if (res.S != 1 && res.v != 1) {
    res.upd_linearized_pixels = 0;
    res.upd_trans_w_only = 0;
  }
  if ((res.S != 1 && res.pad_w == 0) ||
      (res.R != 1 && res.pad_h == 0) ) {
    res.upd_linearized_pixels = 0;
    res.upd_trans_w_only = 0;
  }

  /* For large images facilitate the "large" transposes by blocking the pixel/reduction domains  */
  if (res.ofw >= 56 && res.ofh >=56 && res.R == 1 && res.S == 1 && res.u == 1 && res.v == 1) {
    res.upd_linearized_pixels = 0;
    res.upd_trans_w_only = 1;
  }

  res.on_the_fly_input_packing = 0;
  res.upd_pack_input_upfront = 0;
  res.use_hybrid_imgofm_parallelization = 0;
  res.upd_linearized_tasklist = 0;

  if (res.upd_linearized_pixels == 1) {
    /* Logistics to pad accumulation chainlength */
    compute_pixels = res.ofw * res.ofh + 2 * res.pad_w * (res.ofh-1);
    remainder_pixels = (compute_pixels % multiple_target == 0) ? 0 : (compute_pixels/multiple_target+1)*multiple_target - compute_pixels;
    accum_length_pixels = compute_pixels + remainder_pixels;

    /* In this case compact input upfront */
    if (res.R == 1 && res.S == 1 && (res.u != 1 || res.v != 1)) {
      res.upd_pack_input_upfront = 1;
    }

    /* Logistics for input transpose and additional pixel padding */
    max_init_offset = 2 * res.pad_h * IFWP + 2 * res.pad_w;
    max_compute_offset_input = max_init_offset + accum_length_pixels;
    input_compute_pad = (max_compute_offset_input > IFWP*IFHP) ? max_compute_offset_input - IFWP*IFHP : 0;
    res.input_pixels = IFWP * IFHP + input_compute_pad;
    if (res.upd_pack_input_upfront) {
      res.input_pixels = accum_length_pixels;
    }
    res.output_pixels = accum_length_pixels;
    res.pixel_blocking = accum_length_pixels;
    res.n_used_pixels = accum_length_pixels;
    res.compute_pixels = compute_pixels;

    res.use_intermediate_f32_wt_tensor = (res.pixel_blocking == res.n_used_pixels) ? 0 : 1;

    if (res.ofw <= 14) {
      res.use_hybrid_imgofm_parallelization = 1;
      res.weight_copies = libxsmm_dnn_conv_setup_weight_copies_upd(&res);
      if (res.ofw == 14 && res.K >= 1024) {
        res.use_hybrid_imgofm_parallelization = 0;
        res.weight_copies = res.threads;
      }
    } else {
      res.weight_copies = res.threads;
    }
  }

  if (res.upd_linearized_pixels == 0) {
    res.weight_copies = res.threads;
    if (res.v !=1) {
      res.on_the_fly_input_packing = 1;
    }
    remainder_pixels = (res.ofw % multiple_target == 0) ? 0 : (res.ofw/multiple_target+1)*multiple_target - res.ofw;
    res.ofwp_extended = OFWP + remainder_pixels;
    res.ifwp_extended = IFWP + remainder_pixels;
    if (res.ifwp_extended % 2 == 1) {
      res.ifwp_extended = res.ifwp_extended + 1;
    }
    res.output_pixels = OFHP * res.ofwp_extended;
    /* coverity[identical_branches] */
    res.batchreduce_h_pixels = (res.upd_trans_w_only) ? 1 : 1; /* TODO: identical_branches */
    res.use_intermediate_f32_wt_tensor = (res.batchreduce_h_pixels == res.ofh) ? 0 : 1;
  }

  if (res.N != res.threads) {
    res.use_intermediate_f32_wt_tensor = 1;
    res.use_hybrid_imgofm_parallelization = 0;
    res.weight_copies = LIBXSMM_MIN(res.N, res.threads);
  }

  *inout_cfg = res;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_loop_order_upd( libxsmm_dnn_conv_config* cfg ) {
  int result = 1;
  if (cfg->ofh == 28 && cfg->R == 1 && cfg->u == 1 && cfg->C == 128 && cfg->K == 512) {
    result = 0;
  }
  if (cfg->ofh == 28 && cfg->R == 3 && cfg->u == 1 && cfg->C == 128 && cfg->K == 128) {
    result = 0;
  }
  if (cfg->ofw == 28 && cfg->R == 1 && cfg->C == 256 && cfg->K == 512) {
    result = 0;
  }
  if (cfg->ofw == 14 && !(cfg->R == 1 && cfg->C == 1024 && cfg->K == 256)) {
    result = 0;
  }
  if (cfg->ofw == 7) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_pack_input_upd( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  /* Pack input only for very small images, 1x1 convs, with large K to amortize the relevant overhead */
  if ((cfg->ofh <= 7) && (cfg->R == 1) && (cfg->S == 1) && (cfg->u != 1) && (cfg->v != 1) && (cfg->K >= 2048)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_avoid_rim_fmas_upd( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  /* Avoid rim FMAs only for small images  */
  if ( (cfg->ofh <= 7) && (cfg->R == 3) && (cfg->S == 3) && (cfg->pad_w == 1) && (cfg->pad_h == 1)) {
    result = 1;
  }
  if (cfg->N != cfg->threads) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_upd_ofw_rb( libxsmm_dnn_conv_config* cfg ) {
  int result = 1;
  result = cfg->ofw;
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_upd_ofh_rb( libxsmm_dnn_conv_config* cfg ) {
  int result = 1;
  /* Restrict the reduction chain which is ofw_rb*ofh_rb*/
  if (cfg->ofh <= 28 ) {
    result = cfg->ofh;
  }
  /* In the following scenario with strided convolutions and non batch reduce kernel make sure we have ofh_rb = 1  */
  if ((cfg->u != 1) && (cfg->v != 1) && (cfg->upd_use_batchreduce == 0) && (cfg->upd_pack_input == 0)) {
    result = 1;
  }
  /* If using linearized taskview and have strided convs, make sure ofh_rb is 1.. */
  if (cfg->upd_linearized_tasklist == 1 && cfg->upd_avoid_rim_fmas == 0 && cfg->upd_pack_input == 0 && cfg->u != 1) {
    result = 1;
  }
  if (cfg->upd_linearized_tasklist == 1 && cfg->upd_use_batchreduce == 0 && (cfg->R != 1 || cfg->S != 1)) {
    result = 1;
  }
  if (cfg->upd_linearized_tasklist == 0 && cfg->upd_use_batchreduce == 0 && (cfg->R != 1 || cfg->S != 1)) {
    result = 1;
  }
  if (cfg->ofw == 56 && cfg->R == 1) {
    result = 2;
  }
  if (cfg->upd_linearized_tasklist == 1 && cfg->upd_use_batchreduce == 1 && cfg->upd_avoid_rim_fmas == 1) {
    result = cfg->ofh;
  }

  if ((cfg->N != cfg->threads) && (cfg->R > 1 || cfg->S > 1 ) && (cfg->u > 1 || cfg->v > 1 )) {
    result = 1;
  }

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_upd_IFM( libxsmm_dnn_conv_config* cfg ) {
  int result = 1;
  if (cfg->ofh == 56 && cfg->R == 1 && cfg->S == 1 && cfg->u == 1 && cfg->v == 1) {
    result = 4;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_upd_OFM( libxsmm_dnn_conv_config* cfg ) {
  int result = 1;
  LIBXSMM_UNUSED(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_img_batchreduce_block( libxsmm_dnn_conv_config* cfg ) {
  int result = 1;
  LIBXSMM_UNUSED(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_use_batchreduce_upd( libxsmm_dnn_conv_config* cfg ) {
  int result = 1;
  /* If W is large, no need for batchreduce kernel */
  if (cfg->ofw >= 56) {
    result = 0;
  }
  /* If we have packed the input, then disable batch-reduce GEMM */
  if (cfg->upd_pack_input == 1) {
    result = 0;
  }
  if (cfg->upd_linearized_tasklist == 1 && cfg->upd_avoid_rim_fmas == 0) {
    result = 0;
  }
  if (cfg->upd_linearized_tasklist == 1 && cfg->upd_avoid_rim_fmas == 1) {
    result = 1;
  }

  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_linearized_tasklist_upd( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  /* Use linearized task-list (i.e. no reduction) only if small images and large filters */
  if (cfg->ofh <= 10 && cfg->ofw <= 10) {
    result = 1;
  }
  if (cfg->ofw == 7 && cfg->N == 92 && cfg->threads == 92 && cfg->R == 3 && cfg->S == 3 && cfg->u == 1 && cfg->v == 1) {
    result = 0;
  }
  if (cfg->ofh == 14  && cfg->ofw == 14 && cfg->N == 23 && cfg->threads == 23) {
    result = 1;
  }
#if 0
  if ((cfg->blocksofm * cfg->blocksifm * cfg->R * cfg->S > (cfg->threads * 4)) && (cfg->ofh <= 56)) {
    result = 1;
  }
#endif
  if (cfg->u == 2 && cfg->v == 2 && cfg->K == 512) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_init_upd_gemm_flags( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  LIBXSMM_UNUSED(cfg);
  return result;
}

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_upd_padding_copy( libxsmm_dnn_conv_config* cfg ) {
  int result = 0;
  if ( (cfg->pad_h != cfg->pad_h_in) || (cfg->pad_w != cfg->pad_w_in) ) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INLINE void libxsmm_dnn_conv_setup_upd_scratch( libxsmm_dnn_conv_config* cfg ) {
  cfg->upd_packing_padding_scratch_size = 0;
  /* packing of input */
  if ( cfg->upd_pack_input != 0 ) {
    cfg->upd_packing_padding_scratch_size = (size_t)cfg->N * cfg->C *
      cfg->H/cfg->u *
      cfg->W/cfg->v *
      LIBXSMM_TYPESIZE(cfg->datatype_in);
  }
  /* logical padding with copying in the fly */
  if ( cfg->upd_padding_copy != 0 ) {
    cfg->upd_packing_padding_scratch_size = (size_t)cfg->N * cfg->C *
      (cfg->H + 2*cfg->pad_h) *
      (cfg->W + 2*cfg->pad_w) *
      LIBXSMM_TYPESIZE(cfg->datatype_in);
  }
  /* output/input buffer to transpose when we use bf16 */
  if ( cfg->datatype_in == LIBXSMM_DATATYPE_BF16 ) {
#if 0
    if  (cfg->target_archid >= LIBXSMM_X86_AVX512_SPR) {
      int OFHP = (cfg->upd_padding_copy == 1) ? cfg->ofhp + 2 * cfg->pad_h : cfg->ofhp;
      int IFHP = (cfg->upd_padding_copy == 1) ? cfg->ifhp + 2 * cfg->pad_h : cfg->ifhp;

      if (cfg->upd_linearized_pixels == 1) {
        cfg->upd_lp_output_full_scratch_size = (size_t) (cfg->N * cfg->output_pixels * cfg->K * sizeof(cfg->datatype_in));
        cfg->upd_lp_input_full_scratch_size = (size_t) (cfg->N * cfg->input_pixels * cfg->C * sizeof(cfg->datatype_in));
      }

      if (cfg->upd_linearized_pixels == 0) {
        cfg->upd_lp_output_full_scratch_size = (size_t) (cfg->N * OFHP * cfg->ofwp_extended * cfg->K * sizeof(cfg->datatype_in));
        cfg->upd_lp_input_full_scratch_size = (size_t) (cfg->N * IFHP * cfg->ifwp_extended * cfg->C * sizeof(cfg->datatype_in));
      }
    } else {
#endif
    if (1) {
      const int multiple_target = 2;
      int IFHP = (cfg->upd_padding_copy == 1) ? cfg->ifhp + 2 * cfg->pad_h : cfg->ifhp;
      int IFWP = (cfg->upd_padding_copy == 1) ? cfg->ifwp + 2 * cfg->pad_w : cfg->ifwp;
      int OFHP = (cfg->upd_padding_copy == 1) ? cfg->ofhp + 2 * cfg->pad_h : cfg->ofhp;
      int OFWP = (cfg->upd_padding_copy == 1) ? cfg->ofwp + 2 * cfg->pad_w : cfg->ofwp;

      if (cfg->upd_linearized_pixels == 1) {
        int compute_pixels = cfg->ofw * cfg->ofh + 2 * cfg->pad_w * (cfg->ofh-1);
        int remainder_pixels = (compute_pixels % multiple_target == 0) ? 0 : (compute_pixels/multiple_target+1)*multiple_target - compute_pixels;
        int accum_length_pixels = compute_pixels + remainder_pixels;

        int max_init_offset = 2 * cfg->pad_h * IFWP + 2 * cfg->pad_w;
        int max_compute_offset_input = max_init_offset + accum_length_pixels;
        int input_compute_pad = (max_compute_offset_input > IFWP*IFHP) ? max_compute_offset_input - IFWP*IFHP : 0;
        int input_pixels = IFWP * IFHP + input_compute_pad;

        if (cfg->upd_pack_input_upfront == 1) {
          input_pixels = accum_length_pixels;
        }

        cfg->upd_lp_output_full_scratch_size = (size_t) (cfg->N * accum_length_pixels * cfg->K * sizeof(cfg->datatype_in));
        cfg->upd_lp_input_full_scratch_size = (size_t) (cfg->N * input_pixels * cfg->C * sizeof(cfg->datatype_in));
      }

      if (cfg->upd_linearized_pixels == 0) {
        int remainder_pixels = (cfg->ofw % multiple_target == 0) ? 0 : (cfg->ofw/multiple_target+1)*multiple_target - cfg->ofw;
        int ofwp_extended = OFWP + remainder_pixels;
        int ifwp_extended = IFWP + remainder_pixels;

        cfg->upd_lp_output_full_scratch_size = (size_t) (cfg->N * OFHP * ofwp_extended * cfg->K * sizeof(cfg->datatype_in));
        cfg->upd_lp_input_full_scratch_size = (size_t) (cfg->N * IFHP * ifwp_extended * cfg->C * sizeof(cfg->datatype_in));
      }
    }
    cfg->upd_lp_filter_full_scratch_size = (size_t)cfg->R * cfg->S * cfg->C * cfg->K * cfg->threads *
      LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32);
  } else {
    cfg->upd_lp_output_full_scratch_size = 0;
    cfg->upd_lp_input_full_scratch_size = 0;
    cfg->upd_lp_filter_full_scratch_size = 0;
  }
  /* filter scratch */
  cfg->upd_filter_scratch_size = (size_t) cfg->R * cfg->S * cfg->C * cfg->K * LIBXSMM_MAX(cfg->threads, cfg->N) * sizeof(float);

  /* align sizes to full cacheline */
  cfg->upd_packing_padding_scratch_size += ( cfg->upd_packing_padding_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->upd_packing_padding_scratch_size % LIBXSMM_CACHELINE);
  cfg->upd_lp_output_full_scratch_size += ( cfg->upd_lp_output_full_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->upd_lp_output_full_scratch_size % LIBXSMM_CACHELINE);
  cfg->upd_lp_input_full_scratch_size += ( cfg->upd_lp_input_full_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->upd_lp_input_full_scratch_size % LIBXSMM_CACHELINE);
  cfg->upd_filter_scratch_size += ( cfg->upd_filter_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->upd_filter_scratch_size % LIBXSMM_CACHELINE);
  cfg->upd_lp_filter_full_scratch_size += ( cfg->upd_lp_filter_full_scratch_size % LIBXSMM_CACHELINE == 0 ) ? 0 :
    LIBXSMM_CACHELINE - (cfg->upd_lp_filter_full_scratch_size % LIBXSMM_CACHELINE);

  /* calculate offsets */
  cfg->upd_packing_padding_scratch_offset = 0;
  cfg->upd_lp_output_full_scratch_offset = cfg->upd_packing_padding_scratch_size;
  cfg->upd_lp_input_full_scratch_offset = cfg->upd_lp_output_full_scratch_offset + cfg->upd_lp_output_full_scratch_size;
  cfg->upd_filter_scratch_offset = cfg->upd_lp_input_full_scratch_offset + cfg->upd_lp_input_full_scratch_size;
  cfg->upd_lp_filter_full_scratch_offset = cfg->upd_filter_scratch_offset + cfg->upd_filter_scratch_size;

  /* set overall scratch size for update */
  cfg->upd_scratch_size = cfg->upd_packing_padding_scratch_size +
    cfg->upd_lp_output_full_scratch_size +
    cfg->upd_lp_input_full_scratch_size +
    cfg->upd_filter_scratch_size +
    cfg->upd_lp_filter_full_scratch_size;
}

LIBXSMM_API_INLINE void libxsmm_dnn_conv_generate_fwd_kernels( libxsmm_dnn_conv_config* inout_cfg) {
  libxsmm_dnn_conv_config res = *inout_cfg;
  if ( res.datatype_in == LIBXSMM_DATATYPE_F32 ) {
    libxsmm_blasint ldx;
    libxsmm_blasint ldA;
    libxsmm_blasint ldC;
    float beta;
    libxsmm_meltw_unary_shape unary_shape;
    libxsmm_meltw_binary_shape binary_shape;
    libxsmm_blasint stride_in;
    libxsmm_blasint stride_out;
    libxsmm_gemm_shape l_shape;
    libxsmm_gemm_batch_reduce_config l_brconfig;
    libxsmm_gemm_ext_unary_argops l_argops;
    libxsmm_gemm_ext_binary_postops l_postops;
    libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
    libxsmm_bitfield l_prefetch_flags = 0;
    int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
    int brgemm_pf_oob = 0;
    const char *const env_brgemm_pf_oob = getenv("BRGEMM_PF_OOB");

    res.A_offsets = NULL;
    res.B_offsets = NULL;

    res.block_fwd_ofm = 1;
    res.block_fwd_oj = res.fwd_ofh_rb;
    ldx = (res.pack_input == 1) ? (libxsmm_blasint)res.ifmblock : (libxsmm_blasint)res.v*res.ifmblock;
    ldA = res.ofmblock;
    ldC = res.ofmblock;
    beta = (res.avoid_acc_load) ? (float)0.0 : (float)1.0;

    l_flags |= res.fwd_flags;
    l_flags |= ( beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
    if ( 0 == env_brgemm_pf_oob ) {
    } else {
      brgemm_pf_oob = atoi(env_brgemm_pf_oob);
    }
    if (brgemm_pf_oob > 0) {
      prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB);
    }
    l_prefetch_flags = prefetch_mode;

    /* Strided kernel  */
    libxsmm_blasint IFW = (res.pack_input == 1) ? res.ofwp : res.ifwp;
    libxsmm_blasint IFH = (res.pack_input == 1) ? res.ofhp : res.ifhp;
    libxsmm_blasint stride_a = res.R * res.S * res.ifmblock * res.ofmblock * sizeof(float);
    libxsmm_blasint stride_b = IFW * IFH * res.ifmblock * sizeof(float);

    l_shape.m = res.ofmblock;
    l_shape.n = res.fwd_gemm_pixels;
    l_shape.k = res.ifmblock;
    l_shape.lda = ldA;
    l_shape.ldb = ldx;
    l_shape.ldc = ldC;
    l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
    l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
    l_shape.out_type  = LIBXSMM_DATATYPE_F32;
    l_shape.comp_type = LIBXSMM_DATATYPE_F32;
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = stride_a;
    l_brconfig.br_stride_b_hint = stride_b;
    l_brconfig.br_unroll_hint   = res.blocksifm_blocking;

    memset( &l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops) );
    memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );

    if ((res.fuse_type & LIBXSMM_DNN_CONV_ELTWISE_FUSE_BIAS) > 0) {
      l_postops.d_in_type      = LIBXSMM_DATATYPE_F32;
      l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
      l_postops.d_binary_type  = LIBXSMM_MELTW_TYPE_BINARY_ADD;
      l_postops.ldd            = ldC;
    }
    if ((res.fuse_type & LIBXSMM_DNN_CONV_ELTWISE_FUSE_RELU) > 0) {
      l_argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_RELU;
      l_argops.ldcp           = ldC;
    }

    /* Stride-based kernels  */
    res.fwd_compute_kernel_strd_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if (  res.fwd_compute_kernel_strd_f32.gemm  == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel_strd_f32 failed. Bailing...!\n");
      exit(-1);
    }
    res.fwd_compute_kernel_strd_fused_f32.gemm_ext = libxsmm_dispatch_brgemm_ext_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig,
        l_argops, l_postops );
    if (  res.fwd_compute_kernel_strd_fused_f32.gemm_ext == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel_strd_fused_f32 failed. Bailing...!\n");
      exit(-1);
    }
    if (res.avoid_fmas_in_rim > 0) {
      l_shape.n =  res.fwd_ofh_rb*(res.fwd_ofw_rb-1);
      res.fwd_compute_kernel2_strd_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      if (  res.fwd_compute_kernel2_strd_f32.gemm  == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel_strd_f32 failed. Bailing...!\n");
        exit(-1);
      }
    }

    /* Offset-based kernel */
    IFW = (res.fwd_padding_copy == 1) ? res.ifwp + 2*res.pad_w : ( (res.pack_input == 1) ? res.ofwp : res.ifwp );
    IFH = (res.fwd_padding_copy == 1) ? res.ifhp + 2*res.pad_h : ( (res.pack_input == 1) ? res.ofhp : res.ifhp );
    int n_blocks = res.R * res.S * res.blocksifm_blocking;
    int i = 0, ifm, ki, kj;
    l_shape.n = res.fwd_gemm_pixels;

    if ((res.avoid_fmas_in_rim == 0) && (res.R > 1 || res.S > 1)) {
      res.A_offsets = (unsigned long long*) libxsmm_aligned_malloc(n_blocks * sizeof(unsigned long long), 2097152);
      res.B_offsets = (unsigned long long*) libxsmm_aligned_malloc(n_blocks * sizeof(unsigned long long), 2097152);
      for (ifm = 0; ifm < res.blocksifm_blocking; ifm++) {
        for (kj = 0; kj < res.R; kj++) {
          for (ki = 0; ki < res.S; ki++) {
            res.A_offsets[i] = (ifm * res.R * res.S * res.ifmblock * res.ofmblock +
                kj * res.S * res.ifmblock * res.ofmblock +
                ki * res.ifmblock * res.ofmblock) * sizeof(float);
            res.B_offsets[i] = (ifm * IFH * IFW * res.ifmblock +
                kj * IFW * res.ifmblock +
                ki * res.ifmblock) * sizeof(float);
            i++;
          }
        }
      }

      l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
      l_brconfig.br_stride_a_hint = 0;
      l_brconfig.br_stride_b_hint = 0;

      res.fwd_compute_kernel_offs_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      if (  res.fwd_compute_kernel_offs_f32.gemm  == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel_offs_f32 failed. Bailing...!\n");
        exit(-1);
      }
      res.fwd_compute_kernel_offs_fused_f32.gemm_ext = libxsmm_dispatch_brgemm_ext_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig,
          l_argops, l_postops );
      if (  res.fwd_compute_kernel_offs_fused_f32.gemm_ext  == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel_offs_fused_f32 failed. Bailing...!\n");
        exit(-1);
      }
    }

    /* Eltwise TPPs */
    stride_in             = res.ifmblock * res.v;
    stride_out            = res.ifmblock;
    unary_shape.m         = res.ifmblock;
    unary_shape.n         = res.ofw;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_F32;
    unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
    unary_shape.out_type  = LIBXSMM_DATATYPE_F32;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    res.strided_copy_kernel_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.strided_copy_kernel_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP strided_copy_kernel_f32 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ifmblock;
    stride_out            = res.ifmblock;
    unary_shape.m         = res.ifmblock;
    unary_shape.n         = 1;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    res.ifmblock_copy_kernel_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.ifmblock_copy_kernel_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP ifmblock_copy_kernel_f32 failed. Bailing...!\n");
      exit(-1);
    }

    res.ifmblock_zero_kernel_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.ifmblock_zero_kernel_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP ifmblock_zero_kernel_f32 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ofmblock;
    stride_out            = res.ofmblock;
    unary_shape.m         = res.ofmblock;
    unary_shape.n         = 1;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    res.ofmblock_zero_kernel_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.ofmblock_zero_kernel_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP ofmblock_zero_kernel_f32 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ofwp * res.ofmblock;
    stride_out            = res.ofw * res.ofmblock;
    unary_shape.m         = res.ofw * res.ofmblock;
    unary_shape.n         = 1;
    unary_shape.ldi       = stride_out;
    unary_shape.ldo       = stride_out;
    res.ofw_x_ofmblock_zero_kernel_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.ofw_x_ofmblock_zero_kernel_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP ofw_x_ofmblock_zero_kernel_f32 failed. Bailing...!\n");
      exit(-1);
    }

    stride_out            = res.ofwp * res.ofmblock;
    unary_shape.m         = res.ofw * res.ofmblock;
    unary_shape.n         = res.ofh;
    unary_shape.ldi       = stride_out;
    unary_shape.ldo       = stride_out;
    res.ofh_x_ofw_x_ofmblock_zero_kernel_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.ofh_x_ofw_x_ofmblock_zero_kernel_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP ofh_x_ofw_x_ofmblock_zero_kernel_f32 failed. Bailing...!\n");
      exit(-1);
    }


    if ((res.fuse_type & LIBXSMM_DNN_CONV_ELTWISE_FUSE_RELU) > 0) {
      stride_in             = res.ofmblock;
      stride_out            = res.ofmblock;
      unary_shape.m         = res.ofmblock;
      unary_shape.n         = res.fwd_ofw_rb;
      unary_shape.ldi       = stride_in;
      unary_shape.ldo       = stride_out;
      res.relu_kernel_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_RELU, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
      if (  res.relu_kernel_f32  == NULL ) {
        fprintf( stderr, "JIT for TPP relu_kernel_f32 failed. Bailing...!\n");
        exit(-1);
      }
    }

    if ((res.fuse_type & LIBXSMM_DNN_CONV_ELTWISE_FUSE_BIAS) > 0) {
      stride_in              = res.ofmblock;
      stride_out             = res.ofmblock;
      binary_shape.m         = res.ofmblock;
      binary_shape.n         = res.fwd_ofw_rb;
      binary_shape.in0_type  = LIBXSMM_DATATYPE_F32;
      binary_shape.in1_type  = LIBXSMM_DATATYPE_F32;
      binary_shape.comp_type = LIBXSMM_DATATYPE_F32;
      binary_shape.out_type  = LIBXSMM_DATATYPE_F32;
      binary_shape.ldi       = stride_in;
      binary_shape.ldi2      = stride_in;
      binary_shape.ldo       = stride_out;
      res.colbias_add_kernel_f32 = libxsmm_dispatch_meltw_binary_v2( LIBXSMM_MELTW_TYPE_BINARY_ADD, binary_shape, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) ;
      if (  res.colbias_add_kernel_f32  == NULL ) {
        fprintf( stderr, "JIT for TPP colbias_add_kernel_f32 failed. Bailing...!\n");
        exit(-1);
      }
    }
  }

  if ( res.datatype_in == LIBXSMM_DATATYPE_BF16 ) {
    libxsmm_blasint ldx;
    libxsmm_blasint ldA;
    libxsmm_blasint ldC;
    float beta;
    libxsmm_meltw_unary_shape unary_shape;
    libxsmm_meltw_binary_shape binary_shape;
    libxsmm_blasint stride_in;
    libxsmm_blasint stride_out;
    libxsmm_gemm_shape l_shape;
    libxsmm_gemm_batch_reduce_config l_brconfig;
    libxsmm_gemm_ext_unary_argops l_argops;
    libxsmm_gemm_ext_binary_postops l_postops;
    libxsmm_bitfield l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
    libxsmm_bitfield l_prefetch_flags = 0;
    int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
    int brgemm_pf_oob = 0;
    const char *const env_brgemm_pf_oob = getenv("BRGEMM_PF_OOB");

    res.A_offsets = NULL;
    res.B_offsets = NULL;

    res.block_fwd_ofm = 1;
    res.block_fwd_oj = res.fwd_ofh_rb;
    ldx = (res.pack_input == 1) ? (libxsmm_blasint)res.ifmblock : (libxsmm_blasint)res.v*res.ifmblock;
    ldA = res.ofmblock;
    ldC = res.ofmblock;
    beta = (res.avoid_acc_load) ? (float)0.0 : (float)1.0;

    l_flags |= res.fwd_flags;
    l_flags |= ( beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
    if ( 0 == env_brgemm_pf_oob ) {
    } else {
      brgemm_pf_oob = atoi(env_brgemm_pf_oob);
    }
    if (brgemm_pf_oob > 0) {
      prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB);
    }
    l_prefetch_flags = prefetch_mode;

    /* Strided kernel  */
    libxsmm_blasint IFW = (res.pack_input == 1) ? res.ofwp : res.ifwp;
    libxsmm_blasint IFH = (res.pack_input == 1) ? res.ofhp : res.ifhp;
    libxsmm_blasint stride_a = res.R * res.S * res.ifmblock * res.ofmblock * sizeof(libxsmm_bfloat16);
    libxsmm_blasint stride_b = IFW * IFH * res.ifmblock * sizeof(libxsmm_bfloat16);

    l_shape.m = res.ofmblock;
    l_shape.n = res.fwd_gemm_pixels;
    l_shape.k = res.ifmblock;
    l_shape.lda = ldA;
    l_shape.ldb = ldx;
    l_shape.ldc = ldC;
    l_shape.a_in_type = LIBXSMM_DATATYPE_BF16;
    l_shape.b_in_type = LIBXSMM_DATATYPE_BF16;
    l_shape.out_type  = LIBXSMM_DATATYPE_BF16;
    l_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = stride_a;
    l_brconfig.br_stride_b_hint = stride_b;
    l_brconfig.br_unroll_hint   = res.blocksifm_blocking;

    memset( &l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops) );
    memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );

    if ((res.fuse_type & LIBXSMM_DNN_CONV_ELTWISE_FUSE_BIAS) > 0) {
      l_postops.d_in_type      = LIBXSMM_DATATYPE_BF16;
      l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
      l_postops.d_binary_type  = LIBXSMM_MELTW_TYPE_BINARY_ADD;
      l_postops.ldd            = ldC;
    }
    if ((res.fuse_type & LIBXSMM_DNN_CONV_ELTWISE_FUSE_RELU) > 0) {
      l_argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_RELU;
      l_argops.ldcp           = ldC;
    }

    /* Stride-based kernels  */
    res.fwd_compute_kernel_strd_bf16.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if (  res.fwd_compute_kernel_strd_bf16.gemm  == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel_strd_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    l_shape.out_type  = LIBXSMM_DATATYPE_F32;
    res.fwd_compute_kernel_strd_bf16f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if (  res.fwd_compute_kernel_strd_bf16f32.gemm  == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel_strd_bf16f32 failed. Bailing...!\n");
      exit(-1);
    }
    l_shape.out_type  = LIBXSMM_DATATYPE_BF16;

    res.fwd_compute_kernel_strd_fused_bf16.gemm_ext = libxsmm_dispatch_brgemm_ext_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig,
        l_argops, l_postops );
    if (  res.fwd_compute_kernel_strd_fused_bf16.gemm_ext == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel_strd_fused_bf16 failed. Bailing...!\n");
      exit(-1);
    }
    if (res.avoid_fmas_in_rim > 0) {
      l_shape.n =  res.fwd_ofh_rb*(res.fwd_ofw_rb-1);
      res.fwd_compute_kernel2_strd_bf16.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      if (  res.fwd_compute_kernel2_strd_bf16.gemm  == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel_strd_bf16 failed. Bailing...!\n");
        exit(-1);
      }

      l_shape.out_type  = LIBXSMM_DATATYPE_F32;
      res.fwd_compute_kernel2_strd_bf16f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      if (  res.fwd_compute_kernel2_strd_bf16f32.gemm  == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel2_strd_bf16f32 failed. Bailing...!\n");
        exit(-1);
      }
      l_shape.out_type  = LIBXSMM_DATATYPE_BF16;
    }

    /* Offset-based kernel */
    IFW = (res.fwd_padding_copy == 1) ? res.ifwp + 2*res.pad_w : ( (res.pack_input == 1) ? res.ofwp : res.ifwp );
    IFH = (res.fwd_padding_copy == 1) ? res.ifhp + 2*res.pad_h : ( (res.pack_input == 1) ? res.ofhp : res.ifhp );
    int n_blocks = res.R * res.S * res.blocksifm_blocking;
    int i = 0, ifm, ki, kj;
    l_shape.n = res.fwd_gemm_pixels;

    if ((res.avoid_fmas_in_rim == 0) && (res.R > 1 || res.S > 1)) {
      res.A_offsets = (unsigned long long*) libxsmm_aligned_malloc(n_blocks * sizeof(unsigned long long), 2097152);
      res.B_offsets = (unsigned long long*) libxsmm_aligned_malloc(n_blocks * sizeof(unsigned long long), 2097152);
      for (ifm = 0; ifm < res.blocksifm_blocking; ifm++) {
        for (kj = 0; kj < res.R; kj++) {
          for (ki = 0; ki < res.S; ki++) {
            res.A_offsets[i] = (ifm * res.R * res.S * res.ifmblock * res.ofmblock +
                kj * res.S * res.ifmblock * res.ofmblock +
                ki * res.ifmblock * res.ofmblock) * sizeof(libxsmm_bfloat16);
            res.B_offsets[i] = (ifm * IFH * IFW * res.ifmblock +
                kj * IFW * res.ifmblock +
                ki * res.ifmblock) * sizeof(libxsmm_bfloat16);
            i++;
          }
        }
      }

      l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
      l_brconfig.br_stride_a_hint = 0;
      l_brconfig.br_stride_b_hint = 0;

      res.fwd_compute_kernel_offs_bf16.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      if (  res.fwd_compute_kernel_offs_bf16.gemm  == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel_offs_bf16 failed. Bailing...!\n");
        exit(-1);
      }
      res.fwd_compute_kernel_offs_fused_bf16.gemm_ext = libxsmm_dispatch_brgemm_ext_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig,
          l_argops, l_postops );
      if (  res.fwd_compute_kernel_offs_fused_bf16.gemm_ext  == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel_offs_fused_bf16 failed. Bailing...!\n");
        exit(-1);
      }
    }

    /* Eltwise TPPs */
    stride_in             = res.ifmblock * res.v;
    stride_out            = res.ifmblock;
    unary_shape.m         = res.ifmblock;
    unary_shape.n         = res.ofw;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
    unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    res.strided_copy_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.strided_copy_kernel_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP strided_copy_kernel_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ifmblock;
    stride_out            = res.ifmblock;
    unary_shape.m         = res.ifmblock;
    unary_shape.n         = 1;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    res.ifmblock_copy_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.ifmblock_copy_kernel_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP ifmblock_copy_kernel_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    res.ifmblock_zero_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.ifmblock_zero_kernel_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP ifmblock_zero_kernel_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ofmblock;
    stride_out            = res.ofmblock;
    unary_shape.m         = res.ofmblock;
    unary_shape.n         = 1;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    res.ofmblock_zero_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.ofmblock_zero_kernel_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP ofmblock_zero_kernel_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ofwp * res.ofmblock;
    stride_out            = res.ofw * res.ofmblock;
    unary_shape.m         = res.ofw * res.ofmblock;
    unary_shape.n         = 1;
    unary_shape.ldi       = stride_out;
    unary_shape.ldo       = stride_out;
    res.ofw_x_ofmblock_zero_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.ofw_x_ofmblock_zero_kernel_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP ofw_x_ofmblock_zero_kernel_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    stride_out            = res.ofwp * res.ofmblock;
    unary_shape.m         = res.ofw * res.ofmblock;
    unary_shape.n         = res.ofh;
    unary_shape.ldi       = stride_out;
    unary_shape.ldo       = stride_out;
    res.ofh_x_ofw_x_ofmblock_zero_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.ofh_x_ofw_x_ofmblock_zero_kernel_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP ofh_x_ofw_x_ofmblock_zero_kernel_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    unary_shape.in0_type   = LIBXSMM_DATATYPE_F32;
    unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
    unary_shape.out_type  = LIBXSMM_DATATYPE_F32;

    stride_out            = res.ofwp * res.ofmblock;
    unary_shape.m         = res.ofw * res.ofmblock;
    unary_shape.n         = res.ofh;
    unary_shape.ldi       = stride_out;
    unary_shape.ldo       = stride_out;
    res.ofh_x_ofw_x_ofmblock_zero_kernel_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.ofh_x_ofw_x_ofmblock_zero_kernel_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP ofh_x_ofw_x_ofmblock_zero_kernel_f32 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ofwp * res.ofmblock;
    stride_out            = res.ofw * res.ofmblock;
    unary_shape.m         = res.ofw * res.ofmblock;
    unary_shape.n         = 1;
    unary_shape.ldi       = stride_out;
    unary_shape.ldo       = stride_out;
    res.ofw_x_ofmblock_zero_kernel_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.ofw_x_ofmblock_zero_kernel_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP ofw_x_ofmblock_zero_kernel_f32 failed. Bailing...!\n");
      exit(-1);
    }
    unary_shape.in0_type   = LIBXSMM_DATATYPE_F32;
    unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;
    stride_out            = res.ofwp * res.ofmblock;
    unary_shape.m         = res.ofw * res.ofmblock;
    unary_shape.n         = 1;
    unary_shape.ldi       = stride_out;
    unary_shape.ldo       = stride_out;
    res.cvt_kernel_fp32bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.cvt_kernel_fp32bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP cvt_kernel_fp32bf16 failed. Bailing...!\n");
      exit(-1);
    }
    unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
    unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;

    if ((res.fuse_type & LIBXSMM_DNN_CONV_ELTWISE_FUSE_RELU) > 0) {
      stride_in             = res.ofmblock;
      stride_out            = res.ofmblock;
      unary_shape.m         = res.ofmblock;
      unary_shape.n         = res.fwd_ofw_rb;
      unary_shape.ldi       = stride_in;
      unary_shape.ldo       = stride_out;
      res.relu_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_RELU, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
      if (  res.relu_kernel_bf16  == NULL ) {
        fprintf( stderr, "JIT for TPP relu_kernel_bf16 failed. Bailing...!\n");
        exit(-1);
      }
    }

    if ((res.fuse_type & LIBXSMM_DNN_CONV_ELTWISE_FUSE_BIAS) > 0) {
      stride_in              = res.ofmblock;
      stride_out             = res.ofmblock;
      binary_shape.m         = res.ofmblock;
      binary_shape.n         = res.fwd_ofw_rb;
      binary_shape.in0_type  = LIBXSMM_DATATYPE_BF16;
      binary_shape.in1_type  = LIBXSMM_DATATYPE_BF16;
      binary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
      binary_shape.out_type  = LIBXSMM_DATATYPE_BF16;
      binary_shape.ldi       = stride_in;
      binary_shape.ldi2      = stride_in;
      binary_shape.ldo       = stride_out;
      res.colbias_add_kernel_bf16 = libxsmm_dispatch_meltw_binary_v2( LIBXSMM_MELTW_TYPE_BINARY_ADD, binary_shape, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) ;
      if (  res.colbias_add_kernel_bf16  == NULL ) {
        fprintf( stderr, "JIT for TPP colbias_add_kernel_bf16 failed. Bailing...!\n");
        exit(-1);
      }
    }
  }

  *inout_cfg = res;
}

LIBXSMM_API_INLINE void libxsmm_dnn_conv_generate_bwd_kernels( libxsmm_dnn_conv_config* inout_cfg) {
  libxsmm_dnn_conv_config res = *inout_cfg;
  if ( res.datatype_in == LIBXSMM_DATATYPE_F32 ) {
    libxsmm_blasint ldA;
    libxsmm_blasint ldB;
    libxsmm_blasint ldC;
    float beta;
    libxsmm_meltw_unary_shape unary_shape;
    libxsmm_blasint stride_in;
    libxsmm_blasint stride_out;
    libxsmm_gemm_shape l_shape;
    libxsmm_gemm_batch_reduce_config l_brconfig;
    libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
    libxsmm_bitfield l_prefetch_flags = 0;
    int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
    int brgemm_pf_oob = 0;
    const char *const env_brgemm_pf_oob = getenv("BRGEMM_PF_OOB");

    res.A_offsets_bwd = NULL;
    res.B_offsets_bwd = NULL;

    ldB = (libxsmm_blasint)res.ofmblock;
    ldA = (libxsmm_blasint)res.ifmblock;
    ldC = (res.spread_input_bwd == 1) ? (libxsmm_blasint)(res.ifmblock * res.v) : (libxsmm_blasint)res.ifmblock;
    beta = (res.avoid_acc_load_bwd ? 0.f : 1.f);

    l_flags |= res.bwd_flags;
    l_flags |= ( beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
    if ( 0 == env_brgemm_pf_oob ) {
    } else {
      brgemm_pf_oob = atoi(env_brgemm_pf_oob);
    }
    if (brgemm_pf_oob > 0) {
      prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB);
    }
    l_prefetch_flags = prefetch_mode;

    /* Strided kernel  */
    libxsmm_blasint stride_a = res.R * res.S * res.ifmblock * res.ofmblock * sizeof(float);
    libxsmm_blasint stride_b = res.ofwp * res.ofhp * res.ofmblock * sizeof(float);

    l_shape.m = res.ifmblock;
    l_shape.n = res.bwd_ofh_rb*res.bwd_ofw_rb;
    l_shape.k = res.ofmblock;
    l_shape.lda = ldA;
    l_shape.ldb = ldB;
    l_shape.ldc = ldC;
    l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
    l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
    l_shape.out_type  = LIBXSMM_DATATYPE_F32;
    l_shape.comp_type = LIBXSMM_DATATYPE_F32;
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = stride_a;
    l_brconfig.br_stride_b_hint = stride_b;
    l_brconfig.br_unroll_hint   = res.blocksofm_blocking;

    /* Stride-based kernels  */
    res.bwd_compute_kernel_strd_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if (  res.bwd_compute_kernel_strd_f32.gemm  == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP bwd_compute_kernel_strd_f32 failed. Bailing...!\n");
      exit(-1);
    }

    if (res.avoid_fmas_in_rim > 0) {
      l_shape.n =  res.bwd_ofh_rb*(res.bwd_ofw_rb-1);
      res.bwd_compute_kernel2_strd_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      if (  res.bwd_compute_kernel2_strd_f32.gemm  == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP bwd_compute_kernel2_strd_f32 failed. Bailing...!\n");
        exit(-1);
      }
    }

    /* Offset-based kernel */
    int n_blocks = res.R * res.S * res.blocksofm_blocking;
    int i = 0, ofm, ki, kj;
    l_shape.n = res.bwd_ofh_rb*res.bwd_ofw_rb;

    if ((res.avoid_fmas_in_rim == 0) && (res.R > 1 || res.S > 1)) {
      res.A_offsets_bwd= (unsigned long long*) libxsmm_aligned_malloc(n_blocks * sizeof(unsigned long long), 2097152);
      res.B_offsets_bwd = (unsigned long long*) libxsmm_aligned_malloc(n_blocks * sizeof(unsigned long long), 2097152);
      for (ofm = 0; ofm < res.blocksofm_blocking; ofm++) {
        for (kj = 0; kj < res.R; kj++) {
          for (ki = 0; ki < res.S; ki++) {
            res.A_offsets_bwd[i] = (ofm * res.R * res.S * res.ifmblock * res.ofmblock +
                kj * res.S * res.ifmblock * res.ofmblock +
                ki * res.ifmblock * res.ofmblock) * sizeof(float);
            res.B_offsets_bwd[i] = (ofm * res.ofhp * res.ofwp * res.ofmblock +
                kj * res.ofwp * res.ofmblock +
                ki * res.ofmblock) * sizeof(float);
            i++;
          }
        }
      }

      l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
      l_brconfig.br_stride_a_hint = 0;
      l_brconfig.br_stride_b_hint = 0;

      res.bwd_compute_kernel_offs_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      if (  res.bwd_compute_kernel_offs_f32.gemm  == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP bwd_compute_kernel_offs_f32 failed. Bailing...!\n");
        exit(-1);
      }
    }

    /* Regular GEMM for fallback codepath */
    ldC = (libxsmm_blasint)(res.v*res.ifmblock);
    l_shape.m = res.ifmblock;
    l_shape.n = res.ofw;
    l_shape.k = res.ofmblock;
    l_shape.lda = res.ifmblock;
    l_shape.ldb = res.ofmblock;
    l_shape.ldc = ldC;
    l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
    l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
    l_shape.out_type  = LIBXSMM_DATATYPE_F32;
    l_shape.comp_type = LIBXSMM_DATATYPE_F32;
    l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');

    res.bwd_compute_kernel_fallback_f32.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );

    /* Eltwise TPPs */
    stride_in             = res.ofmblock;
    stride_out            = res.ifmblock;
    unary_shape.m         = res.ofmblock;
    unary_shape.n         = res.ifmblock;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_F32;
    unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
    unary_shape.out_type  = LIBXSMM_DATATYPE_F32;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;

    res.tr_kernel= libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.tr_kernel  == NULL ) {
      fprintf( stderr, "JIT for TPP tr_kernel failed. Bailing...!\n");
      exit(-1);
    }

    stride_out            = (res.pack_input_bwd == 1) ? res.ofw * res.ifmblock : res.ifwp * res.ifmblock;
    stride_in             = stride_out;
    unary_shape.m         = res.ofw * res.ifmblock;
    unary_shape.n         = res.ofh;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    res.ofh_x_ofw_x_ifmblock_zero_kernel_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;

    if (  res.ofh_x_ofw_x_ifmblock_zero_kernel_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP ofh_x_ofw_x_ifmblock_zero_kernel_f32 failed. Bailing...!\n");
      exit(-1);
    }

    unary_shape.m         = (res.W + (2 * res.pad_w)) * (res.H + (2 * res.pad_h)) * res.ifmblock;
    unary_shape.n         = 1;
    stride_out            = unary_shape.m;
    stride_in             = stride_out;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    res.paddedH_x_paddedW_x_ifmblock_zero_kernel_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;

    if (  res.paddedH_x_paddedW_x_ifmblock_zero_kernel_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP paddedH_x_paddedW_x_ifmblock_zero_kernel_f32 failed. Bailing...!\n");
      exit(-1);
    }

    unary_shape.m         = res.ifwp * res.ifhp * res.ifmblock;
    unary_shape.n         = 1;
    stride_out            = unary_shape.m;
    stride_in             = stride_out;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    res.ifhp_x_ifwp_x_ifmblock_zero_kernel_f32= libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;

    if (  res.ifhp_x_ifwp_x_ifmblock_zero_kernel_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP ifhp_x_ifwp_x_ifmblock_zero_kernel_f32 failed. Bailing...!\n");
      exit(-1);
    }
  }

  if ( res.datatype_in == LIBXSMM_DATATYPE_BF16 ) {
    libxsmm_blasint ldA;
    libxsmm_blasint ldB;
    libxsmm_blasint ldC;
    float beta;
    libxsmm_meltw_unary_shape unary_shape;
    libxsmm_blasint stride_in;
    libxsmm_blasint stride_out;
    libxsmm_gemm_shape l_shape;
    libxsmm_gemm_batch_reduce_config l_brconfig;
    libxsmm_bitfield l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
    libxsmm_bitfield l_prefetch_flags = 0;
    int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
    int brgemm_pf_oob = 0;
    const char *const env_brgemm_pf_oob = getenv("BRGEMM_PF_OOB");

    res.A_offsets_bwd = NULL;
    res.B_offsets_bwd = NULL;

    ldB = (libxsmm_blasint)res.ofmblock;
    ldA = (libxsmm_blasint)res.ifmblock;
    ldC = (res.spread_input_bwd == 1) ? (libxsmm_blasint)(res.ifmblock * res.v) : (libxsmm_blasint)res.ifmblock;
    beta = (res.avoid_acc_load_bwd ? 0.f : 1.f);

    l_flags |= res.bwd_flags;
    l_flags |= ( beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
    if ( 0 == env_brgemm_pf_oob ) {
    } else {
      brgemm_pf_oob = atoi(env_brgemm_pf_oob);
    }
    if (brgemm_pf_oob > 0) {
      prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB);
    }
    l_prefetch_flags = prefetch_mode;

    /* Strided kernel  */
    libxsmm_blasint stride_a = res.R * res.S * res.ifmblock * res.ofmblock * sizeof(libxsmm_bfloat16);
    libxsmm_blasint stride_b = res.ofwp * res.ofhp * res.ofmblock * sizeof(libxsmm_bfloat16);

    l_shape.m = res.ifmblock;
    l_shape.n = res.bwd_ofh_rb*res.bwd_ofw_rb;
    l_shape.k = res.ofmblock;
    l_shape.lda = ldA;
    l_shape.ldb = ldB;
    l_shape.ldc = ldC;
    l_shape.a_in_type = LIBXSMM_DATATYPE_BF16;
    l_shape.b_in_type = LIBXSMM_DATATYPE_BF16;
    l_shape.out_type  = LIBXSMM_DATATYPE_BF16;
    l_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = stride_a;
    l_brconfig.br_stride_b_hint = stride_b;
    l_brconfig.br_unroll_hint   = res.blocksofm_blocking;

    /* Stride-based kernels  */
    res.bwd_compute_kernel_strd_bf16.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if (  res.bwd_compute_kernel_strd_bf16.gemm  == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP bwd_compute_kernel_strd_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    if (res.avoid_fmas_in_rim > 0) {
      l_shape.n =  res.bwd_ofh_rb*(res.bwd_ofw_rb-1);
      res.bwd_compute_kernel2_strd_bf16.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      if (  res.bwd_compute_kernel2_strd_bf16.gemm  == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP bwd_compute_kernel2_strd_bf16 failed. Bailing...!\n");
        exit(-1);
      }
    }

    l_shape.out_type  = LIBXSMM_DATATYPE_F32;
    l_shape.n = res.bwd_ofh_rb*res.bwd_ofw_rb;
    res.bwd_compute_kernel_strd_bf16f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if (  res.bwd_compute_kernel_strd_bf16f32.gemm  == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP bwd_compute_kernel_strd_bf16f32 failed. Bailing...!\n");
      exit(-1);
    }

    if (res.avoid_fmas_in_rim > 0) {
      l_shape.n =  res.bwd_ofh_rb*(res.bwd_ofw_rb-1);
      res.bwd_compute_kernel2_strd_bf16f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      if (  res.bwd_compute_kernel2_strd_bf16f32.gemm  == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP bwd_compute_kernel2_strd_bf16f32 failed. Bailing...!\n");
        exit(-1);
      }
    }
    l_shape.out_type  = LIBXSMM_DATATYPE_BF16;

    /* Offset-based kernel */
    int n_blocks = res.R * res.S * res.blocksofm_blocking;
    int i = 0, ofm, ki, kj;
    l_shape.n = res.bwd_ofh_rb*res.bwd_ofw_rb;

    if ((res.avoid_fmas_in_rim == 0) && (res.R > 1 || res.S > 1)) {
      res.A_offsets_bwd= (unsigned long long*) libxsmm_aligned_malloc(n_blocks * sizeof(unsigned long long), 2097152);
      res.B_offsets_bwd = (unsigned long long*) libxsmm_aligned_malloc(n_blocks * sizeof(unsigned long long), 2097152);
      for (ofm = 0; ofm < res.blocksofm_blocking; ofm++) {
        for (kj = 0; kj < res.R; kj++) {
          for (ki = 0; ki < res.S; ki++) {
            res.A_offsets_bwd[i] = (ofm * res.R * res.S * res.ifmblock * res.ofmblock +
                kj * res.S * res.ifmblock * res.ofmblock +
                ki * res.ifmblock * res.ofmblock) * sizeof(libxsmm_bfloat16);
            res.B_offsets_bwd[i] = (ofm * res.ofhp * res.ofwp * res.ofmblock +
                kj * res.ofwp * res.ofmblock +
                ki * res.ofmblock) * sizeof(libxsmm_bfloat16);
            i++;
          }
        }
      }

      l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
      l_brconfig.br_stride_a_hint = 0;
      l_brconfig.br_stride_b_hint = 0;

      res.bwd_compute_kernel_offs_bf16.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      if (  res.bwd_compute_kernel_offs_bf16.gemm  == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP bwd_compute_kernel_offs_bf16 failed. Bailing...!\n");
        exit(-1);
      }
    }

    /* Regular GEMM for fallback codepath */
    ldC = (libxsmm_blasint)(res.v*res.ifmblock);
    l_shape.m = res.ifmblock;
    l_shape.n = res.ofw;
    l_shape.k = res.ofmblock;
    l_shape.lda = res.ifmblock;
    l_shape.ldb = res.ofmblock;
    l_shape.ldc = ldC;
    l_shape.a_in_type = LIBXSMM_DATATYPE_BF16;
    l_shape.b_in_type = LIBXSMM_DATATYPE_BF16;
    l_shape.out_type  = LIBXSMM_DATATYPE_BF16;
    l_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');

    res.bwd_compute_kernel_fallback_bf16.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );

    /* Eltwise TPPs */
    stride_in             = res.ofmblock;
    stride_out            = res.ifmblock;
    unary_shape.m         = res.ofmblock;
    unary_shape.n         = res.ifmblock;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
    unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;

    res.tr_kernel= libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI_TO_VNNIT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.tr_kernel  == NULL ) {
      fprintf( stderr, "JIT for TPP tr_kernel failed. Bailing...!\n");
      exit(-1);
    }

    stride_out            = (res.pack_input_bwd == 1) ? res.ofw * res.ifmblock : res.ifwp * res.ifmblock;
    stride_in             = stride_out;
    unary_shape.m         = res.ofw * res.ifmblock;
    unary_shape.n         = res.ofh;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    res.ofh_x_ofw_x_ifmblock_zero_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;

    if (  res.ofh_x_ofw_x_ifmblock_zero_kernel_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP ofh_x_ofw_x_ifmblock_zero_kernel_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    unary_shape.in0_type   = LIBXSMM_DATATYPE_F32;
    unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
    unary_shape.out_type  = LIBXSMM_DATATYPE_F32;
    stride_out            = (res.pack_input_bwd == 1) ? res.ofw * res.ifmblock : res.ifwp * res.ifmblock;
    stride_in             = stride_out;
    unary_shape.m         = res.ofw * res.ifmblock;
    unary_shape.n         = res.ofh;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    res.ofh_x_ofw_x_ifmblock_zero_kernel_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;

    if (  res.ofh_x_ofw_x_ifmblock_zero_kernel_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP ofh_x_ofw_x_ifmblock_zero_kernel_f32 failed. Bailing...!\n");
      exit(-1);
    }

    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;
    stride_out            = (res.pack_input_bwd == 1) ? res.ofw * res.ifmblock : res.ifwp * res.ifmblock;
    unary_shape.m         = res.ofw * res.ifmblock;
    unary_shape.n         = 1;
    unary_shape.ldi       = stride_out;
    unary_shape.ldo       = stride_out;
    res.cvt_kernel_bwd_fp32bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.cvt_kernel_bwd_fp32bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP cvt_kernel_bwd_fp32bf16 failed. Bailing...!\n");
      exit(-1);
    }

    unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
    unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;

    unary_shape.m         = (res.W + (2 * res.pad_w)) * (res.H + (2 * res.pad_h)) * res.ifmblock;
    unary_shape.n         = 1;
    stride_out            = unary_shape.m;
    stride_in             = stride_out;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    res.paddedH_x_paddedW_x_ifmblock_zero_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;

    if (  res.paddedH_x_paddedW_x_ifmblock_zero_kernel_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP paddedH_x_paddedW_x_ifmblock_zero_kernel_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    unary_shape.m         = res.ifwp * res.ifhp * res.ifmblock;
    unary_shape.n         = 1;
    stride_out            = unary_shape.m;
    stride_in             = stride_out;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    res.ifhp_x_ifwp_x_ifmblock_zero_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;

    if (  res.ifhp_x_ifwp_x_ifmblock_zero_kernel_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP ifhp_x_ifwp_x_ifmblock_zero_kernel_bf16 failed. Bailing...!\n");
      exit(-1);
    }
  }

  *inout_cfg = res;
}

LIBXSMM_API_INLINE void libxsmm_dnn_conv_generate_upd_kernels( libxsmm_dnn_conv_config* inout_cfg) {
  libxsmm_dnn_conv_config res = *inout_cfg;
  res.A_offsets_upd = NULL;
  res.B_offsets_upd = NULL;
  res.A_offsets2_upd = NULL;
  res.B_offsets2_upd = NULL;
  res.A_offsets3_upd = NULL;
  res.B_offsets3_upd = NULL;

  if ( res.datatype_in == LIBXSMM_DATATYPE_F32 ) {
    libxsmm_blasint LDA;
    libxsmm_blasint LDB;
    libxsmm_blasint LDC;
    float beta;
    libxsmm_meltw_unary_shape unary_shape;
    libxsmm_blasint stride_in;
    libxsmm_blasint stride_out;
    libxsmm_gemm_shape l_shape;
    libxsmm_gemm_batch_reduce_config l_brconfig;
    libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
    libxsmm_bitfield l_prefetch_flags = 0;
    int prefetch_mode = (res.u == 2 || (res.R == 3 && res.ofw == 7) ) ? libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE) : libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BL1);
    int brgemm_pf_oob = 0;
    const char *const env_brgemm_pf_oob = getenv("BRGEMM_PF_OOB");
    const int img_work = res.N;
    const int img_chunksize = (img_work % res.threads == 0) ? (img_work / res.threads) : (img_work / res.threads) + 1;
    int n_blocks;
    const int IFWP = (res.upd_padding_copy == 1) ? res.ifwp + 2*res.pad_w :  res.ifwp;
    const int IFW =  (res.upd_pack_input == 1) ? res.ifwp/res.v : IFWP;
    const int IFHP = (res.upd_padding_copy == 1) ? res.ifhp + 2*res.pad_h :  res.ifhp;
    libxsmm_blasint img_block_size = res.N;
    LDA = res.ofmblock;
    LDB = (res.upd_pack_input == 1) ? res.ifmblock : res.v * res.ifmblock;
    LDC = res.ofmblock;

    if ( 0 == env_brgemm_pf_oob ) {
    } else {
      brgemm_pf_oob = atoi(env_brgemm_pf_oob);
    }
    if (brgemm_pf_oob > 0) {
      prefetch_mode = prefetch_mode | libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB);
    }
    l_prefetch_flags = prefetch_mode;

    /* Regular GEMM  -- no tasklist*/
    l_shape.m = res.ofmblock;
    l_shape.n = res.ifmblock;
    l_shape.k = res.upd_ofw_rb * res.upd_ofh_rb;
    l_shape.lda = LDA;
    l_shape.ldb = LDB;
    l_shape.ldc = LDC;
    l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
    l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
    l_shape.out_type  = LIBXSMM_DATATYPE_F32;
    l_shape.comp_type = LIBXSMM_DATATYPE_F32;

    beta = ((img_chunksize == 1) && (res.upd_ofh_rb == res.ofh) && (res.upd_ofw_rb == res.ofw)) ? 0.f : 1.f;
    l_flags = LIBXSMM_GEMM_FLAGS('N', 'T');
    l_flags |= ( beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
    res.upd_compute_kernel_no_linearized_tasklist_f32.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
    if (  res.upd_compute_kernel_no_linearized_tasklist_f32.gemm  == NULL ) {
      fprintf( stderr, "JIT for GEMM TPP upd_compute_kernel_no_linearized_tasklist_f32 failed. Bailing...!\n");
      exit(-1);
    }

    /* Regular GEMM  -- tasklist */
    beta = ((res.N == 1) && (res.upd_ofh_rb == res.ofh) && (res.upd_ofw_rb == res.ofw)) ? 0.f : 1.f;
    l_flags = LIBXSMM_GEMM_FLAGS('N', 'T');
    l_flags |= ( beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
    res.upd_compute_kernel_linearized_tasklist_f32.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
    if (  res.upd_compute_kernel_linearized_tasklist_f32.gemm  == NULL ) {
      fprintf( stderr, "JIT for GEMM TPP upd_compute_kernel_linearized_tasklist_f32 failed. Bailing...!\n");
      exit(-1);
    }

    /* Offset BRGEMM -- tasklist */
    libxsmm_blasint img_br, j_br, i = 0;
    n_blocks = LIBXSMM_MAX(res.N * res.upd_ofh_rb, res.ofh);
    res.A_offsets_upd = (unsigned long long*) libxsmm_aligned_malloc(n_blocks * sizeof(unsigned long long), 2097152);
    res.B_offsets_upd = (unsigned long long*) libxsmm_aligned_malloc(n_blocks * sizeof(unsigned long long), 2097152);
    res.A_offsets2_upd = (unsigned long long*) libxsmm_aligned_malloc(n_blocks * sizeof(unsigned long long), 2097152);
    res.B_offsets2_upd = (unsigned long long*) libxsmm_aligned_malloc(n_blocks * sizeof(unsigned long long), 2097152);
    res.A_offsets3_upd = (unsigned long long*) libxsmm_aligned_malloc(n_blocks * sizeof(unsigned long long), 2097152);
    res.B_offsets3_upd = (unsigned long long*) libxsmm_aligned_malloc(n_blocks * sizeof(unsigned long long), 2097152);

    if (res.upd_linearized_tasklist == 1) {
      i = 0;
      for (img_br = 0; img_br < img_block_size; img_br++) {
        for (j_br = 0; j_br < res.upd_ofh_rb; j_br++) {
          res.A_offsets_upd[i] = ((img_br * res.blocksofm * res.ofhp * res.ofwp * res.ofmblock) +
                                 (j_br * res.ofwp * res.ofmblock)) * sizeof(float);
          res.B_offsets_upd[i] = ((img_br * res.blocksifm * IFHP * IFWP * res.ifmblock) +
                                 (j_br * res.u * IFWP * res.ifmblock)) * sizeof(float);
          i++;
        }
      }

      i = 0;
      for (img_br = 0; img_br < img_block_size; img_br++) {
        for (j_br = 1; j_br < res.upd_ofh_rb; j_br++) {
          res.A_offsets2_upd[i] = ((img_br * res.blocksofm * res.ofhp * res.ofwp * res.ofmblock) +
                                 (j_br * res.ofwp * res.ofmblock)) * sizeof(float);
          res.B_offsets2_upd[i] = ((img_br * res.blocksifm * IFHP * IFWP * res.ifmblock) +
                                 (j_br * res.u * IFWP * res.ifmblock)) * sizeof(float);
          i++;
        }
      }

      i = 0;
      for (img_br = 0; img_br < img_block_size; img_br++) {
        for (j_br = 0; j_br < res.upd_ofh_rb-1; j_br++) {
          res.A_offsets3_upd[i] = ((img_br * res.blocksofm * res.ofhp * res.ofwp * res.ofmblock) +
                                 (j_br * res.ofwp * res.ofmblock)) * sizeof(float);
          res.B_offsets3_upd[i] = ((img_br * res.blocksifm * IFHP * IFWP * res.ifmblock) +
                                 (j_br * res.u * IFWP * res.ifmblock)) * sizeof(float);
          i++;
        }
      }
    } else {
      if (res.N != res.threads) {
        for (j_br = 0; j_br < res.ofh; j_br++) {
          res.A_offsets_upd[i] = (j_br * res.ofwp * res.ofmblock) * sizeof(float);
          res.B_offsets_upd[i] = (j_br * res.u * IFW * res.ifmblock) * sizeof(float);
          i++;
        }
      } else {
        for (img_br = 0; img_br < img_block_size; img_br++) {
          for (j_br = 0; j_br < res.upd_ofh_rb; j_br++) {
            res.A_offsets_upd[i] = ((img_br * res.blocksofm * res.ofhp * res.ofwp * res.ofmblock) +
                                   (j_br * res.ofwp * res.ofmblock)) * sizeof(float);
            res.B_offsets_upd[i] = ((img_br * res.blocksifm * IFHP * IFWP * res.ifmblock) +
                                   (j_br * res.u * IFWP * res.ifmblock)) * sizeof(float);
            i++;
          }
        }
      }
    }

    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
    l_brconfig.br_stride_a_hint = 0;
    l_brconfig.br_stride_b_hint = 0;

    beta = ((res.upd_ofh_rb == res.ofh) && (res.upd_ofw_rb == res.ofw)) ? 0.f : 1.f;
    l_flags = LIBXSMM_GEMM_FLAGS('N', 'T');
    l_flags |= ( beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
    l_shape.k = res.upd_ofw_rb;
    res.upd_compute_kernel_linearized_tasklist_offs_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if (  res.upd_compute_kernel_linearized_tasklist_offs_f32.gemm  == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP upd_compute_kernel_linearized_tasklist_offs_f32 failed. Bailing...!\n");
      exit(-1);
    }

    l_shape.k = res.upd_ofw_rb-1;
    res.upd_compute_kernel2_linearized_tasklist_offs_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if (  res.upd_compute_kernel2_linearized_tasklist_offs_f32.gemm  == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP upd_compute_kernel2_linearized_tasklist_offs_f32 failed. Bailing...!\n");
      exit(-1);
    }

    beta = 0.f;
    l_flags = LIBXSMM_GEMM_FLAGS('N', 'T');
    l_flags |= ( beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
    l_shape.k = res.upd_ofw_rb;
    res.upd_compute_kernel_flat_linearized_tasklist_offs_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if (  res.upd_compute_kernel_flat_linearized_tasklist_offs_f32.gemm  == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP upd_compute_kernel_flat_linearized_tasklist_offs_f32 failed. Bailing...!\n");
      exit(-1);
    }

    beta = ((res.upd_ofh_rb == res.ofh) && (res.upd_ofw_rb == res.ofw)) ? 0.f : 1.f;
    l_flags = LIBXSMM_GEMM_FLAGS('N', 'T');
    l_flags |= ( beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
    res.upd_compute_kernel_hybrid_linearized_tasklist_offs_f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if (  res.upd_compute_kernel_hybrid_linearized_tasklist_offs_f32.gemm  == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP upd_compute_kernel_hybrid_linearized_tasklist_offs_f32 failed. Bailing...!\n");
      exit(-1);
    }

    /* Eltwise TPPs */
    stride_in             = res.K * res.C * res.R * res.S;
    stride_out            = res.K * res.C * res.R * res.S;
    unary_shape.m         = res.K * res.C * res.R * res.S;
    unary_shape.n         = 1;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_F32;
    unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
    unary_shape.out_type  = LIBXSMM_DATATYPE_F32;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;

    res.zero_weights_kernel_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.zero_weights_kernel_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP zero_weights_kernel_f32 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ifmblock * res.ofmblock;
    stride_out            = res.ifmblock * res.ofmblock;
    unary_shape.m         = res.ifmblock * res.ofmblock;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    res.zero_ifmblock_x_ofmblock_kernel_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.zero_ifmblock_x_ofmblock_kernel_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP zero_ifmblock_x_ofmblock_kernel_f32 failed. Bailing...!\n");
      exit(-1);
    }

    /* Reduction kernels.. we generate 2 variants depending on threads/available work */
    if (res.weight_copies > 1) {
      const int fm_blocking = (res.ofmblock % 16 == 0) ? 16 : res.ofmblock;
      const int reduce_work = res.blocksofm * res.blocksifm * res.R * res.S * (res.ofmblock/fm_blocking) * res.ifmblock;
      const int reduce_chunksize = (reduce_work % res.threads == 0) ? (reduce_work / res.threads) : (reduce_work / res.threads) + 1;
      const int chunk0 = reduce_chunksize * fm_blocking;
      const int chunk1 = (reduce_work - (reduce_work/reduce_chunksize) * reduce_chunksize) * fm_blocking;
      stride_in             = res.K * res.C * res.R * res.S;
      stride_out            = chunk0;
      unary_shape.m         = chunk0;
      unary_shape.n         = res.weight_copies;
      unary_shape.ldi       = stride_in;
      unary_shape.ldo       = stride_out;
      res.wt_reduce_kernel0_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
      if (  res.wt_reduce_kernel0_f32  == NULL ) {
        fprintf( stderr, "JIT for TPP wt_reduce_kernel0_f32 failed. Bailing...!\n");
        exit(-1);
      }

      if (chunk1 > 0) {
        stride_out            = chunk1;
        unary_shape.m         = chunk1;
        unary_shape.ldi       = stride_in;
        unary_shape.ldo       = stride_out;
        res.wt_reduce_kernel1_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
        if (  res.wt_reduce_kernel1_f32  == NULL ) {
          fprintf( stderr, "JIT for TPP wt_reduce_kernel1_f32 failed. Bailing...!\n");
          exit(-1);
        }
      }
    }
  }

  if ( res.datatype_in == LIBXSMM_DATATYPE_BF16 ) {
    const int IFHP = (res.upd_padding_copy == 1) ? res.ifhp + 2*res.pad_h :  res.ifhp;
    libxsmm_blasint LDA = res.ofmblock;
    libxsmm_blasint LDB = IFHP*res.ifwp_extended;
    libxsmm_blasint LDC = res.ofmblock;
    float beta;
    libxsmm_meltw_unary_shape unary_shape;
    libxsmm_blasint stride_in;
    libxsmm_blasint stride_out;
    libxsmm_gemm_shape l_shape;
    libxsmm_gemm_batch_reduce_config l_brconfig;
    libxsmm_bitfield l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
    libxsmm_bitfield l_prefetch_flags = 0;
    int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
    int brgemm_pf_oob = 0;
    const char *const env_brgemm_pf_oob = getenv("BRGEMM_PF_OOB");
    beta = (res.use_intermediate_f32_wt_tensor ? 1.f : 0.f);
    l_flags |= ( beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
    if ( 0 == env_brgemm_pf_oob ) {
    } else {
      brgemm_pf_oob = atoi(env_brgemm_pf_oob);
    }
    if (brgemm_pf_oob > 0) {
      prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB);
    }
    l_prefetch_flags = prefetch_mode;

    /* Strided kernel  */
    libxsmm_blasint stride_a = res.ofwp_extended * res.ofmblock * sizeof(libxsmm_bfloat16);
    libxsmm_blasint stride_b = res.ifwp_extended * sizeof(libxsmm_bfloat16);
    l_shape.m = res.ofmblock;
    l_shape.n = res.ifmblock;
    if (res.ofw % 2 == 0) {
      l_shape.k = res.ofw;
    } else {
      l_shape.k = res.ofw + 1;
    }
    l_shape.lda = LDA;
    l_shape.ldb = LDB;
    l_shape.ldc = LDC;
    l_shape.a_in_type = LIBXSMM_DATATYPE_BF16;
    l_shape.b_in_type = LIBXSMM_DATATYPE_BF16;
    l_shape.out_type  = LIBXSMM_DATATYPE_F32;
    l_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = stride_a;
    l_brconfig.br_stride_b_hint = stride_b;
    l_brconfig.br_unroll_hint   = 0;

    if ((stride_a > 0) && (stride_b > 0)) {
      /* Stride-based kernels  */
      res.upd_compute_kernel1_bf16f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      if (  res.upd_compute_kernel1_bf16f32.gemm  == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP upd_compute_kernel1_bf16f32 failed. Bailing...!\n");
        exit(-1);
      }

      if (res.ofw % 2 == 1) {
         l_shape.k = res.ofw+1;
      }
      res.upd_compute_kernel2_bf16f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      if (  res.upd_compute_kernel2_bf16f32.gemm  == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP upd_compute_kernel2_bf16f32 failed. Bailing...!\n");
        exit(-1);
      }
    }

    if (res.pixel_blocking % 2 == 0) {
      l_shape.k = LIBXSMM_MAX(2,res.pixel_blocking);
      l_shape.ldb = LIBXSMM_MAX(l_shape.k, res.input_pixels);
      l_brconfig.br_unroll_hint = 0;
      stride_a = res.blocksofm * res.output_pixels * res.ofmblock * sizeof(libxsmm_bfloat16);
      stride_b = res.blocksifm * res.ifmblock * res.input_pixels * sizeof(libxsmm_bfloat16);
      l_brconfig.br_stride_a_hint = stride_a;
      l_brconfig.br_stride_b_hint = stride_b;

      res.upd_compute_kernel3_bf16f32.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      if (  res.upd_compute_kernel3_bf16f32.gemm  == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP upd_compute_kernel3_bf16f32 failed. Bailing...!\n");
        exit(-1);
      }

      /* Regular GEMM */
      l_shape.m = res.ofmblock;
      l_shape.n = res.ifmblock;
      l_shape.k =  LIBXSMM_MAX(2,res.pixel_blocking);
      res.upd_compute_kernel4_bf16f32.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
      if (  res.upd_compute_kernel4_bf16f32.gemm  == NULL ) {
        fprintf( stderr, "JIT for GEMM TPP upd_compute_kernel4_bf16f32 failed. Bailing...!\n");
        exit(-1);
      }
    }

    /* Generate unary kernels */
    stride_in             = res.ofmblock;
    stride_out            = res.ofmblock;
    unary_shape.m         = res.ofmblock;
    unary_shape.n         = res.ifmblock;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
    unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;

    res.upd_weight_vnni_format_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.upd_weight_vnni_format_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP upd_weight_vnni_format_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ofmblock;
    stride_out            = res.ofmblock;
    unary_shape.m         = res.ofmblock;
    unary_shape.n         = res.ifmblock;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_F32;
    unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;

    res.upd_weight_cvt_f32bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.upd_weight_cvt_f32bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP upd_weight_cvt_f32bf16 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ifmblock * res.ofmblock * res.S;
    stride_out            = res.ifmblock * res.ofmblock * res.S;
    unary_shape.m         = res.ifmblock * res.ofmblock * res.S;
    unary_shape.n         = 1;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_F32;
    unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
    unary_shape.out_type  = LIBXSMM_DATATYPE_F32;

    res.zero_partial_weights_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.zero_partial_weights_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP zero_partial_weights_f32 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.C * res.K * res.R * res.S;
    stride_out            = res.C * res.K * res.R * res.S;
    unary_shape.m         = res.C * res.K * res.R * res.S;
    unary_shape.n         = 1;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_F32;
    unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
    unary_shape.out_type  = LIBXSMM_DATATYPE_F32;

    res.zero_full_weights_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.zero_full_weights_f32  == NULL ) {
      fprintf( stderr, "JIT for TPP zero_full_weights_f32 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ifmblock * res.input_pixels;
    stride_out            = res.ifmblock * res.input_pixels;
    unary_shape.m         = res.ifmblock * res.input_pixels;
    unary_shape.n         = 1;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
    unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;

    res.zero_ifmblock_input_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.zero_ifmblock_input_pixels_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP zero_ifmblock_input_pixels_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.C * res.ifhp * res.ifwp_extended;
    stride_out            = res.C * res.ifhp * res.ifwp_extended;
    unary_shape.m         = res.C * res.ifhp * res.ifwp_extended;
    unary_shape.n         = 1;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
    unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;

    res.zero_ifmblock_input_pixels_extended_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.zero_ifmblock_input_pixels_extended_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP zero_ifmblock_input_pixels_extended_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ofmblock * res.output_pixels;
    stride_out            = res.ofmblock * res.output_pixels;
    unary_shape.m         = res.ofmblock * res.output_pixels;
    unary_shape.n         = 1;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;

    res.zero_ofmblock_output_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.zero_ofmblock_output_pixels_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP zero_ofmblock_output_pixels_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    if (res.input_pixels > 0) {
      stride_in             = res.ifmblock;
      stride_out            = res.input_pixels;
      unary_shape.m         = res.ifmblock;
      unary_shape.n         = res.ifwp;
      unary_shape.ldi       = stride_in;
      unary_shape.ldo       = stride_out;
      unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
      unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
      unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;

      res.transpose_input_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
      if (  res.transpose_input_pixels_bf16  == NULL ) {
        fprintf( stderr, "JIT for TPP transpose_input_pixels_bf16 failed. Bailing...!\n");
        exit(-1);
      }

      stride_in             = res.v * res.ifmblock;
      stride_out            = res.input_pixels;
      unary_shape.m         = res.ifmblock;
      unary_shape.n         = res.ifwp/res.v;
      unary_shape.ldi       = stride_in;
      unary_shape.ldo       = stride_out;
      unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
      unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
      unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;

      res.transposeNpack_input_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
      if (  res.transposeNpack_input_pixels_bf16  == NULL ) {
        fprintf( stderr, "JIT for TPP transposeNpack_input_pixels_bf16 failed. Bailing...!\n");
        exit(-1);
      }
    }

    stride_in             = res.ofmblock;
    stride_out            = res.ofmblock;
    unary_shape.m         = res.ofmblock;
    unary_shape.n         = res.ofwp;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
    unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;

    if (res.ofwp % 2 == 1) {
      res.vnni_output_w_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    } else {
      res.vnni_output_w_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    }
    if (  res.vnni_output_w_pixels_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP vnni_output_w_pixels_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ofmblock;
    stride_out            = res.ofmblock;
    unary_shape.m         = res.ofmblock;
    unary_shape.n         = res.ofwp-1;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
    unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;

    if ((res.ofwp-1) % 2 == 1) {
      res.vnni_output_w2_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    } else {
      res.vnni_output_w2_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    }
    if (  res.vnni_output_w2_pixels_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP vnni_output_w2_pixels_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ofmblock;
    stride_out            = res.ofmblock;
    unary_shape.m         = res.ofmblock;
    unary_shape.n         = res.compute_pixels;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
    unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;

    if (res.compute_pixels % 2 == 1) {
      res.vnni_output_compute_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    } else {
      res.vnni_output_compute_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    }
    if (  res.vnni_output_compute_pixels_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP vnni_output_compute_pixels_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    res.upd_remaining_pixels = res.output_pixels - ((res.compute_pixels+1)/2)*2;
    if (res.upd_remaining_pixels > 0) {
      stride_in             = res.upd_remaining_pixels * res.ofmblock;
      stride_out            = res.upd_remaining_pixels * res.ofmblock;
      unary_shape.m         = res.upd_remaining_pixels * res.ofmblock;
      unary_shape.n         = 1;
      unary_shape.ldi       = stride_in;
      unary_shape.ldo       = stride_out;

      res.vnni_output_zero_remaining_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
      if (  res.vnni_output_zero_remaining_pixels_bf16  == NULL ) {
        fprintf( stderr, "JIT for TPP vnni_output_zero_remaining_pixels_bf16 failed. Bailing...!\n");
        exit(-1);
      }
    }

    stride_in             = res.ifmblock;
    stride_out            = res.ifhp * res.ifwp_extended;
    unary_shape.m         = res.ifmblock;
    unary_shape.n         = res.ifwp;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
    unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;

    res.transpose_input_pixels_ifwp_extended_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.transpose_input_pixels_ifwp_extended_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP transpose_input_pixels_ifwp_extended_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ifmblock * res.v;
    stride_out            = IFHP * res.ifwp_extended;
    unary_shape.m         = res.ifmblock;
    unary_shape.n         = res.ofw;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
    unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;

    res.transpose_input_pixels_ifwp_strided_extended_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.transpose_input_pixels_ifwp_strided_extended_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP transpose_input_pixels_ifwp_strided_extended_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    stride_in             = res.ifmblock;
    stride_out            = IFHP * res.ifwp_extended;
    unary_shape.m         = res.ifmblock;
    unary_shape.n         = res.ifwp;
    unary_shape.ldi       = stride_in;
    unary_shape.ldo       = stride_out;
    unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
    unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
    unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;

    res.transpose_input_pixels_ifwp_extended2_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ) ;
    if (  res.transpose_input_pixels_ifwp_extended2_bf16  == NULL ) {
      fprintf( stderr, "JIT for TPP transpose_input_pixels_ifwp_extended2_bf16 failed. Bailing...!\n");
      exit(-1);
    }

    /* Reduction kernels.. we generate 2 variants depending on threads/available work */
    if (res.weight_copies > 1) {
      int active_copies = res.weight_copies;
      const int fm_blocking = (res.ofmblock % 16 == 0) ? 16 : res.ofmblock;
      const int reduce_work = res.blocksofm * res.blocksifm * res.R * res.S * (res.ofmblock/fm_blocking) * res.ifmblock;
      const int reduce_chunksize = (reduce_work % res.threads == 0) ? (reduce_work / res.threads) : (reduce_work / res.threads) + 1;
      const int chunk0 = reduce_chunksize * fm_blocking;
      const int chunk1 = (reduce_work - (reduce_work/reduce_chunksize) * reduce_chunksize) * fm_blocking;
      const int img_work = res.N;
      const int img_chunksize = (img_work % res.threads == 0) ? (img_work / res.threads) : (img_work / res.threads) + 1;

      stride_in             = res.K * res.C * res.R * res.S;
      stride_out            = chunk0;
      unary_shape.m         = chunk0;
      unary_shape.in0_type   = LIBXSMM_DATATYPE_BF16;
      unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
      unary_shape.out_type  = LIBXSMM_DATATYPE_BF16;

      /* In this case calculate how many weight copies have been indeed computed  */
      if (res.N != res.threads) {
        active_copies = 1;
        while (active_copies * img_chunksize < res.N) {
          active_copies++;
        }
      }

      unary_shape.n         = active_copies;
      unary_shape.ldi       = stride_in;
      unary_shape.ldo       = stride_out;
      res.wt_reduce_kernel0_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
      if (  res.wt_reduce_kernel0_bf16  == NULL ) {
        fprintf( stderr, "JIT for TPP wt_reduce_kernel0_bf16 failed. Bailing...!\n");
        exit(-1);
      }

      if (chunk1 > 0) {
        stride_out            = chunk1;
        unary_shape.m         = chunk1;
        unary_shape.ldi       = stride_in;
        unary_shape.ldo       = stride_out;
        res.wt_reduce_kernel1_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
        if (  res.wt_reduce_kernel1_bf16  == NULL ) {
          fprintf( stderr, "JIT for TPP wt_reduce_kernel1_bf16 failed. Bailing...!\n");
          exit(-1);
        }
      }
    }
  }

  *inout_cfg = res;
}

LIBXSMM_API libxsmm_dnn_conv_config setup_libxsmm_dnn_conv( libxsmm_datatype cnn_dtype_in, libxsmm_datatype cnn_dtype_out, libxsmm_blasint N, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint R, libxsmm_blasint S,
    libxsmm_blasint stride_h, libxsmm_blasint stride_w,
    libxsmm_blasint pad_h, libxsmm_blasint pad_w,
    libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in,
    libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
    libxsmm_blasint bc, libxsmm_blasint bk, libxsmm_blasint threads, libxsmm_dnn_conv_eltwise_fuse fuse_type, libxsmm_blasint overwrite_output, libxsmm_blasint avoid_bwd_wt_trans, libxsmm_blasint zero_fwd_output_rim) {
  libxsmm_dnn_conv_config res;

  memset(&res, 0, sizeof(libxsmm_dnn_conv_config));

  /* init libxsmm */
  LIBXSMM_INIT

  /* Generic parameter setup  */
  res.N = N;
  res.H = H;
  res.W = W;
  res.C = C;
  res.K = K;
  res.R = R;
  res.S = S;
  res.u = stride_h;
  res.v = stride_w;
  res.pad_h = pad_h;
  res.pad_w = pad_w;
  res.pad_h_in = pad_h_in;
  res.pad_w_in = pad_w_in;
  res.pad_h_out = pad_h_out;
  res.pad_w_out = pad_w_out;
  res.threads = threads;
  res.target_archid = libxsmm_target_archid;
  res.datatype_in   = cnn_dtype_in;
  res.datatype_out  = cnn_dtype_out;
  res.ifhp = res.H + 2*res.pad_h_in;
  res.ifwp = res.W + 2*res.pad_w_in;
  res.ofh = (res.H + 2*res.pad_h - res.R) / res.u + 1;
  res.ofw = (res.W + 2*res.pad_w - res.S) / res.v + 1;
  res.ofhp = res.ofh + 2*res.pad_h_out;
  res.ofwp = res.ofw + 2*res.pad_w_out;
  res.ifmblock = 1;
  res.ofmblock = 1;
  res.blocksifm = res.C;
  res.blocksofm = res.K;
  res.fwd_ofw_rb = 1;
  res.fwd_ofh_rb = 1;
  res.bwd_ofw_rb = 1;
  res.bwd_ofh_rb = 1;
  res.upd_ofw_rb = 1;
  res.upd_ofh_rb = 1;
  res.fm_lp_block = 1;
  res.blocksifm_blocking = 1;
  res.blocksofm_blocking = 1;
  res.avoid_bwd_wt_trans = avoid_bwd_wt_trans;
  res.overwrite_output   = overwrite_output;
  res.zero_fwd_output_rim= zero_fwd_output_rim;
  res.fuse_type          = fuse_type;
  res.bc = bc;
  res.bk = bk;

  /* Use helper functions to setup convolutions */
  res.ifmblock      = libxsmm_dnn_conv_setup_ifmblock(&res);
  res.ofmblock      = libxsmm_dnn_conv_setup_ofmblock(&res);
  res.fm_lp_block   = libxsmm_dnn_conv_setup_fm_lp_block(&res);
  res.blocksifm     = libxsmm_dnn_conv_setup_blocksifm(&res);
  res.blocksofm     = libxsmm_dnn_conv_setup_blocksofm(&res);

  /* FWD parameter setup  */
  res.fwd_ofw_rb              = libxsmm_dnn_conv_setup_fwd_ofw_rb(&res);
  res.pack_input              = libxsmm_dnn_conv_setup_pack_input_fwd(&res);
  res.fwd_ofh_rb              = libxsmm_dnn_conv_setup_fwd_ofh_rb(&res);
  res.fwd_gemm_pixels         = libxsmm_dnn_conv_setup_fwd_pixels_gemm(&res);
  res.block_fwd_oj            = libxsmm_dnn_conv_setup_fwd_block_H(&res);
  res.loop_order              = libxsmm_dnn_conv_setup_loop_order_fwd(&res);
  res.blocksifm_blocking      = libxsmm_dnn_conv_setup_blocksifm_blocking(&res);
  res.block_fwd_ofm           = libxsmm_dnn_conv_setup_block_fwd_OFM(&res);
  res.block_fwd_ifm           = libxsmm_dnn_conv_setup_block_fwd_IFM(&res);
  res.avoid_fmas_in_rim       = libxsmm_dnn_conv_setup_avoid_rim_fmas_fwd(&res);
  res.use_ofm_parallelization = libxsmm_dnn_conv_setup_use_ofm_parallelization(&res);
  res.shuffle_filter_accesses = libxsmm_dnn_conv_setup_shuffle_filter_accesses(&res);
  res.avoid_acc_load          = libxsmm_dnn_conv_setup_avoid_acc_load(&res);
  res.fwd_flags               = libxsmm_dnn_conv_setup_init_fwd_gemm_flags(&res);
  res.use_fallback_fwd_loops  = libxsmm_dnn_conv_setup_fallback_loops_fwd(&res);
  res.fwd_padding_copy        = libxsmm_dnn_conv_setup_fwd_padding_copy(&res);
  /* Generate FWD kernels  */
  libxsmm_dnn_conv_generate_fwd_kernels(&res);

  /* BWD parameter setup  */
  res.bwd_ofw_rb = libxsmm_dnn_conv_setup_bwd_ofw_rb(&res);
  res.bwd_ofh_rb = libxsmm_dnn_conv_setup_bwd_ofh_rb(&res);
  res.bwd_gemm_pixels = libxsmm_dnn_conv_setup_bwd_pixels_gemm(&res);
  res.pack_input_bwd = libxsmm_dnn_conv_setup_pack_input_bwd(&res);
  res.spread_input_bwd = libxsmm_dnn_conv_setup_spread_input_bwd(&res);
  res.blocksofm_blocking = libxsmm_dnn_conv_setup_blocksofm_blocking(&res);
  res.avoid_acc_load_bwd = libxsmm_dnn_conv_setup_avoid_acc_load_bwd(&res);
  res.use_ifm_parallelization = libxsmm_dnn_conv_setup_use_ifm_parallelization(&res);
  res.block_bwd_ofm = libxsmm_dnn_conv_setup_block_bwd_OFM(&res);
  res.block_bwd_ifm = libxsmm_dnn_conv_setup_block_bwd_IFM(&res);
  res.block_bwd_oj = libxsmm_dnn_conv_setup_bwd_block_H(&res);
  res.use_fallback_bwd_loops = libxsmm_dnn_conv_setup_fallback_loops_bwd(&res);
  res.bwd_flags = libxsmm_dnn_conv_setup_init_bwd_gemm_flags(&res);
  /* Generate BWD kernels  */
  libxsmm_dnn_conv_generate_bwd_kernels(&res);

  /* UPD parameter setup */
  res.upd_linearized_tasklist = libxsmm_dnn_conv_setup_linearized_tasklist_upd(&res);
  res.upd_avoid_rim_fmas = libxsmm_dnn_conv_setup_avoid_rim_fmas_upd(&res);
  res.upd_pack_input = libxsmm_dnn_conv_setup_pack_input_upd(&res);
  res.upd_use_batchreduce = libxsmm_dnn_conv_setup_use_batchreduce_upd(&res);
  res.upd_ofw_rb = libxsmm_dnn_conv_setup_upd_ofw_rb(&res);
  res.upd_ofh_rb = libxsmm_dnn_conv_setup_upd_ofh_rb(&res);
  res.upd_loop_order = libxsmm_dnn_conv_setup_loop_order_upd(&res);
  res.weight_copies = libxsmm_dnn_conv_setup_weight_copies_upd(&res);
  res.block_upd_ofm = libxsmm_dnn_conv_setup_block_upd_OFM(&res);
  res.block_upd_ifm = libxsmm_dnn_conv_setup_block_upd_IFM(&res);
  res.upd_loop_order = libxsmm_dnn_conv_setup_loop_order_upd(&res);
  res.upd_padding_copy = libxsmm_dnn_conv_setup_upd_padding_copy(&res);

  if (cnn_dtype_in == LIBXSMM_DATATYPE_BF16) {
    libxsmm_dnn_conv_setup_bf16_upd_algorithms(&res);
  }

  /* Generate UPD kernels  */
  libxsmm_dnn_conv_generate_upd_kernels(&res);

  /* let's configure  scratch */
  libxsmm_dnn_conv_setup_fwd_scratch( &res );
  libxsmm_dnn_conv_setup_bwd_scratch( &res );
  libxsmm_dnn_conv_setup_upd_scratch( &res );
  res.scratch_size = res.fwd_scratch_size + res.bwd_scratch_size + res.upd_scratch_size;

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  return res;
}

LIBXSMM_API_INLINE void libxsmm_dnn_conv_free_offset_brgemm_aux_arrays( libxsmm_dnn_conv_config* cfg) {
  if (cfg->A_offsets != NULL) {
    libxsmm_free(cfg->A_offsets);
  }
  if (cfg->B_offsets != NULL) {
    libxsmm_free(cfg->B_offsets);
  }
  if (cfg->A_offsets_bwd != NULL) {
    libxsmm_free(cfg->A_offsets_bwd);
  }
  if (cfg->B_offsets_bwd != NULL) {
    libxsmm_free(cfg->B_offsets_bwd);
  }
  if (cfg->A_offsets_upd != NULL) {
    libxsmm_free(cfg->A_offsets_upd);
  }
  if (cfg->B_offsets_upd != NULL) {
    libxsmm_free(cfg->B_offsets_upd);
  }
  if (cfg->A_offsets2_upd != NULL) {
    libxsmm_free(cfg->A_offsets2_upd);
  }
  if (cfg->B_offsets2_upd != NULL) {
    libxsmm_free(cfg->B_offsets2_upd);
  }
  if (cfg->A_offsets3_upd != NULL) {
    libxsmm_free(cfg->A_offsets3_upd);
  }
  if (cfg->B_offsets3_upd != NULL) {
    libxsmm_free(cfg->B_offsets3_upd);
  }
}

LIBXSMM_API void destroy_libxsmm_dnn_conv(libxsmm_dnn_conv_config* cfg) {

  libxsmm_dnn_conv_free_offset_brgemm_aux_arrays(cfg);

  libxsmm_barrier_destroy(cfg->barrier);

  /* when/if libxsmm_matrix_eqn_destroy gets added, destructors for equations should go here */
}





