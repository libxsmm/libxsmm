/******************************************************************************
** Copyright (c) 2018, Intel Corporation                                     **
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
#include "libxsmm_dnn_dryruns.h"
#include "generator_common.h"
#include "libxsmm_main.h"
#include <libxsmm.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#define MIXED 0
#define KHWC 1
#define HWKC 2
#define CHWK 3
#define HWCK 4


LIBXSMM_API_INTERN void tune_fwd_blockings(libxsmm_dnn_layer *handle);
LIBXSMM_API_INTERN void tune_fwd_blockings(libxsmm_dnn_layer *handle) {
  int BLOCKSIFM_BLOCKING = handle->blocksifm_blocking;
  /* Some cache blocking tuning here... */
  int loop_order = MIXED;
  int blockifm = 8;
  int block_j = 14;

  /* Loop order tuning  */
  if (handle->desc.H >= 28 && handle->desc.R == 1) {
    loop_order = HWKC;
  }

  /* Feature map block tuning */
  while (blockifm % BLOCKSIFM_BLOCKING != 0) {
    blockifm++;
  }

  handle->block_fwd_ofm = LIBXSMM_MIN(handle->blocksofm, 16);
  handle->block_fwd_ifm = blockifm;

  /* Spatial dimension block tuning  */
  if ((handle->ofh == 7 && handle->desc.u == 2) || (handle->ofh == 14 && handle->desc.R != 3 ) ||  handle->ofh == 27 || (handle->ofh == 28 && handle->desc.R == 1) || handle->ofh == 48 || handle->ofh == 54 || handle->ofh == 56 || handle->ofh == 112 ) {
    block_j = 4;
  }
  while ( block_j % handle->fwd_ofh_rb != 0 ) {
    block_j--;
  }

  handle->block_fwd_oj = block_j;
  handle->loop_order = loop_order;
}

LIBXSMM_API_INLINE void tune_upd_blockings(libxsmm_dnn_layer *handle) {
  if ( handle->ofh == 56 ) {
    /* Pixel block is 196 Kbytes */
    handle->block_upd_ofm = handle->blocksofm;
    handle->block_upd_ifm = 1;
  }

  if ( handle->ofh == 28 ) {
    /* Pixel block is 49 Kbytes */
    handle->block_upd_ofm = 3;
    handle->block_upd_ifm = 3;
  }

  if ( handle->ofh == 14 || handle->ofh == 28 || handle->ofh == 56 ) {
    /* Pixel block is 12.25 Kbytes */
    handle->block_upd_ofm = 8;
    handle->block_upd_ifm = 32;
  }

  if ( handle->ofh == 7 ) {
    /* Pixel block is 3.06 Kbytes */
    handle->block_upd_ofm = 8;
    handle->block_upd_ifm = 16;
  }

  if (  handle->ofh == 28 || handle->ofh == 35  || handle->ofh == 56 || handle->ofh == 149 || handle->ofh == 71  ||  handle->ofh == 147 || handle->ofh == 73   ) {     /* Pixel block is 12.25 Kbytes */
    handle->block_upd_ofm = 32;
    handle->block_upd_ifm = 16;
  }

  handle->block_upd_ofm = 64;
  handle->block_upd_ifm = 64;
}

LIBXSMM_API_INLINE int find_rb(int W, int H, int *wrb1_res, int *hrb1_res, int *wrb2_res, int *hrb2_res) {
  const int min_r = 15;
  const int max_r = 28;
  int n_variants = 0;
  unsigned int wrb1 = 0, hrb1 = 0, wrb2 = 0, hrb2 = 0;
  unsigned int foo1, foo2;

  /* Case 1: min_r <= W <= max_r  */
  if (min_r <= W && W <= max_r) {
    n_variants = 1;
    wrb1 = W;
    hrb1 = 1;
  }
  /* Case 2: max_r < W  */
  if (max_r < W) {
    libxsmm_compute_equalized_blocking(W, max_r, &foo1, &wrb1, &foo2, &wrb2);
    if (wrb2 == 0) {
      n_variants = 1;
    } else {
      n_variants = 2;
    }
    hrb1 = 1;
    hrb2 = 1;
  }

  /* Case 3: W < min_r */
  if (W < min_r) {
    wrb1 = W;
    wrb2 = W;
    libxsmm_compute_equalized_blocking(H, max_r/W, &foo1, &hrb1, &foo2, &hrb2);
    if (hrb2 == 0) {
      n_variants = 1;
    } else {
      n_variants = 2;
    }
  }

#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
  printf("Problem has W = %d and H = %d\n", W, H);
  if (n_variants == 1) {
    printf("Have 1 variant with wrb = %d and hrb = %d\n",wrb1, hrb1);
  } else {
    printf("Have 2 variants\n");
    printf("Variant 1 with wrb = %d and hrb = %d\n",wrb1, hrb1);
    printf("Variant 2 with wrb = %d and hrb = %d\n",wrb2, hrb2);
  }
#endif

  *wrb1_res = wrb1;
  *hrb1_res = hrb1;
  *wrb2_res = wrb2;
  *hrb2_res = hrb2;
  return n_variants;
}

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_setup_feature_map_blocks( libxsmm_dnn_layer* handle, int *noarch ) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  /* Determine if we have low precision kernel generation */
  if ((handle->datatype_out != LIBXSMM_DNN_DATATYPE_F32) || (handle->datatype_in != LIBXSMM_DNN_DATATYPE_F32)) {
    handle->use_lp_kernel = 1;
  } else {
    handle->use_lp_kernel = 0;
  }

  if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
    handle->ifmblock = (handle->desc.C >=16) ? 16 : handle->desc.C;
    handle->ofmblock = (handle->desc.K >=16) ? 16 : handle->desc.K;
    handle->fm_lp_block = 1;
    handle->ifmblock_hp = handle->ifmblock * handle->fm_lp_block;
    handle->ofmblock_lp = handle->ofmblock / handle->fm_lp_block;
  } else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
    handle->ifmblock = (handle->desc.C >=16) ? 8 : handle->desc.C/2;
    handle->ofmblock = (handle->desc.K >=16) ? 16 : handle->desc.K/2;
    handle->fm_lp_block = 2;
    handle->ifmblock_hp = handle->ifmblock * handle->fm_lp_block;
    handle->ofmblock_lp = handle->ofmblock / handle->fm_lp_block;
    if ( (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_F32_BF16_CVT_RNE) == LIBXSMM_DNN_CONV_OPTION_F32_BF16_CVT_RNE ) {
      handle->f32_bf16_cvt_rne = 1;
    }
  } else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) && ((handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32) || (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32)) ) {
    handle->ifmblock = (handle->desc.C >=16) ? 8 : (handle->desc.C/2);
    handle->ofmblock = (handle->desc.K >=16) ? 16 : (handle->desc.K/2);
    handle->fm_lp_block = 2;
    handle->ifmblock_hp = handle->ifmblock * handle->fm_lp_block;
    handle->ofmblock_lp = handle->ofmblock / handle->fm_lp_block;
    if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC ) {
      *noarch = 1;
    }
  } else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32) && ((handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0)) {
    handle->ifmblock = (handle->desc.C >=16) ? 4 : (handle->desc.C/4);
    handle->ofmblock = (handle->desc.K >=16) ? 16 : (handle->desc.K/4);
    handle->fm_lp_block = 4;
    handle->ifmblock_hp = handle->ifmblock * handle->fm_lp_block;
    handle->ofmblock_lp = handle->ofmblock / handle->fm_lp_block;
    if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC || libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM ) {
      *noarch = 1;
    }
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
    return status;
  }

  /* Let's calculate how many blocks we need for the feature maps */
  if (handle->use_lp_kernel == 1) {
    if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
      handle->blocksifm = handle->desc.C / handle->ifmblock_hp;
      handle->blocksofm = handle->desc.K / handle->ofmblock;
      handle->blocksifm_lp = handle->desc.C / handle->ifmblock_hp;
      handle->blocksofm_lp = handle->desc.K / handle->ofmblock;
    } else {
      handle->blocksifm = handle->desc.C / handle->ifmblock_hp;
      handle->blocksofm = handle->desc.K / handle->ofmblock;
      handle->blocksifm_lp = handle->desc.C / handle->ifmblock_hp;
      handle->blocksofm_lp = handle->desc.K / handle->ofmblock;
    }
  } else {
    handle->blocksifm = handle->desc.C / handle->ifmblock;
    handle->blocksofm = handle->desc.K / handle->ofmblock;
    handle->blocksifm_lp = handle->blocksifm;
    handle->blocksofm_lp = handle->blocksofm;
  }

  handle->block_fwd_ofm = 16;
  handle->block_bwd_ifm = 16;

  /* Let's check one more time that we can actually block */
  if ( (handle->desc.C % (handle->ifmblock * handle->fm_lp_block) != 0) || (handle->desc.K % (handle->ofmblock) != 0)) {
    handle->custom_format_type = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1;
    *noarch = 1;
  }

  return status;
}

LIBXSMM_API_INTERN void libxsmm_dnn_setup_scratch( libxsmm_dnn_layer* handle ) {
  handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);
  /* backward transpose filters */
  handle->scratch1 = 0;
  handle->scratch1_size = handle->blocksifm_lp * handle->ifmblock * handle->blocksofm * handle->ofmblock
    * handle->desc.R * handle->desc.S * handle->fm_lp_block * libxsmm_dnn_typesize(handle->datatype_in);
  if (handle->fm_lp_block > 1) {
    /* If low precision, we need extra buffer to store intermediate weight tensor */
    handle->scratch1_size *= 2;
  }

  /* weight update transpose of minibatch */
  handle->scratch3 = 0;
  handle->scratch3_size = handle->desc.N * handle->blocksifm_lp * handle->ifmblock * handle->ifhp * (handle->ifwp+8)
    * handle->fm_lp_block * libxsmm_dnn_typesize(handle->datatype_in);

  /* minibatch parallel execution of weight update kernel */
  if ( ((handle->blocksifm * handle->blocksofm) < handle->desc.threads) || (handle->use_thread_private_jit) ) {
    handle->upd_use_thread_fil = 1;
    handle->scratch4 = 0;
    handle->scratch4_size = 2 * handle->desc.threads * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S * libxsmm_dnn_typesize(handle->datatype_out);
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
    handle->scratch2_size = handle->desc.N * handle->blocksofm * handle->ofmblock * (handle->ofhp+2*handle->desc.pad_h) * (handle->ofwp+8+2*handle->desc.pad_w) * libxsmm_dnn_typesize(handle->datatype_in);
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) {
      /* Allocate scratch to dump results before downconvert  */
      handle->scratch2_size += handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S * sizeof(float);
    }
  } else {
    handle->scratch2 = 0;
    handle->scratch2_size = 0;
  }
}

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_setup_generic( libxsmm_dnn_layer* handle ) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  int tmp_max_c_block = 16;
  int tmp_max_k_block = 16;
  int tmp_block = 0;

  if ( handle->desc.C < tmp_max_c_block ) {
    handle->ifmblock = handle->desc.C;
  } else {
    for ( tmp_block = 1; tmp_block <= tmp_max_c_block; tmp_block *= 2 ) {
      if ( handle->desc.C % tmp_block == 0 ) handle->ifmblock = tmp_block;
    }
  }
  handle->blocksifm = handle->desc.C / handle->ifmblock;

  if ( handle->desc.K < tmp_max_k_block ) {
    handle->ofmblock = handle->desc.K;
  } else {
    for ( tmp_block = 1; tmp_block <= tmp_max_k_block; tmp_block *= 2 ) {
      if ( handle->desc.K % tmp_block == 0 ) handle->ofmblock = tmp_block;
    }
  }
  handle->blocksofm = handle->desc.K / handle->ofmblock;

  handle->fwd_ofh_rb = 1;
  handle->fwd_ofw_rb = handle->ofw;
  handle->bwd_ofh_rb = 1;
  handle->bwd_ofw_rb = handle->ofw;
  handle->fm_lp_block = 1;
  handle->use_thread_private_jit = 0;

  /* here we need to handle BF16 again */
  if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.C % 2 == 0) && (handle->desc.K % 2 == 0) ) {
    handle->fm_lp_block = 2;
    handle->ifmblock = (handle->desc.C >=16) ? 8 : handle->desc.C/2;
    handle->ofmblock = (handle->desc.K >=16) ? 8 : handle->desc.K/2;
    handle->ifmblock_hp = handle->ifmblock * handle->fm_lp_block;
    handle->ofmblock_lp = handle->ofmblock * handle->fm_lp_block;
    handle->blocksifm = handle->desc.C / (handle->ifmblock * handle->fm_lp_block);
    handle->blocksofm = handle->desc.K / (handle->ofmblock * handle->fm_lp_block);
    handle->blocksifm_lp = handle->blocksifm;
    handle->blocksofm_lp = handle->blocksofm;
  }

  /* Adjust blocking factors if custom_2 format is requested */
  if ((handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2)) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32)  {
      /* In this case of custom_2 format, regardless of requested padding, all the pad_in/pad_out parameters should be 0 */
      if ( ((handle->desc.pad_h > 0) && ((handle->desc.pad_h_in != 0) || (handle->desc.pad_h_out != 0))) || ((handle->desc.pad_w > 0) && ((handle->desc.pad_w_in != 0) || (handle->desc.pad_w_out !=0))) ) {
        status = LIBXSMM_DNN_ERR_INVALID_PADDING;
        free(handle);
        handle = 0;
        return status;
      }
      if ( (handle->desc.N % 16 == 0) && (handle->desc.C % 16 == 0) && (handle->desc.K % 16 == 0) ) {
        handle->nbImg = 16;
        handle->ifmblock = 16;
        handle->ofmblock = 16;
        handle->fm_lp_block = 1;
      } else {
        /* Fallback to custom_1 format, when using custom_2 format N should be divisible by 16 */
        handle->custom_format_type = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1;
      }
    } else {
      /* Fallback to custom_1 format, for now custom_2 format is supported only for float */
      handle->custom_format_type = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1;
    }
  }

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

  /* prepare barrier */
  handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);

  /* backward transpose filters, as we want to call small GEMMs we need that scratch */
  handle->scratch1 = 0;
  handle->scratch1_size = handle->blocksifm * handle->ifmblock * handle->blocksofm * handle->ofmblock
    * handle->desc.R * handle->desc.S * libxsmm_dnn_typesize(handle->datatype_in);
  if (handle->fm_lp_block > 1) {
    /* If low precision, we need extra buffer to store intermediate weight tensor */
    handle->scratch1_size *= 2;
  }

  handle->scratch3 = 0;
  handle->scratch3_size = 0;
  handle->scratch4 = 0;
  handle->scratch4_size = 0;
  return status;
}

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_setup_fwd( libxsmm_dnn_layer* handle, int *noarch ) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  int wrb1 = 0, wrb2 = 0, hrb1 = 0, hrb2 = 0, n_variants = 1;
  int i = 0; /* general counting helper */
  handle->use_fwd_for_bwd = 0;

  /* Find register blocking and number of variants  */
  if (handle->desc.N >= handle->desc.threads) {
    n_variants = find_rb(handle->ofw, handle->ofh, &wrb1, &hrb1, &wrb2, &hrb2);
    handle->fwd_ofw_rb = wrb1;
    handle->fwd_ofh_rb = hrb1;

    if (n_variants == 2) {
      if (wrb1 == wrb2) {
        handle->h_variants = 1;
        handle->w_variants = 0;
      } else {
        handle->h_variants = 0;
        handle->w_variants = 1;
      }
      handle->fwd_ofw_rb_2 = wrb2;
      handle->fwd_ofh_rb_2 = hrb2;
    }
  } else {
    if ((handle->ofw < 15) && (handle->ofh % 2 == 0) ) {
      handle->fwd_ofw_rb = handle->ofw;
      handle->fwd_ofh_rb = 2;
    } else {
      for (i = 28; i > 1; --i) {
        if (handle->ofw % i == 0) break;
      }
      handle->fwd_ofw_rb = i;
      handle->fwd_ofh_rb = 1;
    }
  }
  handle->n_variants = n_variants;

  /* if we have 1x1 let's bring some ifms into the kernel for forward to increase accumulation chain length on AVX512 */
  if ( (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1) ) {
    if ( ((handle->ifmblock*handle->fm_lp_block)%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*512) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksifm_blocking = 512;
    } else if ( ((handle->ifmblock*handle->fm_lp_block)%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*256) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksifm_blocking = 256;
    } else if ( ((handle->ifmblock*handle->fm_lp_block)%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*128) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksifm_blocking = 128;
    } else if ( ((handle->ifmblock*handle->fm_lp_block)%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*64) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksifm_blocking = 64;
    } else if ( ((handle->ifmblock*handle->fm_lp_block)%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*32) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksifm_blocking = 32;
    } else if ( ((handle->ifmblock*handle->fm_lp_block)%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*16) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksifm_blocking = 16;
    } else if ( ((handle->ifmblock*handle->fm_lp_block)%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*8) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksifm_blocking = 8;
    } else if ( ((handle->ifmblock*handle->fm_lp_block)%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*4) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksifm_blocking = 4;
    } else if ( ((handle->ifmblock*handle->fm_lp_block)%16 == 0) && (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*2) == 0) && (handle->desc.R == 1) && (handle->desc.S == 1) ) {
      handle->blocksifm_blocking = 2;
    } else {
      handle->blocksifm_blocking = 1;
    }
  } else {
    handle->blocksifm_blocking = 1;
  }

  /* FIXME: KNM specific tuning for Resnet */
  if ( (handle->desc.C == 1024 && handle->desc.K == 256) || (handle->desc.C == 2048 && handle->desc.K == 512) ) {
    handle->blocksifm_blocking = 8;
  }

  /* Restrict acc chain for overflow handling only if combo is int16/int32 */
  if (handle->use_lp_kernel == 1 && (handle->datatype_in != LIBXSMM_DNN_DATATYPE_BF16) ) {
    if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) && ((handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32) || (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32)) ) {
      if (handle->blocksifm_blocking * handle->ifmblock * handle->fm_lp_block > 256) {
        handle->blocksifm_blocking = 16;
        while ( handle->desc.C%(handle->blocksifm_blocking * handle->ifmblock * handle->fm_lp_block) != 0  ) {
          handle->blocksifm_blocking--;
        }
      }
    }
  }

  /* When we chose overwrite and we loop over all ifms, then let's use streaming stores */
  if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->desc.C == handle->blocksifm_blocking*handle->ifmblock*handle->fm_lp_block) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16 || handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 || handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32) ) {
    handle->use_nts_fwd = 1;
  } else {
    handle->use_nts_fwd = 0;
  }

  /* Adjust blocking factors if custom_2 format is requested */
  if ((handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2)) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) {
      /* Calculate number of image blocks in case of custom_2 format */
      handle->nBImg = handle->desc.N / handle->nbImg;
      /* In this case of custom_2 format, regardless of requested padding, all the pad_in/pad_out parameters should be 0 */
      if ( ((handle->desc.pad_h > 0) && ((handle->desc.pad_h_in != 0) || (handle->desc.pad_h_out != 0))) || ((handle->desc.pad_w > 0) && ((handle->desc.pad_w_in != 0) || (handle->desc.pad_w_out !=0))) ) {
        status = LIBXSMM_DNN_ERR_INVALID_PADDING;
        free(handle);
        handle = 0;
        return status;
      }
      if ((handle->desc.N % 16 == 0) && (handle->desc.K % 16 == 0) && (handle->desc.C % 16 == 0) ) {
        handle->nbImg = 16;
        handle->ifmblock = 16;
        handle->ofmblock = 16;
        handle->fm_lp_block = 1;
      } else {
        /* Fallback to custom_1 format, when using custom_2 format N should be divisible by 16 */
        handle->custom_format_type = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1;
      }
    } else {
      /* Fallback to custom_1 format, for now custom_2 format is supported only for float */
      handle->custom_format_type = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1;
    }
  }

  /* allocate appropriate buffers if the input must be padded */
  if ((handle->desc.pad_h_in == 0) && (handle->desc.pad_w_in == 0) && (handle->desc.pad_h_out == 0) && (handle->desc.pad_w_out == 0) && ((handle->desc.pad_h > 0) || (handle->desc.pad_w > 0))) {
    const size_t fwdbwd_scratch_size_a = handle->desc.C * (handle->ifhp+2*handle->desc.pad_h) * (handle->ifwp+2*handle->desc.pad_w) * libxsmm_dnn_typesize(handle->datatype_out);
    const size_t fwdbwd_scratch_size_b = handle->desc.K * (handle->ofhp+2*handle->desc.pad_h) * (handle->ofwp+2*handle->desc.pad_w) * libxsmm_dnn_typesize(handle->datatype_in);
    const size_t fwdbwd_scratch_size = LIBXSMM_MAX(fwdbwd_scratch_size_a, fwdbwd_scratch_size_b);
    handle->fwdbwd_scratch_size = LIBXSMM_UP2(fwdbwd_scratch_size, LIBXSMM_CACHELINE) * handle->desc.threads;
    handle->minibatch_scratch_size = libxsmm_dnn_typesize(handle->datatype_out) * LIBXSMM_MAX(
        handle->desc.N * handle->blocksifm_lp * handle->ifmblock * handle->fm_lp_block * (handle->ifhp+2*handle->desc.pad_h) * (handle->ifwp+2*handle->desc.pad_w+8),
        handle->desc.N * handle->blocksofm_lp * handle->ofmblock * handle->fm_lp_block * (handle->ofhp+2*handle->desc.pad_h) * (handle->ofwp+2*handle->desc.pad_w));
    handle->max_scratch5_size = LIBXSMM_MAX(handle->minibatch_scratch_size, handle->fwdbwd_scratch_size);
    handle->padding_flag = 1;
    handle->scratch5 = 0;
  } else {
    handle->padding_flag = 0;
  }

  if (*noarch == 0) {
    /* Initialize descriptor to be used by generator  */
    libxsmm_convolution_forward_descriptor descriptor;
    libxsmm_mcopy_descriptor matcopy_descriptor;
    libxsmm_mcopy_descriptor matzero_descriptor;
    /* init descriptors */
    memset( &descriptor, 0, sizeof(libxsmm_convolution_forward_descriptor) );
    memset( &matcopy_descriptor, 0, sizeof(libxsmm_mcopy_descriptor) );
    memset( &matzero_descriptor, 0, sizeof(libxsmm_mcopy_descriptor) );

    descriptor.input_L2_prefetching = 0;
    descriptor.lookahead = 0;
    if ( (handle->desc.R == 1) && (handle->desc.S == 1) ) {
      if ( (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC) || (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) ) {
        descriptor.extra_L2_prefetching = 1;
        descriptor.lookahead = 4;
      } else {
        descriptor.extra_L2_prefetching = 0;
        descriptor.lookahead = 0;
      }
    }

    if (handle->desc.R == 1 && handle->desc.S == 1) {
      descriptor.unroll_kh = 1;
      descriptor.unroll_kw = 1;
    } else if (handle->desc.R > 1 && handle->desc.S == 1 ) {
      descriptor.unroll_kh = 1;
      descriptor.unroll_kw = 0;
    } else if (handle->desc.R == 1 && handle->desc.S > 1 ) {
      descriptor.unroll_kh = 1;
      descriptor.unroll_kw = 1;
    } else {
      descriptor.unroll_kh = 0;
      descriptor.unroll_kw = 1;
    }

    descriptor.use_nts = handle->use_nts_fwd;
    descriptor.f32_bf16_cvt_rne = handle->f32_bf16_cvt_rne;

    if (handle->padding_flag == 1) {
      descriptor.ifh_padded = handle->ifhp + 2 * handle->desc.pad_h;
      descriptor.ifw_padded = handle->ifwp + 2 * handle->desc.pad_w;
      matcopy_descriptor.n = handle->ifhp;
      if (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) {
        matcopy_descriptor.m = handle->ifwp * handle->ifmblock * handle->fm_lp_block;
        matcopy_descriptor.ldi = handle->ifwp * handle->ifmblock * handle->fm_lp_block;
        matcopy_descriptor.ldo = (handle->ifwp + 2*handle->desc.pad_w) * handle->ifmblock * handle->fm_lp_block;
      } else { /* Assumes NHWC format */
        matcopy_descriptor.m = handle->ifwp * handle->blocksifm_lp * handle->ifmblock * handle->fm_lp_block;
        matcopy_descriptor.ldi = handle->ifwp * handle->blocksifm_lp * handle->ifmblock * handle->fm_lp_block;
        matcopy_descriptor.ldo = (handle->ifwp + 2*handle->desc.pad_w) * handle->blocksifm_lp * handle->ifmblock * handle->fm_lp_block;
      }
      matcopy_descriptor.prefetch = 1;
      matcopy_descriptor.unroll_level = 2;
      matcopy_descriptor.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype_in);
      matcopy_descriptor.flags = 0;
    } else {
      descriptor.ifh_padded = handle->ifhp;
      descriptor.ifw_padded = handle->ifwp;
    }
    descriptor.kh = handle->desc.R;
    descriptor.kw = handle->desc.S;
    descriptor.stride_h = handle->desc.u;
    descriptor.stride_w = handle->desc.v;
    descriptor.blocks_ofm = handle->blocksofm;
    descriptor.blocks_ifm = handle->blocksifm_lp;
    descriptor.blocks_ifm_blocking = handle->blocksifm_blocking;
    descriptor.weight_stride = 1;
    descriptor.use_fwd_generator_for_bwd = 0;
    descriptor.stride_h_store = 1;
    descriptor.stride_w_store = 1;
    descriptor.ofm_block = handle->ofmblock;
    descriptor.ifm_block = handle->ifmblock;
    descriptor.ifm_block_hp = handle->ifmblock_hp;
    descriptor.ofh_padded = handle->ofhp;
    descriptor.ofw_padded = handle->ofwp;
    descriptor.ofh_rb = handle->fwd_ofh_rb;
    descriptor.ofw_rb = handle->fwd_ofw_rb;
    descriptor.fm_lp_block = handle->fm_lp_block;
    descriptor.datatype = handle->datatype_in;
    descriptor.datatype_itm = handle->datatype_out;
    descriptor.option = handle->desc.options;
    descriptor.format = (libxsmm_dnn_tensor_format)(handle->buffer_format | handle->filter_format);
    descriptor.perform_relu_in_kernel = 0;

    if ( ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS) > 0) && (handle->use_nts_fwd == 1) && (handle->use_fwd_for_bwd == 0) ) {
      descriptor.compute_batch_stats = 1;
      handle->compute_batch_stats_in_kernel = 1;
    } else {
      descriptor.compute_batch_stats = 0;
      handle->compute_batch_stats_in_kernel = 0;
    }

    if ( ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) && (handle->use_nts_fwd == 1) && (handle->use_fwd_for_bwd == 0) ) {
      descriptor.compute_max = 1;
      handle->compute_max_in_kernel_fwd = 1;
    } else {
      descriptor.compute_max = 0;
      handle->compute_max_in_kernel_fwd = 0;
    }

    if ( (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) ) {
      handle->code_fwd[0].xgemm.smm = libxsmm_smmdispatch(16, 16, 16, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    } else {
      if ( handle->n_variants == 1 ) {
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
        handle->code_fwd[0].pmm = libxsmm_create_xconv_forward(&descriptor);
      }
      if (handle->padding_flag == 1) {
        handle->matcopy_fwd[0].xmatcopy = libxsmm_dispatch_mcopy(&matcopy_descriptor);
      }
    }
    /* use jit code path */
    handle->use_fwd_generic = 0;

    handle->n_entries_fwd = (int*) malloc(handle->desc.threads * sizeof(int));
    memset( handle->n_entries_fwd, 0, handle->desc.threads * sizeof(int) );
    handle->compute_fwd_indices_ptrs = (int**) malloc(handle->desc.threads * sizeof(int*));
    memset( handle->compute_fwd_indices_ptrs, 0, handle->desc.threads * sizeof(int*));
    handle->bn_indices_ptrs = (int**) malloc(handle->desc.threads * sizeof(int*));
    memset( handle->bn_indices_ptrs, 0, handle->desc.threads * sizeof(int*));
    handle->kernel_fwd_variant_ptrs = (char**) malloc(handle->desc.threads * sizeof(char*));
    memset( handle->kernel_fwd_variant_ptrs, 0, handle->desc.threads * sizeof(char*) );
    handle->n_fwd_code_segments = (int*) malloc(handle->desc.threads * sizeof(int));
    memset( handle->n_fwd_code_segments, 0, handle->desc.threads * sizeof(int) );
    handle->fwd_code_segments = (segment_t**) malloc(handle->desc.threads * sizeof(segment_t*));
    memset( handle->fwd_code_segments, 0, handle->desc.threads * sizeof(segment_t*) );
    handle->ofh_fwd_start = (int*) malloc(handle->desc.threads * sizeof(int));
    memset( handle->ofh_fwd_start, 0, handle->desc.threads * sizeof(int) );
    handle->ofh_fwd_end = (int*) malloc(handle->desc.threads * sizeof(int));
    memset( handle->ofh_fwd_end, 0, handle->desc.threads * sizeof(int) );

    descriptor.n_variants = handle->n_variants;
    if ( handle->n_variants == 2 ) {
      descriptor.ofh_padded = handle->ofhp;
      descriptor.ofw_padded = handle->ofwp;
      descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
      descriptor.ofh_rb = hrb1;
      descriptor.ofw_rb = wrb1;
      handle->code_fwd[0].pmm = libxsmm_create_xconv_forward(&descriptor);
      descriptor.ofh_rb = hrb2;
      descriptor.ofw_rb = wrb2;
      descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
      handle->code_fwd[1].pmm = libxsmm_create_xconv_forward(&descriptor);
      handle->fwd_ofh_rb = hrb1;
      descriptor.ofh_rb = hrb1;
      descriptor.ofw_rb = wrb1;
    }
    for (i = 0; i < handle->desc.threads; i++) {
      handle->compute_fwd_indices_ptrs[i] = NULL;
      handle->kernel_fwd_variant_ptrs[i] = NULL;
      handle->fwd_code_segments[i] = NULL;
    }

    /* In case of logical padding also add a kernel that copies only one line of the image;
     * in case we exploit intra-image parallelism we should avoid copying entire image for
     * each thread but only the minimum required number of input pixels... */
    if (handle->padding_flag == 1) {
      matcopy_descriptor.n = 1;
      handle->matcopy_fwd[2].xmatcopy = libxsmm_dispatch_mcopy(&matcopy_descriptor);
    }

    /* In case overwrite is requested, generate zeroing kernel */
    if ( (handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0 )  {
      matzero_descriptor.n = 1;
      if (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) {
        matzero_descriptor.m = handle->ofh*handle->ofwp*handle->ofmblock;
        matzero_descriptor.ldi = handle->ofh*handle->ofwp*handle->ofmblock;
        matzero_descriptor.ldo = handle->ofh*handle->ofwp*handle->ofmblock;
      } else { /* Assumes NHWC format */
        status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
      }
      matzero_descriptor.prefetch = 0;
      matzero_descriptor.unroll_level = 2;
      matzero_descriptor.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype_out);
      matzero_descriptor.flags = LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE;
      handle->matcopy_fwd[1].xmatcopy = libxsmm_dispatch_mcopy(&matzero_descriptor);

      if (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) {
        matzero_descriptor.m = handle->ofwp*handle->ofmblock;
        matzero_descriptor.ldi = handle->ofwp*handle->ofmblock;
        matzero_descriptor.ldo = handle->ofwp*handle->ofmblock;
      } else { /* Assumes NHWC format */
        status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
      }
      handle->matcopy_fwd[3].xmatcopy = libxsmm_dispatch_mcopy(&matzero_descriptor);
    }

    /* Perform the dryrun and generate thread private jit indices to be used for the convolutions */
    tune_fwd_blockings(handle);
    status = libxsmm_dnn_perform_fwd_dryrun_direct(handle);

#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
    { /* compute kernel stream overhead */
      int ks_overhead = 0;
      ks_overhead += handle->desc.threads*4*sizeof(int);
      ks_overhead += handle->desc.threads*sizeof(int*);
      ks_overhead += handle->desc.threads*sizeof(char*);
      ks_overhead += handle->desc.threads*sizeof(segment_t*);
      for ( i = 0; i < handle->desc.threads; ++i ) {
        ks_overhead += ((handle->n_entries_fwd[i]*3)+3)*sizeof(int);
        ks_overhead += handle->n_entries_fwd[i]*sizeof(char);
        ks_overhead += handle->n_fwd_code_segments[i]*sizeof(segment_t);
      }
      printf("KS Overhead FWD in KB: %i \n", ks_overhead/1024 );
    }
#endif
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_setup_bwd( libxsmm_dnn_layer* handle, int *noarch ) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  int wrb1 = 0, wrb2 = 0, hrb1 = 0, hrb2 = 0;
  /* Let's check if we can use algorithmic duality for backward convolution! */
  /* TODO: Enable duality even in cases of image parallelism */
  if ( (handle->use_thread_private_jit > 0) && (handle->desc.N >= handle->desc.threads) && ( (handle->desc.R == 1 && handle->desc.S == 1 && handle->desc.pad_h == 0 && handle->desc.pad_w == 0) || (handle->desc.u == 1 && handle->desc.v == 1) ) && !((handle->desc.R > 1 && handle->desc.pad_h == 0) || (handle->desc.S > 1 && handle->desc.pad_w == 0)) )  {
    handle->exploit_duality = 1;
  } else {
    handle->exploit_duality = 0;
  }
  find_rb(handle->ofw, handle->ofh, &wrb1, &hrb1, &wrb2, &hrb2);

  /* if we have 1x1 let's bring some ifms into the kernel for forward to increase accumulation chain length on AVX512 */
  if ( (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1) ) {
    if ( ((handle->ofmblock)%16 == 0) &&  (handle->desc.K%(handle->ofmblock*512) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksofm_blocking = 512;
    } else if ( ((handle->ofmblock)%16 == 0) &&  (handle->desc.K%(handle->ofmblock*256) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksofm_blocking = 256;
    } else if ( ((handle->ofmblock)%16  == 0) &&  (handle->desc.K%(handle->ofmblock*128) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksofm_blocking = 128;
    } else if ( ((handle->ofmblock)%16  == 0) &&  (handle->desc.K%(handle->ofmblock*64) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksofm_blocking = 64;
    } else if ( ((handle->ofmblock)%16  == 0) &&  (handle->desc.K%(handle->ofmblock*32) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksofm_blocking = 32;
    } else if ( ((handle->ofmblock)%16  == 0) &&  (handle->desc.K%(handle->ofmblock*16) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksofm_blocking = 16;
    } else if ( ((handle->ofmblock)%16  == 0) &&  (handle->desc.K%(handle->ofmblock*8) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksofm_blocking = 8;
    } else if ( ((handle->ofmblock)%16  == 0) &&  (handle->desc.K%(handle->ofmblock*4) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
      handle->blocksofm_blocking = 4;
    } else if ( ((handle->ofmblock)%16  == 0) && (handle->desc.K%(handle->ofmblock*2) == 0) && (handle->desc.R == 1) && (handle->desc.S == 1) ) {
      handle->blocksofm_blocking = 2;
    } else {
      handle->blocksofm_blocking = 1;
    }
  } else {
    handle->blocksofm_blocking = 1;
  }


  /* FIXME: KNM specific tuning for Resnet */
  if ( (handle->desc.C == 256 && handle->desc.K == 1024) || (handle->desc.C == 512 && handle->desc.K == 2048) ||  (handle->desc.C == 1024 && handle->desc.K == 2048) ) {
    handle->blocksofm_blocking = 8;
  }

  /* Restrict acc chain for overflow handling only if combo is int16/int32  */
  if (handle->use_lp_kernel == 1) {
    if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) && ((handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32) || (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32)) ) {
      if (handle->blocksofm_blocking * handle->ofmblock > 256) {
        handle->blocksofm_blocking = 16;
        while ( handle->desc.K%(handle->blocksofm_blocking * handle->ofmblock * handle->fm_lp_block) != 0  ) {
          handle->blocksofm_blocking--;
        }
      }
    }
  }

  /* When we chose overwrite and we loop over all ofms, then let's use streaming stores */
  if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->desc.K == handle->blocksofm_blocking*handle->ofmblock) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 || handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32 ) ) {
    handle->use_nts_bwd = 1;
  } else {
    handle->use_nts_bwd = 0;
  }

  /* FIXME: SKX specific tuning for GooglenetV3 */
  if ((libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE || libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL) && handle->desc.K/16 <= 8) {
    handle->use_nts_bwd = 0;
  }

  if (*noarch == 0) {
    libxsmm_convolution_forward_descriptor fwd_equivalent_descriptor;
    libxsmm_mcopy_descriptor matcopy_descriptor;
    libxsmm_mcopy_descriptor matzero_descriptor_overwrite;

    /* init descriptors */
    memset( &fwd_equivalent_descriptor, 0, sizeof(libxsmm_convolution_forward_descriptor) );
    memset( &matcopy_descriptor, 0, sizeof(libxsmm_mcopy_descriptor) );
    memset( &matzero_descriptor_overwrite, 0, sizeof(libxsmm_mcopy_descriptor) );

    fwd_equivalent_descriptor.input_L2_prefetching = 0;
    fwd_equivalent_descriptor.lookahead = 0;
    if ( (handle->desc.R == 1) && (handle->desc.S == 1) ) {
      if ( (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC) || (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) ) {
        fwd_equivalent_descriptor.extra_L2_prefetching = 1;
        fwd_equivalent_descriptor.lookahead = 4;
      } else {
        fwd_equivalent_descriptor.extra_L2_prefetching = 0;
        fwd_equivalent_descriptor.lookahead = 0;
      }
    }

    if (handle->padding_flag == 1) {
      if (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) {
        matcopy_descriptor.n = handle->ifhp;
        matcopy_descriptor.m = handle->ifwp * handle->ifmblock * handle->fm_lp_block;
        matcopy_descriptor.ldi = handle->ifwp * handle->ifmblock * handle->fm_lp_block;
        matcopy_descriptor.ldo = (handle->ifwp + 2*handle->desc.pad_w) * handle->ifmblock * handle->fm_lp_block;
      } else { /* Assumes NHWC format */
        matcopy_descriptor.n = 1;
        matcopy_descriptor.m =  handle->ifmblock;
        matcopy_descriptor.ldi = handle->ifmblock;
        matcopy_descriptor.ldo = handle->ifmblock;
      }
      matcopy_descriptor.prefetch = 1;
      matcopy_descriptor.unroll_level = 2;
      matcopy_descriptor.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype_in);
      matcopy_descriptor.flags = 0;
    }

    if ( handle->exploit_duality == 1 ) {
      handle->bwd_ofh_rb =  handle->fwd_ofh_rb;
      handle->bwd_ofw_rb =  handle->fwd_ofw_rb;
      if (handle->padding_flag == 1) {
        fwd_equivalent_descriptor.ifh_padded = handle->ofhp + 2 * handle->desc.pad_h;
        fwd_equivalent_descriptor.ifw_padded = handle->ofwp + 2 * handle->desc.pad_w;
      } else {
        fwd_equivalent_descriptor.ifh_padded = handle->ofhp;
        fwd_equivalent_descriptor.ifw_padded = handle->ofwp;
      }
      fwd_equivalent_descriptor.kh = handle->desc.R;
      fwd_equivalent_descriptor.kw = handle->desc.S;
      fwd_equivalent_descriptor.unroll_kw = 1;
      fwd_equivalent_descriptor.unroll_kh = 0;
      if (handle->desc.R == 1 && handle->desc.S == 1) {
        fwd_equivalent_descriptor.unroll_kh = 1;
        fwd_equivalent_descriptor.unroll_kw = 1;
      } else if (handle->desc.R > 1 && handle->desc.S == 1 ) {
        fwd_equivalent_descriptor.unroll_kh = 1;
        fwd_equivalent_descriptor.unroll_kw = 0;
      } else if (handle->desc.R == 1 && handle->desc.S > 1 ) {
        fwd_equivalent_descriptor.unroll_kh = 1;
        fwd_equivalent_descriptor.unroll_kw = 1;
      } else {
        fwd_equivalent_descriptor.unroll_kh = 0;
        fwd_equivalent_descriptor.unroll_kw = 1;
      }
      fwd_equivalent_descriptor.stride_h = 1;
      fwd_equivalent_descriptor.stride_w = 1;
      fwd_equivalent_descriptor.blocks_ofm = handle->blocksifm;
      fwd_equivalent_descriptor.blocks_ifm = handle->blocksofm;
      fwd_equivalent_descriptor.ofm_block = handle->ifmblock_hp;
      fwd_equivalent_descriptor.ifm_block = handle->ofmblock_lp;
      fwd_equivalent_descriptor.ofh_padded = handle->ifhp;
      fwd_equivalent_descriptor.ofw_padded = handle->ifwp;
      fwd_equivalent_descriptor.ofh_rb = handle->fwd_ofh_rb;
      fwd_equivalent_descriptor.ofw_rb = handle->fwd_ofw_rb;
      fwd_equivalent_descriptor.fm_lp_block = handle->fm_lp_block;
      fwd_equivalent_descriptor.datatype = handle->datatype_in;
      fwd_equivalent_descriptor.datatype_itm = handle->datatype_out;
      fwd_equivalent_descriptor.option = handle->desc.options;
      fwd_equivalent_descriptor.format = (libxsmm_dnn_tensor_format)(handle->buffer_format | handle->filter_format);
      fwd_equivalent_descriptor.blocks_ifm_blocking = handle->blocksofm_blocking;
      fwd_equivalent_descriptor.weight_stride = 1;
      fwd_equivalent_descriptor.use_fwd_generator_for_bwd = 1;
      fwd_equivalent_descriptor.stride_h_store = handle->desc.u;
      fwd_equivalent_descriptor.stride_w_store = handle->desc.v;
      fwd_equivalent_descriptor.use_nts = handle->use_nts_bwd;
      fwd_equivalent_descriptor.compute_batch_stats = 0;
      fwd_equivalent_descriptor.compute_max = 0;
      if ( ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) && (handle->use_nts_bwd == 1))  {
        fwd_equivalent_descriptor.compute_max = 1;
        handle->compute_max_in_kernel_bwd = 1;
      } else {
        fwd_equivalent_descriptor.compute_max = 0;
        handle->compute_max_in_kernel_bwd = 0;
      }
      fwd_equivalent_descriptor.perform_relu_in_kernel = (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) && (handle->use_nts_bwd == 1)) ? 1 : 0;
      if (handle->padding_flag == 1) {
        matcopy_descriptor.n = handle->ofhp;
        matcopy_descriptor.m = handle->ofwp * handle->ofmblock;
        matcopy_descriptor.ldi = handle->ofwp * handle->ofmblock;
        matcopy_descriptor.ldo = (handle->ofwp + 2*handle->desc.pad_w) * handle->ofmblock;
        matcopy_descriptor.prefetch = 1;
        matcopy_descriptor.unroll_level = 2;
        matcopy_descriptor.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype_in);
        matcopy_descriptor.flags = 0;
      }
    }


    /* TODO check JIT errors */
    if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  || libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE || libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM || libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL ) {
      if ( (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) ) {
        handle->code_bwd[0].xgemm.smm = libxsmm_smmdispatch(16, 16, 16, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
      } else {
        if (handle->exploit_duality == 1) {
          if (handle->padding_flag == 1) {
            handle->matcopy_bwd[0].xmatcopy = libxsmm_dispatch_mcopy(&matcopy_descriptor);
            matcopy_descriptor.n = 1;
            handle->matcopy_bwd[2].xmatcopy = libxsmm_dispatch_mcopy(&matcopy_descriptor);
          }
          fwd_equivalent_descriptor.n_variants = handle->n_variants;
          fwd_equivalent_descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
          if ( handle->n_variants == 1) {
            handle->code_bwd[0].pmm = libxsmm_create_xconv_forward(&fwd_equivalent_descriptor);
          } else {
            fwd_equivalent_descriptor.ofh_rb = hrb1;
            fwd_equivalent_descriptor.ofw_rb = wrb1;
            handle->code_bwd[0].pmm = libxsmm_create_xconv_forward(&fwd_equivalent_descriptor);
            fwd_equivalent_descriptor.ofh_rb = hrb2;
            fwd_equivalent_descriptor.ofw_rb = wrb2;
            handle->code_bwd[1].pmm = libxsmm_create_xconv_forward(&fwd_equivalent_descriptor);
            handle->bwd_ofh_rb = hrb1;
            handle->bwd_ofw_rb = wrb1;
            fwd_equivalent_descriptor.ofh_rb = hrb1;
            fwd_equivalent_descriptor.ofw_rb = wrb1;
          }

          /* enable jit code */
          handle->use_bwd_generic = 0;
        } else {
          /* disable jit code, use generic */
          handle->bwd_ofh_rb = 1;
          handle->bwd_ofw_rb = handle->ofw;
          handle->use_bwd_generic = 1;
        }
      }
    } else {
      assert(0/*should not happen*/);
    }

    {
      libxsmm_dnn_layer mirror_handle;
      handle->n_entries_bwd = (int*) malloc(handle->desc.threads * sizeof(int));
      memset( handle->n_entries_bwd, 0, handle->desc.threads * sizeof(int) );
      handle->compute_bwd_indices_ptrs = (int**) malloc(handle->desc.threads * sizeof(int*));
      memset( handle->compute_bwd_indices_ptrs, 0, handle->desc.threads * sizeof(int*) );
      handle->kernel_bwd_variant_ptrs = (char**) malloc(handle->desc.threads * sizeof(char*));
      memset( handle->kernel_bwd_variant_ptrs, 0, handle->desc.threads * sizeof(char*));
      handle->n_bwd_code_segments = (int*) malloc(handle->desc.threads * sizeof(int));
      memset( handle->n_bwd_code_segments, 0, handle->desc.threads * sizeof(int) );
      handle->bwd_code_segments = (segment_t**) malloc(handle->desc.threads * sizeof(segment_t*));
      memset( handle->bwd_code_segments, 0, handle->desc.threads * sizeof(segment_t*) );
      handle->ofh_bwd_start = (int*) malloc(handle->desc.threads * sizeof(int));
      memset( handle->ofh_bwd_start, 0, handle->desc.threads * sizeof(int) );
      handle->ofh_bwd_end = (int*) malloc(handle->desc.threads * sizeof(int));
      memset( handle->ofh_bwd_end, 0, handle->desc.threads * sizeof(int));
      handle->n_entries_trans_bwd = (int*) malloc(handle->desc.threads * sizeof(int));
      memset( handle->n_entries_trans_bwd, 0, handle->desc.threads * sizeof(int));
      handle->transpose_bwd_indices_ptrs = (int**) malloc(handle->desc.threads * sizeof(int*));
      memset( handle->transpose_bwd_indices_ptrs, 0, handle->desc.threads * sizeof(int*) );

      mirror_handle = *handle;
      mirror_handle.use_fwd_for_bwd = 1;
      mirror_handle.blocksifm_blocking = handle->blocksofm_blocking;
      mirror_handle.fwd_ofh_rb = handle->bwd_ofh_rb;
      mirror_handle.fwd_ofw_rb = handle->bwd_ofw_rb;
      mirror_handle.blocksofm = handle->blocksifm;
      mirror_handle.blocksofm_lp = handle->blocksifm_lp;
      mirror_handle.ifhp = handle->ofhp;
      mirror_handle.ifwp = handle->ofwp;
      mirror_handle.ofhp = handle->ifhp;
      mirror_handle.ofwp = handle->ifwp;
      mirror_handle.use_nts_fwd = handle->use_nts_bwd;
      mirror_handle.block_fwd_ofm = handle->block_fwd_ifm;
      mirror_handle.blocksifm = handle->blocksofm;
      mirror_handle.blocksifm_lp = handle->blocksofm_lp;
      mirror_handle.ofh = (handle->desc.H + 2 * handle->desc.pad_h - handle->desc.R) / handle->desc.v + 1;
      mirror_handle.ofw = (handle->desc.W + 2 * handle->desc.pad_w - handle->desc.S) / handle->desc.u + 1;
      mirror_handle.ifmblock = handle->ofmblock_lp;
      mirror_handle.ofmblock = handle->ifmblock_hp;
      mirror_handle.compute_fwd_indices_ptrs =  handle->compute_bwd_indices_ptrs;
      mirror_handle.n_entries_fwd = handle->n_entries_bwd;
      mirror_handle.kernel_fwd_variant_ptrs = handle->kernel_bwd_variant_ptrs;
      mirror_handle.n_fwd_code_segments = handle->n_bwd_code_segments;
      mirror_handle.fwd_code_segments = handle->bwd_code_segments;
      mirror_handle.ofh_fwd_start = handle->ofh_bwd_start;
      mirror_handle.ofh_fwd_end = handle->ofh_bwd_end;
      mirror_handle.perform_relu_in_kernel = (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) && (handle->use_nts_bwd == 1)) ? 1 : 0;
      handle->perform_relu_in_kernel = (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) && (handle->use_nts_bwd == 1)) ? 1 : 0;

      tune_fwd_blockings(&mirror_handle);
      status = libxsmm_dnn_perform_fwd_dryrun_direct(&mirror_handle);
    }

    /* In case overwrite is requested, generate zeroing kernel */
    if ( (handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0 )  {
      matzero_descriptor_overwrite.n = 1;
      if (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) {
        matzero_descriptor_overwrite.m = handle->desc.H*handle->ifwp*handle->ifmblock_hp;
        matzero_descriptor_overwrite.ldi = handle->desc.H*handle->ifwp*handle->ifmblock_hp;
        matzero_descriptor_overwrite.ldo = handle->desc.H*handle->ifwp*handle->ifmblock_hp;
      } else { /* Assumes NHWC format */
        status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
      }
      matzero_descriptor_overwrite.prefetch = 0;
      matzero_descriptor_overwrite.unroll_level = 2;
      matzero_descriptor_overwrite.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype_out);
      matzero_descriptor_overwrite.flags = LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE;
      handle->matcopy_bwd[1].xmatcopy = libxsmm_dispatch_mcopy(&matzero_descriptor_overwrite);

      if (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) {
        matzero_descriptor_overwrite.m = handle->ifwp*handle->ifmblock_hp;
        matzero_descriptor_overwrite.ldi = handle->ifwp*handle->ifmblock_hp;
        matzero_descriptor_overwrite.ldo = handle->ifwp*handle->ifmblock_hp;
      } else { /* Assumes NHWC format */
        status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
      }
      handle->matcopy_bwd[3].xmatcopy = libxsmm_dispatch_mcopy(&matzero_descriptor_overwrite);
    }

#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
    { /* compute kernel stream overhead */
      int ks_overhead = 0;
      int i = 0;
      ks_overhead += handle->desc.threads*5*sizeof(int);
      ks_overhead += handle->desc.threads*2*sizeof(int*);
      ks_overhead += handle->desc.threads*sizeof(char*);
      ks_overhead += handle->desc.threads*sizeof(segment_t*);
      for ( i = 0; i < handle->desc.threads; ++i ) {
        ks_overhead += ((handle->n_entries_bwd[i]*3)+3)*sizeof(int);
        ks_overhead += handle->n_entries_bwd[i]*sizeof(char);
        ks_overhead += handle->n_bwd_code_segments[i]*sizeof(segment_t);
        ks_overhead += (handle->n_entries_trans_bwd[i]+1)*sizeof(int);
      }
      printf("KS Overhead BWD in KB: %i \n", ks_overhead/1024);
    }
#endif
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_setup_upd( libxsmm_dnn_layer* handle, int *noarch ) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  int i = 0; /* general counting helper */
  handle->blocksimg_blocking = 1;

  /* FIXME: Do we still need that? Don't we unroll aggressively anyway here? */
  for (i = LIBXSMM_MIN(28, handle->ofh); i > 1; i--) {
    if (handle->ofh % i == 0) break;
  }
  handle->upd_ofh_rb = i;
  for (i = LIBXSMM_MIN(28, handle->ofw); i > 1; i--) {
    if (handle->ofw % i == 0) break;
  }
  handle->upd_ofw_rb = i;
  handle->blocksimg_blocking = 1;

  if (*noarch == 0)  {
    if ( handle->desc.N % handle->desc.threads == 0 ) {
      libxsmm_convolution_weight_update_descriptor descriptor;
      libxsmm_mcopy_descriptor matcopy_descriptor;
      libxsmm_mcopy_descriptor matzero_descriptor;

      /* init descriptors */
      memset( &descriptor, 0, sizeof(libxsmm_convolution_weight_update_descriptor) );
      memset( &matcopy_descriptor, 0, sizeof(libxsmm_mcopy_descriptor) );
      memset( &matzero_descriptor, 0, sizeof(libxsmm_mcopy_descriptor) );

      if (handle->padding_flag == 1) {
        descriptor.ifh_padded = handle->ifhp + 2 * handle->desc.pad_h;
        descriptor.ifw_padded = handle->ifwp + 2 * handle->desc.pad_w;
        matzero_descriptor.n = 1;
        if (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) {
          matcopy_descriptor.n = handle->ifhp;
          matcopy_descriptor.m = handle->ifwp * handle->ifmblock_hp;
          matzero_descriptor.m = descriptor.ifh_padded * descriptor.ifw_padded * handle->ifmblock_hp;
          matcopy_descriptor.ldi = handle->ifwp * handle->ifmblock_hp;
          matzero_descriptor.ldi = descriptor.ifh_padded * descriptor.ifw_padded * handle->ifmblock_hp;
          matcopy_descriptor.ldo = (handle->ifwp + 2*handle->desc.pad_w) * handle->ifmblock_hp;
          matzero_descriptor.ldo = descriptor.ifh_padded * descriptor.ifw_padded * handle->ifmblock_hp;
        } else { /* Assumes NHWC format */
          matcopy_descriptor.n = 1;
          matcopy_descriptor.m = handle->ifwp * handle->blocksifm * handle->ifmblock;
          matcopy_descriptor.ldi = handle->ifwp * handle->blocksifm * handle->ifmblock;
          matcopy_descriptor.ldo = (handle->ifwp + 2*handle->desc.pad_w) * handle->blocksifm * handle->ifmblock;
          matzero_descriptor.m = descriptor.ifw_padded * handle->blocksifm * handle->ifmblock;
          matzero_descriptor.ldi = descriptor.ifw_padded * handle->blocksifm * handle->ifmblock;
          matzero_descriptor.ldo = descriptor.ifw_padded * handle->blocksifm * handle->ifmblock;
        }
        matcopy_descriptor.prefetch = 1;
        matzero_descriptor.prefetch = 0;
        matcopy_descriptor.unroll_level = 2;
        matzero_descriptor.unroll_level = 2;
        matcopy_descriptor.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype_in);
        matzero_descriptor.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype_in);
        matcopy_descriptor.flags = 0;
        matzero_descriptor.flags = LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE;
      } else {
        descriptor.ifh_padded = handle->ifhp;
        descriptor.ifw_padded = handle->ifwp;
      }
      descriptor.ofm_block = handle->ofmblock;
      descriptor.ifm_block = handle->ifmblock;
      descriptor.ofm_block_lp = handle->ofmblock_lp;
      descriptor.ifm_block_hp = handle->ifmblock_hp;
      descriptor.fm_lp_block = handle->fm_lp_block;
      descriptor.kh = handle->desc.R;
      descriptor.kw = handle->desc.S;
      descriptor.stride_h = handle->desc.u;
      descriptor.stride_w = handle->desc.v;
      descriptor.blocks_ofm = handle->blocksofm;
      descriptor.blocks_ifm = handle->blocksifm;
      descriptor.ofh_padded = handle->ofhp;
      descriptor.ofw_padded = handle->ofwp;
      descriptor.ofw = handle->ofw;
      descriptor.ofh = handle->ofh;
      descriptor.ofh_rb = handle->upd_ofh_rb;
      descriptor.ofw_rb = handle->upd_ofw_rb;
      descriptor.ofh_unroll = 0;
      descriptor.ofw_unroll = 0;
      descriptor.datatype = handle->datatype_in;
      descriptor.datatype_itm = handle->datatype_out;
      descriptor.option = handle->desc.options;
      descriptor.format = (libxsmm_dnn_tensor_format)(handle->buffer_format | handle->filter_format);
      descriptor.use_nts = 0;
      descriptor.blocks_img = 1;
      descriptor.ncopies = handle->desc.threads;

      /* TODO check JIT errors */
      if ( (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
            libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
            libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM ||
            libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL) &&
          !((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32)) )
      {
        const unsigned int wu_each_iter_code_size = 10 * descriptor.ifm_block;
        const unsigned int wu_max_code_size = 8000;
        int upper_limit_ofw_rb = wu_max_code_size / wu_each_iter_code_size, upper_limit_ofh_rb = 0;
        unsigned int chunk_size;

        descriptor.ifm_unroll = 1;
        handle->enforce_sfma_kernel = 0;
        for (i = LIBXSMM_MIN(upper_limit_ofw_rb, LIBXSMM_MIN(56,handle->ofw)); i >= 1; i--) {
          if (handle->ofw % i == 0) break;
        }
        descriptor.ofw_rb = i;

        assert(0 != descriptor.ofw_rb);
        assert(0 != wu_each_iter_code_size);
        upper_limit_ofh_rb = wu_max_code_size / (descriptor.ofw_rb * wu_each_iter_code_size);
        for (i = LIBXSMM_MIN(upper_limit_ofh_rb, handle->ofh); i >= 1; i--) {
          if (handle->ofh % i == 0) break;
        }
        descriptor.ofh_rb =  i;

        chunk_size = 6 * descriptor.ofw_rb *  descriptor.ofh_rb * handle->ifmblock * (unsigned int)libxsmm_dnn_typesize(handle->datatype_out);
        while( chunk_size > 32000) {
          for (i = descriptor.ofh_rb-1; i >= 1; i--) {
            if (handle->ofh % i == 0) break;
          }
          if (i == 0) i = 1;
          descriptor.ofh_rb =  i;
          chunk_size = 6 * descriptor.ofw_rb *  descriptor.ofh_rb * handle->ifmblock * (unsigned int)libxsmm_dnn_typesize(handle->datatype_out);
          if ( i == 1) break;
        }

        if ( handle->ofh * handle->ofw * wu_each_iter_code_size <= wu_max_code_size) {
          descriptor.ofh_unroll = 1;
          descriptor.ofw_unroll = 1;
        } else if ((handle->ofh * descriptor.ofw_rb * wu_each_iter_code_size <= wu_max_code_size)) {
          descriptor.ofh_unroll = 1;
          descriptor.ofw_unroll = 0;
        } else if ((handle->ofw * descriptor.ofh_rb * wu_each_iter_code_size <= wu_max_code_size)) {
          descriptor.ofh_unroll = 0;
          descriptor.ofw_unroll = 1;
        } else {
          descriptor.ofh_unroll = 0;
          descriptor.ofw_unroll = 0;
        }

        if (handle->desc.R != 1 && handle->desc.S != 1  && (handle->desc.C == 3 && handle->desc.K == 64 && handle->ofw%14 == 0) && (handle->desc.u != 1 || handle->desc.v != 1)) {
          descriptor.use_fastpath = 0;
          handle->use_fastpath = 0;
          handle->enforce_sfma_kernel = 0;
        } else {
          descriptor.use_fastpath = 1;
          handle->use_fastpath = 1;
        }

        descriptor.ofw_rb = 14;
        descriptor.ofh_rb = 4;

        handle->upd_ofh_rb = descriptor.ofh_rb;
        handle->upd_ofw_rb = descriptor.ofw_rb;
        descriptor.transpose_ofw_ifm = 0;
        handle->use_hybrid_wu_parallelism = 0;

        if ( handle->use_lp_kernel == 1 && ((libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL || libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE) && (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16 || handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) ) ) {
          handle->use_vperm_transposes = 1;
        } else {
          handle->use_vperm_transposes = 0;
        }

        if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16 || handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) {
          if (libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE || libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL) {
            if (handle->ofwp % 2 == 0) {
              handle->avoid_output_trans = 1;
            } else {
              handle->avoid_output_trans = 0;
            }
            descriptor.avoid_output_trans = handle->avoid_output_trans;
            if ( ((handle->desc.R == 1 && handle->desc.S == 1) || handle->padding_flag == 0) && handle->desc.u == 1 && handle->desc.v == 1) {
              handle->avoid_input_trans = 1;
            } else {
              handle->avoid_input_trans = 0;
            }
          }
        } else if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) {
          if (libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE || libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL ) {
            handle->avoid_output_trans = 0;
            handle->avoid_input_trans = 0;
          }
        }

        if (handle->use_fastpath == 1) {
          /* Here starts logic for calculating RB factors for UPD when KS are enabled  */
          int ifw_padded, qfma_padding, kernel_ifw_padded/*, kernel_ofw_padded*/;
          int kernel_ofw_compute;
          int kernel_ofw_fake_pixels;
          int kernel_ofw;
          int padding_target;
          int output_lp_padding = 0;
          int enforce_sfma_kernel = 0;
          handle->output_lp_padding = 0;

          if ((libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE || libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC || libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL || ( (handle->desc.R!=1 || handle->desc.S!=1 || handle->desc.pad_h!=0 || handle->desc.pad_w!=0) && (handle->desc.u!=1 || handle->desc.v!=1) )) && (handle->use_lp_kernel == 0)  ) {
            enforce_sfma_kernel = 1;
          }

          handle->enforce_sfma_kernel = enforce_sfma_kernel;

          if (handle->use_lp_kernel == 0) {
            padding_target = 4;
          } else {
            padding_target = 8;
            output_lp_padding = handle->ofwp%2;
            if (libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE || libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL) {
              if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16 || handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) {
                padding_target = 2;
              } else {
                padding_target = 4;
                output_lp_padding = (handle->ofwp % 4 == 0) ? 0 : padding_target - handle->ofwp % 4;
              }
            }
            handle->output_lp_padding = output_lp_padding;
          }

          if (handle->desc.R == 1 && handle->desc.S == 1 && handle->desc.pad_h == 0 && handle->desc.pad_w == 0 && (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM || ((libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE || libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC || libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL) && handle->use_lp_kernel == 1)) && (handle->desc.u != 1 || handle->desc.v != 1)) {
            handle->resize_input = 1;
            handle->trans_ofw_ifm = 1;
            handle->ifwp_resized = handle->ifwp/handle->desc.u;
            handle->ifhp_resized = handle->ifhp/handle->desc.v;
            descriptor.stride_h = 1;
            descriptor.stride_w = 1;
            descriptor.ifh_padded = handle->ifhp_resized;
          } else {
            handle->resize_input = 0;
          }

          if (handle->resize_input == 1 ) {
            ifw_padded = handle->ifwp_resized;
          } else {
            ifw_padded = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
          }

          if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM && (enforce_sfma_kernel == 0) ) {
            qfma_padding = (handle->desc.W % padding_target == 0) ? 0 : padding_target - handle->desc.W % padding_target;
          } else if (libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE || libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL || libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC || enforce_sfma_kernel == 1) {
            if (handle->use_lp_kernel == 1) {
              qfma_padding = (ifw_padded % padding_target == 0) ? 0 : padding_target - ifw_padded % padding_target;
            } else {
              qfma_padding = 0;
            }
          } else {
            qfma_padding = 0;
          }

          kernel_ifw_padded = ifw_padded + qfma_padding;
          handle->qfma_input_pad = qfma_padding;
          descriptor.ifw_padded = kernel_ifw_padded;

          descriptor.ofw_padded = handle->ofwp+output_lp_padding;
          descriptor.ofh_padded = handle->ofhp;

          if (handle->desc.R == 1 && handle->desc.S == 1) {
            kernel_ofw_compute = handle->ofwp+output_lp_padding;
          } else {
            if (handle->padding_flag == 1) {
              kernel_ofw_compute = handle->ofwp+output_lp_padding;
            } else {
              kernel_ofw_compute = handle->ofw+output_lp_padding;
            }
          }

          if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM && (enforce_sfma_kernel == 0)) {
            kernel_ofw_fake_pixels = (kernel_ofw_compute % padding_target == 0) ? 0 : padding_target - kernel_ofw_compute % padding_target;
          } else {
            kernel_ofw_fake_pixels = 0;
          }

          kernel_ofw = kernel_ofw_compute + kernel_ofw_fake_pixels;
          descriptor.ofw_fake_pixels = kernel_ofw_fake_pixels;
          descriptor.ofw_rb = kernel_ofw;
          descriptor.ofh_rb = handle->ofh;

          while (   descriptor.ofw_rb  *  descriptor.ofh_rb > 196 ) {
            descriptor.ofh_rb = descriptor.ofh_rb / 2;
          }

          if (descriptor.ofh_rb == 0) {
            descriptor.ofh_rb = 1;
          }

          while (  handle->ofh % descriptor.ofh_rb != 0 ) {
            descriptor.ofh_rb--;
          }

          if (handle->ofh == 14 && (libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE || libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC || libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL || enforce_sfma_kernel==1)) {
            descriptor.ofh_rb = 2;
          }

          descriptor.use_nts = 1;
          descriptor.blocks_h = handle->ofh / descriptor.ofh_rb;
          handle->upd_ofh_rb = descriptor.ofh_rb * descriptor.blocks_h;
          handle->upd_ofw_rb = kernel_ofw;

          if ( handle->ofh == 28) {
            descriptor.use_nts = 0;
            descriptor.blocks_h = 1;
            handle->upd_ofh_rb = 2;
            descriptor.ofh_rb = 2;
            if ( handle->blocksofm == 32 && handle->blocksifm == 16 ) {
              handle->upd_ofh_rb = 7;
              descriptor.ofh_rb = 7;
            }
            if ( handle->blocksofm == 8 && handle->blocksifm == 16 ) {
              handle->upd_ofh_rb = 1;
              descriptor.ofh_rb = 1;
            }
            if ( handle->desc.R == 3 && handle->desc.S == 3 ) {
              handle->upd_ofh_rb = 7;
              descriptor.ofh_rb = 7;
            }
          }

          if ( handle->ofh == 56 ) {
            descriptor.use_nts = 0;
            descriptor.ofh_rb = 1;
            descriptor.blocks_h = 1;
            handle->upd_ofh_rb = 1;
            if ( handle->desc.R == 3 && handle->desc.S == 3 ) {
              handle->upd_ofh_rb = 2;
              descriptor.ofh_rb = 2;
            }
          }
          if ( handle->ofh == 35 || handle->ofh == 149 || handle->ofh == 71  ||  handle->ofh == 147 || handle->ofh == 73   ) {
            descriptor.use_nts = 0;
            descriptor.ofh_rb = 1;
            descriptor.blocks_h = 1;
            handle->upd_ofh_rb = 1;
            if ( handle->desc.R == 3 && handle->desc.S == 3 ) {
              handle->upd_ofh_rb = 1;
              descriptor.ofh_rb = 1;
            }
          }

          if (handle->ofh == 28 || handle->ofh == 35 || handle->ofh == 56 ||  handle->ofh == 71  || handle->ofh == 149 ||  handle->ofh == 147 || handle->ofh == 73  || ( handle->ofh == 14 && handle->desc.threads > 1 && ( handle->desc.C == 512 && (handle->desc.K == 1024 || handle->desc.K == 256) ) )) {
            if ((descriptor.use_nts == 1) && (handle->desc.threads != handle->desc.N)) {
              descriptor.use_nts = 0;
            }
            handle->use_hybrid_wu_parallelism = 0;
            handle->weight_copies = handle->desc.threads;
            descriptor.ncopies = handle->weight_copies;
            handle->blocksimg_blocking = 1;
            descriptor.blocks_img = 1;
            handle->reduce_weights = 1;
          } else {
            int spread_out = 0;
            /*if (handle->desc.threads % 7 == 0) {
              spread_out = 7;
              } else*/
            if ( handle->desc.threads % 4 == 0) {
              spread_out = 4;
            } else if (handle->desc.threads % 2 == 0) {
              spread_out = 2;
            } else {
              spread_out = 1;
            }
            if (spread_out == 1 && handle->desc.threads > 1) {
              handle->use_hybrid_wu_parallelism = 0;
              handle->weight_copies = handle->desc.threads;
              handle->blocksimg_blocking = 1;
              descriptor.blocks_img = 1;
            } else {
              handle->use_hybrid_wu_parallelism = 1;
              handle->weight_copies = handle->desc.threads/spread_out;
              descriptor.ncopies = handle->weight_copies;
              handle->blocksimg_blocking = spread_out * (handle->desc.N/handle->desc.threads);
              if (handle->blocksimg_blocking <= 0) handle->blocksimg_blocking = 1;
              descriptor.blocks_img = handle->blocksimg_blocking;
              handle->reduce_weights = 1;
            }
          }

          if ((libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL || libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE || libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC) &&  (handle->ofh == 7 || (handle->ofh == 14 && handle->desc.R == 1 && handle->desc.S == 1 && handle->desc.u == 1 && handle->desc.v == 1)) ) {
            descriptor.use_nts = 0;
            descriptor.blocks_h = handle->ofh / descriptor.ofh_rb;
            handle->upd_ofh_rb = descriptor.ofh_rb * descriptor.blocks_h;
            handle->use_hybrid_wu_parallelism = 1;
            handle->weight_copies = 1;
            descriptor.ncopies = 1;
            handle->blocksimg_blocking = 1;
            descriptor.blocks_img = handle->blocksimg_blocking;
            handle->reduce_weights = 0;
          } else {
            handle->reduce_weights = 1;
          }
        }

        handle->use_nts_upd = descriptor.use_nts;

        /* NONE */
        if (handle->padding_flag == 1) {
          handle->matcopy_upd[0].xmatcopy = libxsmm_dispatch_mcopy(&matcopy_descriptor);
          handle->matcopy_upd[1].xmatcopy = libxsmm_dispatch_mcopy(&matzero_descriptor);
        }
        descriptor.transpose_ofw_ifm = 0;
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
        if ( (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) ) {
          handle->code_upd[0].xgemm.smm = libxsmm_smmdispatch(16, 16, 16, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        } else {
          /*handle->code_upd[0].pmm = libxsmm_create_xconv_update_weights(&descriptor);*/
        }
        /*ALL*/
        descriptor.transpose_ofw_ifm = 0;
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
        handle->code_upd[0].pmm = libxsmm_create_xconv_update_weights(&descriptor);
        /*TRANSPOSE ALL*/
        descriptor.transpose_ofw_ifm = 1;
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
        handle->code_upd[1].pmm = libxsmm_create_xconv_update_weights(&descriptor);

        /* enable JIT code path */
        handle->use_upd_generic = 0;
      } else {
        assert(0/*should not happen*/);
      }

      if ( handle->use_thread_private_jit ) {
        handle->trans_ofw_ifm = 0;
        /* Determine if we will be using thread private filters  */
        if ( (handle->blocksifm_lp * handle->blocksofm < handle->desc.threads) ) {
          handle->use_thread_private_filter = 1;
          /* determine if we will transpose input  */
          if ( ((libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM && handle->enforce_sfma_kernel == 0 ) && (handle->upd_ofw_rb%4 == 0)) || ((libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC) || ((libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE || libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL) && handle->use_lp_kernel == 1 && ( ((handle->desc.R !=1 || handle->desc.S != 1) && handle->padding_flag == 1) || handle->desc.u != 1 || handle->desc.v != 1 || handle->desc.W%2 != 0 )) ) ) {
            handle->trans_ofw_ifm = 1;
          }
        } else {
          handle->use_thread_private_filter = 0;
          /* determine if we will transpose input  */
          if ( ((libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM && handle->enforce_sfma_kernel == 0) && (handle->upd_ofw_rb%4 == 0)) || ((libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC)  || ((libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE || libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL) && handle->use_lp_kernel == 1 && (((handle->desc.R !=1 || handle->desc.S != 1) && handle->padding_flag == 1) || handle->desc.u != 1 || handle->desc.v != 1 ||  handle->desc.W%2 != 0 )) ) ) {
            handle->trans_ofw_ifm = 1;
            if ( handle->desc.R !=1 && handle->desc.S != 1 && ( handle->desc.u !=1 || handle->desc.v != 1 )  ) {
              handle->trans_ofw_ifm = 0;
            }
          }
        }

        if ((libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE || libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL) && handle->use_lp_kernel == 1 && handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) {
          handle->trans_ofw_ifm = 1;
        }

        if (handle->use_fastpath == 0 || handle->enforce_sfma_kernel == 1) {
          handle->trans_ofw_ifm = 0;
        }

        handle->n_entries_upd = (int*) malloc(handle->desc.threads * sizeof(int));
        memset( handle->n_entries_upd, 0, handle->desc.threads * sizeof(int) );
        handle->compute_upd_indices_ptrs = (int**) malloc(handle->desc.threads * sizeof(int*));
        memset( handle->compute_upd_indices_ptrs, 0, handle->desc.threads * sizeof(int*) );
        handle->kernel_upd_variant_ptrs = (char**) malloc(handle->desc.threads * sizeof(char*));
        memset( handle->kernel_upd_variant_ptrs, 0, handle->desc.threads * sizeof(char*) );
        handle->n_upd_code_segments = (int*) malloc(handle->desc.threads * sizeof(int));
        memset( handle->n_upd_code_segments, 0, handle->desc.threads * sizeof(int) );
        handle->upd_code_segments = (segment_t**) malloc(handle->desc.threads * sizeof(segment_t*));
        memset( handle->upd_code_segments, 0, handle->desc.threads * sizeof(segment_t*));
        handle->n_entries_init_upd = (int*) malloc(handle->desc.threads * sizeof(int));
        memset( handle->n_entries_init_upd, 0, handle->desc.threads * sizeof(int) );
        handle->init_upd_indices_ptrs = (int**) malloc(handle->desc.threads * sizeof(int*));
        memset( handle->init_upd_indices_ptrs, 0, handle->desc.threads * sizeof(int*) );
        handle->n_entries_copy_upd = (int*) malloc(handle->desc.threads * sizeof(int));
        memset( handle->n_entries_copy_upd, 0, handle->desc.threads * sizeof(int) );
        handle->copy_upd_indices_ptrs = (int**) malloc(handle->desc.threads * sizeof(int*));
        memset( handle->copy_upd_indices_ptrs, 0, handle->desc.threads * sizeof(int*) );

        matzero_descriptor.n = 1;
        matzero_descriptor.m = handle->blocksofm*handle->blocksifm*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock;
        matzero_descriptor.ldi = handle->blocksofm*handle->blocksifm*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock;
        matzero_descriptor.ldo = handle->blocksofm*handle->blocksifm*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock;
        matzero_descriptor.prefetch = 0;
        matzero_descriptor.unroll_level = 6;
        matzero_descriptor.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype_out);
        matzero_descriptor.flags = LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE;
        handle->matcopy_upd[2].xmatcopy = libxsmm_dispatch_mcopy(&matzero_descriptor);

        /* Perform the dry-run and generate thread private jit indices to be used for the convolutions */
        tune_upd_blockings(handle);
        status = libxsmm_dnn_perform_upd_dryrun_direct(handle);

#if defined(LIBXSMM_DNN_HANDLE_DEBUG)
        { /* compute kernel stream overhead */
          int ks_overhead = 0;
          ks_overhead += handle->desc.threads*4*sizeof(int);
          ks_overhead += handle->desc.threads*3*sizeof(int*);
          ks_overhead += handle->desc.threads*sizeof(char*);
          ks_overhead += handle->desc.threads*sizeof(segment_t*);
          for ( i = 0; i < handle->desc.threads; ++i ) {
            ks_overhead += ((handle->n_entries_upd[i]*3)+3)*sizeof(int);
            ks_overhead += handle->n_entries_upd[i]*sizeof(char);
            ks_overhead += handle->n_upd_code_segments[i]*sizeof(segment_t);
            ks_overhead += (handle->n_entries_copy_upd[i]+1)*sizeof(int);
            ks_overhead += (handle->n_entries_init_upd[i]+1)*sizeof(int);
          }
          printf("KS Overhead UPD in KB: %i \n", ks_overhead/1024 );
        }
#endif
      }
    } else {
      handle->use_upd_generic = 1;
    }
  } /* end of weight-update handle */

  return status;
}


#undef MIXED
#undef KHWC
#undef HWKC
#undef CHWK
#undef HWCK

