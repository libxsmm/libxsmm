/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Alexander Heinecke, Rajkishore Barik, Ankush Mandal, Evangelos Georganas (Intel Corp.)
 ******************************************************************************/
#include "libxsmm_dnn_handle.h"
#include "libxsmm_main.h"
#include <libxsmm.h>
#include "libxsmm_dnn_dryruns.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_internal_create_conv_handle_direct( libxsmm_dnn_layer* handle ) {
  /* flag to test if we found an architecture which is supported */
  int noarch = 1;
  /* general counting helper */
  int i = 0;
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  const char *const env = getenv("LIBXSMM_DNN_INTERNAL_FORMAT");
  const char *const env_jit = getenv("LIBXSMM_DNN_THREAD_PRIVATE_JIT");
  /* const char *const env_ifm = getenv("DISABLE_IFM");*/
  int disable_ifm_in = 0;

  /*if ( 0 == env_ifm || 0 == *env_ifm) {
    disable_ifm_in = 0;
  } else {
    disable_ifm_in = 1;
  }*/

  /* 
  if (handle->ofh == 7 && handle->desc.u == 2) {
    disable_ifm_in = 1;
  }  
  */

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

  handle->use_fwd_for_bwd = 0;
  if ( 0 == env_jit || 0 == *env_jit) {
    /* By default do not do any thread private jitting */
    handle->use_thread_private_jit = 0;
  } else {
    handle->use_thread_private_jit = 1;
    /* If we do not have AVX512 arch disable kernel streams  */
    if (libxsmm_target_archid != LIBXSMM_X86_AVX512_MIC  &&
        libxsmm_target_archid != LIBXSMM_X86_AVX512_CORE &&
        libxsmm_target_archid != LIBXSMM_X86_AVX512_KNM ) {
      handle->use_thread_private_jit = 0;
    }
    /* If we do not follow FP32 path disable kernel streams  */
    if ( (handle->datatype != LIBXSMM_DNN_DATATYPE_F32) || (handle->datatype_itm != LIBXSMM_DNN_DATATYPE_F32) ) {
      handle->use_thread_private_jit = 0;
    }
    /* If we use any options/fuse ops, disable kernel streams */
    if ( (handle->desc.fuse_ops != LIBXSMM_DNN_CONV_FUSE_NONE) ) {
      handle->use_thread_private_jit = 0;
    }
    /* If we do not run on custom/custom format, disable kernel streams */
    if (handle->buffer_format != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM || handle->filter_format != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM ) {
      handle->use_thread_private_jit = 0;
    }
  }

  if ( (handle->use_thread_private_jit > 0) && ( (handle->desc.R == 1 && handle->desc.S == 1) || (handle->desc.u == 1 && handle->desc.v == 1) ) )  {
    handle->exploit_duality = 1;  
  } else {
    handle->exploit_duality = 0;
  }

  /* now architecture specific */
  if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
      libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
      libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
  {
    noarch = 0;
#define LIBXSMM_FWD_OFH_BLOCKING
#if defined(LIBXSMM_FWD_OFH_BLOCKING)
    if ( ((handle->ofw < 15) && (handle->ofh % 2 == 0) && (handle->desc.S == 1)) ||
        ((handle->ofw < 15) && (handle->ofh % 2 == 0) && (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM)) ) {
      handle->fwd_ofw_rb = handle->ofw;
      handle->fwd_ofh_rb = 2;
      /* on AVX512_CORE and int this only works for smaller 13 */
      if ( (((handle->datatype == LIBXSMM_DNN_DATATYPE_I16) ||
              (handle->datatype == LIBXSMM_DNN_DATATYPE_I8)) &&
            (libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE)) &&
          (handle->ofw > 12) ) {
        handle->fwd_ofh_rb = 1;
      }
    }
    else {
#endif
      /* we need additional temp registers when running with int on AVX512_CORE */
      if ( ((libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE) &&
            (handle->datatype == LIBXSMM_DNN_DATATYPE_I16) &&
            (handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I32)) ||
          ((libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE) &&
           (handle->datatype == LIBXSMM_DNN_DATATYPE_I8) &&
           (handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I16) &&
           ((handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) ) {
        for (i = 26; i > 1; --i) {
          if (handle->ofw % i == 0) break;
        }
        /* for 32 accumulation we need even one register more */
      } else if ( (handle->datatype == LIBXSMM_DNN_DATATYPE_I8) &&
          (handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) &&
          ((handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) {
        for (i = 25; i > 1; --i) {
          if (handle->ofw % i == 0) break;
        }
      } else {
        for (i = 28; i > 1; --i) {
          if (handle->ofw % i == 0) break;
        }
      }
      handle->fwd_ofw_rb = i;
      handle->fwd_ofh_rb = 1;
#if defined(LIBXSMM_FWD_OFH_BLOCKING)
    }
#endif

#define LIBXSMM_BWD_OFW_BLOCKING
#if defined(LIBXSMM_BWD_OFW_BLOCKING)
    handle->bwd_ofh_rb = 1;
    for (i = LIBXSMM_MIN(24, handle->ofw); i > 1; i--) {
      if (handle->ofw % i == 0) break;
    }
    handle->bwd_ofw_rb = i;
#endif

#define LIBXSMM_UPD_OFH_BLOCKING
#if defined(LIBXSMM_UPD_OFH_BLOCKING)
    for (i = LIBXSMM_MIN(28, handle->ofh); i > 1; i--) {
      if (handle->ofh % i == 0) break;
    }
    handle->upd_ofh_rb = i;
    for (i = LIBXSMM_MIN(28, handle->ofw); i > 1; i--) {
      if (handle->ofw % i == 0) break;
    }
    handle->upd_ofw_rb = i;
#endif

    /* calculate blockings */
    if ( (handle->datatype == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32) ) {
      handle->ifmblock = (handle->desc.C >=16) ? 16 : handle->desc.C;
      handle->ofmblock = (handle->desc.K >=16) ? 16 : handle->desc.K;
      handle->fm_lp_block = 1;
    }
    else if ( (handle->datatype == LIBXSMM_DNN_DATATYPE_I16) && (handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) ) {
      handle->ifmblock = (handle->desc.C >=16) ? 16 : (handle->desc.C/2);
      handle->ofmblock = (handle->desc.K >=16) ? 16 : (handle->desc.K/2);
      handle->fm_lp_block = 2;
      if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC ) {
        status = LIBXSMM_DNN_WARN_FALLBACK;
        handle->ifmblock = 1;
        handle->ofmblock = 1;
        handle->fm_lp_block = 1;
        noarch = 1;
      }
    }
    else if ( (handle->datatype == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I16)
        && ((handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) {
      handle->ifmblock = (handle->desc.C >=32) ? 32 : (handle->desc.C/2);
      handle->ofmblock = (handle->desc.K >=32) ? 32 : (handle->desc.K/2);
      handle->fm_lp_block = 2;
      if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM ) {
        status = LIBXSMM_DNN_WARN_FALLBACK;
        handle->ifmblock = 1;
        handle->ofmblock = 1;
        handle->fm_lp_block = 1;
        noarch = 1;
      }
    }
    else if ( (handle->datatype == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I32)
        && ((handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) {
      handle->ifmblock = (handle->desc.C >=16) ? 16 : (handle->desc.C/4);
      handle->ofmblock = (handle->desc.K >=16) ? 16 : (handle->desc.K/4);
      handle->fm_lp_block = 4;
      if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM ) {
        status = LIBXSMM_DNN_WARN_FALLBACK;
        handle->ifmblock = 1;
        handle->ofmblock = 1;
        handle->fm_lp_block = 1;
        noarch = 1;
      }
    }
    else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      free(handle);
      handle = 0;
      return status;
    }

    /* if we have 1x1 let's bring some ifms into the kernel for forward to increase accumulation chain length on AVX512 */
    /* @TODO we limit ifm blocking in JIT to custom format 1 for now */
    if ( (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1) ) {
      if ( (handle->ifmblock%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*512) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksifm_blocking = 512;
      } else if ( (handle->ifmblock%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*256) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksifm_blocking = 256;
      } else if ( (handle->ifmblock%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*128) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksifm_blocking = 128;
      } else if ( (handle->ifmblock%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*64) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksifm_blocking = 64;
      } else if ( (handle->ifmblock%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*32) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksifm_blocking = 32;
      } else if ( (handle->ifmblock%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*16) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksifm_blocking = 16;
      } else if ( (handle->ifmblock%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*8) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksifm_blocking = 8;
      } else if ( (handle->ifmblock%16 == 0) &&  (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*4) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksifm_blocking = 4;
      } else if ( (handle->ifmblock%16 == 0) && (handle->desc.C%(handle->ifmblock*handle->fm_lp_block*2) == 0) && (handle->desc.R == 1) && (handle->desc.S == 1) ) {
        handle->blocksifm_blocking = 2;
      } else {
        handle->blocksifm_blocking = 1;
      }
    } else {
      handle->blocksifm_blocking = 1;
    }

    /* Ditto for ofms in BWD  */
    if ( (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1) ) {
      if ( (handle->ofmblock%16 == 0) &&  (handle->desc.K%(handle->ofmblock*handle->fm_lp_block*512) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksofm_blocking = 512;
      } else if ( (handle->ofmblock%16 == 0) &&  (handle->desc.K%(handle->ofmblock*handle->fm_lp_block*256) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksofm_blocking = 256;
      } else if ( (handle->ofmblock%16 == 0) &&  (handle->desc.K%(handle->ofmblock*handle->fm_lp_block*128) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksofm_blocking = 128;
      } else if ( (handle->ofmblock%16 == 0) &&  (handle->desc.K%(handle->ofmblock*handle->fm_lp_block*64) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksofm_blocking = 64;
      } else if ( (handle->ofmblock%16 == 0) &&  (handle->desc.K%(handle->ofmblock*handle->fm_lp_block*32) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksofm_blocking = 32;
      } else if ( (handle->ofmblock%16 == 0) &&  (handle->desc.K%(handle->ofmblock*handle->fm_lp_block*16) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksofm_blocking = 16;
      } else if ( (handle->ofmblock%16 == 0) &&  (handle->desc.K%(handle->ofmblock*handle->fm_lp_block*8) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksofm_blocking = 8;
      } else if ( (handle->ofmblock%16 == 0) &&  (handle->desc.K%(handle->ofmblock*handle->fm_lp_block*4) == 0) && (handle->desc.R == 1) &&  (handle->desc.S == 1) ) {
        handle->blocksofm_blocking = 4;
      } else if ( (handle->ofmblock%16 == 0) && (handle->desc.K%(handle->ofmblock*handle->fm_lp_block*2) == 0) && (handle->desc.R == 1) && (handle->desc.S == 1) ) {
        handle->blocksofm_blocking = 2;
      } else {
        handle->blocksofm_blocking = 1;
      }
    } else {
      handle->blocksofm_blocking = 1;
    }

    /* Logic for L2 tiling  */
    unsigned int input_block_size = 28 * handle->blocksifm_blocking * 64;
    unsigned int output_block_size = 28 * 64;
    unsigned int weight_ofm_block = LIBXSMM_MIN(16, handle->blocksofm);
    unsigned int weight_ifm_block = LIBXSMM_MIN(16, handle->blocksifm);
    unsigned int weight_block_size = (handle->blocksifm_blocking * 64) * 64/2; /*  (weight_ofm_block * 64)/2; */
    unsigned int total_size = input_block_size + output_block_size + weight_block_size;

#if 0
    if ( handle->blocksifm_blocking != 1 ) {
      while (total_size > 450000 ) {
        handle->blocksifm_blocking =  handle->blocksifm_blocking / 2;
        input_block_size = 28 * handle->blocksifm_blocking * 64;
        weight_block_size =  (handle->blocksifm_blocking * 64) * 64/2;   /*  (weight_ofm_block * 64)/2; */
        total_size = input_block_size + output_block_size + weight_block_size;
        if (handle->blocksifm_blocking == 1 ) {
          break;
        }
      }
      input_block_size = 28 * handle->blocksifm_blocking * 64;
      weight_block_size =  (handle->blocksifm_blocking * 64) * 64/2; /*  ((weight_ofm_block+2) * 64/2);*/
      total_size = input_block_size + output_block_size + weight_block_size;

      /* Maximize reuse by bringing more ofms inside */
#if 0
      while ( total_size < 450000 ) {
         weight_ofm_block += 2;
         input_block_size = 28 * handle->blocksifm_blocking * 64;
         weight_block_size =  (handle->blocksifm_blocking * 64) * ((weight_ofm_block+2) * 64)/2; 
         total_size = input_block_size + output_block_size + weight_block_size;
      }
#endif
      handle->block_fwd_ofm = weight_ofm_block;

      /*printf("Picked IFM blocking equal to %d and OFM blocking equal to %d (total_size is %d)\n", handle->blocksifm_blocking, weight_ofm_block, total_size/1024);*/
    }

#endif

    handle->block_fwd_ofm = weight_ofm_block;
    handle->block_bwd_ifm = weight_ifm_block;

    /*if (disable_ifm_in == 1) {   
       handle->blocksifm_blocking = 1;
    }*/

    /* when we chose overwrite and we loop over all ifms, then let's use streaming stores */
    if (    ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0)
         && (handle->desc.C == handle->blocksifm_blocking*handle->ifmblock*handle->fm_lp_block)
         && (handle->datatype == LIBXSMM_DNN_DATATYPE_F32) ) {
      handle->use_nts_fwd = 1;
    } else {
      handle->use_nts_fwd = 0;
    }

    if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) 
         && (handle->desc.K == handle->blocksofm_blocking*handle->ofmblock*handle->fm_lp_block) 
         && (handle->datatype == LIBXSMM_DNN_DATATYPE_F32) ) {
      handle->use_nts_bwd = 1;
    } else {
      handle->use_nts_bwd = 0;
    }

    /* Adjust blocking factors if custom_2 format is requested */
    if ((handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2)) {
      if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32) {
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

  } else if ( libxsmm_target_archid == LIBXSMM_X86_AVX2 ) {
    noarch = 0;

    /* get max. blocking forward */
    handle->fwd_ofh_rb = 1;
    if ( handle->ofw > 3 ) {
      handle->fwd_ofw_rb = 3;
      handle->fwd_ofw_rb_2 = handle->ofw % 3;
    } else {
      handle->fwd_ofw_rb = handle->ofw;
      handle->fwd_ofw_rb_2 = 0;
    }

    /* get max. blocking backward, ofw is blocked internally */
    handle->bwd_ofw_rb = handle->ofw;
    handle->bwd_ofh_rb = 1;

#define LIBXSMM_UPD_OFH_BLOCKING
#if defined(LIBXSMM_UPD_OFH_BLOCKING)
    for (i = LIBXSMM_MIN(3, handle->ofh); i > 1; i--) {
      if (handle->ofh % i == 0) break;
    }
    handle->upd_ofh_rb = i;
    for (i = LIBXSMM_MIN(3, handle->ofw); i > 1; i--) {
      if (handle->ofw % i == 0) break;
    }
    handle->upd_ofw_rb = i;
#endif

    /* calculate blockings */
    if ( (handle->datatype == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32) ) {
      handle->ifmblock = (handle->desc.C >=32) ? 32 : handle->desc.C;
      handle->ofmblock = (handle->desc.K >=32) ? 32 : handle->desc.K;
      handle->fm_lp_block = 1;

      /* let's find out if we need a smaller blocking */
      if ( handle->desc.C % handle->ifmblock != 0 ) {
        if ( handle->desc.C % 16 == 0 ) {
          handle->ifmblock = 16;
        } else if ( handle->desc.C % 8 == 0 ) {
          handle->ifmblock = 8;
        } else {
          noarch = 1;
          status = LIBXSMM_DNN_WARN_FALLBACK;
          handle->ifmblock = 1;
          handle->ofmblock = 1;
        }
      }

      if ( (handle->desc.K % handle->ofmblock != 0) && (noarch == 0) ) {
        if ( handle->desc.K % 16 == 0 ) {
          handle->ofmblock = 16;
        } else if ( handle->desc.K % 8 == 0 ) {
          handle->ofmblock = 8;
        } else {
          noarch = 1;
          status = LIBXSMM_DNN_WARN_FALLBACK;
          handle->ifmblock = 1;
          handle->ofmblock = 1;
        }
      }
    }
    else if ( (handle->datatype == LIBXSMM_DNN_DATATYPE_I16) && (handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) ) {
      status = LIBXSMM_DNN_WARN_FALLBACK;
      handle->ifmblock = 1;
      handle->ofmblock = 1;
      handle->fm_lp_block = 1;
      noarch = 1;
    }
    else if ( (handle->datatype == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I16)
        && ((handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) {
      status = LIBXSMM_DNN_WARN_FALLBACK;
      handle->ifmblock = 1;
      handle->ofmblock = 1;
      handle->fm_lp_block = 1;
      noarch = 1;
    }
    else if ( (handle->datatype == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I32)
        && ((handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) {
      status = LIBXSMM_DNN_WARN_FALLBACK;
      handle->ifmblock = 1;
      handle->ofmblock = 1;
      handle->fm_lp_block = 1;
      noarch = 1;
    }
    else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      free(handle);
      handle = 0;
      return status;
    }
    /* Adjust blocking factors if custom_2 format is requested */
    if ((handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2)) {
      if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32)  {
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

  } else {
    status = LIBXSMM_DNN_WARN_FALLBACK;
    handle->ifmblock = 1;
    handle->ofmblock = 1;
    handle->fm_lp_block = 1;
  }

  /* Let's calculate how many blocks we need */
  handle->blocksifm = handle->desc.C / (handle->ifmblock * handle->fm_lp_block);
  handle->blocksofm = handle->desc.K / (handle->ofmblock * handle->fm_lp_block);

  /* Calculate number of image blocks in case of custom_2 format */
  if ( (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) ) {
    handle->nBImg = handle->desc.N / handle->nbImg;
  }

  /* Let's check that we can actually block */
  if ( (handle->desc.C % (handle->ifmblock * handle->fm_lp_block) != 0) ||
      (handle->desc.K % (handle->ofmblock * handle->fm_lp_block) != 0)    )
  {
    handle->custom_format_type = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1;
    status = LIBXSMM_DNN_WARN_FALLBACK;
    handle->ifmblock = 1;
    handle->ofmblock = 1;
    handle->fm_lp_block = 1;
    handle->blocksifm = handle->desc.C / handle->ifmblock;
    handle->blocksofm = handle->desc.K / handle->ofmblock;
  }

  /* Check if padded needs to be applied in the input and allocate appropriate buffers */
  /* Anand: changing below check for pad to either/or pad_h or pad_w instead of and */
  if ((handle->desc.pad_h_in == 0) && (handle->desc.pad_w_in == 0) && (handle->desc.pad_h_out == 0) && (handle->desc.pad_w_out == 0) && ((handle->desc.pad_h > 0) || (handle->desc.pad_w > 0))) {
    handle->padding_flag = 1;
    handle->scratch5  = 0;
    handle->minibatch_scratch_size = LIBXSMM_MAX(handle->desc.N * handle->blocksifm * handle->ifmblock * handle->fm_lp_block * (handle->ifhp+2*handle->desc.pad_h) * (handle->ifwp+2*handle->desc.pad_w+4) * libxsmm_dnn_typesize(handle->datatype_itm), handle->desc.N * handle->blocksofm * handle->ofmblock * handle->fm_lp_block * (handle->ofhp+2*handle->desc.pad_h) * (handle->ofwp+2*handle->desc.pad_w) * libxsmm_dnn_typesize(handle->datatype_itm));
    handle->fwdbwd_scratch_size = handle->desc.threads * handle->blocksifm * handle->ifmblock * handle->fm_lp_block * (handle->ifhp+2*handle->desc.pad_h) * (handle->ifwp+2*handle->desc.pad_w) * libxsmm_dnn_typesize(handle->datatype_itm);
    handle->max_scratch5_size = (handle->minibatch_scratch_size > handle->fwdbwd_scratch_size) ? handle->minibatch_scratch_size : handle->fwdbwd_scratch_size;
  } else {
    handle->padding_flag = 0;
  }

  /* TODO: we need to add much more checks here .... */

  if (noarch == 0) {
    /* Forward path */
    { libxsmm_convolution_forward_descriptor descriptor;
      libxsmm_matcopy_descriptor matcopy_descriptor;
      libxsmm_matcopy_descriptor matzero_descriptor;
      descriptor.use_nts =  handle->use_nts_fwd;

      if (handle->desc.R == 1 && handle->desc.S == 1) {
        descriptor.unroll_kh = 1;
        descriptor.unroll_kw = 1;
      }
      else {
        descriptor.unroll_kh = 0;
        descriptor.unroll_kw = 1;
      }
      if ( ((handle->datatype == LIBXSMM_DNN_DATATYPE_I8) ||
            (handle->datatype == LIBXSMM_DNN_DATATYPE_I16) ) &&
          (libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE) &&
          handle->desc.R > 1 && handle->desc.S > 1 && handle->fwd_ofh_rb == 1 ) {
        /* we need 3 instrad of 1 instruction for FMA -> do not perform any unrolling in kh/kw to control code size */
        descriptor.unroll_kh = 0;
        descriptor.unroll_kw = 0;
      }
      if (handle->padding_flag == 1) {
        descriptor.ifh_padded = handle->ifhp + 2 * handle->desc.pad_h;
        descriptor.ifw_padded = handle->ifwp + 2 * handle->desc.pad_w;
        matcopy_descriptor.n = handle->ifhp;
        if (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) {
          matcopy_descriptor.m = handle->ifwp * handle->ifmblock * handle->fm_lp_block;
          matcopy_descriptor.ldi = handle->ifwp * handle->ifmblock * handle->fm_lp_block;
          matcopy_descriptor.ldo = (handle->ifwp + 2*handle->desc.pad_w) * handle->ifmblock * handle->fm_lp_block;
        } else { /* Assumes NHWC format */
          matcopy_descriptor.m = handle->ifwp * handle->blocksifm * handle->ifmblock * handle->fm_lp_block;
          matcopy_descriptor.ldi = handle->ifwp * handle->blocksifm * handle->ifmblock * handle->fm_lp_block;
          matcopy_descriptor.ldo = (handle->ifwp + 2*handle->desc.pad_w) * handle->blocksifm * handle->ifmblock * handle->fm_lp_block;
        }
        matcopy_descriptor.prefetch = 1;
        matcopy_descriptor.unroll_level = 2;
        matcopy_descriptor.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype);
        matcopy_descriptor.flags = 0;
      } else {
        descriptor.ifh_padded = handle->ifhp;
        descriptor.ifw_padded = handle->ifwp;
      }
      descriptor.kh = handle->desc.R;
      descriptor.kw = handle->desc.S;
      descriptor.stride_h = handle->desc.u;
      descriptor.stride_w = handle->desc.v;
      descriptor.blocks_ofm = handle->blocksofm*handle->fm_lp_block;
      descriptor.blocks_ifm = handle->blocksifm;
      descriptor.blocks_ifm_blocking = handle->blocksifm_blocking;
      descriptor.weight_stride = 1;
      descriptor.use_fwd_generator_for_bwd = 0;
      descriptor.stride_h_store = 1;
      descriptor.stride_w_store = 1;
      descriptor.ofm_block = handle->ofmblock;
      descriptor.ifm_block = handle->ifmblock;
      descriptor.ofh_padded = handle->ofhp;
      descriptor.ofw_padded = handle->ofwp;
      descriptor.ofh_rb = handle->fwd_ofh_rb;
      descriptor.ofw_rb = handle->fwd_ofw_rb;
      descriptor.fm_lp_block = handle->fm_lp_block;
      descriptor.datatype = handle->datatype;
      descriptor.datatype_itm = handle->datatype_itm;
      descriptor.option = handle->desc.options;
      descriptor.format = (libxsmm_dnn_tensor_format)(handle->buffer_format | handle->filter_format);
      /* TODO check JIT errors */
      if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
      {
        if ( (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) ) {
          handle->code_fwd[0].smm = libxsmm_smmdispatch(16, 16, 16, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        } else {
          descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
          handle->code_fwd[0].pmm = libxsmm_create_xconv_forward(&descriptor);
          descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NO_WEIGHT;
          handle->code_fwd[1].pmm = libxsmm_create_xconv_forward(&descriptor);
          descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
          handle->code_fwd[2].pmm = libxsmm_create_xconv_forward(&descriptor);
          descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NO_OUTPUT;
          handle->code_fwd[3].pmm = libxsmm_create_xconv_forward(&descriptor);
          if (handle->padding_flag == 1) {
            handle->matcopy_fwd[0].xmatcopy = libxsmm_xmatcopydispatch(&matcopy_descriptor);
          }
        }
      } else if (libxsmm_target_archid == LIBXSMM_X86_AVX2) {
        /* we don't do prefetching and kh/kw unrolling (ignored in kernel generator) for AVX2 */
        descriptor.unroll_kh = 0;
        descriptor.unroll_kw = 0;
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
        if ( (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) ) {
          handle->code_fwd[0].smm = libxsmm_smmdispatch(16, 16, 16, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        } else {
          handle->code_fwd[0].pmm = libxsmm_create_xconv_forward(&descriptor);
        }
        if (handle->fwd_ofw_rb_2 != 0) {
          descriptor.ofw_rb = handle->fwd_ofw_rb_2;
          handle->code_fwd[1].pmm = libxsmm_create_xconv_forward(&descriptor);
        } else {
          handle->code_fwd[1].pmm = handle->code_fwd[0].pmm;
        }
        handle->code_fwd[2].pmm = handle->code_fwd[0].pmm;
        handle->code_fwd[3].pmm = handle->code_fwd[0].pmm;
        if (handle->padding_flag == 1) {
          handle->matcopy_fwd[0].xmatcopy = libxsmm_xmatcopydispatch(&matcopy_descriptor);
        }
      } else {
        assert(0/*should not happen*/);
      }

      if ( handle->use_thread_private_jit ) {
        int ks_overhead = 0;

        handle->n_entries_fwd = (int*) malloc(handle->desc.threads * sizeof(int));
        memset( handle->n_entries_fwd, 0, handle->desc.threads * sizeof(int) );
        handle->compute_fwd_indices_ptrs = (int**) malloc(handle->desc.threads * sizeof(int*));
        memset( handle->compute_fwd_indices_ptrs, 0, handle->desc.threads * sizeof(int*));
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


        if ( handle->ofw == 7) {
          int hrb_save =  descriptor.ofh_rb;
          int wrb_save =  descriptor.ofw_rb;
          descriptor.ofh_padded = handle->ofhp;
          descriptor.ofw_padded = handle->ofwp;
          descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
          descriptor.ofh_rb = 4;
          descriptor.ofw_rb = 7;
          handle->code_fwd[2].pmm = libxsmm_create_xconv_forward(&descriptor);
          descriptor.ofh_rb = 3;
          descriptor.ofw_rb = 7;
          descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
          handle->code_fwd[3].pmm = libxsmm_create_xconv_forward(&descriptor);
          handle->fwd_ofh_rb = 4;
          descriptor.ofh_rb = hrb_save;
          descriptor.ofw_rb = wrb_save;
        }

        for (i = 0; i < handle->desc.threads; i++) {
          handle->compute_fwd_indices_ptrs[i] = NULL;
          handle->kernel_fwd_variant_ptrs[i] = NULL;
          handle->fwd_code_segments[i] = NULL;
        }

        /* In case of logical padding also add a kernel that copies only one line of the image -- in case we exploit intra-image parallelism we should avoid copying entire image for each thread but only the minimum required number of input pixels... */
        if (handle->padding_flag == 1) {
          matcopy_descriptor.n = 1;
          handle->matcopy_fwd[2].xmatcopy = libxsmm_xmatcopydispatch(&matcopy_descriptor);
        }

        /* In case overwrite is requested, generate zero-ing kernel */
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
          matzero_descriptor.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype);
          matzero_descriptor.flags = LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE;
          handle->matcopy_fwd[1].xmatcopy = libxsmm_xmatcopydispatch(&matzero_descriptor);

          if (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) {
            matzero_descriptor.m = handle->ofwp*handle->ofmblock;
            matzero_descriptor.ldi = handle->ofwp*handle->ofmblock;
            matzero_descriptor.ldo = handle->ofwp*handle->ofmblock;
          } else { /* Assumes NHWC format */
            status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
          }
          handle->matcopy_fwd[3].xmatcopy = libxsmm_xmatcopydispatch(&matzero_descriptor);

        }

        /* Perform the dryrun and generate thread private jit indices to be used for the convolutions */
        status = libxsmm_dnn_perform_fwd_dryrun_direct(handle);

        /* compute kernel stream overhead */
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
    }
    /* Backward path */
    { libxsmm_convolution_backward_descriptor descriptor;
      libxsmm_matcopy_descriptor matcopy_descriptor;
      libxsmm_matcopy_descriptor matcopyback_descriptor;
      libxsmm_convolution_forward_descriptor fwd_equivalent_descriptor;
      libxsmm_matcopy_descriptor matzero_descriptor_overwrite;

      if (handle->padding_flag == 1) {
        descriptor.ifh_padded = handle->ifhp + 2 * handle->desc.pad_h;
        descriptor.ifw_padded = handle->ifwp + 2 * handle->desc.pad_w;
        if (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) {
          matcopy_descriptor.n = handle->ifhp;
          matcopyback_descriptor.n = handle->ifhp;
          matcopy_descriptor.m = handle->ifwp * handle->ifmblock;
          matcopyback_descriptor.m = handle->ifwp * handle->ifmblock;
          matcopy_descriptor.ldi = handle->ifwp * handle->ifmblock;
          matcopyback_descriptor.ldi = (handle->ifwp + 2*handle->desc.pad_w) * handle->ifmblock;
          matcopy_descriptor.ldo = (handle->ifwp + 2*handle->desc.pad_w) * handle->ifmblock;
          matcopyback_descriptor.ldo = handle->ifwp * handle->ifmblock;
        } else { /* Assumes NHWC format */
          matcopy_descriptor.n = 1;
          matcopy_descriptor.m =  handle->ifmblock;
          matcopy_descriptor.ldi = handle->ifmblock;
          matcopy_descriptor.ldo = handle->ifmblock;
          matcopyback_descriptor.n = 1;
          matcopyback_descriptor.m = handle->ifmblock;
          matcopyback_descriptor.ldi = handle->ifmblock;
          matcopyback_descriptor.ldo = handle->ifmblock;
        }
        matcopy_descriptor.prefetch = 1;
        matcopyback_descriptor.prefetch = 0;
        matcopy_descriptor.unroll_level = 2;
        matcopyback_descriptor.unroll_level = 2;
        matcopy_descriptor.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype);
        matcopyback_descriptor.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype);
        matcopy_descriptor.flags = 0;
        matcopyback_descriptor.flags = 0;
      } else {
        descriptor.ifh_padded = handle->ifhp;
        descriptor.ifw_padded = handle->ifwp;
      }
      descriptor.kh = handle->desc.R;
      descriptor.kw = handle->desc.S;
      descriptor.unroll_kw = 1;
      descriptor.unroll_kh = 1;
      descriptor.stride_h = handle->desc.u;
      descriptor.stride_w = handle->desc.v;
      descriptor.blocks_ofm = handle->blocksofm;
      descriptor.blocks_ifm = handle->blocksifm;
      /* ORIG::descriptor.ofm_block= ((nOfm % VLEN == 0) ? VLEN : 1); */
      descriptor.ofm_block = handle->ofmblock;
      /* ORIG::descriptor.ifm_block = VLEN; */
      descriptor.ifm_block = handle->ifmblock;
      descriptor.ofh_padded = handle->ofhp;
      descriptor.ofw_padded = handle->ofwp;
      descriptor.ofh_rb = handle->bwd_ofh_rb;
      descriptor.ofw_rb = handle->bwd_ofw_rb;
      descriptor.fm_lp_block = handle->fm_lp_block;
      descriptor.ofw = handle->ofw;
      /*
         descriptor.ofw_unroll = 1;
         descriptor.peeled = 0;*/
      descriptor.datatype = handle->datatype;
      descriptor.datatype_itm = handle->datatype_itm;
      descriptor.option = handle->desc.options;
      descriptor.format = (libxsmm_dnn_tensor_format)(handle->buffer_format | handle->filter_format);

      /* Use the fwd convolution generator for bwd under some assumptions  */
      /* if ( (handle->use_thread_private_jit > 0) && ( (handle->desc.R == 1 && handle->desc.S == 1) || (handle->desc.stride_h == 1 && handle->desc.stride_w == 1) ) )  { */
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
        fwd_equivalent_descriptor.stride_h = 1;
        fwd_equivalent_descriptor.stride_w = 1;
        fwd_equivalent_descriptor.blocks_ofm = handle->blocksifm;
        fwd_equivalent_descriptor.blocks_ifm = handle->blocksofm;
        fwd_equivalent_descriptor.ofm_block = handle->ifmblock;
        fwd_equivalent_descriptor.ifm_block = handle->ofmblock;
        fwd_equivalent_descriptor.ofh_padded = handle->ifhp;
        fwd_equivalent_descriptor.ofw_padded = handle->ifwp;
        fwd_equivalent_descriptor.ofh_rb = handle->bwd_ofh_rb;
        fwd_equivalent_descriptor.ofw_rb = handle->bwd_ofw_rb;
        fwd_equivalent_descriptor.fm_lp_block = handle->fm_lp_block;
        fwd_equivalent_descriptor.datatype = handle->datatype;
        fwd_equivalent_descriptor.datatype_itm = handle->datatype_itm;
        fwd_equivalent_descriptor.option = handle->desc.options;
        fwd_equivalent_descriptor.format = (libxsmm_dnn_tensor_format)(handle->buffer_format | handle->filter_format);
        fwd_equivalent_descriptor.blocks_ifm_blocking = handle->blocksofm_blocking;
        fwd_equivalent_descriptor.weight_stride = 1; 
        fwd_equivalent_descriptor.use_fwd_generator_for_bwd = 1;
        fwd_equivalent_descriptor.stride_h_store = handle->desc.u;
        fwd_equivalent_descriptor.stride_w_store = handle->desc.v;
        fwd_equivalent_descriptor.use_nts = handle->use_nts_bwd;
        if (handle->padding_flag == 1) {
          matcopy_descriptor.n = handle->ofhp;
          matcopy_descriptor.m = handle->ofwp * handle->ofmblock * handle->fm_lp_block;
          matcopy_descriptor.ldi = handle->ofwp * handle->ofmblock * handle->fm_lp_block;
          matcopy_descriptor.ldo = (handle->ofwp + 2*handle->desc.pad_w) * handle->ofmblock * handle->fm_lp_block;
          matcopy_descriptor.prefetch = 1;
          matcopy_descriptor.unroll_level = 2;
          matcopy_descriptor.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype);
          matcopy_descriptor.flags = 0;
        } 
      }


      /* TODO check JIT errors */
      if ( /*(*/libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
        libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM/*) &&
                                                           ((handle->filter_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM))*/ )
          {
#if 0
            /* control code size */
            const unsigned int max_code_size = 20000/*16384*/;
            const unsigned int bp_each_iter_code_size = 12/*16*/;
            if ((descriptor.ofw * descriptor.kw * descriptor.ofm_block * bp_each_iter_code_size <= max_code_size)) {
              descriptor.ofw_unroll = 1;
              descriptor.unroll_kw = 1;
            } else if (descriptor.kw * descriptor.ofm_block * bp_each_iter_code_size <= max_code_size) {
              unsigned int upper_bound_ofw_rb = (max_code_size) / (descriptor.kw * descriptor.ofm_block * bp_each_iter_code_size);
              for (i = LIBXSMM_MIN(upper_bound_ofw_rb+1, 24); i >= 10; i--) {
                if (handle->ofw % i == 0) break;
              }
              if (i>=10) {
                descriptor.ofw_rb =  i;
                descriptor.ofw_unroll = 0;
                descriptor.unroll_kw = 1;
              } else {
                if (descriptor.ofw_rb*descriptor.kw * descriptor.ofm_block * bp_each_iter_code_size <= max_code_size) {
                  descriptor.ofw_unroll = 0;
                  descriptor.unroll_kw = 1;
                } else {
                  descriptor.ofw_unroll = 0;
                  descriptor.unroll_kw = 0;
                }
              }
            } else if (descriptor.ofw_rb*descriptor.kw * descriptor.ofm_block * bp_each_iter_code_size <= max_code_size) {
              descriptor.ofw_unroll = 0;
              descriptor.unroll_kw = 1;
            } else {
              descriptor.ofw_unroll = 0;
              descriptor.unroll_kw = 0;
            }
#endif
            /*descriptor.prefetch_output_ahead = 0;*/
            /* TODO: Decide the unroll factor and register blocking using some heuristics as above */
            descriptor.unroll_kh = 0;
            descriptor.unroll_kw = 1;
            if ( handle->fwd_ofw_rb != 7 ) {
              descriptor.ofh_rb = handle->fwd_ofh_rb;
              handle->bwd_ofh_rb = handle->fwd_ofh_rb;
              descriptor.ofw_rb = handle->fwd_ofw_rb;
              handle->bwd_ofw_rb = handle->fwd_ofw_rb;
            }
#if 0
            !defined(NDEBUG)
            printf("DEBUG JIT of conv (NON-PEELED):\n  arch: %s\n  type: %s\n  kw: %u\n  unroll_kw: %u\n  kh: %u\n  unroll_kh: %u\n  ofm_block: %u\n  ifm_block: %u\n"
                "  ofh_padded: %u\n  ofw_padded: %u\n  ofh_rb: %u\n  ofw_rb: %u\n  ifh_padded: %u\n  ifw_padded: %u\n"
                "  stride_h: %u\n  stride_w: %u\n",
                libxsmm_get_target_arch(),
                "backward",     /* type */
                descriptor.kw,         /* kernel width */
                descriptor.unroll_kw,  /* kernel width, unrolled */
                descriptor.kh,         /* kernel width */
                descriptor.unroll_kh,  /* kernel width, unrolled */
                descriptor.ofm_block,  /* should be VLEN */
                descriptor.ifm_block,  /* should be VLEN */
                descriptor.ofh_padded, /* this we need for 2D register block */
                descriptor.ofw_padded, /* this we use for 1D and 2D register block */
                descriptor.ofh_rb,     /* UR, register block of ofh */
                descriptor.ofw_rb,     /* UR, register block of ofw */
                descriptor.ifh_padded,
                descriptor.ifw_padded,
                descriptor.stride_h,   /* this we use for offsets in the input */
                descriptor.stride_w   /* this we use for offsets in the input */);
#endif

            /* NONE */
            if (handle->padding_flag == 1) {
              handle->matcopy_bwd[0].xmatcopy = libxsmm_xmatcopydispatch(&matcopy_descriptor);
              /*handle->matcopy_bwd[1].xmatcopy = libxsmm_xmatcopydispatch(&matcopyback_descriptor);*/
            }

            /*descriptor.prefetch_output_ahead = 0;*/
            descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
            if ( (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) ) {
              handle->code_bwd[0].smm = libxsmm_smmdispatch(16, 16, 16, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
            } else {
              handle->code_bwd[0].pmm = libxsmm_create_xconv_backward(&descriptor);
            }
            /* PREFETCH_NO_WEIGHT */
            descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NO_WEIGHT_L2;
            handle->code_bwd[1].pmm = libxsmm_create_xconv_backward(&descriptor);
            /*ALL*/
            /*descriptor.prefetch_output_ahead = 0;*/
            descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
            handle->code_bwd[2].pmm = libxsmm_create_xconv_backward(&descriptor);

            /*if (handle->use_thread_private_jit > 0) {*/
            if (handle->exploit_duality == 1) {
              fwd_equivalent_descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
              if ( handle->bwd_ofw_rb != 7) {
                handle->code_bwd[4].pmm = libxsmm_create_xconv_forward(&fwd_equivalent_descriptor);
              } else {
                int hrb_save =  fwd_equivalent_descriptor.ofh_rb;
                int wrb_save =  fwd_equivalent_descriptor.ofw_rb;
                fwd_equivalent_descriptor.ofh_rb = 4;
                fwd_equivalent_descriptor.ofw_rb = 7;
                handle->code_bwd[4].pmm = libxsmm_create_xconv_forward(&fwd_equivalent_descriptor);
                fwd_equivalent_descriptor.ofh_rb = 3;
                fwd_equivalent_descriptor.ofw_rb = 7;
                handle->code_bwd[5].pmm = libxsmm_create_xconv_forward(&fwd_equivalent_descriptor);
                handle->bwd_ofh_rb = 4;
                fwd_equivalent_descriptor.ofh_rb = hrb_save;
                fwd_equivalent_descriptor.ofw_rb = wrb_save;
              }
            }
#if 0
            /* PEELED VERSION */
            for (i = LIBXSMM_MIN(24, handle->ofw); i > 1; i--) {
              if (handle->ofw % i == 0) break;
            }
            descriptor.ofw_rb = i;
            descriptor.peeled = 1;

            /* control the code size using the following heuristic */
            if ((descriptor.ofw * descriptor.kw * descriptor.kh * descriptor.ofm_block * bp_each_iter_code_size <= max_code_size)) {
              descriptor.ofw_unroll = 1;
              descriptor.unroll_kw = 1;
              descriptor.unroll_kh = 1;
            } else if (descriptor.kw * descriptor.kh * descriptor.ofm_block * bp_each_iter_code_size <= max_code_size) {
              unsigned int upper_bound_ofw_rb = (max_code_size) / (descriptor.kw * descriptor.kh * descriptor.ofm_block * bp_each_iter_code_size);
              for (i = LIBXSMM_MIN(upper_bound_ofw_rb+1, 24); i >= 10; i--) {
                if (handle->ofw % i == 0) break;
              }
              if (i>=10) {
                descriptor.ofw_rb =  i;
                descriptor.ofw_unroll = 0;
                descriptor.unroll_kw = 1;
                descriptor.unroll_kh = 1;
              } else {
                if (descriptor.ofw_rb*descriptor.kw * descriptor.kh * descriptor.ofm_block * bp_each_iter_code_size <= max_code_size) {
                  descriptor.ofw_unroll = 0;
                  descriptor.unroll_kw = 1;
                  descriptor.unroll_kh = 1;
                } else {
                  descriptor.ofw_unroll = 0;
                  descriptor.unroll_kw = 0;
                  descriptor.unroll_kh = 1;
                }
              }
            } else if (descriptor.ofw_rb* descriptor.kh * descriptor.kw * descriptor.ofm_block * bp_each_iter_code_size <= max_code_size) {
              descriptor.ofw_unroll = 0;
              descriptor.unroll_kw = 1;
              descriptor.unroll_kh = 1;
            } else {
              descriptor.ofw_unroll = 0;
              descriptor.unroll_kw = 0;
              descriptor.unroll_kh = 1; /* always unroll kh */
            }
            descriptor.prefetch_output_ahead = 0;
#if !defined(NDEBUG)
            printf("DEBUG JIT of conv (PEELED):\n  arch: %s\n  type: %s\n  kw: %u\n  unroll_kw: %u\n  kh: %u\n  unroll_kh: %u\n  ofm_block: %u\n  ifm_block: %u\n"
                "  ofh_padded: %u\n  ofw_padded: %u\n  ofh_rb: %u\n  ofw_rb: %u\n  ifh_padded: %u\n  ifw_padded: %u\n"
                "  stride_h: %u\n  stride_w: %u\n  ofw: %u\n  ofw_unroll: %u\n  peeled: %u\n  prefetch_output: %u\n",
                libxsmm_get_target_arch(),
                "backward",       /* type */
                descriptor.kw,         /* kernel width */
                descriptor.unroll_kw,  /* kernel width, unrolled */
                descriptor.kh,         /* kernel width */
                descriptor.unroll_kh,  /* kernel width, unrolled */
                descriptor.ofm_block,  /* should be VLEN */
                descriptor.ifm_block,  /* should be VLEN */
                descriptor.ofh_padded, /* this we need for 2D register block */
                descriptor.ofw_padded, /* this we use for 1D and 2D register block */
                descriptor.ofh_rb,     /* UR, register block of ofh */
                descriptor.ofw_rb,     /* UR, register block of ofw */
                descriptor.ifh_padded,
                descriptor.ifw_padded,
                descriptor.stride_h,   /* this we use for offsets in the input */
                descriptor.stride_w,   /* this we use for offsets in the input */
                descriptor.ofw, /* upper bound of oi loop */
                descriptor.ofw_unroll, /*unroll for ofw loop */
                descriptor.peeled, /*peeled?*/
                descriptor.prefetch_output_ahead /* prefetch output ahead */ );
#endif

            /* NONE */
            descriptor.prefetch_output_ahead = 0;
            descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
            handle->code_bwd[2].pmm = libxsmm_create_xconv_backward(&descriptor);
            /* NO_WEIGHT_L2 */
            descriptor.prefetch_output_ahead = 0;
            descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NO_WEIGHT_L2;
            handle->code_bwd[3].pmm = libxsmm_create_xconv_backward(&descriptor);
#endif
          } else if (/*(*/libxsmm_target_archid == LIBXSMM_X86_AVX2/*) ||
                                                                     ((handle->filter_format != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) || (handle->buffer_format != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM))*/ ) {
            /* we don't do prefetching and kh/kw unrolling (ignored in kernel generator) for AVX2 */
            if (handle->padding_flag == 1) {
              handle->matcopy_bwd[0].xmatcopy = libxsmm_xmatcopydispatch(&matcopy_descriptor);
              handle->matcopy_bwd[1].xmatcopy = libxsmm_xmatcopydispatch(&matcopyback_descriptor);
            }
            descriptor.unroll_kh = 0;
            descriptor.unroll_kw = 0;
            descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
            if ( (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) ) {
              handle->code_bwd[0].smm = libxsmm_smmdispatch(16, 16, 16, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
            } else {
              handle->code_bwd[0].pmm = libxsmm_create_xconv_backward(&descriptor);
            }
            handle->code_bwd[1].pmm = handle->code_bwd[0].pmm;
            handle->code_bwd[2].pmm = handle->code_bwd[0].pmm;
            handle->code_bwd[3].pmm = handle->code_bwd[0].pmm;
          } else {
            assert(0/*should not happen*/);
          }

      if ( handle->use_thread_private_jit ) {
        int ks_overhead = 0;

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

        if (handle->exploit_duality == 1) {
          libxsmm_dnn_layer *mirror_handle = (libxsmm_dnn_layer*)  malloc(sizeof(libxsmm_dnn_layer));
          memcpy( mirror_handle, handle ,sizeof(libxsmm_dnn_layer));
          mirror_handle->use_fwd_for_bwd = 1;
          mirror_handle->blocksifm_blocking = handle->blocksofm_blocking;
          mirror_handle->fwd_ofh_rb = handle->bwd_ofh_rb;
          mirror_handle->fwd_ofw_rb = handle->bwd_ofw_rb;
          mirror_handle->blocksofm = handle->blocksifm;
          mirror_handle->ifhp = handle->ofhp;
          mirror_handle->ifwp = handle->ofwp;
          mirror_handle->ofhp = handle->ifhp;
          mirror_handle->ofwp = handle->ifwp;
          mirror_handle->use_nts_fwd = handle->use_nts_bwd;
          mirror_handle->block_fwd_ofm = handle->block_fwd_ifm;
          mirror_handle->blocksifm = handle->blocksofm;
          mirror_handle->ofh = (handle->desc.H + 2 * handle->desc.pad_h - handle->desc.S) / handle->desc.v + 1;
          mirror_handle->ofw = (handle->desc.W + 2 * handle->desc.pad_w - handle->desc.R) / handle->desc.u + 1;
          mirror_handle->ifmblock = handle->ofmblock;
          mirror_handle->ofmblock = handle->ifmblock;
          mirror_handle->compute_fwd_indices_ptrs =  handle->compute_bwd_indices_ptrs;
          mirror_handle->n_entries_fwd = handle->n_entries_bwd;
          mirror_handle->kernel_fwd_variant_ptrs = handle->kernel_bwd_variant_ptrs;
          mirror_handle->n_fwd_code_segments = handle->n_bwd_code_segments;
          mirror_handle->fwd_code_segments = handle->bwd_code_segments;
          mirror_handle->ofh_fwd_start = handle->ofh_bwd_start;
          mirror_handle->ofh_fwd_end = handle->ofh_bwd_end;
          status = libxsmm_dnn_perform_fwd_dryrun_direct(mirror_handle);

          /* In case overwrite is requested, generate zero-ing kernel */
          if ( (handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0 )  {
            matzero_descriptor_overwrite.n = 1;
            if (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) {
              matzero_descriptor_overwrite.m = handle->desc.H*handle->ifwp*handle->ifmblock;
              matzero_descriptor_overwrite.ldi = handle->desc.H*handle->ifwp*handle->ifmblock;
              matzero_descriptor_overwrite.ldo = handle->desc.H*handle->ifwp*handle->ifmblock;
            } else { /* Assumes NHWC format */
              status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
            }
            matzero_descriptor_overwrite.prefetch = 0;
            matzero_descriptor_overwrite.unroll_level = 2;
            matzero_descriptor_overwrite.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype);
            matzero_descriptor_overwrite.flags = LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE;
            handle->matcopy_bwd[1].xmatcopy = libxsmm_xmatcopydispatch(&matzero_descriptor_overwrite);
          }        
        } else {
           status = libxsmm_dnn_perform_bwd_dryrun_direct(handle);
        }

        /* compute kernel stream overhead */
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

   } /* End of backward */
      /* TODO weight update path */
      { libxsmm_convolution_weight_update_descriptor descriptor;
        libxsmm_matcopy_descriptor matcopy_descriptor;
        libxsmm_matcopy_descriptor matzero_descriptor;
        if (handle->padding_flag == 1) {
          descriptor.ifh_padded = handle->ifhp + 2 * handle->desc.pad_h;
          descriptor.ifw_padded = handle->ifwp + 2 * handle->desc.pad_w;
          matzero_descriptor.n = 1;
          if (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) {
            matcopy_descriptor.n = handle->ifhp;
            matcopy_descriptor.m = handle->ifwp * handle->ifmblock;
            matzero_descriptor.m = descriptor.ifh_padded * descriptor.ifw_padded * handle->ifmblock;
            matcopy_descriptor.ldi = handle->ifwp * handle->ifmblock;
            matzero_descriptor.ldi = descriptor.ifh_padded * descriptor.ifw_padded * handle->ifmblock;
            matcopy_descriptor.ldo = (handle->ifwp + 2*handle->desc.pad_w) * handle->ifmblock;
            matzero_descriptor.ldo = descriptor.ifh_padded * descriptor.ifw_padded * handle->ifmblock;
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
          matcopy_descriptor.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype);
          matzero_descriptor.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype);
          matcopy_descriptor.flags = 0;
          matzero_descriptor.flags = LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE;
        } else {
          descriptor.ifh_padded = handle->ifhp;
          descriptor.ifw_padded = handle->ifwp;
        }
        descriptor.ofm_block = handle->ofmblock;
        descriptor.ifm_block = handle->ifmblock;
        descriptor.kh = handle->desc.R;
        descriptor.kw = handle->desc.S;
        descriptor.unroll_kw = (descriptor.ifm_block == 1) ? 1 : 0;
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
        descriptor.datatype = handle->datatype;
        descriptor.datatype_itm = handle->datatype_itm;
        descriptor.option = handle->desc.options;
        descriptor.format = (libxsmm_dnn_tensor_format)(handle->buffer_format | handle->filter_format);
        descriptor.ncopies = handle->desc.threads;

        /* TODO check JIT errors */
        if ( /*(*/libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
            libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM /*)*/ /*&&
                                                                    ((handle->filter_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM))*/ )
            {
              const unsigned int wu_each_iter_code_size = 10 * (descriptor.ifm_block == 1 ? descriptor.kw : descriptor.ifm_block);
              const unsigned int wu_max_code_size = 8000;
              int upper_limit_ofw_rb = wu_max_code_size / wu_each_iter_code_size, upper_limit_ofh_rb = 0;
              unsigned int chunk_size;

              descriptor.ifm_unroll = 1;

              for (i = LIBXSMM_MIN(upper_limit_ofw_rb, LIBXSMM_MIN(56,handle->ofw)); i >= 1; i--) {
                if (handle->ofw % i == 0) break;
              }
              descriptor.ofw_rb = i;

              upper_limit_ofh_rb = wu_max_code_size / (descriptor.ofw_rb * wu_each_iter_code_size);
              for (i = LIBXSMM_MIN(upper_limit_ofh_rb, handle->ofh); i >= 1; i--) {
                if (handle->ofh % i == 0) break;
              }
              descriptor.ofh_rb =  i;

              chunk_size = 6 * descriptor.ofw_rb *  descriptor.ofh_rb * handle->ifmblock * (unsigned int)libxsmm_dnn_typesize(handle->datatype);
              while( chunk_size > 32000) {
                for (i = descriptor.ofh_rb-1; i >= 1; i--) {
                  if (handle->ofh % i == 0) break;
                }
                if (i == 0) i = 1;
                descriptor.ofh_rb =  i;
                chunk_size = 6 * descriptor.ofw_rb *  descriptor.ofh_rb * handle->ifmblock * (unsigned int)libxsmm_dnn_typesize(handle->datatype);
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

              /* Here starts logic for calculating RB factors for UPD when KS are enabled  */
              int ifw_padded, qfma_padding, kernel_ifw_padded, kernel_ofw_padded;
              int kernel_ofw_compute;
              int kernel_ofw_fake_pixels;
              int kernel_ofw;

              ifw_padded = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
              qfma_padding = (ifw_padded % 4 == 0) ? 0 : 4 - ifw_padded % 4;
              kernel_ifw_padded = ifw_padded + qfma_padding;
              handle->qfma_input_pad = qfma_padding;
              descriptor.ifw_padded = kernel_ifw_padded;
              descriptor.ofw_padded = handle->ofwp;
              descriptor.ofh_padded = handle->ofhp;

              if (handle->desc.R == 1 && handle->desc.S == 1) {
                kernel_ofw_compute = handle->ofwp;
              } else {
                if (handle->padding_flag == 1) {
                  kernel_ofw_compute = handle->ofwp;
                } else {
                  kernel_ofw_compute = handle->ofw;
                }
              }
              kernel_ofw_fake_pixels = (kernel_ofw_compute % 4 == 0) ? 0 : 4 - kernel_ofw_compute % 4;
              kernel_ofw = kernel_ofw_compute + kernel_ofw_fake_pixels;
              /*printf("Compute pixels are: %d , fake pixels are %d, ofw is %d and ofw_padded is %d, ofw_rb is %d\n",kernel_ofw_compute, kernel_ofw_fake_pixels, handle->ofw, descriptor.ofw_padded, kernel_ofw ); */
              descriptor.ofw_fake_pixels = kernel_ofw_fake_pixels;
              descriptor.ofw_rb = kernel_ofw;  
              descriptor.ofh_rb = handle->ofh;

              while (   descriptor.ofw_rb  *  descriptor.ofh_rb > 196 ) {
                descriptor.ofh_rb = descriptor.ofh_rb / 2;
              }

              while (  handle->ofh % descriptor.ofh_rb != 0 ) {
                descriptor.ofh_rb--;
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

              descriptor.transpose_ofw_ifm = 0;
#if 0
              !defined(NDEBUG)
                printf("DEBUG JIT of conv:\n  arch: %s\n  type: %s\n  ofm_block: %u\n  ifm_block: %u\n"
                    "  ofh_padded: %u\n  ofw_padded: %u\n  ofh_rb: %u\n  ofw_rb: %u\n  ifh_padded: %u\n  ifw_padded: %u\n"
                    "  stride_h: %u\n  stride_w: %u\n  ifm_unroll: %u\n  ofh: %u\n  ofh_unroll: %u\n  ofw: %u\n  ofw_unroll: %u\n  kw:%u\n  unroll_kw=%u\n  kh: %u\n  transpose: %u\n",
                    libxsmm_get_target_arch(),
                    "weight-update",        /* type */
                    descriptor.ofm_block,  /* should be VLEN */
                    descriptor.ifm_block,  /* should be VLEN */
                    descriptor.ofh_padded, /* this we need for 2D register block */
                    descriptor.ofw_padded, /* this we use for 1D and 2D register block */
                    descriptor.ofh_rb,     /* UR, register block of ofh */
                    descriptor.ofw_rb,     /* UR, register block of ofw */
                    descriptor.ifh_padded,
                    descriptor.ifw_padded,
                    descriptor.stride_h,   /* this we use for offsets in the input */
                    descriptor.stride_w,   /* this we use for offsets in the input */
                    descriptor.ifm_unroll, /*should unroll ifm loop -- yes or no */
                    descriptor.ofh,        /*ofh */
                    descriptor.ofh_unroll,        /*ofh_unroll */
                    descriptor.ofw,        /*ofw */
                    descriptor.ofw_unroll,        /*ofw_unroll */
                    descriptor.kw, /*kw unroll */
                    descriptor.unroll_kw, /* unroll factor of kw */
                    descriptor.kh, /*kh unroll */
                    descriptor.transpose_ofw_ifm /*transpose */);
#endif
              /* NONE */
              if (handle->padding_flag == 1) {
                handle->matcopy_upd[0].xmatcopy = libxsmm_xmatcopydispatch(&matcopy_descriptor);
                handle->matcopy_upd[1].xmatcopy = libxsmm_xmatcopydispatch(&matzero_descriptor);
              }

              descriptor.transpose_ofw_ifm = 0;
              descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
              if ( (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) ) {
                handle->code_upd[0].smm = libxsmm_smmdispatch(16, 16, 16, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
              } else {
                handle->code_upd[0].pmm = libxsmm_create_xconv_update_weights(&descriptor);
              }
              /*ALL*/
              descriptor.transpose_ofw_ifm = 0;
              descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
              handle->code_upd[1].pmm = libxsmm_create_xconv_update_weights(&descriptor);
              /* NO_OUTPUT_L2 */
              descriptor.transpose_ofw_ifm = 0;
              descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NO_OUTPUT_L2;
              handle->code_upd[2].pmm = libxsmm_create_xconv_update_weights(&descriptor);
              /* TRANSPOSE NONE */
              descriptor.transpose_ofw_ifm = 1;
              descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
              handle->code_upd[3].pmm = libxsmm_create_xconv_update_weights(&descriptor);
              /*TRANSPOSE ALL*/
              descriptor.transpose_ofw_ifm = 1;
              descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
              handle->code_upd[4].pmm = libxsmm_create_xconv_update_weights(&descriptor);
              /* TRANSPOSE NO_OUTPUT_L2 */
              descriptor.transpose_ofw_ifm = 1;
              descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NO_OUTPUT_L2;
              handle->code_upd[5].pmm = libxsmm_create_xconv_update_weights(&descriptor);
            } else if (/*(*/libxsmm_target_archid == LIBXSMM_X86_AVX2/*)*/ /*||
                                                                             ((handle->filter_format != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) || (handle->buffer_format != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM))*/ ) {
              /* we don't do prefetching and kh/kw unrolling (ignored in kernel generator) for AVX2 */
              descriptor.unroll_kw = 0;
              descriptor.ifm_unroll = 0;
              descriptor.transpose_ofw_ifm = 0;
              descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
              if (handle->padding_flag == 1) {
                handle->matcopy_upd[0].xmatcopy = libxsmm_xmatcopydispatch(&matcopy_descriptor);
                handle->matcopy_upd[1].xmatcopy = libxsmm_xmatcopydispatch(&matzero_descriptor);
              }

              if ( (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) ) {
                handle->code_upd[0].smm = libxsmm_smmdispatch(16, 16, 16, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
              } else {
                handle->code_upd[0].pmm = libxsmm_create_xconv_update_weights(&descriptor);
              }
              handle->code_upd[1].pmm = handle->code_upd[0].pmm;
              handle->code_upd[2].pmm = handle->code_upd[0].pmm;
              handle->code_upd[3].pmm = handle->code_upd[0].pmm;
              handle->code_upd[4].pmm = handle->code_upd[0].pmm;
              handle->code_upd[5].pmm = handle->code_upd[0].pmm;
            } else {
              assert(0/*should not happen*/);
            }

        if ( handle->use_thread_private_jit ) {
          int ks_overhead = 0;

          handle->trans_ofw_ifm = 0;
          /* Determine if we will be using thread private filters  */
          if ((handle->ifmblock == 1) || (handle->blocksifm * handle->blocksofm < handle->desc.threads) ) {
            handle->use_thread_private_filter = 1;
            /* determine if we will transpose input  */
            if ( (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) && (handle->desc.v == 1) && (handle->upd_ofw_rb%4 == 0) ) {
              handle->trans_ofw_ifm = 1;
            }
          } else {
            handle->use_thread_private_filter = 0;
            /* determine if we will transpose input  */
            if ( (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) && (handle->desc.v == 1) && (handle->upd_ofw_rb%4 == 0) ) {
              handle->trans_ofw_ifm = 1;
            }
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
          matzero_descriptor.typesize = (unsigned char)libxsmm_dnn_typesize(handle->datatype);
          matzero_descriptor.flags = LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE;
          handle->matcopy_upd[2].xmatcopy = libxsmm_xmatcopydispatch(&matzero_descriptor);

          /* Perform the dryrun and generate thread private jit indices to be used for the convolutions */
          status = libxsmm_dnn_perform_upd_dryrun_direct(handle);

          /* compute kernel stream overhead */
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
      } /* end of weight-update handle */
      {
        handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);

        /* backward transpose filters */
        handle->scratch1 = 0;
        handle->scratch1_size = handle->blocksifm * handle->ifmblock * handle->blocksofm * handle->ofmblock
          * handle->desc.R * handle->desc.S * handle->fm_lp_block * libxsmm_dnn_typesize(handle->datatype);
        if (handle->fm_lp_block > 1) {
          /* If low precision, we need extra buffer to store intermediate weight tensor */
          handle->scratch1_size *= 2;
        }

        /* weight update transpose of minibatch */
        handle->scratch3 = 0;
        handle->scratch3_size = handle->desc.N * handle->blocksifm * handle->ifmblock * handle->ifhp * (handle->ifwp+4)
          * handle->fm_lp_block * libxsmm_dnn_typesize(handle->datatype);

        /* minibatch parallel execution of weight update kernel */
        if ((handle->ifmblock == 1) || ((handle->blocksifm * handle->blocksofm) < handle->desc.threads) || (handle->use_thread_private_jit)) {
          handle->upd_use_thread_fil = 1;
          handle->scratch4 = 0;
          handle->scratch4_size = handle->desc.threads * handle->blocksifm * handle->ifmblock * handle->blocksofm * handle->ofmblock
            * handle->desc.R * handle->desc.S * handle->fm_lp_block * libxsmm_dnn_typesize(handle->datatype);
          handle->scratch4_size += handle->desc.threads * handle->block_upd_ofm * handle->block_upd_ifm * handle->desc.R
            * handle->desc.S * handle->ifmblock * handle->ofmblock * libxsmm_dnn_typesize(handle->datatype);

          /* enable external reduce of filter scratch */
          if ( (handle->options & LIBXSMM_DNN_CONV_OPTION_WU_EXT_FILTER_REDUCE) > 0 ) {
            handle->upd_use_external_reduce = 1;
          }
        } else {
          handle->scratch4 = 0;
          handle->scratch4_size = 0;
          handle->upd_use_thread_fil = 0;
        }

        /* low percision intermediate output buffer */
        if ( handle->datatype != handle->datatype_itm ) {
          handle->scratch6 = 0;
          handle->scratch6_size = handle->desc.N * handle->blocksofm * handle->ofmblock * handle->ofhp * handle->ofwp * handle->fm_lp_block
            * libxsmm_dnn_typesize(handle->datatype_itm);
          /* For backward code, have to correct the size */
          handle->scratch7 = 0;
          handle->scratch7_size = handle->desc.N * handle->blocksifm * handle->ifmblock * handle->ifhp * handle->ifwp * handle->fm_lp_block
            * libxsmm_dnn_typesize(handle->datatype_itm);
        } else {
          handle->scratch6 = 0;
          handle->scratch6_size = 0;
          /* For backward code */
          handle->scratch7 = 0;
          handle->scratch7_size = 0;
        }
      }
    }
      else {
        handle->code_fwd[0].xconv.sconv = 0;
        handle->code_fwd[1].xconv.sconv = 0;
        handle->code_fwd[2].xconv.sconv = 0;
        handle->code_fwd[3].xconv.sconv = 0;
        /* Backward path */
        handle->code_bwd[0].xconv.sconv = 0;
        handle->code_bwd[1].xconv.sconv = 0;
        handle->code_bwd[2].xconv.sconv = 0;
        handle->code_bwd[3].xconv.sconv = 0;
        /* weight update path */
        handle->code_upd[0].xconv.sconv = 0;
        handle->code_upd[1].xconv.sconv = 0;
        handle->code_upd[2].xconv.sconv = 0;
        handle->code_upd[3].xconv.sconv = 0;
        handle->code_upd[4].xconv.sconv = 0;
        handle->code_upd[5].xconv.sconv = 0;

        handle->barrier = 0;

        handle->scratch1 = 0;
        handle->scratch1_size = 0;
        handle->scratch3 = 0;
        handle->scratch3_size = 0;
        handle->scratch4 = 0;
        handle->scratch4_size = 0;
        /* low percision intermediate output buffer */
        if ( handle->datatype != handle->datatype_itm ) {
          handle->scratch6 = 0;
          handle->scratch6_size = handle->desc.N * handle->blocksofm * handle->ofmblock * handle->ofhp * handle->ofwp * handle->fm_lp_block
            * libxsmm_dnn_typesize(handle->datatype_itm);
        } else {
          handle->scratch6 = 0;
          handle->scratch6_size = 0;
        }
      }

      return status;
    }

    /* This function finds the prime factors of a number */
    LIBXSMM_API_INLINE void internal_dnn_handle_factors(
        unsigned int num,
        unsigned int num_factors[] )
    {
      unsigned int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
      int i;
      unsigned int total_primes = 10;
      unsigned int index = 0;

      for ( i = total_primes-1; i >= 0; i-- ) {
        while((num % primes[i]) == 0) {
          num_factors[index] = primes[i];
          index++;
          num = num/primes[i];
        }
      }
    }


    /**
     * This function finds the unroll factor for (itiles*jtiles*bimg)
     * such that ur <= max_acc
     * The following loop may not give an optimal solution (knapsack problem)
     * Eg, 12 = 3*2*2, MAX_ACC = 4, this algorithm: 3, best: 2*2
     */
    LIBXSMM_API_INLINE void internal_dnn_handle_factors_all(
        unsigned int  product,
        unsigned int* ur,
        unsigned int  max_acc)
    {
      unsigned int i;
      unsigned int fact[10];

      for ( i = 0; i < 10; i++ ) {
        fact[i] = 1;
      }
      internal_dnn_handle_factors(product, fact);

      *ur = 1;
      for ( i = 0; fact[i] != 1; i++ ) {
        if ( (fact[i] * (*ur)) <= max_acc ) {
          *ur = (*ur)*fact[i];
        }
      }
    }


    LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_internal_create_conv_handle_winograd_check( libxsmm_dnn_layer* handle ) {
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
            libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE  ||
            libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM ) &&
          (handle->datatype == LIBXSMM_DNN_DATATYPE_F32) &&
          (handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32) &&
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
        const int alpha = 6; /* The value of alpha can be either 4 or 6 */
        const int tileSize = alpha - 2;
        int allowed_unroll = 0;
        int max_acc = 0;
        int flagBenchmark = 0;
        LIBXSMM_UNUSED(flagBenchmark/*TODO*/);

        /* Forward path */
        { wino_desc_fp.alpha = alpha;
          wino_desc_fp.jtiles = (handle->ofh + tileSize - 1) / tileSize;
          wino_desc_fp.itiles = (handle->ofw + tileSize - 1) / tileSize;

          /* LUT for DeepBench */
          if ((240 == handle->ofw) && (24 == handle->ofh) && (16 == handle->desc.N) && (16 == handle->desc.C) && (32 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 1;
            wino_desc_fp.ur = 6;
            flagBenchmark = 1;
            /*status = LIBXSMM_DNN_WARN_FALLBACK;*/
          } else if ((120 == handle->ofw) && (12 == handle->ofh) && (16 == handle->desc.N) && (32 == handle->desc.C) && (64 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 1;
            wino_desc_fp.ur = 6;
            flagBenchmark = 1;
            /*status = LIBXSMM_DNN_WARN_FALLBACK;*/
          } else if ((60 == handle->ofw) && (6 == handle->ofh) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 1;
            wino_desc_fp.ur = 6;
            flagBenchmark = 1;
            /*status = LIBXSMM_DNN_WARN_FALLBACK;*/
          } else if ((54 == handle->ofw) && (54 == handle->ofh) && (8 == handle->desc.N) && (64 == handle->desc.C) && (64 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 1;
            wino_desc_fp.ur = 7;
            flagBenchmark = 1;
            /*status = LIBXSMM_DNN_WARN_FALLBACK;*/
          } else if ((27 == handle->ofw) && (27 == handle->ofh) && (8 == handle->desc.N) && (128 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 1;
            wino_desc_fp.ur = 7;
            flagBenchmark = 1;
            /*status = LIBXSMM_DNN_WARN_FALLBACK;*/
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 16;
            flagBenchmark = 1;
            /*status = LIBXSMM_DNN_WARN_FALLBACK;*/
          } else if ((7 == handle->ofw) && (7 == handle->ofh) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 16;
            flagBenchmark = 1;
            /*status = LIBXSMM_DNN_WARN_FALLBACK;*/
          } else if ((112 == handle->ofw) && (112 == handle->ofh) && (8 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 1;
            wino_desc_fp.ur = 14;
            flagBenchmark = 1;
          } else if ((56 == handle->ofw) && (56 == handle->ofh) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 1;
            wino_desc_fp.ur = 14;
            flagBenchmark = 1;
          } else if ((28 == handle->ofw) && (28 == handle->ofh) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 2;
            wino_desc_fp.ur = 14;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 16;
            flagBenchmark = 1;
          } else if ((7 == handle->ofw) && (7 == handle->ofh) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 16;
            flagBenchmark = 1;
            /*status = LIBXSMM_DNN_WARN_FALLBACK;*/
          } else if ((112 == handle->ofw) && (112 == handle->ofh) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 1;
            wino_desc_fp.ur = 14;
            flagBenchmark = 1;
          } else if ((56 == handle->ofw) && (56 == handle->ofh) && (16 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 1;
            wino_desc_fp.ur = 14;
            flagBenchmark = 1;
          } else if ((28 == handle->ofw) && (28 == handle->ofh) && (16 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 2;
            wino_desc_fp.ur = 14;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 4;
            wino_desc_fp.ur = 16;
            flagBenchmark = 1;
          } else if ((7 == handle->ofw) && (7 == handle->ofh) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 16;
            wino_desc_fp.ur = 16;
            flagBenchmark = 1;
          }

          /* LUT for AlexNet */
          else if ((13 == handle->ofw) && (13 == handle->ofh) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 16;
            flagBenchmark = 1;
          } else if ((13 == handle->ofw) && (13 == handle->ofh) && (64 <= handle->desc.N) && (384 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 16;
            flagBenchmark = 1;
          } else if ((13 == handle->ofw) && (13 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 16;
            flagBenchmark = 1;
          }

          /* LUT for GoogLenetV1 */
          else if ((56 == handle->ofw) && (56 == handle->ofh) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (192 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 1;
            wino_desc_fp.ur = 14;
            flagBenchmark = 1;
          } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 4;
            wino_desc_fp.ur = 14;
            flagBenchmark = 1;
          } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (192 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 4;
            wino_desc_fp.ur = 14;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (208 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 16;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (112 == handle->desc.C) && (224 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 16;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 16;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (144 == handle->desc.C) && (288 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 16;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 16;
            flagBenchmark = 1;
          } else if ((7 == handle->ofw) && (7 == handle->ofh) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 4;
            flagBenchmark = 1;
          } else if ((7 == handle->ofw) && (7 == handle->ofh) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 4;
            flagBenchmark = 1;
          }

          /* LUT for Overfeat */
          else if ((12 == handle->ofw) && (12 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 12;
            flagBenchmark = 1;
          } else if ((12 == handle->ofw) && (12 == handle->ofh) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (1024 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 12;
            flagBenchmark = 1;
          } else if ((12 == handle->ofw) && (12 == handle->ofh) && (64 <= handle->desc.N) && (1024 == handle->desc.C) && (1024 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 12;
            flagBenchmark = 1;
          }

          /* LUT for VGGA */
          else if ((112 == handle->ofw) && (112 == handle->ofh) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 1;
            wino_desc_fp.ur = 14;
            flagBenchmark = 1;
          } else if ((56 == handle->ofw) && (56 == handle->ofh) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 1;
            wino_desc_fp.ur = 14;
            flagBenchmark = 1;
          } else if ((56 == handle->ofw) && (56 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 1;
            wino_desc_fp.ur = 14;
            flagBenchmark = 1;
          } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 4;
            wino_desc_fp.ur = 14;
            flagBenchmark = 1;
          } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 4; /*2;*/
            wino_desc_fp.ur = 14;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_fp.bimg = 8;
            wino_desc_fp.ur = 4;
            flagBenchmark = 1;
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
            internal_dnn_handle_factors_all( wino_desc_fp.itiles*wino_desc_fp.jtiles*wino_desc_fp.bimg, &(wino_desc_fp.ur), max_acc );
            /* ur should be at least 14 to hide qfma latency */
            wino_desc_fp.ur = LIBXSMM_MIN(LIBXSMM_MAX(wino_desc_fp.ur, 14), wino_desc_fp.itiles*wino_desc_fp.jtiles*wino_desc_fp.bimg);
          }

          /* The following condition checks whether we have encountered an input which is listed in our benchmark LUT */
          /* if (flagBenchmark) printf("In benchmark\n"); */
          /* ur_ifm = blocksifm so that we don't need to zero-initialize M and use streaming store */
          wino_desc_fp.ur_ifm = handle->blocksifm;
          wino_desc_fp.blocks_ifm = handle->blocksifm;

          handle->cwino_fwd = wino_desc_fp;

          /* TODO check JIT errors */
          if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
              libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
              libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
          {
            wino_desc_fp.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
            handle->code_fwd[0].pmm = libxsmm_create_xconv_wino_forward(&wino_desc_fp);
            wino_desc_fp.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1;
            handle->code_fwd[1].pmm = libxsmm_create_xconv_wino_forward(&wino_desc_fp);
            wino_desc_fp.prefetch = (libxsmm_convolution_prefetch_type)(LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 | LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2);
            handle->code_fwd[2].pmm = libxsmm_create_xconv_wino_forward(&wino_desc_fp);
            /* wino_desc_fp.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NO_OUTPUT; */
            /* handle->code_fwd[3].pmm = libxsmm_create_xconv_wino_forward(&wino_desc_fp); */
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
            flagBenchmark = 1;
          } else if ((120 == handle->desc.W) && (12 == handle->desc.H) && (16 == handle->desc.N) && (32 == handle->desc.C) && (64 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 1;
            wino_desc_bp.ur = 6;
            flagBenchmark = 1;
          } else if ((60 == handle->desc.W) && (6 == handle->desc.H) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 1;
            wino_desc_bp.ur = 6;
            flagBenchmark = 1;
          } else if ((54 == handle->desc.W) && (54 == handle->desc.H) && (8 == handle->desc.N) && (64 == handle->desc.C) && (64 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 1;
            wino_desc_bp.ur = 7;
            flagBenchmark = 1;
          } else if ((27 == handle->desc.W) && (27 == handle->desc.H) && (8 == handle->desc.N) && (128 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 1;
            wino_desc_bp.ur = 7;
            flagBenchmark = 1;
          } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 16;
            flagBenchmark = 1;
          } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 16;
            flagBenchmark = 1;
          } else if ((112 == handle->desc.W) && (112 == handle->desc.H) && (8 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 1;
            wino_desc_bp.ur = 14;
            flagBenchmark = 1;
          } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 1;
            wino_desc_bp.ur = 14;
            flagBenchmark = 1;
          } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 2;
            wino_desc_bp.ur = 14;
            flagBenchmark = 1;
          } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 16;
            flagBenchmark = 1;
          } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 16;
            flagBenchmark = 1;
          } else if ((112 == handle->desc.W) && (112 == handle->desc.H) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 1;
            wino_desc_bp.ur = 14;
            flagBenchmark = 1;
          } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (16 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 1;
            wino_desc_bp.ur = 14;
            flagBenchmark = 1;
          } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (16 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 2;
            wino_desc_bp.ur = 14;
            flagBenchmark = 1;
          } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 4;
            wino_desc_bp.ur = 16;
            flagBenchmark = 1;
          } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 16;
            wino_desc_bp.ur = 16;
            flagBenchmark = 1;
          }

          /* LUT for AlexNet */
          else if ((13 == handle->desc.W) && (13 == handle->desc.H) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 16;
            flagBenchmark = 1;
          } else if ((13 == handle->desc.W) && (13 == handle->desc.H) && (64 <= handle->desc.N) && (384 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 16;
            flagBenchmark = 1;
          } else if ((13 == handle->desc.W) && (13 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 16;
            flagBenchmark = 1;
          }

          /* LUT for GoogLenetV1 */
          else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (192 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 1;
            wino_desc_bp.ur = 14;
            flagBenchmark = 1;
          } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 4;
            wino_desc_bp.ur = 14;
            flagBenchmark = 1;
          } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (192 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 4;
            wino_desc_bp.ur = 14;
            flagBenchmark = 1;
          } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (208 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 16;
            flagBenchmark = 1;
          } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (112 == handle->desc.C) && (224 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 16;
            flagBenchmark = 1;
          } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 16;
            flagBenchmark = 1;
          } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (144 == handle->desc.C) && (288 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 16;
            flagBenchmark = 1;
          } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 16;
            flagBenchmark = 1;
          } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 4;
            flagBenchmark = 1;
          } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 4;
            flagBenchmark = 1;
          }

          /* LUT for Overfeat */
          else if ((12 == handle->desc.W) && (12 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 12;
            flagBenchmark = 1;
          } else if ((12 == handle->desc.W) && (12 == handle->desc.H) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (1024 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 12;
            flagBenchmark = 1;
          } else if ((12 == handle->desc.W) && (12 == handle->desc.H) && (64 <= handle->desc.N) && (1024 == handle->desc.C) && (1024 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 12;
            flagBenchmark = 1;
          }

          /* LUT for VGGA */
          else if ((112 == handle->desc.W) && (112 == handle->desc.H) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 1;
            wino_desc_bp.ur = 14;
            flagBenchmark = 1;
          } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 1;
            wino_desc_bp.ur = 14;
            flagBenchmark = 1;
          } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 1;
            wino_desc_bp.ur = 14;
            flagBenchmark = 1;
          } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 4;
            wino_desc_bp.ur = 14;
            flagBenchmark = 1;
          } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 4;
            wino_desc_bp.ur = 14;
            flagBenchmark = 1;
          } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_bp.bimg = 8;
            wino_desc_bp.ur = 4;
            flagBenchmark = 1;
          }

          /* General scenario */
          else {
            wino_desc_bp.bimg = wino_desc_fp.bimg;
            if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
              max_acc = 24;
            } else {
              max_acc = 26;
            }
            internal_dnn_handle_factors_all( wino_desc_bp.itiles*wino_desc_bp.jtiles*wino_desc_bp.bimg, &(wino_desc_bp.ur), max_acc );
            wino_desc_bp.ur = LIBXSMM_MIN(LIBXSMM_MAX(wino_desc_bp.ur, 14), wino_desc_bp.itiles*wino_desc_bp.jtiles*wino_desc_bp.bimg);
          }

          wino_desc_bp.ur_ifm = handle->blocksofm;
          wino_desc_bp.blocks_ifm = handle->blocksofm;

          handle->cwino_bwd = wino_desc_bp;

          /* TODO check JIT errors */
          if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
              libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
              libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
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
            flagBenchmark = 1;
            /*status = LIBXSMM_DNN_WARN_FALLBACK;*/
          } else if ((120 == handle->ofw) && (12 == handle->ofh) && (16 == handle->desc.N) && (32 == handle->desc.C) && (64 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 1;
            wino_desc_wu.ur = 1;
            flagBenchmark = 1;
            /*status = LIBXSMM_DNN_WARN_FALLBACK;*/
          } else if ((60 == handle->ofw) && (6 == handle->ofh) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 1;
            wino_desc_wu.ur = 1;
            flagBenchmark = 1;
            /*status = LIBXSMM_DNN_WARN_FALLBACK;*/
          } else if ((54 == handle->ofw) && (54 == handle->ofh) && (8 == handle->desc.N) && (64 == handle->desc.C) && (64 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 1;
            if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
              wino_desc_wu.ur = 1;
            } else {
              wino_desc_wu.ur = 2;
            }
            flagBenchmark = 1;
          } else if ((27 == handle->ofw) && (27 == handle->ofh) && (8 == handle->desc.N) && (128 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 1; /*8;*/
            wino_desc_wu.ur = 1; /*2;*/
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 8;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((7 == handle->ofw) && (7 == handle->ofh) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 8;
            wino_desc_wu.ur = 4;
            flagBenchmark = 1;
          } else if ((112 == handle->ofw) && (112 == handle->ofh) && (8 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 1;
            wino_desc_wu.ur = 1;
            flagBenchmark = 1;
          } else if ((56 == handle->ofw) && (56 == handle->ofh) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 1; /*2;*/
            if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
              wino_desc_wu.ur = 1;
            } else {
              wino_desc_wu.ur = 2;
            }
            flagBenchmark = 1;
          } else if ((28 == handle->ofw) && (28 == handle->ofh) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 2; /*4;*/
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 8;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((7 == handle->ofw) && (7 == handle->ofh) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 8;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((112 == handle->ofw) && (112 == handle->ofh) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 1;
            wino_desc_wu.ur = 1;
            flagBenchmark = 1;
          } else if ((56 == handle->ofw) && (56 == handle->ofh) && (16 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 1;
            if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
              wino_desc_wu.ur = 1;
            } else {
              wino_desc_wu.ur = 2;
            }
            flagBenchmark = 1;
          } else if ((28 == handle->ofw) && (28 == handle->ofh) && (16 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 2; /*16;*/
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 4; /*16;*/
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((7 == handle->ofw) && (7 == handle->ofh) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 16;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          }

          /* LUT for AlexNet */
          else if ((13 == handle->ofw) && (13 == handle->ofh) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 8; /*16;*/
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((13 == handle->ofw) && (13 == handle->ofh) && (64 <= handle->desc.N) && (384 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 8; /*16;*/
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((13 == handle->ofw) && (13 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 8; /*16;*/
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          }

          /* LUT for GoogLenetV1 */
          else if ((56 == handle->ofw) && (56 == handle->ofh) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (192 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 1;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 4;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (192 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 4;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (208 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 8; /*16;*/
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (112 == handle->desc.C) && (224 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 8; /*16;*/
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 8; /*16;*/
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (144 == handle->desc.C) && (288 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 8; /*16;*/
            wino_desc_wu.ur = 1;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 8; /*16;*/
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((7 == handle->ofw) && (7 == handle->ofh) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 32;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((7 == handle->ofw) && (7 == handle->ofh) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 32;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          }

          /* LUT for Overfeat */
          else if ((12 == handle->ofw) && (12 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 32;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((12 == handle->ofw) && (12 == handle->ofh) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (1024 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 32;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((12 == handle->ofw) && (12 == handle->ofh) && (64 <= handle->desc.N) && (1024 == handle->desc.C) && (1024 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 32;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          }

          /* LUT for VGGA */
          else if ((112 == handle->ofw) && (112 == handle->ofh) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 1;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((56 == handle->ofw) && (56 == handle->ofh) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 1;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((56 == handle->ofw) && (56 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 1;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 4;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((28 == handle->ofw) && (28 == handle->ofh) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 4;
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
          } else if ((14 == handle->ofw) && (14 == handle->ofh) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (6 == alpha)) {
            wino_desc_wu.bimg = 8; /*16;*/
            wino_desc_wu.ur = 2;
            flagBenchmark = 1;
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
              internal_dnn_handle_factors_all( wino_desc_wu.itiles*wino_desc_wu.jtiles*wino_desc_wu.bimg/4, &(wino_desc_wu.ur), allowed_unroll );
            } else {
              internal_dnn_handle_factors_all( wino_desc_wu.itiles*wino_desc_wu.jtiles*wino_desc_wu.bimg,   &(wino_desc_wu.ur), allowed_unroll );
            }
          }

          wino_desc_wu.ur_ifm = 1;

          handle->cwino_upd = wino_desc_wu;
          /* TODO check JIT errors */
          if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
              libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
              libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
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
          handle->scratch1_size = alpha*alpha*handle->desc.C*handle->desc.K*libxsmm_dnn_typesize(handle->datatype);
          handle->scratch3 = 0;
          handle->scratch3_size = alpha*alpha*ijtiles*handle->desc.N * handle->desc.C * libxsmm_dnn_typesize(handle->datatype);
          handle->scratch4 = 0;
          handle->scratch4_size = alpha*alpha*ijtiles*handle->desc.N * handle->desc.K * libxsmm_dnn_typesize(handle->datatype_itm);
          handle->scratch6 = 0;
          handle->scratch6_size = 0;
          handle->scratchIw = 0;
          handle->scratchIw_size = ijtiles*alpha*alpha*16*libxsmm_dnn_typesize(handle->datatype)*handle->desc.threads;
          handle->scratchOw = 0;
          handle->scratchOw_size = ijtiles*alpha*alpha*16*libxsmm_dnn_typesize(handle->datatype_itm)*handle->desc.threads;
          handle->scratchVk = 0;
          handle->scratchVk_size = handle->scratch3_size;
          handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);
        }
      } else {
        handle->code_fwd[0].xconv.sconv = 0;
        handle->code_fwd[1].xconv.sconv = 0;
        handle->code_fwd[2].xconv.sconv = 0;
        handle->code_fwd[3].xconv.sconv = 0;
        /* Backward path */
        handle->code_bwd[0].xconv.sconv = 0;
        handle->code_bwd[1].xconv.sconv = 0;
        handle->code_bwd[2].xconv.sconv = 0;
        handle->code_bwd[3].xconv.sconv = 0;
        /* weight update path */
        handle->code_upd[0].xconv.sconv = 0;
        handle->code_upd[1].xconv.sconv = 0;
        handle->code_upd[2].xconv.sconv = 0;
        handle->code_upd[3].xconv.sconv = 0;
        handle->code_upd[4].xconv.sconv = 0;
        handle->code_upd[5].xconv.sconv = 0;
        handle->barrier  = 0;
        handle->scratch1 = 0;
        handle->scratch1_size = 0;
        handle->scratch3 = 0;
        handle->scratch3_size = 0;
        handle->scratch4 = 0;
        handle->scratch4_size = 0;
        handle->scratch6 = 0;
        handle->scratch6_size = 0;
      }

      return status;
    }

