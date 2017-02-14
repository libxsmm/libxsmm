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
/* Alexander Heinecke, Rajkishore Barik (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_handle.h"
#include "libxsmm_main.h"
#include <libxsmm.h>

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

  /* now architecture specific */
  if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
      libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
      libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
  {
    noarch = 0;
#define LIBXSMM_FWD_OFH_BLOCKING
#if defined(LIBXSMM_FWD_OFH_BLOCKING)
    if ( ((handle->ofw < 15) && (handle->ofh % 2 == 0) && (handle->desc.S == 1)) ||
         ((handle->ofw < 15) && (handle->ofh % 2 == 0) && (libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_KNM)) ) {
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
      /* for 32 accumuation we need even one register more */
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
      handle->ifmblock = (handle->desc.C >=16) ? 16 : handle->desc.C/2;
      handle->ofmblock = (handle->desc.K >=16) ? 16 : handle->desc.K/2;
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
      handle->ifmblock = (handle->desc.C >=32) ? 32 : handle->desc.C/2;
      handle->ofmblock = (handle->desc.K >=32) ? 32 : handle->desc.K/2;
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
      handle->ifmblock = (handle->desc.C >=16) ? 16 : handle->desc.C/4;
      handle->ofmblock = (handle->desc.K >=16) ? 16 : handle->desc.K/4;
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

    /* RB: updated to reflect the scenario that ifm=3 */
    
#if 0
    if (handle->desc.C < 16) {
      handle->ifmblock = 1;
    }
#endif
   
    /* Check if padded needs to be applied in the input and allocate appropriate buffers */
    if ((handle->desc.pad_h_in == 0) && (handle->desc.pad_w_in == 0) && (handle->desc.pad_h > 0) && (handle->desc.pad_w > 0)) {
      handle->padding_flag = 1;
      handle->scratch5 = 0;
      handle->minibatch_scratch_size = handle->desc.N * handle->blocksifm * handle->ifmblock * handle->fm_lp_block * (handle->ifhp+2*handle->desc.pad_h) * (handle->ifwp+2*handle->desc.pad_w) * libxsmm_dnn_typesize(handle->datatype);
      handle->fwdbwd_scratch_size = handle->desc.threads * handle->blocksifm * handle->ifmblock * handle->fm_lp_block * (handle->ifhp+2*handle->desc.pad_h) * (handle->ifwp+2*handle->desc.pad_w) * libxsmm_dnn_typesize(handle->datatype);
      handle->max_scratch5_size = (handle->minibatch_scratch_size > handle->fwdbwd_scratch_size) ? handle->minibatch_scratch_size : handle->fwdbwd_scratch_size ;
    } else {
      handle->padding_flag = 0;
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
  } else {
    status = LIBXSMM_DNN_WARN_FALLBACK;
    handle->ifmblock = 1;
    handle->ofmblock = 1;
    handle->fm_lp_block = 1;
  }

  /* Let's calculate how many blocks we need */
  handle->blocksifm = handle->desc.C / (handle->ifmblock * handle->fm_lp_block);
  handle->blocksofm = handle->desc.K / (handle->ofmblock * handle->fm_lp_block);

  /* Let's check that we can actually block */
  if ( (handle->desc.C % (handle->ifmblock * handle->fm_lp_block) != 0) ||
       (handle->desc.K % (handle->ofmblock * handle->fm_lp_block) != 0)    )
  {
    status = LIBXSMM_DNN_WARN_FALLBACK;
    handle->ifmblock = 1;
    handle->ofmblock = 1;
    handle->fm_lp_block = 1;
    handle->blocksifm = handle->desc.C / handle->ifmblock;
    handle->blocksofm = handle->desc.K / handle->ofmblock;
  }
  /* Check if padded needs to be applied in the input and allocate appropriate buffers */
  if ((handle->desc.pad_h_in == 0) && (handle->desc.pad_w_in == 0) && (handle->desc.pad_h > 0) && (handle->desc.pad_w > 0)) {
    handle->padding_flag = 1;
    handle->scratch5  = 0;
    handle->minibatch_scratch_size = handle->desc.N * handle->blocksifm * handle->ifmblock * handle->fm_lp_block * (handle->ifhp+2*handle->desc.pad_h) * (handle->ifwp+2*handle->desc.pad_w) * libxsmm_dnn_typesize(handle->datatype);
    handle->fwdbwd_scratch_size = handle->desc.threads * handle->blocksifm * handle->ifmblock * handle->fm_lp_block * (handle->ifhp+2*handle->desc.pad_h) * (handle->ifwp+2*handle->desc.pad_w) * libxsmm_dnn_typesize(handle->datatype);
    handle->max_scratch5_size = (handle->minibatch_scratch_size > handle->fwdbwd_scratch_size) ? handle->minibatch_scratch_size : handle->fwdbwd_scratch_size ;
  } else {
    handle->padding_flag = 0;
  }

  /* TODO: we need to add much more checks here .... */

  if (noarch == 0) {
    /* Forward path */
    { libxsmm_convolution_forward_descriptor descriptor;
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
      if (libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_MIC  ||
          libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE ||
          libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_KNM )
      {
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
        handle->code_fwd[0].pmm = libxsmm_create_xconv_forward(&descriptor);
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NO_WEIGHT;
        handle->code_fwd[1].pmm = libxsmm_create_xconv_forward(&descriptor);
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
        handle->code_fwd[2].pmm = libxsmm_create_xconv_forward(&descriptor);
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NO_OUTPUT;
        handle->code_fwd[3].pmm = libxsmm_create_xconv_forward(&descriptor);
      } else if (libxsmm_target_archid == LIBXSMM_X86_AVX2) {
        /* we don't do prefetching and kh/kw unrolling (ignored in kernel generator) for AVX2 */
        descriptor.unroll_kh = 0;
        descriptor.unroll_kw = 0;
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
        handle->code_fwd[0].pmm = libxsmm_create_xconv_forward(&descriptor);
        if (handle->fwd_ofw_rb_2 != 0) {
          descriptor.ofw_rb = handle->fwd_ofw_rb_2;
          handle->code_fwd[1].pmm = libxsmm_create_xconv_forward(&descriptor);
        } else {
          handle->code_fwd[1].pmm = handle->code_fwd[0].pmm;
        }
        handle->code_fwd[2].pmm = handle->code_fwd[0].pmm;
        handle->code_fwd[3].pmm = handle->code_fwd[0].pmm;
      } else {
        assert(0/*should not happen*/);
      }
    }
    /* Backward path */
    { libxsmm_convolution_backward_descriptor descriptor;
      if (handle->padding_flag == 1) {
        descriptor.ifh_padded = handle->ifhp + 2 * handle->desc.pad_h;
        descriptor.ifw_padded = handle->ifwp + 2 * handle->desc.pad_w;
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
      descriptor.ofw = handle->ofw;
      descriptor.ofw_unroll = 1;
      descriptor.peeled = 0;
      descriptor.datatype = handle->datatype;
      descriptor.datatype_itm = handle->datatype_itm;
      descriptor.option = handle->desc.options;
      descriptor.format = (libxsmm_dnn_tensor_format)(handle->buffer_format | handle->filter_format);
      /* TODO check JIT errors */
      if ( (libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_MIC  ||
            libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE ||
            libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_KNM) &&
           ((handle->filter_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM)) )
      {
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
        descriptor.prefetch_output_ahead = 0;
#if !defined(NDEBUG)
        printf("DEBUG JIT of conv (NON-PEELED):\n  arch: %s\n  type: %s\n  kw: %u\n  unroll_kw: %u\n  kh: %u\n  unroll_kh: %u\n  ofm_block: %u\n  ifm_block: %u\n"
         "  ofh_padded: %u\n  ofw_padded: %u\n  ofh_rb: %u\n  ofw_rb: %u\n  ifh_padded: %u\n  ifw_padded: %u\n"
         "  stride_h: %u\n  stride_w: %u\n  ofw: %u\n  ofw_unroll: %u\n  peeled: %u\n  prefetch_output: %u\n",
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
            descriptor.stride_w,   /* this we use for offsets in the input */
            descriptor.ofw, /* upper bound of oi loop */
            descriptor.ofw_unroll, /*unroll for ofw loop */
            descriptor.peeled, /*peeled */
            descriptor.prefetch_output_ahead /* prefetch output ahead */);
#endif

        /* NONE */
        descriptor.prefetch_output_ahead = 0;
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
        handle->code_bwd[0].pmm = libxsmm_create_xconv_backward(&descriptor);
        /*ALL*/
        descriptor.prefetch_output_ahead = 0;
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
        handle->code_bwd[1].pmm = libxsmm_create_xconv_backward(&descriptor);

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
      } else if ((libxsmm_target_archid == LIBXSMM_X86_AVX2) ||
                   ((handle->filter_format != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) || (handle->buffer_format != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM)) ) {
        /* we don't do prefetching and kh/kw unrolling (ignored in kernel generator) for AVX2 */
        descriptor.unroll_kh = 0;
        descriptor.unroll_kw = 0;
        descriptor.ofw_unroll = 0;
        descriptor.prefetch_output_ahead = 0;
        descriptor.peeled = 0;
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
        handle->code_bwd[0].pmm = libxsmm_create_xconv_backward(&descriptor);
        handle->code_bwd[1].pmm = handle->code_bwd[0].pmm;
        handle->code_bwd[2].pmm = handle->code_bwd[0].pmm;
        handle->code_bwd[3].pmm = handle->code_bwd[0].pmm;
      } else {
        assert(0/*should not happen*/);
      }
    } /* End of backward */
    /* TODO weight update path */
    { libxsmm_convolution_weight_update_descriptor descriptor;
      if (handle->padding_flag == 1) {
        descriptor.ifh_padded = handle->ifhp + 2 * handle->desc.pad_h;
        descriptor.ifw_padded = handle->ifwp + 2 * handle->desc.pad_w;
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

      /* TODO check JIT errors */
      if ( /*(*/libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_MIC  ||
            libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE ||
            libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_KNM /*)*/ /*&&
            ((handle->filter_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM))*/ )
      {
        const unsigned int wu_each_iter_code_size = 10 * (descriptor.ifm_block == 1 ? descriptor.kw : descriptor.ifm_block);
        const unsigned int wu_max_code_size = 20000;
        int upper_limit_ofw_rb = wu_max_code_size / wu_each_iter_code_size, upper_limit_ofh_rb = 0;
        descriptor.ifm_unroll = 1;

        for (i = LIBXSMM_MIN(upper_limit_ofw_rb, LIBXSMM_MIN(56,handle->ofw)); i >= 1; i--) {
          if (handle->ofw % i == 0) break;
        }
        descriptor.ofw_rb =  i;
        upper_limit_ofh_rb = wu_max_code_size / (descriptor.ofw_rb * wu_each_iter_code_size);
        for (i = LIBXSMM_MIN(upper_limit_ofh_rb, handle->ofh); i >= 1; i--) {
          if (handle->ofh % i == 0) break;
        }
        descriptor.ofh_rb =  i;

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
        handle->upd_ofh_rb = descriptor.ofh_rb;
        handle->upd_ofw_rb = descriptor.ofw_rb;
        descriptor.transpose_ofw_ifm = 0;
#if !defined(NDEBUG)
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
        descriptor.transpose_ofw_ifm = 0;
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
        handle->code_upd[0].pmm = libxsmm_create_xconv_update_weights(&descriptor);
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
        handle->code_upd[0].pmm = libxsmm_create_xconv_update_weights(&descriptor);
        handle->code_upd[1].pmm = handle->code_upd[0].pmm;
        handle->code_upd[2].pmm = handle->code_upd[0].pmm;
        handle->code_upd[3].pmm = handle->code_upd[0].pmm;
        handle->code_upd[4].pmm = handle->code_upd[0].pmm;
        handle->code_upd[5].pmm = handle->code_upd[0].pmm;
      } else {
        assert(0/*should not happen*/);
      }
    } /* end of weight-update handle */
    {
      handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);

      /* backward transpose filters */
      handle->scratch1 = 0;
      handle->scratch1_size = handle->blocksifm * handle->ifmblock * handle->blocksofm * handle->ofmblock
                                * handle->desc.R * handle->desc.S * handle->fm_lp_block * libxsmm_dnn_typesize(handle->datatype);

      /* weight update transpose of minibatch */
      handle->scratch3 = 0;
      handle->scratch3_size = handle->desc.N * handle->blocksifm * handle->ifmblock * handle->ifhp * handle->ifwp
                                * handle->fm_lp_block * libxsmm_dnn_typesize(handle->datatype);

      /* minibatch parallel execution of weight update kernel */
      if ((handle->ifmblock == 1) || ((handle->blocksifm * handle->blocksofm) < (2*handle->desc.threads))) {
        handle->upd_use_thread_fil = 1;
        handle->scratch4 = 0;
        handle->scratch4_size = handle->desc.threads * handle->blocksifm * handle->ifmblock * handle->blocksofm * handle->ofmblock
          * handle->desc.R * handle->desc.S * handle->fm_lp_block * libxsmm_dnn_typesize(handle->datatype);

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
      } else {
        handle->scratch6 = 0;
        handle->scratch6_size = 0;
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
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_dnn_handle_factors(
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
 * This function finds the loop increments (ur_i, ur_j, ur_m) of (itiles, jtiles, bimg)
 * such that ur_i*ur_j*ur_m <= max_acc
 * The following loop may not give an optimal solution (knapsack problem)
 * Eg, 12 = 3*2*2, MAX_ACC = 4, this algorithm: 3, best: 2*2
 */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_dnn_handle_factors_ijm(
                  unsigned int itiles,
                  unsigned int  jtiles,
                  unsigned int  bimg,
                  unsigned int* ur_i,
                  unsigned int* ur_j,
                  unsigned int* ur_m,
                  unsigned int  max_acc)
{
  unsigned int i;
  unsigned int j;
  unsigned int k;
  unsigned int index;
  int found;
  unsigned int cur_acc;
  unsigned int fact[10];
  unsigned int cur_fact[10];
  unsigned int fact_i[10];
  unsigned int fact_j[10];
  unsigned int fact_m[10];

  for ( i = 0; i < 10; i++ ) {
    fact[i] = 1;
    cur_fact[i] = 1;
  }
  internal_dnn_handle_factors(itiles*jtiles*bimg, fact);

  cur_acc = 1;
  index = 0;
  for ( i = 0; fact[i] != 1; i++ ) {
    if ( (fact[i] * cur_acc) <= max_acc ) {
      cur_acc = cur_acc*fact[i];
      cur_fact[index] = fact[i];
      index++;
    }
  }

  for ( i = 0; i < 10; i++ ) {
    fact_i[i] = 1;
    fact_j[i] = 1;
    fact_m[i] = 1;
  }
  internal_dnn_handle_factors(itiles, fact_i);
  internal_dnn_handle_factors(jtiles, fact_j);
  internal_dnn_handle_factors(bimg,   fact_m);

  *ur_i = 1;
  *ur_j = 1;
  *ur_m = 1;

  for ( i= 0; cur_fact[i] != 1; i++ ) {
    found = 0;
    for ( j = 0; fact_i[j] != 1; j++ ) {
      if ( cur_fact[i] == fact_i[j] ) {
        *ur_i = (*ur_i)*fact_i[j];
        found = 1;
        /* Remove this element from fact_i */
        for ( k = j; k < 9; k++ ) {
          fact_i[k] = fact_i[k+1];
        }
        break;
      }
    }
    if ( found == 1 )
      continue;

    for ( j = 0; fact_j[j] != 1; j++ ) {
      if ( cur_fact[i] == fact_j[j] ) {
        *ur_j = (*ur_j)*fact_j[j];
        found = 1;
        /* Remove this element from fact_j */
        for ( k = j; k < 9; k++ ) {
          fact_j[k] = fact_j[k+1];
        }
        break;
      }
    }
    if ( found == 1 )
      continue;

    for ( j = 0; fact_m[j] != 1; j++ ) {
      if ( cur_fact[i] == fact_m[j] ) {
        *ur_m = (*ur_m)*fact_m[j];
        found = 1;
        /* Remove this element from fact_m */
        for ( k = j; k < 9; k++ ) {
          fact_m[k] = fact_m[k+1];
        }
        break;
      }
    }
    if ( found == 1 ) {
      continue;
    }

#if !defined(NDEBUG)
    fprintf(stderr, "LIBXSMM error: Control should not reach here FACT=%u\n", cur_fact[i]);
    assert(0);
#endif
  }
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_internal_create_conv_handle_winograd_check( libxsmm_dnn_layer* handle ) {
  /* flag to test if we found an architecture which is supported */
  int noarch = 1;
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

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
    handle->ofmblock = 16 ;
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
      if ((240 == handle->desc.W) && (24 == handle->desc.H) && (16 == handle->desc.N) && (16 == handle->desc.C) && (32 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 15;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        status = LIBXSMM_DNN_WARN_FALLBACK;
        flagBenchmark = 1;
      } else if ((120 == handle->desc.W) && (12 == handle->desc.H) && (16 == handle->desc.N) && (32 == handle->desc.C) && (64 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 15;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
        status = LIBXSMM_DNN_WARN_FALLBACK;
      } else if ((60 == handle->desc.W) && (6 == handle->desc.H) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 15;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
        status = LIBXSMM_DNN_WARN_FALLBACK;
      } else if ((54 == handle->desc.W) && (54 == handle->desc.H) && (8 == handle->desc.N) && (64 == handle->desc.C) && (64 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 14;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
        status = LIBXSMM_DNN_WARN_FALLBACK;
      } else if ((27 == handle->desc.W) && (27 == handle->desc.H) && (8 == handle->desc.N) && (128 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 7;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
        status = LIBXSMM_DNN_WARN_FALLBACK;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 4;
        wino_desc_fp.ur_j = 4;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
        status = LIBXSMM_DNN_WARN_FALLBACK;
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 8;
        wino_desc_fp.ur_i = 2;
        wino_desc_fp.ur_j = 2;
        wino_desc_fp.ur_m = 2;
        flagBenchmark = 1;
        status = LIBXSMM_DNN_WARN_FALLBACK;
      } else if ((112 == handle->desc.W) && (112 == handle->desc.H) && (8 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 14;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
        status = LIBXSMM_DNN_WARN_FALLBACK;
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 7;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 7;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 8;
        wino_desc_fp.ur_i = 4;
        wino_desc_fp.ur_j = 2;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 8;
        wino_desc_fp.ur_i = 2;
        wino_desc_fp.ur_j = 2;
        wino_desc_fp.ur_m = 2;
        flagBenchmark = 1;
        status = LIBXSMM_DNN_WARN_FALLBACK;
      } else if ((112 == handle->desc.W) && (112 == handle->desc.H) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 14;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
        status = LIBXSMM_DNN_WARN_FALLBACK;
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (16 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 7;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (16 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 7;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 4;
        wino_desc_fp.ur_i = 4;
        wino_desc_fp.ur_j = 2;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 16;
        wino_desc_fp.ur_i = 2;
        wino_desc_fp.ur_j = 2;
        wino_desc_fp.ur_m = 2;
        flagBenchmark = 1;
      }

      /* LUT for AlexNet */
      else if ((13 == handle->desc.W) && (13 == handle->desc.H) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 8;
        wino_desc_fp.ur_i = 4;
        wino_desc_fp.ur_j = 4;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((13 == handle->desc.W) && (13 == handle->desc.H) && (64 <= handle->desc.N) && (384 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 8;
        wino_desc_fp.ur_i = 4;
        wino_desc_fp.ur_j = 4;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((13 == handle->desc.W) && (13 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 8;
        wino_desc_fp.ur_i = 4;
        wino_desc_fp.ur_j = 4;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      }

      /* LUT for GoogLenetV1 */
      else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (192 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 14;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 4;
        wino_desc_fp.ur_i = 7;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (192 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 4;
        wino_desc_fp.ur_i = 7;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (208 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 4;
        wino_desc_fp.ur_j = 4;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (112 == handle->desc.C) && (224 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 4;
        wino_desc_fp.ur_i = 4;
        wino_desc_fp.ur_j = 4;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 4;
        wino_desc_fp.ur_j = 4;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (144 == handle->desc.C) && (288 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 4;
        wino_desc_fp.ur_i = 4;
        wino_desc_fp.ur_j = 4;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 4;
        wino_desc_fp.ur_i = 4;
        wino_desc_fp.ur_j = 4;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 8;
        wino_desc_fp.ur_i = 2;
        wino_desc_fp.ur_j = 2;
        wino_desc_fp.ur_m = 2;
        flagBenchmark = 1;
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 16;
        wino_desc_fp.ur_i = 2;
        wino_desc_fp.ur_j = 2;
        wino_desc_fp.ur_m = 2;
        flagBenchmark = 1;
      }

      /* LUT for Overfeat */
      else if ((12 == handle->desc.W) && (12 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 8;
        wino_desc_fp.ur_i = 3;
        wino_desc_fp.ur_j = 3;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((12 == handle->desc.W) && (12 == handle->desc.H) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (1024 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 8;
        wino_desc_fp.ur_i = 3;
        wino_desc_fp.ur_j = 3;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((12 == handle->desc.W) && (12 == handle->desc.H) && (64 <= handle->desc.N) && (1024 == handle->desc.C) && (1024 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 8;
        wino_desc_fp.ur_i = 3;
        wino_desc_fp.ur_j = 3;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      }

      /* LUT for VGGA */
      else if ((112 == handle->desc.W) && (112 == handle->desc.H) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 7;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 7;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 7;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 7;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 2;
        wino_desc_fp.bimg = 2;
        wino_desc_fp.ur_i = 7;
        wino_desc_fp.ur_j = 1;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_fp.vratio = 1;
        wino_desc_fp.bimg = 8;
        wino_desc_fp.ur_i = 4;
        wino_desc_fp.ur_j = 4;
        wino_desc_fp.ur_m = 1;
        flagBenchmark = 1;
      }

      /* General scenario */
      else if ((handle->desc.C >= 128) && (handle->desc.K >= 128)) {
        if (((handle->desc.N % 8) == 0) && (handle->desc.C >= 256) && (handle->desc.K >= 256)) {
          wino_desc_fp.bimg = 8;
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
        internal_dnn_handle_factors_ijm( wino_desc_fp.itiles, wino_desc_fp.jtiles, wino_desc_fp.bimg,
                     &(wino_desc_fp.ur_i), &(wino_desc_fp.ur_j), &(wino_desc_fp.ur_m), max_acc );
        if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
          wino_desc_fp.vratio = 1;
        } else if (wino_desc_fp.ur_i * wino_desc_fp.ur_j * wino_desc_fp.ur_m <= 13 && handle->blocksofm % 2 == 0 && handle->blocksifm % 2 == 0) {
          wino_desc_fp.vratio = 2;
        } else {
          wino_desc_fp.vratio = 1;
        }
      } else {
        if ((handle->desc.N % 2) == 0) {
          wino_desc_fp.bimg = 2;
        } else {
          wino_desc_fp.bimg = 1;
        }
        if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
          max_acc = 24;
        } else {
          max_acc = 26;
        }
        internal_dnn_handle_factors_ijm( wino_desc_fp.itiles, wino_desc_fp.jtiles, wino_desc_fp.bimg,
                     &(wino_desc_fp.ur_i), &(wino_desc_fp.ur_j), &(wino_desc_fp.ur_m), max_acc );
        if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
          wino_desc_fp.vratio = 1;
        } else if (wino_desc_fp.ur_i * wino_desc_fp.ur_j * wino_desc_fp.ur_m <= 13 && handle->blocksofm % 2 == 0 && handle->blocksifm % 2 == 0) {
          wino_desc_fp.vratio = 2;
        } else {
          wino_desc_fp.vratio = 1;
        }
      }

      /* The following condition checks whether we have encountered an input which is listed in our benchmark LUT */
      /* if (flagBenchmark) printf("In benchmark\n"); */

      handle->cwino_fwd = wino_desc_fp;

      /* TODO check JIT errors */
      if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
      {
        wino_desc_fp.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
        handle->code_fwd[0].pmm = libxsmm_create_xconv_wino_forward(&wino_desc_fp);
        wino_desc_fp.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
        handle->code_fwd[1].pmm = libxsmm_create_xconv_wino_forward(&wino_desc_fp);
        /* wino_desc_fp.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NO_WEIGHT; */
        /* handle->code_fwd[2].pmm = libxsmm_create_xconv_wino_forward(&wino_desc_fp); */
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
      if ((240 == handle->desc.W) && (24 == handle->desc.H) && (16 == handle->desc.N) && (16 == handle->desc.C) && (32 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 15;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((120 == handle->desc.W) && (12 == handle->desc.H) && (16 == handle->desc.N) && (32 == handle->desc.C) && (64 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 15;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((60 == handle->desc.W) && (6 == handle->desc.H) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 15;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((54 == handle->desc.W) && (54 == handle->desc.H) && (8 == handle->desc.N) && (64 == handle->desc.C) && (64 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 14;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((27 == handle->desc.W) && (27 == handle->desc.H) && (8 == handle->desc.N) && (128 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 7;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 4;
        wino_desc_bp.ur_j = 4;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 8;
        wino_desc_bp.ur_i = 2;
        wino_desc_bp.ur_j = 2;
        wino_desc_bp.ur_m = 2;
        flagBenchmark = 1;
      } else if ((112 == handle->desc.W) && (112 == handle->desc.H) && (8 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 14;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 7;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 7;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 8;
        wino_desc_bp.ur_i = 4;
        wino_desc_bp.ur_j = 2;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 8;
        wino_desc_bp.ur_i = 2;
        wino_desc_bp.ur_j = 2;
        wino_desc_bp.ur_m = 2;
        flagBenchmark = 1;
      } else if ((112 == handle->desc.W) && (112 == handle->desc.H) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 14;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (16 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 7;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (16 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 7;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 4;
        wino_desc_bp.ur_i = 4;
        wino_desc_bp.ur_j = 2;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 16;
        wino_desc_bp.ur_i = 2;
        wino_desc_bp.ur_j = 2;
        wino_desc_bp.ur_m = 2;
        flagBenchmark = 1;
      }

      /* LUT for AlexNet */
      else if ((13 == handle->desc.W) && (13 == handle->desc.H) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 8;
        wino_desc_bp.ur_i = 4;
        wino_desc_bp.ur_j = 4;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((13 == handle->desc.W) && (13 == handle->desc.H) && (64 <= handle->desc.N) && (384 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 8;
        wino_desc_bp.ur_i = 4;
        wino_desc_bp.ur_j = 4;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((13 == handle->desc.W) && (13 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 8;
        wino_desc_bp.ur_i = 4;
        wino_desc_bp.ur_j = 4;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      }

      /* LUT for GoogLenetV1 */
      else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (192 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 14;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 4;
        wino_desc_bp.ur_i = 7;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (192 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 4;
        wino_desc_bp.ur_i = 7;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (208 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 4;
        wino_desc_bp.ur_j = 4;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (112 == handle->desc.C) && (224 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 4;
        wino_desc_bp.ur_i = 4;
        wino_desc_bp.ur_j = 4;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 4;
        wino_desc_bp.ur_j = 4;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (144 == handle->desc.C) && (288 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 4;
        wino_desc_bp.ur_i = 4;
        wino_desc_bp.ur_j = 4;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 4;
        wino_desc_bp.ur_i = 4;
        wino_desc_bp.ur_j = 4;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 8;
        wino_desc_bp.ur_i = 2;
        wino_desc_bp.ur_j = 2;
        wino_desc_bp.ur_m = 2;
        flagBenchmark = 1;
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 16;
        wino_desc_bp.ur_i = 2;
        wino_desc_bp.ur_j = 2;
        wino_desc_bp.ur_m = 2;
        flagBenchmark = 1;
      }

      /* LUT for Overfeat */
      else if ((12 == handle->desc.W) && (12 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 8;
        wino_desc_bp.ur_i = 3;
        wino_desc_bp.ur_j = 3;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((12 == handle->desc.W) && (12 == handle->desc.H) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (1024 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 8;
        wino_desc_bp.ur_i = 3;
        wino_desc_bp.ur_j = 3;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((12 == handle->desc.W) && (12 == handle->desc.H) && (64 <= handle->desc.N) && (1024 == handle->desc.C) && (1024 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 8;
        wino_desc_bp.ur_i = 3;
        wino_desc_bp.ur_j = 3;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      }

      /* LUT for VGGA */
      else if ((112 == handle->desc.W) && (112 == handle->desc.H) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 7;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 7;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 7;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 7;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 2;
        wino_desc_bp.bimg = 2;
        wino_desc_bp.ur_i = 7;
        wino_desc_bp.ur_j = 1;
        wino_desc_bp.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_bp.vratio = 1;
        wino_desc_bp.bimg = 8;
        wino_desc_bp.ur_i = 4;
        wino_desc_bp.ur_j = 4;
        wino_desc_bp.ur_m = 1;
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
        internal_dnn_handle_factors_ijm( wino_desc_bp.itiles, wino_desc_bp.jtiles, wino_desc_bp.bimg,
                     &(wino_desc_bp.ur_i), &(wino_desc_bp.ur_j), &(wino_desc_bp.ur_m), max_acc );
        if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
          wino_desc_bp.vratio = 1;
        } else if (wino_desc_bp.ur_i * wino_desc_bp.ur_j * wino_desc_bp.ur_m <= 13 && handle->blocksofm % 2 == 0 && handle->blocksifm % 2 == 0) {
          wino_desc_bp.vratio = 2;
        } else {
          wino_desc_bp.vratio = 1;
        }
      }

      handle->cwino_bwd = wino_desc_bp;

      /* TODO check JIT errors */
      if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
      {
        /* NONE */
        wino_desc_bp.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
        handle->code_bwd[0].pmm = libxsmm_create_xconv_wino_backward(&wino_desc_bp);
        /* ALL */
        wino_desc_bp.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
        handle->code_bwd[1].pmm = libxsmm_create_xconv_wino_backward(&wino_desc_bp);
      } else {
        assert(0/*should not happen*/);
      }
    } /* End of backward */
    /* Weight update path */
    { wino_desc_wu.alpha = alpha;
      wino_desc_wu.jtiles = wino_desc_fp.jtiles;
      wino_desc_wu.itiles = wino_desc_fp.itiles;

      /* LUT for DeepBench */
      if ((240 == handle->desc.W) && (24 == handle->desc.H) && (16 == handle->desc.N) && (16 == handle->desc.C) && (32 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 15;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((120 == handle->desc.W) && (12 == handle->desc.H) && (16 == handle->desc.N) && (32 == handle->desc.C) && (64 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 3;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((60 == handle->desc.W) && (6 == handle->desc.H) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 15;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((54 == handle->desc.W) && (54 == handle->desc.H) && (8 == handle->desc.N) && (64 == handle->desc.C) && (64 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 7;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((27 == handle->desc.W) && (27 == handle->desc.H) && (8 == handle->desc.N) && (128 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 7;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 4;
        wino_desc_wu.ur_j = 2;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 8;
        wino_desc_wu.ur_i = 2;
        wino_desc_wu.ur_j = 2;
        wino_desc_wu.ur_m = 2;
        flagBenchmark = 1;
      } else if ((112 == handle->desc.W) && (112 == handle->desc.H) && (8 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 7;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (8 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 7;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (8 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 7;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 8;
        wino_desc_wu.ur_i = 4;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 2;
        flagBenchmark = 1;
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (8 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 8;
        wino_desc_wu.ur_i = 2;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 4;
        flagBenchmark = 1;
      } else if ((112 == handle->desc.W) && (112 == handle->desc.H) && (16 == handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 7;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (16 == handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 7;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (16 == handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 7;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 16;
        wino_desc_wu.ur_i = 4;
        wino_desc_wu.ur_j = 2;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (16 == handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 16;
        wino_desc_wu.ur_i = 2;
        wino_desc_wu.ur_j = 2;
        wino_desc_wu.ur_m = 2;
        flagBenchmark = 1;
      }

      /* LUT for AlexNet */
      else if ((13 == handle->desc.W) && (13 == handle->desc.H) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 4;
        wino_desc_wu.ur_i = 4;
        wino_desc_wu.ur_j = 2;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((13 == handle->desc.W) && (13 == handle->desc.H) && (64 <= handle->desc.N) && (384 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 4;
        wino_desc_wu.ur_i = 4;
        wino_desc_wu.ur_j = 2;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((13 == handle->desc.W) && (13 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 4;
        wino_desc_wu.ur_i = 4;
        wino_desc_wu.ur_j = 2;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      }

      /* LUT for GoogLenetV1 */
      else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (192 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 4;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 2;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 8;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (192 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 16;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (96 == handle->desc.C) && (208 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 16;
        wino_desc_wu.ur_i = 4;
        wino_desc_wu.ur_j = 4;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (112 == handle->desc.C) && (224 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 16;
        wino_desc_wu.ur_i = 4;
        wino_desc_wu.ur_j = 4;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 8;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (144 == handle->desc.C) && (288 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 16;
        wino_desc_wu.ur_i = 4;
        wino_desc_wu.ur_j = 4;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 4;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (64 <= handle->desc.N) && (160 == handle->desc.C) && (320 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 16;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((7 == handle->desc.W) && (7 == handle->desc.H) && (64 <= handle->desc.N) && (192 == handle->desc.C) && (384 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 8;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 2;
        flagBenchmark = 1;
      }

      /* LUT for Overfeat */
      else if ((12 == handle->desc.W) && (12 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 16;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((12 == handle->desc.W) && (12 == handle->desc.H) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (1024 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 16;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      } else if ((12 == handle->desc.W) && (12 == handle->desc.H) && (64 <= handle->desc.N) && (1024 == handle->desc.C) && (1024 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 16;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      }

      /* LUT for VGGA */
      else if ((112 == handle->desc.W) && (112 == handle->desc.H) && (64 <= handle->desc.N) && (64 == handle->desc.C) && (128 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 2;
        flagBenchmark = 1;
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (64 <= handle->desc.N) && (128 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 4;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 2;
        flagBenchmark = 1;
      } else if ((56 == handle->desc.W) && (56 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (256 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 1;
        wino_desc_wu.bimg = 4;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 2;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (256 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 2;
        flagBenchmark = 1;
      } else if ((28 == handle->desc.W) && (28 == handle->desc.H) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 2;
        flagBenchmark = 1;
      } else if ((14 == handle->desc.W) && (14 == handle->desc.H) && (64 <= handle->desc.N) && (512 == handle->desc.C) && (512 == handle->desc.K) && (1 == handle->desc.pad_h) && (6 == alpha)) {
        wino_desc_wu.vratio = 2;
        wino_desc_wu.bimg = 2;
        wino_desc_wu.ur_i = 1;
        wino_desc_wu.ur_j = 1;
        wino_desc_wu.ur_m = 1;
        flagBenchmark = 1;
      }

      /* General scenario */
      else if ((handle->desc.C >= 256) && (handle->desc.K >= 256)) {
        if (((handle->desc.N % 16) == 0) && (handle->desc.C >= 512) && (handle->desc.K >= 512)) {
          wino_desc_wu.bimg = 16;
        } else if (((handle->desc.N % 8) == 0) && (handle->desc.C >= 256) && (handle->desc.K >= 256)) {
          wino_desc_wu.bimg = 8;
        } else if ((handle->desc.N % 2) == 0) {
          wino_desc_wu.bimg = 2;
        } else {
          wino_desc_wu.bimg = 1;
        }
        allowed_unroll = 512 / (wino_desc_wu.bimg*wino_desc_wu.itiles*wino_desc_wu.jtiles);
        allowed_unroll = (allowed_unroll > 26) ? 26 : allowed_unroll;
        internal_dnn_handle_factors_ijm( wino_desc_wu.itiles, wino_desc_wu.jtiles, wino_desc_wu.bimg,
                     &(wino_desc_wu.ur_i), &(wino_desc_wu.ur_j), &(wino_desc_wu.ur_m), allowed_unroll );
        if (wino_desc_wu.ur_i * wino_desc_wu.ur_j * wino_desc_wu.ur_m <= 13 && handle->blocksofm % 2 == 0 && handle->blocksifm % 2 == 0) {
          wino_desc_wu.vratio = 2;
        } else {
          wino_desc_wu.vratio = 1;
        }
      } else {
        if ((handle->desc.N % 2) == 0) {
          wino_desc_wu.bimg = 2;
        } else {
          wino_desc_wu.bimg = 1;
        }
        allowed_unroll = 512 / (wino_desc_wu.bimg*wino_desc_wu.itiles*wino_desc_wu.jtiles);
        allowed_unroll = (allowed_unroll > 26) ? 26 : allowed_unroll;
        internal_dnn_handle_factors_ijm( wino_desc_wu.itiles, wino_desc_wu.jtiles, wino_desc_wu.bimg,
                     &(wino_desc_wu.ur_i), &(wino_desc_wu.ur_j), &(wino_desc_wu.ur_m), allowed_unroll );
        if (wino_desc_wu.ur_i * wino_desc_wu.ur_j * wino_desc_wu.ur_m <= 13 && handle->blocksofm % 2 == 0 && handle->blocksifm % 2 == 0) {
          wino_desc_wu.vratio = 2;
        } else {
          wino_desc_wu.vratio = 1;
        }
      }

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
      handle->scratchIw_size = ijtiles*alpha*alpha*32*libxsmm_dnn_typesize(handle->datatype)*handle->desc.threads;
      handle->scratchOw = 0;
      handle->scratchOw_size = ijtiles*alpha*alpha*32*libxsmm_dnn_typesize(handle->datatype_itm)*handle->desc.threads;
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

