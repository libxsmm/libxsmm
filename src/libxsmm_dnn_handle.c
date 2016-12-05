/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
/* Alexander Heinecke (Intel Corp.)
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


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_internal_create_conv_handle_direct_check( libxsmm_dnn_conv_handle* handle ) {
  /* flag to test if we found an architecture which is supported */
  int noarch = 1;
  /* general counting helper */
  int i = 0;
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* now architecture specific */
  if (libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_MIC  ||
      libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE)
  {
    noarch = 0;
#define LIBXSMM_FWD_OFH_BLOCKING
#if defined(LIBXSMM_FWD_OFH_BLOCKING)
    if ((handle->ofw < 15) && (handle->ofh % 2 == 0) && (handle->desc.S == 1)) {
      handle->fwd_ofw_rb = handle->ofw;
      handle->fwd_ofh_rb = 2;
      /* on AVX512_CORE and int this only works for smaller 13 */
      if ( (((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) ||
           (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8)) &&
           (libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE)) &&
           (handle->ofw > 12) ) {
        handle->fwd_ofh_rb = 1;
      }
    }
    else {
#endif
      /* we need additional temp registers when running with int on AVX512_CORE */
      if ( ((libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE) &&
           (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) &&
           (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32)) ||
           ((libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE) &&
           (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) &&
           (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I16) &&
           (handle->desc.options == LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) ) ) {
        for (i = 26; i > 1; --i) {
          if (handle->ofw % i == 0) break;
        }
      /* for 32 accumuation we need even one register more */
      } else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) &&
           (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32) &&
           (handle->desc.options == LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED)  ) {
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
    for(i = LIBXSMM_MIN(24, handle->ofw); i > 1; i--) {
      if(handle->ofw % i == 0) break;
    }
    handle->bwd_ofw_rb = i;
#endif

#define LIBXSMM_UPD_OFH_BLOCKING
#if defined(LIBXSMM_UPD_OFH_BLOCKING)
    for(i = LIBXSMM_MIN(28, handle->ofh); i > 1; i--) {
      if(handle->ofh % i == 0) break;
    }
    handle->upd_ofh_rb = i;
    for(i = LIBXSMM_MIN(28, handle->ofw); i > 1; i--) {
      if(handle->ofw % i == 0) break;
    }
    handle->upd_ofw_rb = i;
#endif

    /* calculate blockings */
    if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
      handle->ifmblock = (handle->desc.C >=16) ? 16 : handle->desc.C;
      handle->ofmblock = (handle->desc.K >=16) ? 16 : handle->desc.K;
      handle->fm_lp_block = 1;
    }
    else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32) ) {
      handle->ifmblock = (handle->desc.C >=16) ? 16 : handle->desc.C/2;
      handle->ofmblock = (handle->desc.K >=16) ? 16 : handle->desc.K;
      handle->fm_lp_block = 2;
      if ( libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_MIC ) {
        status = LIBXSMM_DNN_WARN_FALLBACK;
        handle->ifmblock = 1;
        handle->ofmblock = 1;
        handle->fm_lp_block = 1;
        noarch = 1;
      }
    }
    else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I16)
                 && (handle->desc.options == LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) ) {
      handle->ifmblock = (handle->desc.C >=32) ? 32 : handle->desc.C/2;
      handle->ofmblock = (handle->desc.K >=32) ? 32 : handle->desc.K;
      handle->fm_lp_block = 2;
      if ( libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_MIC ) {
        status = LIBXSMM_DNN_WARN_FALLBACK;
        handle->ifmblock = 1;
        handle->ofmblock = 1;
        handle->fm_lp_block = 1;
        noarch = 1;
      }
    }
    else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32)
                 && (handle->desc.options == LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) ) {
      handle->ifmblock = (handle->desc.C >=16) ? 16 : handle->desc.C/4;
      handle->ofmblock = (handle->desc.K >=16) ? 16 : handle->desc.K;
      handle->fm_lp_block = 4;
      if ( libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_MIC ) {
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
    if (handle->desc.C < 16) {
      handle->ifmblock = 1;
    }
  } else if ( libxsmm_get_target_archid() == LIBXSMM_X86_AVX2 ) {
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
    for(i = LIBXSMM_MIN(3, handle->ofh); i > 1; i--) {
      if(handle->ofh % i == 0) break;
    }
    handle->upd_ofh_rb = i;
    for(i = LIBXSMM_MIN(3, handle->ofw); i > 1; i--) {
      if(handle->ofw % i == 0) break;
    }
    handle->upd_ofw_rb = i;
#endif

    /* calculate blockings */
    if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
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
    else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32) ) {
      status = LIBXSMM_DNN_WARN_FALLBACK;
      handle->ifmblock = 1;
      handle->ofmblock = 1;
      handle->fm_lp_block = 1;
      noarch = 1;
    }
    else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I16)
                && (handle->desc.options == LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) ) {
      status = LIBXSMM_DNN_WARN_FALLBACK;
      handle->ifmblock = 1;
      handle->ofmblock = 1;
      handle->fm_lp_block = 1;
      noarch = 1;
    }
    else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32)
                && (handle->desc.options == LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) ) {
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
  handle->blocksofm = handle->desc.K / handle->ofmblock;

  /* Let's check that we can actually block */
  if (handle->desc.C % (handle->ifmblock * handle->fm_lp_block) != 0 ||
      handle->desc.K % handle->ofmblock != 0)
  {
    status = LIBXSMM_DNN_WARN_FALLBACK;
    handle->ifmblock = 1;
    handle->ofmblock = 1;
    handle->fm_lp_block = 1;
    handle->blocksifm = handle->desc.C / handle->ifmblock;
    handle->blocksofm = handle->desc.K / handle->ofmblock;
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
      if ( ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) ||
           (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) ) &&
           (libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE) &&
           handle->desc.R > 1 && handle->desc.S > 1 && handle->fwd_ofh_rb == 1 ) {
        /* we need 3 instrad of 1 instruction for FMA -> do not perform any unrolling in kh/kw to control code size */
        descriptor.unroll_kh = 0;
        descriptor.unroll_kw = 0;
      }
      descriptor.ifh_padded = handle->desc.H;
      descriptor.ifw_padded = handle->desc.W;
      descriptor.kh = handle->desc.R;
      descriptor.kw = handle->desc.S;
      descriptor.stride_h = handle->desc.u;
      descriptor.stride_w = handle->desc.v;
      descriptor.blocks_ofm = handle->blocksofm;
      descriptor.blocks_ifm = handle->blocksifm;
      descriptor.ofm_block = handle->ofmblock;
      descriptor.ifm_block = handle->ifmblock;
      descriptor.ofh_padded = handle->ofhp;
      descriptor.ofw_padded = handle->ofwp;
      descriptor.ofh_rb = handle->fwd_ofh_rb;
      descriptor.ofw_rb = handle->fwd_ofw_rb;
      descriptor.fm_lp_block = handle->fm_lp_block;
      descriptor.datatype_in = handle->datatype_in;
      descriptor.datatype_out = handle->datatype_out;
      descriptor.option = handle->desc.options;
      descriptor.format = (libxsmm_dnn_conv_format)(handle->buffer_format | handle->filter_format);
      /* TODO check JIT errors */
      if (libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_MIC  ||
          libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE)
      {
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
        handle->code_fwd[0].pmm = libxsmm_create_xconv_forward(&descriptor);
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NO_WEIGHT;
        handle->code_fwd[1].pmm = libxsmm_create_xconv_forward(&descriptor);
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
        handle->code_fwd[2].pmm = libxsmm_create_xconv_forward(&descriptor);
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NO_OUTPUT;
        handle->code_fwd[3].pmm = libxsmm_create_xconv_forward(&descriptor);
      } else if (libxsmm_get_target_archid() == LIBXSMM_X86_AVX2) {
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
        /* shouldn't happend */
      }
    }
    /* Backward path */
    { libxsmm_convolution_backward_descriptor descriptor;
      descriptor.ifh_padded = handle->desc.H;
      descriptor.ifw_padded = handle->desc.W;
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
      descriptor.datatype_in = handle->datatype_in;
      descriptor.datatype_out = handle->datatype_out;
      descriptor.option = handle->desc.options;
      descriptor.format = (libxsmm_dnn_conv_format)(handle->buffer_format | handle->filter_format);
      /* TODO check JIT errors */
      if ( (libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_MIC  ||
            libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE) &&
           ((handle->filter_format == LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) && (handle->buffer_format == LIBXSMM_DNN_CONV_FORMAT_LIBXSMM)) )
      {
        /* control code size */
        const unsigned int max_code_size = 20000/*16384*/;
        const unsigned int bp_each_iter_code_size = 12/*16*/;
        if ((descriptor.ofw * descriptor.kw * descriptor.ofm_block * bp_each_iter_code_size <= max_code_size)) {
          descriptor.ofw_unroll = 1;
          descriptor.unroll_kw = 1;
        } else if (descriptor.kw * descriptor.ofm_block * bp_each_iter_code_size <= max_code_size) {
          unsigned int upper_bound_ofw_rb = (max_code_size) / (descriptor.kw * descriptor.ofm_block * bp_each_iter_code_size);
          for(i = LIBXSMM_MIN(upper_bound_ofw_rb+1, 24); i >= 10; i--) {
            if(handle->ofw % i == 0) break;
          }
          if(i>=10) {
            descriptor.ofw_rb =  i;
            descriptor.ofw_unroll = 0;
            descriptor.unroll_kw = 1;
          } else {
            if(descriptor.ofw_rb*descriptor.kw * descriptor.ofm_block * bp_each_iter_code_size <= max_code_size) {
              descriptor.ofw_unroll = 0;
              descriptor.unroll_kw = 1;
            } else {
              descriptor.ofw_unroll = 0;
              descriptor.unroll_kw = 0;
            }
          }
        } else if(descriptor.ofw_rb*descriptor.kw * descriptor.ofm_block * bp_each_iter_code_size <= max_code_size) {
          descriptor.ofw_unroll = 0;
          descriptor.unroll_kw = 1;
        } else {
          descriptor.ofw_unroll = 0;
          descriptor.unroll_kw = 0;
        }
        /* NONE */
        descriptor.prefetch_output_ahead = 0;
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
        handle->code_bwd[0].pmm = libxsmm_create_xconv_backward(&descriptor);
        /*ALL*/
        descriptor.prefetch_output_ahead = 0;
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
        handle->code_bwd[1].pmm = libxsmm_create_xconv_backward(&descriptor);

        /* PEELED VERSION */
        for(i = LIBXSMM_MIN(24, handle->ofw); i > 1; i--) {
          if(handle->ofw % i == 0) break;
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
          for(i = LIBXSMM_MIN(upper_bound_ofw_rb+1, 24); i >= 10; i--) {
            if(handle->ofw % i == 0) break;
          }
          if(i>=10) {
            descriptor.ofw_rb =  i;
            descriptor.ofw_unroll = 0;
            descriptor.unroll_kw = 1;
            descriptor.unroll_kh = 1;
          } else {
            if(descriptor.ofw_rb*descriptor.kw * descriptor.kh * descriptor.ofm_block * bp_each_iter_code_size <= max_code_size) {
              descriptor.ofw_unroll = 0;
              descriptor.unroll_kw = 1;
              descriptor.unroll_kh = 1;
            } else {
              descriptor.ofw_unroll = 0;
              descriptor.unroll_kw = 0;
              descriptor.unroll_kh = 1;
            }
          }
        } else if(descriptor.ofw_rb* descriptor.kh * descriptor.kw * descriptor.ofm_block * bp_each_iter_code_size <= max_code_size) {
          descriptor.ofw_unroll = 0;
          descriptor.unroll_kw = 1;
          descriptor.unroll_kh = 1;
        } else {
          descriptor.ofw_unroll = 0;
          descriptor.unroll_kw = 0;
          descriptor.unroll_kh = 1; /* always unroll kh */
        }

        /* NONE */
        descriptor.prefetch_output_ahead = 0;
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
        handle->code_bwd[2].pmm = libxsmm_create_xconv_backward(&descriptor);
        /* NO_WEIGHT_L2 */
        descriptor.prefetch_output_ahead = 0;
        descriptor.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NO_WEIGHT_L2;
        handle->code_bwd[3].pmm = libxsmm_create_xconv_backward(&descriptor);
      } else if ((libxsmm_get_target_archid() == LIBXSMM_X86_AVX2) ||
                   ((handle->filter_format != LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) || (handle->buffer_format != LIBXSMM_DNN_CONV_FORMAT_LIBXSMM)) ) {
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
        /* shouldn't happend */
      }
    } /* End of backward */
    /* TODO weight update path */
    { libxsmm_convolution_weight_update_descriptor descriptor;
      descriptor.ifh_padded = handle->desc.H;
      descriptor.ifw_padded = handle->desc.W;
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
      descriptor.datatype_in = handle->datatype_in;
      descriptor.datatype_out = handle->datatype_out;
      descriptor.option = handle->desc.options;
      descriptor.format = (libxsmm_dnn_conv_format)(handle->buffer_format | handle->filter_format);

      /* TODO check JIT errors */
      if ( (libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_MIC  ||
            libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE) &&
           ((handle->filter_format == LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) && (handle->buffer_format == LIBXSMM_DNN_CONV_FORMAT_LIBXSMM)) )
      {
        const unsigned int wu_each_iter_code_size = 10 * (descriptor.ifm_block == 1 ? descriptor.kw : descriptor.ifm_block);
        const unsigned int wu_max_code_size = 20000;
        int upper_limit_ofw_rb = wu_max_code_size / wu_each_iter_code_size, upper_limit_ofh_rb = 0;
        descriptor.ifm_unroll = 1;

        for(i = LIBXSMM_MIN(upper_limit_ofw_rb, handle->ofw); i >= 1; i--) {
          if(handle->ofw % i == 0) break;
        }
        descriptor.ofw_rb =  i;
        upper_limit_ofh_rb = wu_max_code_size / (descriptor.ofw_rb * wu_each_iter_code_size);
        for(i = LIBXSMM_MIN(upper_limit_ofh_rb, handle->ofh); i >= 1; i--) {
          if(handle->ofh % i == 0) break;
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
      } else if ((libxsmm_get_target_archid() == LIBXSMM_X86_AVX2) ||
                   ((handle->filter_format != LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) || (handle->buffer_format != LIBXSMM_DNN_CONV_FORMAT_LIBXSMM)) ) {
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
        /* shouldn't happend */
      }
    } /* end of weight-update handle */
    {
      handle->scratch1 = libxsmm_aligned_malloc( /* populating scratch register for transpose */
        handle->blocksifm * handle->ifmblock * handle->blocksofm * handle->ofmblock * handle->desc.R * handle->desc.S * handle->fm_lp_block * libxsmm_dnn_typesize(handle->datatype_in),
        LIBXSMM_ALIGNMENT);

      handle->scratch2 = libxsmm_barrier_create(handle->desc.threads, 1);

/*#ifdef LIBXSMM_WU_TRANSPOSE_OFW_IFM*/
      handle->scratch3 = libxsmm_aligned_malloc( /* allocate raw data */
        handle->desc.N * handle->blocksifm * handle->ifmblock * handle->desc.H * handle->desc.W * handle->fm_lp_block * libxsmm_dnn_typesize(handle->datatype_in),
        LIBXSMM_ALIGNMENT);
/*#endif*/
      if (handle->ifmblock == 1) {
        handle->upd_use_thread_fil = 1;
        handle->scratch4 = libxsmm_aligned_malloc(
          handle->desc.threads * handle->blocksifm * handle->ifmblock * handle->blocksofm * handle->ofmblock
          * handle->desc.R * handle->desc.S * handle->fm_lp_block * libxsmm_dnn_typesize(handle->datatype_in),
          LIBXSMM_ALIGNMENT);
      } else {
        handle->scratch4 = 0;
        handle->upd_use_thread_fil = 0;
      }
      if ( ((libxsmm_get_target_archid() == LIBXSMM_X86_AVX2) ||
             ((handle->filter_format != LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) || (handle->buffer_format != LIBXSMM_DNN_CONV_FORMAT_LIBXSMM)) )
             && (handle->upd_use_thread_fil == 0)) {
        if ( (handle->desc.threads*2) > (handle->blocksifm*handle->blocksofm) ) {
          handle->upd_use_thread_fil = 1;
          handle->scratch4 = libxsmm_aligned_malloc(
            handle->desc.threads * handle->blocksifm * handle->ifmblock * handle->blocksofm * handle->ofmblock
            * handle->desc.R * handle->desc.S * handle->fm_lp_block * libxsmm_dnn_typesize(handle->datatype_in),
            LIBXSMM_ALIGNMENT);
        }
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
    handle->scratch1 = 0;
    handle->scratch2 = 0;
/*#ifdef LIBXSMM_WU_TRANSPOSE_OFW_IFM*/
    handle->scratch3 = 0;
/*#endif*/
  }

  return status;
}
