/******************************************************************************
** Copyright (c) 2015-2019, Intel Corporation                                **
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

#include <libxsmm_generator.h>
#include "generator_common.h"
#include "generator_convolution_common.h"
#include "generator_convolution_forward_avx512.h"
#include "generator_convolution_weight_update_avx512.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_convolution_forward_kernel( libxsmm_generated_code*                        io_generated_code,
                                                   const libxsmm_convolution_forward_descriptor*  i_conv_desc,
                                                   const char*                                    i_arch ) {
  /* add instruction set mismatch check to code, header */
  libxsmm_generator_isa_check_header( io_generated_code, i_arch );

  /* select datatype */
  if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
    if ( (strcmp(i_arch, "knl") == 0) ||
         (strcmp(i_arch, "knm") == 0) ||
         (strcmp(i_arch, "skx") == 0) ||
         (strcmp(i_arch, "clx") == 0) ) {
      libxsmm_generator_convolution_forward_avx512_kernel( io_generated_code, i_conv_desc, i_arch );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }
  } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 ) {
    if ( (strcmp(i_arch, "skx") == 0) ||
         (strcmp(i_arch, "clx") == 0) ||
         (strcmp(i_arch, "knm") == 0) ) {
      /* call actual kernel generation with revised parameters */
      libxsmm_generator_convolution_forward_avx512_kernel( io_generated_code, i_conv_desc, i_arch );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }
  } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_BF16 ) {
    if ( (strcmp(i_arch, "skx") == 0) ||
         (strcmp(i_arch, "clx") == 0)) {
      /* call actual kernel generation with revised parameters */
      libxsmm_generator_convolution_forward_avx512_kernel( io_generated_code, i_conv_desc, i_arch );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }
  } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
    if ( (strcmp(i_arch, "skx") == 0) ||
         (strcmp(i_arch, "clx") == 0) ||
         (strcmp(i_arch, "knm") == 0)   ) {
      /* call actual kernel generation with revised parameters */
      libxsmm_generator_convolution_forward_avx512_kernel( io_generated_code, i_conv_desc, i_arch );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }
  } else if ( (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32
                     && (i_conv_desc->option & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) {
    if ( (strcmp(i_arch, "skx") == 0) ||
         (strcmp(i_arch, "clx") == 0)   ) {
      libxsmm_generator_convolution_forward_avx512_kernel( io_generated_code, i_conv_desc, i_arch );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }
  } else {
    /* TODO fix this error */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* add instruction set mismatch check to code, footer */
  libxsmm_generator_isa_check_footer( io_generated_code, i_arch );
}


/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_convolution_weight_update_kernel( libxsmm_generated_code*                        io_generated_code,
                                                   const libxsmm_convolution_weight_update_descriptor*  i_conv_desc,
                                                   const char*                                          i_arch ) {
  /* add instruction set mismatch check to code, header */
  libxsmm_generator_isa_check_header( io_generated_code, i_arch );

  /* select datatype */
  if ( (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32) ||
       (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32) ||
       (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) ||
       (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_BF16)
       )
  {
    if ( (strcmp(i_arch, "knl") == 0) ||
         (strcmp(i_arch, "knm") == 0) ||
         (strcmp(i_arch, "skx") == 0) ||
         (strcmp(i_arch, "clx") == 0)   ) {
      libxsmm_generator_convolution_weight_update_avx512_kernel( io_generated_code, i_conv_desc, i_arch );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }
  } else {
    /* TODO fix this error */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* add instruction set mismatch check to code, footer */
  libxsmm_generator_isa_check_footer( io_generated_code, i_arch );
}

