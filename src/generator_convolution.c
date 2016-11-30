/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libxsmm_generator.h>
#include "generator_common.h"
#include "generator_convolution_common.h"
#include "generator_convolution_forward_avx2.h"
#include "generator_convolution_forward_avx512.h"
#include "generator_convolution_backward_avx2.h"
#include "generator_convolution_backward_avx512.h"
#include "generator_convolution_weight_update_avx2.h"
#include "generator_convolution_weight_update_avx512.h"

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_kernel( libxsmm_generated_code*                        io_generated_code,
                                                   const libxsmm_convolution_forward_descriptor*  i_conv_desc,
                                                   const char*                                    i_arch ) {
  /* add instruction set mismatch check to code, header */
  libxsmm_generator_isa_check_header( io_generated_code, i_arch );

  /* select datatype */
  if ( i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
    if ( (strcmp(i_arch, "knl") == 0) ||
         (strcmp(i_arch, "skx") == 0)    ) {
      libxsmm_generator_convolution_forward_avx512_kernel( io_generated_code, i_conv_desc, i_arch );
    } else if ( (strcmp(i_arch, "hsw") == 0) ) {
      libxsmm_generator_convolution_forward_avx2_kernel( io_generated_code, i_conv_desc, i_arch );
    } else {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }
  } else if ( i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_I32 ) {
    if ( (strcmp(i_arch, "skx") == 0) ) {
      libxsmm_generator_convolution_forward_avx512_kernel( io_generated_code, i_conv_desc, i_arch );
    } else {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }
  } else if ( (i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_I16
                     && (i_conv_desc->option & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) {
    if ( (strcmp(i_arch, "skx") == 0) ) {
      libxsmm_generator_convolution_forward_avx512_kernel( io_generated_code, i_conv_desc, i_arch );
    } else {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }
  } else if ( (i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_I32
                     && (i_conv_desc->option & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) {
    if ( (strcmp(i_arch, "skx") == 0) ) {
      libxsmm_generator_convolution_forward_avx512_kernel( io_generated_code, i_conv_desc, i_arch );
    } else {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }
  } else {
    /* TODO fix this error */
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* add instruction set mismatch check to code, footer */
  libxsmm_generator_isa_check_footer( io_generated_code, i_arch );
}

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_kernel( libxsmm_generated_code*                        io_generated_code,
                                                    const libxsmm_convolution_backward_descriptor* i_conv_desc,
                                                    const char*                                    i_arch ) {
  /* add instruction set mismatch check to code, header */
  libxsmm_generator_isa_check_header( io_generated_code, i_arch );

  /* select datatype */
  if ( i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_F32 ) {
    if ( (strcmp(i_arch, "knl") == 0) ||
         (strcmp(i_arch, "skx") == 0)    ) {
      if ( ((i_conv_desc->format & LIBXSMM_DNN_CONV_FORMAT_NHWC) > 0) ||
           ((i_conv_desc->format & LIBXSMM_DNN_CONV_FORMAT_RSCK) > 0) ) {
        libxsmm_generator_convolution_backward_avx2_kernel( io_generated_code, i_conv_desc, i_arch );
      } else {
        libxsmm_generator_convolution_backward_avx512_kernel( io_generated_code, i_conv_desc, i_arch );
      }
    } else if ( (strcmp(i_arch, "hsw") == 0) ) {
      libxsmm_generator_convolution_backward_avx2_kernel( io_generated_code, i_conv_desc, i_arch );
    } else {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }
  } else {
    /* TODO fix this error */
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* add instruction set mismatch check to code, footer */
  libxsmm_generator_isa_check_footer( io_generated_code, i_arch );
}


/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_kernel( libxsmm_generated_code*                        io_generated_code,
                                                   const libxsmm_convolution_weight_update_descriptor*  i_conv_desc,
                                                   const char*                                    i_arch ) {
  /* add instruction set mismatch check to code, header */
  libxsmm_generator_isa_check_header( io_generated_code, i_arch );

  /* select datatype */
  if ( i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
    if ( (strcmp(i_arch, "knl") == 0) ||
         (strcmp(i_arch, "skx") == 0)    ) {
      if ( ((i_conv_desc->format & LIBXSMM_DNN_CONV_FORMAT_NHWC) > 0) ||
           ((i_conv_desc->format & LIBXSMM_DNN_CONV_FORMAT_RSCK) > 0) ) {
        libxsmm_generator_convolution_weight_update_avx2_kernel( io_generated_code, i_conv_desc, i_arch );
      } else {
        libxsmm_generator_convolution_weight_update_avx512_kernel( io_generated_code, i_conv_desc, i_arch );
      }
    } else if ( (strcmp(i_arch, "hsw") == 0) ) {
      libxsmm_generator_convolution_weight_update_avx2_kernel( io_generated_code, i_conv_desc, i_arch );
    } else {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }
  } else {
    /* TODO fix this error */
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* add instruction set mismatch check to code, footer */
  libxsmm_generator_isa_check_footer( io_generated_code, i_arch );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_inlineasm( const char*                                   i_file_out,
                                                      const char*                                   i_routine_name,
                                                      const libxsmm_convolution_forward_descriptor* i_conv_desc,
                                                      const char*                                   i_arch ) {
  /* init generated code object */
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = NULL;
  l_generated_code.buffer_size = 0;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 0;
  l_generated_code.last_error = 0;

  /* add signature to code string */
  if ( i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
    libxsmm_convfunction_signature_fp32( &l_generated_code, i_routine_name );
  } else if ( i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_I32 ) {
    libxsmm_convfunction_signature_int16( &l_generated_code, i_routine_name );
  } else {
    fprintf(stderr, "LIBXSMM ERROR : inline assembly for convolutions is only supported for FP32 and int16!\n");
    return;
  }

  /* generate the actual kernel code for current description depending on the architecture */
  libxsmm_generator_convolution_forward_kernel( &l_generated_code, i_conv_desc, i_arch );

  /* close current function */
  libxsmm_close_function( &l_generated_code );

  /* check for errors during code generation */
  if ( l_generated_code.last_error != 0 ) {
    fprintf(stderr, "LIBXSMM ERROR there was an error generating code. Last known error is:\n%s\n",
      libxsmm_strerror(l_generated_code.last_error));
    return;
  }

  /* append code to source file */
  {
    FILE *const l_file_handle = fopen( i_file_out, "a" );
    if ( l_file_handle != NULL ) {
      fputs( (const char*)l_generated_code.generated_code, l_file_handle );
      fclose( l_file_handle );
    } else {
      fprintf(stderr, "LIBXSMM ERROR libxsmm_generator_conv_inlineasm could not write to into destination source file\n");
      return;
    }
  }

  /* free code memory */
  free( l_generated_code.generated_code );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_directasm( const char*                                   i_file_out,
                                                      const char*                                   i_routine_name,
                                                      const libxsmm_convolution_forward_descriptor* i_conv_desc,
                                                      const char*                                   i_arch ) {
  /* init generated code object */
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = NULL;
  l_generated_code.buffer_size = 0;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 1;
  l_generated_code.last_error = 0;

  /* check if we are not noarch */
  if ( strcmp( i_arch, "noarch" ) == 0 ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_conv_directasm: we cannot create ASM when noarch is specified!\n");
    exit(-1);
  }

  /* add signature to code string */
  if ( i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
    libxsmm_convfunction_signature_fp32( &l_generated_code, i_routine_name );
  } else if ( i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_I32 ) {
    libxsmm_convfunction_signature_int16( &l_generated_code, i_routine_name );
  } else {
    fprintf(stderr, "LIBXSMM ERROR : inline assembly for convolutions is only supported for FP32 and int16!\n");
    return;
  }

  /* generate the actual kernel code for current description depending on the architecture */
  libxsmm_generator_convolution_forward_kernel( &l_generated_code, i_conv_desc, i_arch );

  /* check for errors during code generation */
  if ( l_generated_code.last_error != 0 ) {
    fprintf(stderr, "LIBXSMM ERROR there was an error generating code. Last known error is:\n%s\n",
      libxsmm_strerror(l_generated_code.last_error));
    exit(-1);
  }

  /* append code to source file */
  {
    FILE *const l_file_handle = fopen( i_file_out, "w" );
    if ( l_file_handle != NULL ) {
      fputs( (const char*)l_generated_code.generated_code, l_file_handle );
      fclose( l_file_handle );
    } else {
      fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_conv_directasm: could not write to into destination source file!\n");
      return;
    }
  }

  /* free code memory */
  free( l_generated_code.generated_code );
}

